import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
import csv
try:
    from balance_dataset import balance_dataset
except Exception:
    balance_dataset = None


def build_dataloaders(data_dir, image_size=224, batch_size=32, val_split=0.2, seed=42, weighted_sampler=False, augment="basic"):
    """Build training and validation dataloaders.
    augment: 'basic' (default) or 'strong' for stronger augmentations (ColorJitter, Rotation, RandomErasing)
    """
    if augment == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_tf)
    indices = list(range(len(full_dataset)))
    labels = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=seed, stratify=labels)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=val_tf), val_idx)
    num_workers = min(8, os.cpu_count() or 1)
    if weighted_sampler:
        targets = [full_dataset.samples[i][1] for i in train_idx]
        class_counts = {}
        for t in targets:
            class_counts[t] = class_counts.get(t, 0) + 1
        weights_per_class = {c: 1.0 / class_counts[c] for c in class_counts}
        sample_weights = [weights_per_class[t] for t in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes, device, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    # optionally balance dataset by generating synthetic augmentations for minority classes
    if getattr(args, 'balance_data', False):
        if balance_dataset is None:
            print('balance_dataset module not found; ensure balance_dataset.py exists in project root')
        else:
            tgt = args.balance_target
            if tgt is None:
                tgt = 'max'
            balance_dataset(args.data_dir, target=tgt, max_generated_per_image=args.max_generated_per_image)
    train_loader, val_loader, classes = build_dataloaders(args.data_dir, args.image_size, args.batch_size, val_split=args.val_split, seed=args.seed, weighted_sampler=args.weighted_sampler, augment=args.augment)
    model = build_model(len(classes), device, pretrained=True)
    # label smoothing if requested (PyTorch >=1.10 supports label_smoothing param)
    if args.label_smoothing and args.label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    else:
        criterion = nn.CrossEntropyLoss()
    # allow weight decay for better generalization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    best_acc = 0.0
    last_path = args.output.replace('.pth', '.last.pth') if args.output.endswith('.pth') else args.output + '.last'
    if args.log_dir and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
        if args.log_dir and SummaryWriter is None:
            print('TensorBoard not available: install `tensorboard` to enable logs (conda install -c conda-forge tensorboard)')
    csv_file = open(args.csv_log, 'w', newline='') if args.csv_log else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    start_epoch = 0
    # resume
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state'])
        start_epoch = ck.get('epoch', 0)
        print('Resumed from', args.resume, 'at epoch', start_epoch)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        accumulation_steps = max(1, args.accumulation_steps)
        for step, (images, labels) in enumerate(tqdm(train_loader, desc='train')):
            images = images.to(device)
            labels = labels.to(device)
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels) / accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
                loss.backward()
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            running_loss += loss.item() * images.size(0) * accumulation_steps
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        # val
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='val'):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += images.size(0)
        val_loss = v_loss / v_total if v_total > 0 else 0.0
        val_acc = v_correct / v_total if v_total > 0 else 0.0
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'classes': classes}, last_path)
        if writer:
            writer.add_scalar('loss/train', train_loss, epoch + 1)
            writer.add_scalar('loss/val', val_loss, epoch + 1)
            writer.add_scalar('acc/train', train_acc, epoch + 1)
            writer.add_scalar('acc/val', val_acc, epoch + 1)
        if csv_writer:
            csv_writer.writerow([epoch + 1, f'{train_loss:.6f}', f'{train_acc:.6f}', f'{val_loss:.6f}', f'{val_acc:.6f}'])
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'classes': classes}, args.output)
            print('Saved best', args.output)
    if csv_file:
        csv_file.close()
    if writer:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='Train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default='model.pth')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weighted-sampler', action='store_true', help='use weighted random sampler for training')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--csv-log', type=str, default='metrics.csv')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
    # new options added to help generalization
    parser.add_argument('--augment', type=str, choices=['basic', 'strong'], default='basic', help='augmentation strength to use for training')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing factor (0 disables)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='optimizer weight decay for regularization')
    parser.add_argument('--balance-data', action='store_true', help='generate synthetic images to balance classes before training')
    parser.add_argument('--balance-target', type=str, default=None, help="'max'|'median' or integer target for balancing")
    parser.add_argument('--max-generated-per-image', type=int, default=20, help='safety cap for generated images per source image')
    args = parser.parse_args()
    train(args)
