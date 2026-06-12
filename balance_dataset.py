import random
from pathlib import Path
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
import torchvision.transforms.functional as TF

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def random_augment(pil_img):
    """Apply a sequence of random lightweight augmentations to a PIL image."""
    img = pil_img.copy()
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR)
    # Color jitter approximations
    if random.random() < 0.6:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.6:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.85, 1.25))
    if random.random() < 0.6:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.85, 1.25))
    # Random crop + resize to original size to add variety
    try:
        w, h = img.size
        crop_scale = random.uniform(0.85, 1.0)
        new_w = int(w * crop_scale)
        new_h = int(h * crop_scale)
        if new_w < w and new_h < h:
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)
    except Exception:
        pass
    return img


def balance_dataset(root_dir, target='max', max_generated_per_image=20, seed=42):
    """Balance dataset by creating augmented images for minority classes.

    root_dir: path to dataset with class subfolders
    target: 'max' to match the largest class, 'median' to match median class count, or int target
    max_generated_per_image: safety cap per source image
    """
    random.seed(seed)
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    classes = []
    counts = {}
    for d in sorted(root.iterdir()):
        if d.is_dir():
            cnt = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
            classes.append(d)
            counts[d.name] = cnt
    if not classes:
        print('No class directories found in', root)
        return
    vals = list(counts.values())
    if isinstance(target, int):
        target_count = int(target)
    elif target == 'median':
        import statistics
        target_count = int(statistics.median(vals))
    else:
        target_count = max(vals)

    print('Balancing to target count:', target_count)

    for d in tqdm(classes, desc='classes'):
        files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
        cur = len(files)
        if cur >= target_count:
            continue
        need = target_count - cur
        if not files:
            print('Skipping empty class', d)
            continue
        gen = 0
        i = 0
        while gen < need:
            src = random.choice(files)
            try:
                img = Image.open(src).convert('RGB')
            except Exception:
                continue
            aug = random_augment(img)
            out_name = d / f"aug_{src.stem}_{i}.jpg"
            try:
                aug.save(out_name, quality=90)
                gen += 1
            except Exception:
                pass
            i += 1
            # safety: don't produce endless multiples per single source
            if i > len(files) * max_generated_per_image:
                break
        print(f'Generated {gen} images for class {d.name} (was {cur})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Dataset root with class subfolders')
    parser.add_argument('--target', type=str, default='max', help="'max'|'median' or integer target")
    parser.add_argument('--max-generated-per-image', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    tgt = args.target
    try:
        tgt_int = int(tgt)
    except Exception:
        tgt_int = tgt
    balance_dataset(args.data_dir, target=tgt_int, max_generated_per_image=args.max_generated_per_image, seed=args.seed)
