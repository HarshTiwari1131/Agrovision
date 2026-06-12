from pathlib import Path

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def scan_dataset(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    classes = []
    counts = {}
    total = 0
    for d in sorted(root.iterdir()):
        if d.is_dir():
            cnt = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
            classes.append(d.name)
            counts[d.name] = cnt
            total += cnt
    return classes, counts, total


def load_checkpoint(path, device='cpu'):
    import torch
    ck = torch.load(path, map_location=device)
    return ck.get('classes'), ck.get('model_state')
