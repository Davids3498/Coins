"""Predict the emperor on every image in a directory.

Pipeline mirrors the notebooks:
  image -> resize 256 -> C-RADIO v4-H summary embedding -> linear head -> argmax

Usage:
    python predict.py <image_dir> [--checkpoint PATH] [--data-dir PATH]
                                  [--topk 3] [--batch-size 16] [--ext jpg,jpeg,png]

The label encoder is rebuilt by scanning DATA_DIR's folder structure, exactly the
way the notebook built it during training. If the checkpoint was trained with
class merges (e.g. the 51-class merged head), the same MERGES dict is applied
here so the predicted ids map back to the right names.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_DIR = '/home/david/coin/FOR_TRAINNING'
DEFAULT_CHECKPOINT = '/home/david/coin/weights/emp_model_v2_merged_head_best.pth'

# Must stay in sync with MERGES in emp_model_v2_merged.ipynb.
# Only applied when the checkpoint's output dim < original class count.
MERGES = {
    'GORDIAN II': 'GORDIAN I',
}


def build_label_names(data_dir: Path, num_classes_ckpt: int):
    """Rebuild the same idx->name mapping the training notebook used.

    Returns (idx_to_name, applied_merges_bool).
    """
    unique_labels = sorted({
        re.sub(r'^\d+_', '', p.name)
        for p in data_dir.iterdir()
        if p.is_dir() and (p / 'side_a').exists()
    })
    if not unique_labels:
        sys.exit(f'No emperor folders with side_a/ found under {data_dir}')

    if num_classes_ckpt == len(unique_labels):
        return {i: n for i, n in enumerate(unique_labels)}, False

    if num_classes_ckpt == len(unique_labels) - len(MERGES):
        kept = [n for n in unique_labels if n not in MERGES]
        return {i: n for i, n in enumerate(kept)}, True

    sys.exit(
        f'Checkpoint expects {num_classes_ckpt} classes but folder structure has '
        f'{len(unique_labels)} (and {len(unique_labels) - len(MERGES)} after merges). '
        'Update DATA_DIR or MERGES.'
    )


class ImageDirDataset(Dataset):
    def __init__(self, paths, resolution=256):
        self.paths = paths
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0)
        return img, i


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image_dir', type=Path, help='Directory of images to classify.')
    ap.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_CHECKPOINT))
    ap.add_argument('--data-dir', type=Path, default=Path(DEFAULT_DATA_DIR),
                    help='Training data root — used only to recover class names.')
    ap.add_argument('--topk', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--ext', default='jpg,jpeg,png',
                    help='Comma-separated extensions to include (case-insensitive).')
    ap.add_argument('--recursive', action='store_true',
                    help='Search image_dir recursively.')
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        sys.exit(f'{args.image_dir} is not a directory.')
    if not args.checkpoint.is_file():
        sys.exit(f'Checkpoint not found: {args.checkpoint}')

    exts = {'.' + e.strip().lower().lstrip('.') for e in args.ext.split(',')}
    globber = args.image_dir.rglob if args.recursive else args.image_dir.glob
    paths = sorted(p for p in globber('*') if p.suffix.lower() in exts and p.is_file())
    if not paths:
        sys.exit(f'No images with extensions {sorted(exts)} found in {args.image_dir}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  {len(paths)} image(s)  |  checkpoint: {args.checkpoint.name}',
          file=sys.stderr)

    # Load head first to learn feat_dim and num_classes
    state = torch.load(args.checkpoint, map_location=device)
    weight = state['weight'] if 'weight' in state else state['linear.weight']
    num_classes, feat_dim = weight.shape

    idx_to_name, merged = build_label_names(args.data_dir, num_classes)
    if merged:
        print(f'Detected merged checkpoint ({num_classes} classes) — applied MERGES: {MERGES}',
              file=sys.stderr)

    head = nn.Linear(feat_dim, num_classes).to(device)
    head.load_state_dict(state)
    head.eval()

    # Load C-RADIO v4-H frozen backbone
    print('Loading C-RADIO v4-H...', file=sys.stderr)
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model',
                           version='c-radio_v4-h', progress=False)
    radio = radio.to(device).eval()
    for p in radio.parameters():
        p.requires_grad = False

    loader = DataLoader(ImageDirDataset(paths), batch_size=args.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)

    topk = max(1, args.topk)
    print('path\t' + '\t'.join(f'top{i+1}\tprob{i+1}' for i in range(topk)))

    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                summary, _ = radio(imgs)
            logits = head(summary.float())
            probs = torch.softmax(logits, dim=1)
            top_p, top_i = probs.topk(topk, dim=1)
            top_p = top_p.cpu().numpy()
            top_i = top_i.cpu().numpy()
            for row, dataset_idx in enumerate(idxs.tolist()):
                fields = [str(paths[dataset_idx])]
                for k in range(topk):
                    fields.append(idx_to_name[int(top_i[row, k])])
                    fields.append(f'{top_p[row, k]:.4f}')
                print('\t'.join(fields))


if __name__ == '__main__':
    main()
