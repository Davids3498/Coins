"""Grad-CAM for the distilled MobileNetV3-Large coin classifier.

Shows which pixels of a coin image drove the model's prediction, by backprop-ing
the predicted (or a chosen) class logit into the last conv block's activations
and overlaying the resulting heatmap on the image.

Model: emp_model_distil_student.pth (MobileNetV3-Large, distilled from the v6
ensemble teacher, 92.71% acc). Pipeline mirrors
emp_model_knowledge_distilation.ipynb's val_transform: Resize(256) ->
CenterCrop(224) -> ImageNet normalize.

Usage:
    python3.10 gradcam.py <image_or_dir> [--checkpoint PATH] [--data-dir PATH]
                                          [--out-dir PATH] [--target-class NAME]
                                          [--alpha 0.45]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from matplotlib import cm

DEFAULT_CHECKPOINT = '/home/david/coin/emp_model_distil_student.pth'
DEFAULT_DATA_DIR = '/home/david/coin/FOR_TRAINNING'
DEFAULT_OUT_DIR = '/home/david/coin/gradcam_out'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Must stay in sync with MERGES in prepare.py / predict.py.
MERGES = {'GORDIAN II': 'GORDIAN I'}


def build_label_names(data_dir: Path, num_classes: int):
    unique_labels = sorted({
        re.sub(r'^\d+_', '', p.name)
        for p in data_dir.iterdir()
        if p.is_dir() and (p / 'side_a').exists()
    })
    if num_classes == len(unique_labels) - len(MERGES):
        kept = [n for n in unique_labels if n not in MERGES]
        return {i: n for i, n in enumerate(kept)}
    return {i: n for i, n in enumerate(unique_labels)}


def load_student(checkpoint: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = tvm.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


class GradCAM:
    """Grad-CAM hooked on MobileNetV3's last conv block (model.features[-1])."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam, class_idx, float(probs[0, class_idx].item())


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float) -> Image.Image:
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)
    heatmap_img = Image.fromarray(heatmap).resize(pil_img.size, Image.BILINEAR)
    base = np.asarray(pil_img).astype(np.float32)
    heat = np.asarray(heatmap_img).astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def side_by_side(pil_img: Image.Image, overlay: Image.Image) -> Image.Image:
    w, h = pil_img.size
    canvas = Image.new('RGB', (w * 2 + 10, h), 'white')
    canvas.paste(pil_img, (0, 0))
    canvas.paste(overlay, (w + 10, 0))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image_path', type=Path, help='Image file or directory of images.')
    ap.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_CHECKPOINT))
    ap.add_argument('--data-dir', type=Path, default=Path(DEFAULT_DATA_DIR),
                     help='Training data root — used only to recover class names.')
    ap.add_argument('--out-dir', type=Path, default=Path(DEFAULT_OUT_DIR))
    ap.add_argument('--target-class', default=None,
                     help='Emperor name to explain (default: model\'s own top prediction).')
    ap.add_argument('--alpha', type=float, default=0.45, help='Heatmap overlay opacity.')
    ap.add_argument('--ext', default='jpg,jpeg,png')
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f'Checkpoint not found: {args.checkpoint}')
    if not args.image_path.exists():
        sys.exit(f'{args.image_path} does not exist.')

    exts = {'.' + e.strip().lower().lstrip('.') for e in args.ext.split(',')}
    if args.image_path.is_dir():
        paths = sorted(p for p in args.image_path.glob('*') if p.suffix.lower() in exts)
        if not paths:
            sys.exit(f'No images with extensions {sorted(exts)} found in {args.image_path}')
    else:
        paths = [args.image_path]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  {len(paths)} image(s)  |  checkpoint: {args.checkpoint.name}',
          file=sys.stderr)

    state = torch.load(args.checkpoint, map_location=device)
    num_classes = state['classifier.3.weight'].shape[0]
    idx_to_name = build_label_names(args.data_dir, num_classes)
    name_to_idx = {v: k for k, v in idx_to_name.items()}

    target_idx = None
    if args.target_class:
        key = args.target_class.strip().upper()
        if key not in name_to_idx:
            sys.exit(f'Unknown class {args.target_class!r}. Known: {sorted(name_to_idx)}')
        target_idx = name_to_idx[key]

    model = load_student(args.checkpoint, num_classes, device)
    cam_engine = GradCAM(model)

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    display_crop = T.Compose([T.Resize(256), T.CenterCrop(224)])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        pil_img = Image.open(path).convert('RGB')
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        img_tensor.requires_grad_(False)  # gradient only needed w.r.t. activations, not pixels

        cam, pred_idx, prob = cam_engine(img_tensor, class_idx=target_idx)
        pred_name = idx_to_name[pred_idx]

        cropped = display_crop(pil_img)
        overlay = overlay_heatmap(cropped, cam, args.alpha)
        combo = side_by_side(cropped, overlay)

        out_path = args.out_dir / f'{path.stem}_gradcam.png'
        combo.save(out_path)
        label = 'predicted' if target_idx is None else 'explaining'
        print(f'{path.name}: {label}={pred_name} (p={prob:.3f}) -> {out_path}')


if __name__ == '__main__':
    main()
