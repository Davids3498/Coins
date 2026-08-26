"""ONNX export + int8 quantization for a trained coin_clf student checkpoint.

Standalone: only needs a trained state_dict, not the full dataset. Pass
--data-dir to also verify int8 accuracy on the frozen holdout -- the SAME
11,559-image set evaluate.py and promote.py score against, via
coin_clf.data.build_manifest_holdout. It used to reconstruct a raw-tree 80/20
split of its own, which overlapped the training set by 5,689 images and so
reported an int8 accuracy that was not comparable to any other number here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from coin_clf.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="path to a build_model state_dict (.pth)")
    p.add_argument("--num-classes", type=int, default=51)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--data-dir", default=None,
                    help="if set, also evaluates int8 accuracy on the frozen manifest holdout")
    p.add_argument("--manifest", default=None,
                    help="splits.py manifest pinning the frozen holdout "
                         "(default: data/splits_manifest.json)")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def export_and_quantize(model, output_dir: Path, input_size: int, opset: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "coin_classifier.onnx"
    int8_path = output_dir / "coin_classifier_int8.onnx"

    dummy_input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset,
    )
    onnx.checker.check_model(str(onnx_path))
    size_fp32 = onnx_path.stat().st_size / 1e6
    print(f"Exported: {onnx_path}  ({size_fp32:.1f} MB fp32)")

    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    size_int8 = int8_path.stat().st_size / 1e6
    print(f"Quantized: {int8_path}  ({size_int8:.1f} MB int8)")
    print(f"Size reduction: {size_fp32:.1f}MB -> {size_int8:.1f}MB  ({size_fp32 / size_int8:.1f}x smaller)")

    return onnx_path, int8_path


def check_parity(model, onnx_path: Path, int8_path: Path, input_size: int) -> None:
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        pt_logits = model(dummy).numpy()

    sess_fp32 = ort.InferenceSession(str(onnx_path))
    ort_logits_fp32 = sess_fp32.run(None, {"image": dummy.numpy()})[0]

    sess_int8 = ort.InferenceSession(str(int8_path))
    ort_logits_int8 = sess_int8.run(None, {"image": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_logits - ort_logits_fp32).max())
    print(f"Max logit diff (PyTorch vs ONNX fp32, random input): {max_diff:.6f}")
    print(f"PyTorch pred   : {int(pt_logits.argmax(-1)[0])}")
    print(f"ONNX fp32 pred : {int(ort_logits_fp32.argmax(-1)[0])}")
    print(f"ONNX int8 pred : {int(ort_logits_int8.argmax(-1)[0])}")


def evaluate_int8_on_test_set(int8_path: Path, args: argparse.Namespace) -> float:
    from torch.utils.data import DataLoader

    from coin_clf.data import build_manifest_holdout

    test_ds = build_manifest_holdout(args.data_dir, args.manifest)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    sess_int8 = ort.InferenceSession(str(int8_path))
    correct, total = 0, 0
    for imgs, labels in test_loader:
        logits = sess_int8.run(None, {"image": imgs.numpy()})[0]
        correct += (logits.argmax(-1) == labels.numpy()).sum()
        total += len(labels)
    acc = correct / total
    print(f"ONNX int8 holdout accuracy: {acc:.4f}  ({correct}/{total})")
    return acc


def main() -> None:
    args = parse_args()
    model = build_model(args.num_classes, pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = Path(args.output_dir)
    onnx_path, int8_path = export_and_quantize(model, output_dir, args.input_size, args.opset)
    check_parity(model, onnx_path, int8_path, args.input_size)

    if args.data_dir:
        evaluate_int8_on_test_set(int8_path, args)


if __name__ == "__main__":
    main()
