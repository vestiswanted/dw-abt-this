"""
export_model.py - Convert best.pt (YOLOv5 PyTorch) → best.onnx

Run ONCE before using the aimbot:
    python export_model.py --pt lib/best.pt --out lib/best.onnx --size 640

Requires: torch, ultralytics yolov5 (or the yolov5 package).
You do NOT need a GPU for export – CPU is fine.
"""

import argparse
import sys
from pathlib import Path


def export(pt_path: str, out_path: str, img_size: int) -> None:
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch not installed.  pip install torch torchvision")
        sys.exit(1)

    pt = Path(pt_path)
    if not pt.exists():
        print(f"[ERROR] .pt file not found: {pt}")
        sys.exit(1)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model from {pt} ...")
    model = torch.hub.load(
        "ultralytics/yolov5", "custom",
        path=str(pt),
        force_reload=False,
        _verbose=False,
    )
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)
    print(f"[INFO] Exporting to {out} (opset 12, input {img_size}x{img_size}) ...")

    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"[OK] Exported → {out}")
    print("[INFO] You can now delete torch/torchvision from your runtime env if desired.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv5 .pt to ONNX")
    parser.add_argument("--pt",   default="lib/best.pt",   help="Input .pt path")
    parser.add_argument("--out",  default="lib/best.onnx", help="Output .onnx path")
    parser.add_argument("--size", default=640, type=int,   help="Input image size")
    args = parser.parse_args()

    export(args.pt, args.out, args.size)
