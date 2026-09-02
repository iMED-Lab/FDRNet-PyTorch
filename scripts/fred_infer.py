#!/usr/bin/env python3
"""Run FRED inference on a directory of images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRED_DIR = ROOT / "FRED"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FRED deblurring inference")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def clean_state_dict(state: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state.items()}


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = image_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    sys.path.insert(0, str(FRED_DIR))
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from models.MIMOUNet import build_net

    if not torch.cuda.is_available():
        raise RuntimeError("FRED inference requires a CUDA-enabled PyTorch/MMCV environment")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device)
    state = checkpoint.get("model", checkpoint)
    model = build_net("MIMO-UNetPlus").to(device)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()

    with torch.inference_mode():
        for path in paths:
            array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            height, width = tensor.shape[-2:]
            pad_h = (-height) % 4
            pad_w = (-width) % 4
            tensor = functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
            prediction = model(tensor)[-1][..., :height, :width].clamp(0, 1)
            output = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
            destination = (output_dir / path.relative_to(input_dir)).with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output).save(destination)

    print(f"FRED processed {len(paths)} image(s); outputs: {output_dir}")


if __name__ == "__main__":
    main()
