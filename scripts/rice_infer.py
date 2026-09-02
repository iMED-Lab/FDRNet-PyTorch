#!/usr/bin/env python3
"""Run RICE inference on a directory of images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RICE_DIR = ROOT / "RICE"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RICE illumination enhancement")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--illumination-dir", type=Path)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def clean_state_dict(state: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state.items()}


def save_tensor(tensor, destination: Path) -> None:
    import numpy as np
    from PIL import Image

    array = (tensor[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype("uint8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(destination)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    sys.path.insert(0, str(RICE_DIR))
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from model import Network

    if not torch.cuda.is_available():
        raise RuntimeError("RICE inference requires a CUDA-enabled PyTorch environment")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device)
    state = checkpoint.get("model", checkpoint)
    model = Network(stage=args.stage).to(device)
    missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=False)
    missing = [key for key in missing if key.startswith("enhance.")]
    unexpected = [key for key in unexpected if not key.startswith("_criterion")]
    if missing or unexpected:
        raise RuntimeError(f"Incompatible checkpoint; missing={missing}, unexpected={unexpected}")
    model.eval()

    illumination_dir = args.illumination_dir.resolve() if args.illumination_dir else None
    with torch.inference_mode():
        for path in paths:
            array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            height, width = tensor.shape[-2:]
            tensor = functional.pad(tensor, (0, width % 2, 0, height % 2), mode="replicate")
            illumination_list, result_list, _, _ = model(tensor)
            relative = path.relative_to(input_dir).with_suffix(".png")
            save_tensor(result_list[-1][..., :height, :width], output_dir / relative)
            if illumination_dir:
                save_tensor(illumination_list[-1][..., :height, :width], illumination_dir / relative)

    print(f"RICE processed {len(paths)} image(s); outputs: {output_dir}")


if __name__ == "__main__":
    main()
