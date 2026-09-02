#!/usr/bin/env python3
"""Parameterized RICE training entry point using original images."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RICE_DIR = ROOT / "RICE"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RICE illumination stage")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import RandomCrop, ToTensor

    if not torch.cuda.is_available():
        raise RuntimeError("RICE training requires a CUDA-enabled PyTorch environment")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {data_dir}")

    class OriginalImageDataset(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, index: int):
            image = Image.open(paths[index]).convert("RGB")
            if min(image.size) < args.crop_size:
                raise ValueError(f"Image is smaller than crop size {args.crop_size}: {paths[index]}")
            return ToTensor()(RandomCrop(args.crop_size)(image))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(RICE_DIR))
    from model import Network

    model = Network(stage=args.stage)
    model.enhance.in_conv.apply(model.weights_init)
    model.enhance.conv.apply(model.weights_init)
    model.enhance.out_conv.apply(model.weights_init)
    model = model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=3e-4,
    )
    loader = DataLoader(
        OriginalImageDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images in loader:
            images = images.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            fidelity_loss, smooth_loss, _ = model._loss(images)
            loss = 3.0 * fidelity_loss + smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item() * images.shape[0]

        mean_loss = epoch_loss / len(paths)
        print(f"epoch={epoch:04d} loss={mean_loss:.6f}")
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), output_dir / f"weights_{epoch}.pt")

    torch.save(model.state_dict(), output_dir / "weights_final.pt")
    print(f"RICE checkpoint: {output_dir / 'weights_final.pt'}")


if __name__ == "__main__":
    main()
