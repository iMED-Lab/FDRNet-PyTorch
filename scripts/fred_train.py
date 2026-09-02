#!/usr/bin/env python3
"""Parameterized FRED training entry point without modifying legacy code."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRED_DIR = ROOT / "FRED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FRED deblurring stage")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=25)
    parser.add_argument("--valid-freq", type=int, default=25)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lr-step", type=int, default=500)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def validate_layout(data_dir: Path) -> None:
    required = [
        data_dir / split / kind
        for split in ("train", "valid")
        for kind in ("blur", "sharp")
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing FRED data directories:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_layout(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = args.output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FRED_DIR))
    from models.MIMOUNet import build_net
    from train import _train

    train_args = argparse.Namespace(
        batch_size=args.batch_size,
        data_dir=str(args.data_dir),
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        lr_steps=list(range(args.lr_step, args.epochs + 1, args.lr_step)),
        model_save_dir=str(args.output_dir),
        num_epoch=args.epochs,
        num_worker=args.workers,
        print_freq=args.print_freq,
        result_dir=str(validation_dir),
        resume=args.resume,
        save_freq=args.save_freq,
        valid_freq=args.valid_freq,
        weight_decay=args.weight_decay,
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FRED training requires a CUDA-enabled PyTorch/MMCV environment")
    model = build_net("MIMO-UNetPlus")
    model = model.cuda()
    _train(model, train_args)


if __name__ == "__main__":
    main()
