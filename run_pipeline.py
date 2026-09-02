#!/usr/bin/env python3
"""Unified training and inference launcher for FDRNet."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def execute(command: list[str], dry_run: bool) -> None:
    printable = subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)
    print(f"+ {printable}")
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FDRNet training and two-stage inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train FRED, then train RICE")
    train.add_argument("--fred-data-dir", required=True)
    train.add_argument("--rice-data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--fred-epochs", type=int, default=100)
    train.add_argument("--rice-epochs", type=int, default=800)
    train.add_argument("--fred-batch-size", type=int, default=1)
    train.add_argument("--rice-batch-size", type=int, default=16)
    train.add_argument("--gpu", default="0")
    add_runtime_options(train)

    test = subparsers.add_parser("test", help="Run FRED, then RICE")
    test.add_argument("--input-dir", required=True)
    test.add_argument("--fred-checkpoint", required=True)
    test.add_argument("--rice-checkpoint", required=True)
    test.add_argument("--output-dir", required=True)
    test.add_argument("--device", default="cuda", choices=("cuda",))
    test.add_argument("--gpu", default="0")
    add_runtime_options(test)

    for name, help_text in (("fred-test", "Run only FRED"), ("rice-test", "Run only RICE")):
        stage = subparsers.add_parser(name, help=help_text)
        stage.add_argument("--input-dir", required=True)
        stage.add_argument("--checkpoint", required=True)
        stage.add_argument("--output-dir", required=True)
        stage.add_argument("--device", default="cuda", choices=("cuda",))
        stage.add_argument("--gpu", default="0")
        add_runtime_options(stage)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        output = Path(args.output_dir).resolve()
        fred_command = [
            args.python, str(SCRIPTS / "fred_train.py"),
            "--data-dir", args.fred_data_dir,
            "--output-dir", str(output / "fred"),
            "--epochs", str(args.fred_epochs),
            "--batch-size", str(args.fred_batch_size),
            "--gpu", args.gpu,
        ]
        rice_command = [
            args.python, str(SCRIPTS / "rice_train.py"),
            "--data-dir", args.rice_data_dir,
            "--output-dir", str(output / "rice"),
            "--epochs", str(args.rice_epochs),
            "--batch-size", str(args.rice_batch_size),
            "--gpu", args.gpu,
        ]
        execute(fred_command, args.dry_run)
        execute(rice_command, args.dry_run)
        return

    script = SCRIPTS / ("fred_infer.py" if args.command == "fred-test" else "rice_infer.py")
    if args.command in {"fred-test", "rice-test"}:
        execute([
            args.python, str(script),
            "--input-dir", args.input_dir,
            "--checkpoint", args.checkpoint,
            "--output-dir", args.output_dir,
            "--device", args.device,
            "--gpu", args.gpu,
        ], args.dry_run)
        return

    output = Path(args.output_dir).resolve()
    intermediate = output / "fred"
    final = output / "final"
    execute([
        args.python, str(SCRIPTS / "fred_infer.py"),
        "--input-dir", args.input_dir,
        "--checkpoint", args.fred_checkpoint,
        "--output-dir", str(intermediate),
        "--device", args.device,
        "--gpu", args.gpu,
    ], args.dry_run)
    execute([
        args.python, str(SCRIPTS / "rice_infer.py"),
        "--input-dir", str(intermediate),
        "--checkpoint", args.rice_checkpoint,
        "--output-dir", str(final),
        "--device", args.device,
        "--gpu", args.gpu,
    ], args.dry_run)
    print(f"Intermediate FRED images: {intermediate}")
    print(f"Final FDRNet images: {final}")


if __name__ == "__main__":
    main()
