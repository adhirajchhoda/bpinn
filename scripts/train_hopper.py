#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.train import main as train_main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPINN on Hopper")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hopper"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    forwarded = [
        "train",
        "--artifacts-dir",
        str(args.artifacts_dir),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--physics-weight",
        str(args.physics_weight),
        "--kl-weight",
        str(args.kl_weight),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
    ]
    for flag, value in [
        ("--labels-file", args.labels_file),
        ("--manifest", args.manifest),
        ("--split-config", args.split_config),
    ]:
        if value is not None:
            forwarded.extend([flag, str(value)])

    sys.argv = forwarded
    return train_main()

if __name__ == "__main__":
    raise SystemExit(main())
