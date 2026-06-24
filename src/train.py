
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model import BPINN, BPINNConfig, BPINNLoss, count_parameters
    from src.utils import (
        BPINNArtifactDataset,
        checkpoint_payload,
        choose_dev_sessions,
        collect_samples,
        compute_metrics,
        ensure_dir,
        label_counts,
        load_labels,
        load_manifest,
        load_split_config,
        move_batch_to_device,
        set_seed,
        split_by_session,
        write_json,
    )
else:
    from .model import BPINN, BPINNConfig, BPINNLoss, count_parameters
    from .utils import (
        BPINNArtifactDataset,
        checkpoint_payload,
        choose_dev_sessions,
        collect_samples,
        compute_metrics,
        ensure_dir,
        label_counts,
        load_labels,
        load_manifest,
        load_split_config,
        move_batch_to_device,
        set_seed,
        split_by_session,
        write_json,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPINN on precomputed artifacts")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-session-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--no-flipout", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def evaluate(
    model: BPINN,
    loss_fn: BPINNLoss,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, dict, list, list]:
    model.eval()
    losses = []
    probabilities = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(
                batch["flow"],
                batch["depth"],
                batch["imu"],
                batch["intrinsics"],
                batch["flow_rot"],
            )
            loss, _ = loss_fn(
                output,
                batch["label"],
                batch["flow"],
                batch["bg_mask"],
                num_batches=len(loader),
            )
            losses.append(float(loss.item()))
            probabilities.extend(output.probability.detach().cpu().numpy().tolist())
            labels.extend(batch["label"].detach().cpu().numpy().tolist())

    mean_loss = float(np.mean(losses)) if losses else math.inf
    metrics = compute_metrics(labels, probabilities, threshold=threshold)
    metrics["loss"] = mean_loss
    return mean_loss, metrics, labels, probabilities

def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)

    labels = load_labels(args.labels_file, args.manifest)
    manifest_by_run = load_manifest(args.manifest)
    split_config = load_split_config(args.split_config)

    allowed_sessions = choose_dev_sessions(split_config)
    samples = collect_samples(
        args.artifacts_dir,
        labels=labels,
        manifest_by_run=manifest_by_run,
        allowed_sessions=allowed_sessions,
    )
    samples = [sample for sample in samples if sample.get("label") is not None]
    if len(samples) < 2:
        raise SystemExit("Need at least two labeled samples with precomputed BPINN artifacts")

    train_samples, val_samples, split_metadata = split_by_session(
        samples,
        val_fraction=args.val_session_fraction,
        seed=args.seed,
    )
    train_counts = label_counts(train_samples)
    val_counts = label_counts(val_samples)
    if train_counts["GENUINE"] == 0 or train_counts["ATTACK"] == 0:
        raise SystemExit(f"Train split must contain both classes, got {train_counts}")
    if val_counts["GENUINE"] == 0 or val_counts["ATTACK"] == 0:
        raise SystemExit(f"Validation split must contain both classes, got {val_counts}")

    split_metadata["train_label_counts"] = train_counts
    split_metadata["val_label_counts"] = val_counts
    write_json(output_dir / "split_metadata.json", split_metadata)

    train_loader = DataLoader(
        BPINNArtifactDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        BPINNArtifactDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    config = BPINNConfig(use_flipout=not args.no_flipout)
    model = BPINN(config).to(device)
    loss_fn = BPINNLoss(kl_weight=args.kl_weight, physics_weight=args.physics_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}")
    print(f"Samples: train={len(train_samples)} {train_counts}, val={len(val_samples)} {val_counts}")
    print(f"Parameters: {count_parameters(model):,}")

    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_probabilities = []
        train_labels = []
        physics_losses = []
        skipped = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            output = model(
                batch["flow"],
                batch["depth"],
                batch["imu"],
                batch["intrinsics"],
                batch["flow_rot"],
            )
            loss, components = loss_fn(
                output,
                batch["label"],
                batch["flow"],
                batch["bg_mask"],
                num_batches=len(train_loader),
            )

            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            physics_losses.append(float(components["physics"]))
            train_probabilities.extend(output.probability.detach().cpu().numpy().tolist())
            train_labels.extend(batch["label"].detach().cpu().numpy().tolist())

        scheduler.step()
        if not train_losses:
            raise SystemExit("All training batches were skipped due to non-finite losses")

        train_metrics = compute_metrics(train_labels, train_probabilities)
        train_metrics["loss"] = float(np.mean(train_losses))
        train_metrics["physics_loss"] = float(np.mean(physics_losses)) if physics_losses else math.nan
        train_metrics["skipped_batches"] = skipped

        _, val_metrics, _, _ = evaluate(model, loss_fn, val_loader, device)
        score = val_metrics.get("auc_roc")
        if score is None:
            score = val_metrics.get("accuracy", -math.inf)

        row = {
            "epoch": epoch + 1,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        write_json(output_dir / "training_history.json", history)

        print(
            "Epoch "
            f"{epoch + 1}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics.get('accuracy', 0.0):.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics.get('accuracy', 0.0):.4f}, "
            f"val_auc={val_metrics.get('auc_roc')}"
        )

        if score > best_score:
            best_score = float(score)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(model, optimizer, epoch + 1, val_metrics, split_metadata),
                output_dir / "bpinn_best.pt",
            )
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    final_metrics = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_completed": len(history),
        "final_val": history[-1]["val"],
    }
    torch.save(
        checkpoint_payload(model, optimizer, len(history), final_metrics, split_metadata),
        output_dir / "bpinn_final.pt",
    )
    write_json(output_dir / "training_summary.json", final_metrics)

    print(json.dumps(final_metrics, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
