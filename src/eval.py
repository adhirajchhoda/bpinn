
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (
        BPINNArtifactDataset,
        ID_TO_LABEL,
        choose_dev_sessions,
        choose_holdout_sessions,
        collect_samples,
        compute_metrics,
        load_labels,
        load_manifest,
        load_model_from_checkpoint,
        load_split_config,
        move_batch_to_device,
        write_json,
    )
else:
    from .utils import (
        BPINNArtifactDataset,
        ID_TO_LABEL,
        choose_dev_sessions,
        choose_holdout_sessions,
        collect_samples,
        compute_metrics,
        load_labels,
        load_manifest,
        load_model_from_checkpoint,
        load_split_config,
        move_batch_to_device,
        write_json,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BPINN on precomputed artifacts")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split-config", type=Path, default=None)
    parser.add_argument("--split-metadata", type=Path, default=None)
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "dev", "holdout", "train", "val"],
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/bpinn_eval.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def load_checkpoint_split_metadata(checkpoint: Path) -> dict:
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    if isinstance(raw, dict):
        return raw.get("split_metadata") or {}
    return {}

def allowed_sessions_for_split(args: argparse.Namespace) -> Optional[set]:
    if args.split == "all":
        return None

    split_config = load_split_config(args.split_config)
    if args.split == "dev":
        sessions = choose_dev_sessions(split_config)
    elif args.split == "holdout":
        sessions = choose_holdout_sessions(split_config)
    else:
        metadata = {}
        if args.split_metadata and args.split_metadata.exists():
            metadata = json.loads(args.split_metadata.read_text())
        else:
            metadata = load_checkpoint_split_metadata(args.checkpoint)
        key = "train_sessions" if args.split == "train" else "val_sessions"
        sessions = set(metadata.get(key, []))

    if not sessions:
        raise SystemExit(f"No sessions available for split '{args.split}'")
    return set(sessions)

def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    labels = load_labels(args.labels_file, args.manifest)
    manifest_by_run = load_manifest(args.manifest)
    allowed_sessions = allowed_sessions_for_split(args)

    samples = collect_samples(
        args.artifacts_dir,
        labels=labels,
        manifest_by_run=manifest_by_run,
        allowed_sessions=allowed_sessions,
    )
    if not samples:
        raise SystemExit("No samples found for requested split and artifact directory")

    loader = DataLoader(
        BPINNArtifactDataset(samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = load_model_from_checkpoint(args.checkpoint, device)

    results = []
    labels_out = []
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            run_ids = batch["run_id"]
            session_ids = batch["session_id"]
            attack_types = batch["attack_type"]
            batch = move_batch_to_device(batch, device)

            is_auth, probs, uncertainty = model.predict(
                batch["flow"],
                batch["depth"],
                batch["imu"],
                batch["intrinsics"],
                batch["flow_rot"],
                threshold=args.threshold,
                n_samples=args.n_samples,
            )

            labels_batch = batch["label"].detach().cpu().numpy().tolist()
            probs_list = probs.detach().cpu().numpy().tolist()
            unc_list = uncertainty.detach().cpu().numpy().tolist()
            auth_list = is_auth.detach().cpu().numpy().tolist()

            for i, run_id in enumerate(run_ids):
                label_id = int(labels_batch[i])
                probability = float(probs_list[i])
                result = {
                    "run_id": run_id,
                    "verdict": "VERIFIED" if bool(auth_list[i]) else "REJECTED",
                    "reason_code": "BPINN_THRESHOLD",
                    "confidence": float(abs(probability - args.threshold)),
                    "l2_score": probability,
                    "bpinn_probability": probability,
                    "bpinn_uncertainty": float(unc_list[i]),
                    "label": ID_TO_LABEL.get(label_id),
                    "attack_type": attack_types[i] or None,
                    "session_id": session_ids[i] or None,
                }
                results.append(result)
                if label_id >= 0:
                    labels_out.append(label_id)
                    probabilities.append(probability)

    metrics = compute_metrics(labels_out, probabilities, threshold=args.threshold)
    output = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "n_samples": args.n_samples,
        "evaluated": len(results),
        "metrics": metrics,
        "results": results,
    }
    write_json(args.output, output)
    print(json.dumps({k: output[k] for k in ["split", "evaluated", "metrics"]}, indent=2))
    print(f"Wrote: {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
