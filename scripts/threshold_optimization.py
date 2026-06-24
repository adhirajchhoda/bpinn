#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class Point:
    threshold: float
    far: float
    frr: float
    accuracy: float
    attacks: int
    genuines: int
    false_accepts: int
    false_rejects: int

def load_manifest_labels(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    labels = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            run_id = entry.get("run_id")
            if run_id:
                labels[run_id] = {
                    "label": entry.get("label"),
                    "attack_type": entry.get("attack_type"),
                }
    return labels

def compute_point(rows: List[dict], threshold: float) -> Point:
    attacks = [row for row in rows if row.get("label") == "ATTACK"]
    genuines = [row for row in rows if row.get("label") == "GENUINE"]

    false_accepts = sum(1 for row in attacks if float(row["l2_score"]) >= threshold)
    false_rejects = sum(1 for row in genuines if float(row["l2_score"]) < threshold)

    far = false_accepts / len(attacks) if attacks else 0.0
    frr = false_rejects / len(genuines) if genuines else 0.0
    correct = (len(attacks) - false_accepts) + (len(genuines) - false_rejects)
    total = len(attacks) + len(genuines)

    return Point(
        threshold=float(threshold),
        far=float(far),
        frr=float(frr),
        accuracy=float(correct / total) if total else 0.0,
        attacks=len(attacks),
        genuines=len(genuines),
        false_accepts=int(false_accepts),
        false_rejects=int(false_rejects),
    )

def unwrap_results(payload: object) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise SystemExit("Input must be a JSON list or an eval JSON with a 'results' list")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="BPINN eval JSON")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest JSONL used if labels are absent")
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("outputs/bpinn_threshold_optimization.json"))
    args = parser.parse_args()

    rows = unwrap_results(json.loads(args.input.read_text()))
    labels_by_run = load_manifest_labels(args.manifest) if args.manifest and args.manifest.exists() else {}

    filtered = []
    missing_score = 0
    missing_label = 0
    for row in rows:
        run_id = row.get("run_id")
        score = row.get("l2_score", row.get("bpinn_probability"))
        if run_id is None or score is None:
            missing_score += 1
            continue

        label = row.get("label")
        if label is None and run_id in labels_by_run:
            label = labels_by_run[run_id].get("label")
        if label not in {"GENUINE", "ATTACK"}:
            missing_label += 1
            continue

        filtered.append({"run_id": run_id, "l2_score": float(score), "label": label})

    if not filtered:
        raise SystemExit("No usable rows after filtering")

    scores = sorted({row["l2_score"] for row in filtered})
    thresholds = [scores[0] - 1e-6] + scores + [scores[-1] + 1e-6]
    curve = [compute_point(filtered, threshold) for threshold in thresholds]

    feasible = [point for point in curve if point.far <= args.target_far]
    if feasible:
        chosen = min(feasible, key=lambda point: point.threshold)
    else:
        chosen = min(curve, key=lambda point: (point.far, -point.threshold))

    output = {
        "input": str(args.input),
        "target_far": args.target_far,
        "filtered_rows": len(filtered),
        "dropped_missing_score": missing_score,
        "dropped_missing_label": missing_label,
        "chosen": asdict(chosen),
        "curve": [asdict(point) for point in curve],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["chosen"], indent=2))
    print(f"Wrote: {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
