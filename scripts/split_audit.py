#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    split = json.loads(args.split_config.read_text())
    dev_sessions = set(split.get("dev_sessions", []))
    holdout_sessions = set(split.get("holdout_sessions", []))

    by_split = Counter()
    by_split_label = defaultdict(Counter)
    by_split_attack_type = defaultdict(Counter)
    total = 0
    unknown_session = 0
    overlap = 0

    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        total += 1

        session_id = entry.get("session_id")
        label = entry.get("label")
        attack_type = entry.get("attack_type") or "GENUINE"
        if session_id is None:
            unknown_session += 1
            continue

        in_dev = session_id in dev_sessions
        in_holdout = session_id in holdout_sessions
        if in_dev and in_holdout:
            split_name = "OVERLAP"
            overlap += 1
        elif in_holdout:
            split_name = "holdout"
        elif in_dev:
            split_name = "dev"
        else:
            split_name = "neither"

        by_split[split_name] += 1
        by_split_label[split_name][label] += 1
        if label == "ATTACK":
            by_split_attack_type[split_name][attack_type] += 1

    print("Session split audit")
    print(f"- Split file: {args.split_config}")
    print(f"- Manifest: {args.manifest}")
    print(f"- Total manifest entries: {total}")
    if unknown_session:
        print(f"WARNING: entries missing session_id: {unknown_session}")
    if overlap:
        print(f"ERROR: sessions overlap dev/holdout: {overlap}")
    print()

    for split_name in ["dev", "holdout", "neither", "OVERLAP"]:
        if split_name not in by_split:
            continue
        print(f"{split_name}: {by_split[split_name]}")
        print(f"  labels: {dict(by_split_label[split_name])}")
        if by_split_attack_type.get(split_name):
            print(f"  attacks: {dict(by_split_attack_type[split_name])}")
        print()

    stats = split.get("statistics", {})
    if stats:
        print("Reported statistics:")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")

    return 0 if overlap == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
