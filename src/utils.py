
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import BPINN, BPINNConfig

LABEL_TO_ID = {"ATTACK": 0, "GENUINE": 1}
ID_TO_LABEL = {0: "ATTACK", 1: "GENUINE"}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_labels(labels_file: Optional[Path], manifest: Optional[Path]) -> Dict[str, int]:

    labels: Dict[str, int] = {}

    if manifest and manifest.exists():
        for row in read_jsonl(manifest):
            run_id = row.get("run_id")
            label = row.get("label")
            if run_id and label in LABEL_TO_ID:
                labels[run_id] = LABEL_TO_ID[label]

    if labels_file and labels_file.exists():
        with labels_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = row.get("run_id")
                label = row.get("label")
                if run_id and label in LABEL_TO_ID:
                    labels[run_id] = LABEL_TO_ID[label]

    return labels

def load_manifest(manifest: Optional[Path]) -> Dict[str, dict]:
    if not manifest or not manifest.exists():
        return {}
    return {row["run_id"]: row for row in read_jsonl(manifest) if row.get("run_id")}

def load_split_config(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())

def session_id_for_run(run_id: str, manifest_row: Optional[dict] = None) -> str:
    if manifest_row and manifest_row.get("session_id"):
        return str(manifest_row["session_id"])
    if "_" in run_id:
        return run_id.split("_")[0]
    return run_id[:8]

def required_artifacts_present(clip_dir: Path) -> bool:
    return (
        (clip_dir / "flow_visual.npy").exists()
        and (clip_dir / "depth.npy").exists()
        and (clip_dir / "imu.npy").exists()
        and (clip_dir / "flow_rot.npy").exists()
        and (clip_dir / "bg_mask.npy").exists()
    )

def collect_samples(
    artifacts_dir: Path,
    labels: Optional[Dict[str, int]] = None,
    manifest_by_run: Optional[Dict[str, dict]] = None,
    allowed_sessions: Optional[Iterable[str]] = None,
) -> List[dict]:

    manifest_by_run = manifest_by_run or {}
    allowed = set(allowed_sessions) if allowed_sessions is not None else None
    samples = []

    for clip_dir in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        run_id = clip_dir.name
        if not required_artifacts_present(clip_dir):
            continue

        manifest_row = manifest_by_run.get(run_id, {})
        session_id = session_id_for_run(run_id, manifest_row)
        if allowed is not None and session_id not in allowed:
            continue

        label = labels.get(run_id) if labels else None
        if label is None and manifest_row.get("label") in LABEL_TO_ID:
            label = LABEL_TO_ID[manifest_row["label"]]

        samples.append(
            {
                "run_id": run_id,
                "artifact_dir": str(clip_dir),
                "label": label,
                "session_id": session_id,
                "attack_type": manifest_row.get("attack_type"),
            }
        )

    return samples

def choose_dev_sessions(split_config: dict) -> Optional[set]:
    sessions = split_config.get("dev_sessions")
    return set(sessions) if sessions else None

def choose_holdout_sessions(split_config: dict) -> Optional[set]:
    sessions = split_config.get("holdout_sessions")
    return set(sessions) if sessions else None

def split_by_session(
    samples: Sequence[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], dict]:

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    sessions: Dict[str, List[dict]] = {}
    for sample in samples:
        sessions.setdefault(sample["session_id"], []).append(sample)

    session_ids = sorted(sessions)
    rng = random.Random(seed)
    rng.shuffle(session_ids)

    n_val = max(1, int(round(len(session_ids) * val_fraction)))
    n_val = min(n_val, max(1, len(session_ids) - 1))

    def has_label(session_id: str, label: int) -> bool:
        return any(sample.get("label") == label for sample in sessions[session_id])

    val_sessions = set()
    attack_sessions = [sid for sid in session_ids if has_label(sid, LABEL_TO_ID["ATTACK"])]
    genuine_sessions = [sid for sid in session_ids if has_label(sid, LABEL_TO_ID["GENUINE"])]

    if attack_sessions:
        val_sessions.add(attack_sessions[0])
    if genuine_sessions:
        val_sessions.add(genuine_sessions[0])

    for session_id in session_ids:
        if len(val_sessions) >= n_val:
            break
        val_sessions.add(session_id)

    train_sessions = set(session_ids) - val_sessions
    if not train_sessions and len(session_ids) > 1:
        moved = sorted(val_sessions)[0]
        val_sessions.remove(moved)
        train_sessions.add(moved)

    train = [sample for sample in samples if sample["session_id"] in train_sessions]
    val = [sample for sample in samples if sample["session_id"] in val_sessions]
    metadata = {
        "split_type": "session_grouped",
        "seed": seed,
        "val_fraction": val_fraction,
        "train_sessions": sorted(train_sessions),
        "val_sessions": sorted(val_sessions),
        "train_samples": len(train),
        "val_samples": len(val),
    }
    return train, val, metadata

def _as_chw_flow(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 2:
        array = array.transpose(2, 0, 1)
    if array.ndim != 3 or array.shape[0] != 2:
        raise ValueError(f"Expected flow shape [2,H,W] or [H,W,2], got {array.shape}")
    return array.astype(np.float32)

def _as_hw_depth(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected depth shape [H,W] or [1,H,W], got {array.shape}")
    return array.astype(np.float32)

def _as_imu(array: np.ndarray, target_length: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected IMU shape [6,T] or [T,6], got {array.shape}")
    if array.shape[0] != 6 and array.shape[1] == 6:
        array = array.T
    if array.shape[0] != 6:
        raise ValueError(f"Expected six IMU channels, got {array.shape}")
    array = array.astype(np.float32)
    if array.shape[1] == target_length:
        return array

    old_x = np.linspace(0.0, 1.0, array.shape[1])
    new_x = np.linspace(0.0, 1.0, target_length)
    out = np.zeros((6, target_length), dtype=np.float32)
    for channel in range(6):
        out[channel] = np.interp(new_x, old_x, array[channel])
    return out

def _load_intrinsics(path: Path, height: int, width: int) -> np.ndarray:
    if not path.exists():
        return np.array([500.0, 500.0, width / 2.0, height / 2.0], dtype=np.float32)

    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        values = [raw.get("fx"), raw.get("fy"), raw.get("cx"), raw.get("cy")]
    else:
        values = list(raw)
    if len(values) != 4 or any(value is None for value in values):
        raise ValueError(f"Invalid intrinsics in {path}")
    return np.array(values, dtype=np.float32)

class BPINNArtifactDataset(Dataset):

    def __init__(self, samples: Sequence[dict], imu_length: int = 200) -> None:
        self.samples = list(samples)
        self.imu_length = imu_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        clip_dir = Path(sample["artifact_dir"])

        flow = _as_chw_flow(np.load(clip_dir / "flow_visual.npy"))
        depth = _as_hw_depth(np.load(clip_dir / "depth.npy"))
        imu = _as_imu(np.load(clip_dir / "imu.npy"), self.imu_length)
        flow_rot = _as_chw_flow(np.load(clip_dir / "flow_rot.npy"))
        bg_mask = _as_hw_depth(np.load(clip_dir / "bg_mask.npy"))
        intrinsics = _load_intrinsics(clip_dir / "intrinsics.json", *depth.shape)

        flow = np.clip(np.nan_to_num(flow, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0)
        depth = np.clip(np.nan_to_num(depth, nan=1.0, posinf=10.0, neginf=0.1), 0.1, 10.0)
        imu = np.clip(np.nan_to_num(imu, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0)
        flow_rot = np.clip(
            np.nan_to_num(flow_rot, nan=0.0, posinf=100.0, neginf=-100.0),
            -100.0,
            100.0,
        )
        bg_mask = np.clip(np.nan_to_num(bg_mask, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        intrinsics = np.nan_to_num(intrinsics, nan=0.0, posinf=0.0, neginf=0.0)

        label = -1 if sample.get("label") is None else int(sample["label"])
        return {
            "flow": torch.from_numpy(flow).float(),
            "depth": torch.from_numpy(depth).float(),
            "imu": torch.from_numpy(imu).float(),
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "flow_rot": torch.from_numpy(flow_rot).float(),
            "bg_mask": torch.from_numpy(bg_mask).float(),
            "label": torch.tensor(label).long(),
            "run_id": sample["run_id"],
            "session_id": sample.get("session_id", ""),
            "attack_type": sample.get("attack_type") or "",
        }

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved

def label_counts(samples: Sequence[dict]) -> Dict[str, int]:
    counts = {"GENUINE": 0, "ATTACK": 0, "UNKNOWN": 0}
    for sample in samples:
        label = sample.get("label")
        if label == LABEL_TO_ID["GENUINE"]:
            counts["GENUINE"] += 1
        elif label == LABEL_TO_ID["ATTACK"]:
            counts["ATTACK"] += 1
        else:
            counts["UNKNOWN"] += 1
    return counts

def compute_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict:
    labels_arr = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    valid = labels_arr >= 0
    labels_arr = labels_arr[valid]
    probs = probs[valid]

    if len(labels_arr) == 0:
        return {}

    pred_genuine = probs >= threshold
    true_genuine = labels_arr == LABEL_TO_ID["GENUINE"]
    true_attack = labels_arr == LABEL_TO_ID["ATTACK"]

    false_accepts = int(np.logical_and(true_attack, pred_genuine).sum())
    false_rejects = int(np.logical_and(true_genuine, ~pred_genuine).sum())
    attack_count = int(true_attack.sum())
    genuine_count = int(true_genuine.sum())
    correct = int(
        np.logical_or(
            np.logical_and(true_genuine, pred_genuine),
            np.logical_and(true_attack, ~pred_genuine),
        ).sum()
    )

    metrics = {
        "total": int(len(labels_arr)),
        "genuine": genuine_count,
        "attacks": attack_count,
        "threshold": float(threshold),
        "accuracy": float(correct / len(labels_arr)),
        "far": float(false_accepts / attack_count) if attack_count else 0.0,
        "frr": float(false_rejects / genuine_count) if genuine_count else 0.0,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }

    if len(np.unique(labels_arr)) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            metrics["auc_roc"] = float(roc_auc_score(labels_arr, probs))
        except Exception:
            metrics["auc_roc"] = None

    return metrics

def checkpoint_payload(
    model: BPINN,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: dict,
    split_metadata: dict,
) -> dict:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "metrics": metrics,
        "split_metadata": split_metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload

def load_model_from_checkpoint(path: Path, device: torch.device) -> BPINN:
    try:
        raw = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw_config = raw.get("config") or {}
        if is_dataclass(raw_config):
            raw_config = asdict(raw_config)
        config = BPINNConfig(**raw_config) if isinstance(raw_config, dict) else BPINNConfig()
        state_dict = raw["model_state_dict"]
    else:
        config = BPINNConfig()
        state_dict = raw

    model = BPINN(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))
