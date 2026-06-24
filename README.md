# BPINN

Bayesian physics-informed neural network for media physics verification. This work is patent pending.

Given a video clip with visual optical flow, monocular depth, and IMU readings, BPINN classifies whether the clip is physically consistent with the accompanying sensor stream. Bayesian Flipout layers give you epistemic uncertainty on every prediction, and a differentiable physics head constrains the predicted optical flow using camera-motion equations so that the network cannot just memorize visual statistics without grounding them in geometry.

## Architecture

Two-tower design:

**Classification tower** takes optical flow `[B,2,H,W]`, depth `[B,H,W]`, and IMU `[B,6,T]`, encodes them through CNN/Conv1D encoders, fuses via a Bayesian MLP, and outputs `P(GENUINE)`.

**Physics tower** takes only depth and IMU. A translation head predicts camera velocity `(vx, vy, vz)` and a positive relative-depth scale. An analytic flow head converts predicted translation to dense flow using the v16 interaction matrix and adds precomputed rotational flow from the gyroscope. This tower intentionally never sees `flow_visual`, which prevents it from copying the target into its own prediction.

**Training loss** combines cross entropy, Bayesian KL regularization, and Charbonnier physics loss between observed visual flow and predicted flow over a background mask.

## Artifact format

Each run expects one directory:

```text
data/artifacts/<run_id>/
  flow_visual.npy    # [2,H,W] or [H,W,2]
  depth.npy          # [H,W] or [1,H,W], relative depth
  imu.npy            # [6,T] or [T,6], accel + gyro
  flow_rot.npy       # [2,H,W] or [H,W,2], rotational flow from gyro
  bg_mask.npy        # [H,W], 1 for background pixels
  intrinsics.json    # {"fx": ..., "fy": ..., "cx": ..., "cy": ...}
```

Labels use `GENUINE=1` and `ATTACK=0`.

## Training

```bash
python -m src.train \
  --artifacts-dir data/artifacts \
  --labels-file data/labels.csv \
  --manifest data/manifest.jsonl \
  --split-config data/split_config.json \
  --output-dir outputs/run1 \
  --epochs 50 \
  --batch-size 8 \
  --physics-weight 0.1
```

Writes `bpinn_best.pt`, `bpinn_final.pt`, training history, and split metadata.

## Evaluation

```bash
python -m src.eval \
  --artifacts-dir data/artifacts \
  --checkpoint outputs/run1/bpinn_best.pt \
  --manifest data/manifest.jsonl \
  --split-config data/split_config.json \
  --split holdout \
  --threshold 0.5 \
  --n-samples 10 \
  --output outputs/bpinn_eval_holdout.json
```

Outputs per-run verdicts with `bpinn_probability`, `bpinn_uncertainty`, and aggregate accuracy/FAR/FRR/AUC when labels are available.

## Threshold optimization

Pick a decision threshold on a dev split:

```bash
python scripts/threshold_optimization.py \
  --input outputs/bpinn_eval_dev.json \
  --target-far 0.01 \
  --output outputs/bpinn_threshold_dev_far1.json
```

## HPC (Slurm)

```bash
sbatch scripts/bpinn.slurm
```

Override paths and hyperparameters with env vars (`ARTIFACT_DIR`, `LABELS_FILE`, `EPOCHS`, etc.). See the slurm script for the full list.
