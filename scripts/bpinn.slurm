#!/bin/bash
#SBATCH --job-name=bpinn
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/bpinn_%j.out
#SBATCH --error=logs/bpinn_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
VENV_PATH="${VENV_PATH:-${HOME}/venvs/bpinn}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_DIR}/data/artifacts}"
LABELS_FILE="${LABELS_FILE:-${REPO_DIR}/data/labels.csv}"
MANIFEST_FILE="${MANIFEST_FILE:-${REPO_DIR}/data/manifest.jsonl}"
SPLIT_CONFIG="${SPLIT_CONFIG:-${REPO_DIR}/data/split_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/hopper}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.1}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

module load gnu10
module load python

if [ ! -d "${VENV_PATH}" ]; then
    python -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
PY

python scripts/train_hopper.py \
    --artifacts-dir "${ARTIFACT_DIR}" \
    --labels-file "${LABELS_FILE}" \
    --manifest "${MANIFEST_FILE}" \
    --split-config "${SPLIT_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --physics-weight "${PHYSICS_WEIGHT}"
