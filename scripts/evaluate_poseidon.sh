#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: evaluate_poseidon.sh CHECKPOINT DATA_ROOT [OUTPUT]}"
DATA_ROOT="${2:?Usage: evaluate_poseidon.sh CHECKPOINT DATA_ROOT [OUTPUT]}"
OUTPUT="${3:-${CHECKPOINT}/eval_poseidon.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python "${REPO_ROOT}/evaluate/evaluate_poseidon_fair.py" \
  --checkpoint "${CHECKPOINT}" \
  --data_path "${DATA_ROOT}" \
  --datasets NS-Gauss CE-RP CE-CRP CE-Gauss NS-Sines CE-KH \
  --samples "${SAMPLES:-200}" \
  --rollout_steps 20 \
  --batch_size "${BATCH_SIZE:-16}" \
  --output "${OUTPUT}"
