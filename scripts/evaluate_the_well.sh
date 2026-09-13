#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: evaluate_the_well.sh CHECKPOINT DATA_ROOT SUBSET [OUTPUT]}"
DATA_ROOT="${2:?Usage: evaluate_the_well.sh CHECKPOINT DATA_ROOT SUBSET [OUTPUT]}"
SUBSET="${3:?Choose the The Well subset used to train this checkpoint}"
OUTPUT="${4:-${CHECKPOINT}/eval_the_well.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python "${REPO_ROOT}/evaluate/evaluate_the_well.py" \
  --checkpoint "${CHECKPOINT}" \
  --data_root "${DATA_ROOT}" \
  --datasets "${SUBSET}" \
  --first_target_time 1 \
  --rollout_steps "${ROLLOUT_STEPS:-19}" \
  --samples_per_dataset "${SAMPLES:-200}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --num_workers "${NUM_WORKERS:-1}" \
  --output "${OUTPUT}"
