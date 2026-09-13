#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:?Usage: train_poseidon.sh DATA_ROOT [MODEL_SIZE] [RUN_NAME]}"
MODEL_SIZE="${2:-L}"
RUN_NAME="${3:-dreso_poseidon_${MODEL_SIZE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoint}"
MASTER_PORT="${MASTER_PORT:-29500}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  --nnodes=1 --nproc_per_node=1 "${REPO_ROOT}/train/train.py" \
  --config "${REPO_ROOT}/configs/train_poseidon.yaml" \
  --data_path "${DATA_ROOT}" \
  --checkpoint_path "${OUTPUT_ROOT}" \
  --project_name dreso_poseidon \
  --run_name "${RUN_NAME}" \
  --report_to tensorboard \
  --model_size "${MODEL_SIZE}" \
  --train_time_step_size 1 \
  --train_small_time_transition
