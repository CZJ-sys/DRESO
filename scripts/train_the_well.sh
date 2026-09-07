#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:?Usage: train_the_well.sh DATA_ROOT SUBSET [MODEL_SIZE] [RUN_NAME]}"
SUBSET="${2:?Choose well.acoustic_discontinuous, well.active_matter, well.gray_scott, or well.planetswe}"
MODEL_SIZE="${3:-L}"
RUN_NAME="${4:-dreso_${SUBSET#well.}_${MODEL_SIZE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoint}"
MASTER_PORT="${MASTER_PORT:-29501}"
ABS_POS="${ABS_POS:-false}"

case "${ABS_POS,,}" in
  1|true|yes|on) ABS_POS=true ;;
  0|false|no|off) ABS_POS=false ;;
  *) echo "ABS_POS must be true or false; got ${ABS_POS}." >&2; exit 2 ;;
esac
if [[ "${ABS_POS}" == true && "${RUN_NAME}" != *abspos* ]]; then
  RUN_NAME="${RUN_NAME}_abspos"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  --nnodes=1 --nproc_per_node=1 "${REPO_ROOT}/train/train.py" \
  --config "${REPO_ROOT}/configs/train_the_well.yaml" \
  --datasets "${SUBSET}" \
  --data_path "${DATA_ROOT}" \
  --checkpoint_path "${OUTPUT_ROOT}" \
  --project_name dreso_the_well \
  --run_name "${RUN_NAME}" \
  --report_to tensorboard \
  --model_size "${MODEL_SIZE}" \
  --use_absolute_embeddings "${ABS_POS}" \
  --num_trajectories "${NUM_SAMPLES:-2000}" \
  --num_epochs "${EPOCHS:-50}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --train_time_step_size 1 \
  --train_small_time_transition
