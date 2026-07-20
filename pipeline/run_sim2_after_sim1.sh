#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SESSION="${WAIT_SESSION:-replacement_p100_resume_nccl}"
SIM1_DONE_CKPT="${SIM1_DONE_CKPT:-$ROOT/work_dirs/replacement_experiment/sim1_p100/epoch_20.pth}"
PIPELINE_LOG="${PIPELINE_LOG:-$ROOT/logs/pipeline_runs/replacement_sim2_after_sim1.log}"

mkdir -p "$(dirname "$PIPELINE_LOG")"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$PIPELINE_LOG"
}

log "waiting for tmux session $WAIT_SESSION to finish before starting Sim2 Replacement"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "$SIM1_DONE_CKPT" ]]; then
  log "Sim1 p100 completion checkpoint missing: $SIM1_DONE_CKPT"
  log "not starting Sim2 Replacement"
  exit 1
fi

log "Sim1 p100 completion checkpoint found; starting Sim2 Replacement without GT baseline training"

export CONFIRM_SSD_OVERWRITE="${CONFIRM_SSD_OVERWRITE:-YES}"
export PREPARE_GT_CACHE="${PREPARE_GT_CACHE:-1}"
export RUN_GT_BASELINE="${RUN_GT_BASELINE:-0}"
export RUN_REPLACEMENT="${RUN_REPLACEMENT:-1}"
export SIM_TAG="${SIM_TAG:-sim2}"
export SIM_LABEL="${SIM_LABEL:-Sim2}"
export SIM_ROOT="${SIM_ROOT:-$ROOT/datasets/Sim2_sequences}"
export PLAN_DIR="${PLAN_DIR:-$ROOT/results/replacement_experiment/plan_sim2}"
export RATIOS="${RATIOS:-25 50 75 100}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export GPUS="${GPUS:-0,1,2}"
export NPROC="${NPROC:-3}"
export RTPOSE_DIST_TIMEOUT_SECONDS="${RTPOSE_DIST_TIMEOUT_SECONDS:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export RTPOSE_WORKERS_PER_GPU="${RTPOSE_WORKERS_PER_GPU:-4}"
export RTPOSE_PREFETCH_FACTOR="${RTPOSE_PREFETCH_FACTOR:-1}"
export RTPOSE_TEST_WORKERS_PER_GPU="${RTPOSE_TEST_WORKERS_PER_GPU:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$ROOT/pipeline/replacement_experiment.sh" 2>&1 | tee -a "$PIPELINE_LOG"
