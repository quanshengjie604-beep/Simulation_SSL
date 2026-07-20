#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNSTREAM_DIR="${DOWNSTREAM_DIR:-$ROOT/code/Downstream_analysis}"
PY="${PY:-$ROOT/.venv_rtpose_cu121/bin/python}"

WORK_DIR="${WORK_DIR:-$ROOT/work_dirs/replacement_experiment/sim1_p100}"
CACHE_ROOT="${CACHE_ROOT:-/ssdtemp/users/quansj/rtpose_cache_local}"
CONFIG="${CONFIG:-$DOWNSTREAM_DIR/configs/cruw_pose/hr3d_one_hm_doppler_sp120_replacement.py}"
if [[ -z "${RESUME_FROM:-}" ]]; then
  if [[ -e "$WORK_DIR/latest.pth" ]]; then
    RESUME_FROM="$WORK_DIR/latest.pth"
  else
    RESUME_FROM="$WORK_DIR/epoch_1.pth"
  fi
fi
LOG_PATH="${LOG_PATH:-$ROOT/logs/replacement_experiment/sim1_p100_resume_nccl.log}"

mkdir -p "$(dirname "$LOG_PATH")" "$WORK_DIR"

if [[ ! -x "$PY" ]]; then
  echo "Python executable not found or not executable: $PY" >&2
  exit 2
fi

if [[ ! -f "$RESUME_FROM" ]]; then
  echo "Resume checkpoint missing: $RESUME_FROM" >&2
  exit 2
fi

(
  cd "$DOWNSTREAM_DIR"
  export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2}"
  export PYTHONPATH="$DOWNSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export RTPOSE_CACHE_ROOT="$CACHE_ROOT"
  export RTPOSE_TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
  export RTPOSE_WORK_DIR="$WORK_DIR"
  export RTPOSE_WORKERS_PER_GPU="${RTPOSE_WORKERS_PER_GPU:-4}"
  export RTPOSE_PREFETCH_FACTOR="${RTPOSE_PREFETCH_FACTOR:-1}"
  export RTPOSE_TEST_WORKERS_PER_GPU="${RTPOSE_TEST_WORKERS_PER_GPU:-2}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export RTPOSE_DIST_TIMEOUT_SECONDS="${RTPOSE_DIST_TIMEOUT_SECONDS:-3600}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"

  "$PY" -m torch.distributed.run \
    --nproc_per_node="${NPROC:-3}" \
    tools/train.py "$CONFIG" \
    --launcher pytorch \
    --work_dir "$WORK_DIR" \
    --resume_from "$RESUME_FROM"
) 2>&1 | tee -a "$LOG_PATH"
