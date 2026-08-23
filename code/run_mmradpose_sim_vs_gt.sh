#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HOME="${HOME:-/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export DRJIT_CACHE_DIR="${DRJIT_CACHE_DIR:-/tmp}"
export MMRADPOSE_WITWIN_BACKEND="${MMRADPOSE_WITWIN_BACKEND:-pytorch}"
export MMRADPOSE_TRACE_RESOLUTION="${MMRADPOSE_TRACE_RESOLUTION:-12}"
export MMRADPOSE_SPECULAR_ETA="${MMRADPOSE_SPECULAR_ETA:-0.2}"

/bigdata/users/quansj/miniforge3/envs/witwin/bin/python \
  code/mmradpose_sim_echo_compare_gt.py \
  --device cuda:0 \
  --backend "${MMRADPOSE_WITWIN_BACKEND}" \
  --trace-resolution "${MMRADPOSE_TRACE_RESOLUTION}" \
  --specular-eta "${MMRADPOSE_SPECULAR_ETA}" \
  --mesh-placement gt-fixed \
  --gt-roi-placement gt-aligned \
  --roi-mode tracked-smooth \
  --overwrite
