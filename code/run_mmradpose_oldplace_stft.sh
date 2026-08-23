#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="/bigdata/users/quansj/miniforge3/envs/witwin/bin/python"
FIT_NPZ="results/mmradpose_smplx_micro_doppler/mmradpose_p6_angle0_ac07_r00_F0_149_26kp_fit_smplfit.npz"
LABEL="mmradpose_p6_angle0_ac07_r00_F0_149_26kp_fit_3200hz_duty100_oldplace_fixedrange"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HOME="${HOME:-/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export DRJIT_CACHE_DIR="${DRJIT_CACHE_DIR:-/tmp}"
export MMRADPOSE_WITWIN_BACKEND="${MMRADPOSE_WITWIN_BACKEND:-pytorch}"

"${PYTHON}" code/mmradpose_smplx_to_micro_doppler.py \
  --fit-npz "${FIT_NPZ}" \
  --label "${LABEL}" \
  --backend "${MMRADPOSE_WITWIN_BACKEND}" \
  --radar-profile single77_25fps \
  --range-bin-mode fixed \
  --doppler-transform fft \
  --visibility-mode linear \
  --overwrite
