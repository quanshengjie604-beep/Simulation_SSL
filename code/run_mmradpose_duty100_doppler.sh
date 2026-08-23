#!/usr/bin/env bash
set -euo pipefail

PYTHON="/bigdata/users/quansj/miniforge3/envs/witwin/bin/python"
FIT_NPZ="results/mmradpose_smplx_micro_doppler/mmradpose_p6_angle0_ac07_r00_F0_149_26kp_fit_smplfit.npz"
LABEL="mmradpose_p6_angle0_ac07_r00_F0_149_26kp_fit_3200hz_duty100_fixedrange"

"${PYTHON}" code/mmradpose_smplx_to_micro_doppler.py \
  --fit-npz "${FIT_NPZ}" \
  --label "${LABEL}" \
  --radar-profile single77_25fps \
  --range-bin-mode fixed \
  --doppler-transform fft \
  --visibility-mode linear \
  --overwrite
