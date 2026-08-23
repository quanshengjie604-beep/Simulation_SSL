#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CACHE_MODE="${CACHE_MODE:-preload}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CKPT_DIR="${CKPT_DIR:-self_supervised_pretrain/checkpoints/contrastive_a5_dw_amass_equal}"
TRACKLIST_CSV="${TRACKLIST_CSV:-data/fold_splits/DopplerWild_AMASS_equal_mixed_tracklist.csv}"
DATA_DIR="${DATA_DIR:-data/a5_mixed_unlabeled_tracks_Doppler}"

export PYTHONPATH="$(dirname "${ROOT}"):${ROOT}:${PYTHONPATH:-}"

"${PYTHON}" -m self_supervised_pretrain.main_contrastive \
  --model-name mobilenet_v2 \
  --tracklist-csv "${TRACKLIST_CSV}" \
  --data-dir "${DATA_DIR}" \
  --ckpt-dir "${CKPT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --gpu "${GPU}" \
  --cache-mode "${CACHE_MODE}" \
  --num-workers "${NUM_WORKERS}" \
  --no-persistent-workers \
  --prefetch-factor 1 \
  --cudnn-benchmark \
  --crop-seconds 1.0 \
  --bins-per-second 90 \
  --resize-doppler 256 \
  --train-overlap-ratio 0.5 \
  --uD-mean 15.589631 \
  --uD-std 8.797207 \
  --num-global-views 4 \
  --num-local-views 4 \
  --global-crop-min 0.5 \
  --global-crop-max 1.0 \
  --local-crop-min 0.3 \
  --local-crop-max 0.5 \
  --gaussian-noise-std 0.1 \
  --flip-time-prob 0.1 \
  --flip-freq-prob 0.5 \
  --time-shift-prob 0.5 \
  --time-shift-max-ratio 0.2 \
  --time-mask-prob 0.2 \
  --time-mask-max-ratio 0.05 \
  --freq-mask-prob 0.2 \
  --freq-mask-max-ratio 0.05 \
  --patch-mask-prob 0.2 \
  --patch-mask-num-min 0 \
  --patch-mask-num-max 2 \
  --patch-mask-min-time-ratio 0.02 \
  --patch-mask-max-time-ratio 0.08 \
  --patch-mask-min-freq-ratio 0.02 \
  --patch-mask-max-freq-ratio 0.08 \
  --interference-mix-prob 0.1 \
  --interference-alpha-min 0.2 \
  --interference-alpha-max 0.6 \
  --local-limb-aug-prob 0.4 \
  --local-limb-gauss-sigma 3.0 \
  --local-limb-gauss-alpha 2.0 \
  --local-limb-mask-percentile 60.0 \
  --local-limb-mix 0.85 \
  --local-limb-velnorm-smooth 5 \
  --enforce-disjoint-global-crops
