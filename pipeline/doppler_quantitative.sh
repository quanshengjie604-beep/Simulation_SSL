#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/quansj/miniforge3/envs/witwin/bin/python}"
TRAIN_JSON="${TRAIN_JSON:-$ROOT/datasets/Train_sp120_train_minus_val6.json}"
FILEMETA_JSON="${FILEMETA_JSON:-$ROOT/datasets/filemeta.json}"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/Quantitative_analysis/sim1vsgt}"
GT_SPEC_DIR="${GT_SPEC_DIR:-$OUT_ROOT/doppler_spectrum/GT}"
SIM_SPEC_DIR="${SIM_SPEC_DIR:-$OUT_ROOT/doppler_spectrum/Sim1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/doppler_quantitative_sim1vsgt}"

SAMPLES_PER_SEQUENCE="${SAMPLES_PER_SEQUENCE:-10}"
WINDOW_SECONDS="${WINDOW_SECONDS:-2.5}"
FPS="${FPS:-10.0}"
SEED="${SEED:-20260715}"

mkdir -p "$GT_SPEC_DIR" "$SIM_SPEC_DIR" "$OUT_ROOT" "$LOG_DIR"

if [[ -n "${SEQUENCES:-}" ]]; then
  read -r -a SEQS <<< "${SEQUENCES//,/ }"
  SAMPLE_SEQUENCE_ARGS=(--sequences "${SEQS[@]}")
else
  mapfile -t SEQS < <("$PY" - "$TRAIN_JSON" <<'PY'
import json
import sys
from pathlib import Path

labels = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for seq in sorted(labels, key=lambda item: int(item)):
    print(seq)
PY
)
  SAMPLE_SEQUENCE_ARGS=()
fi

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

missing=0
for seq in "${SEQS[@]}"; do
  if [[ ! -f "$GT_SPEC_DIR/${seq}.npy" ]]; then
    echo "Missing GT Doppler spectrum: $GT_SPEC_DIR/${seq}.npy" >&2
    missing=1
  fi
  if [[ ! -f "$SIM_SPEC_DIR/${seq}.npy" ]]; then
    echo "Missing Sim1 Doppler spectrum: $SIM_SPEC_DIR/${seq}.npy" >&2
    missing=1
  fi
done
if [[ "$missing" != "0" ]]; then
  cat >&2 <<EOF
Generate missing spectra first, for example:
  pipeline/raw2doppler.sh GT
  pipeline/raw2doppler.sh Sim1
EOF
  exit 1
fi

log "Sampling 10 windows per sequence: window=${WINDOW_SECONDS}s output=$OUT_ROOT"
"$PY" "$ROOT/code/Quantitative_analysis/sample_doppler_spectrum_similarity.py" \
  --gt-spectrum-dir "$GT_SPEC_DIR" \
  --sim-spectrum-dir "$SIM_SPEC_DIR" \
  --train "$TRAIN_JSON" \
  --filemeta "$FILEMETA_JSON" \
  --out-dir "$OUT_ROOT" \
  --samples-per-sequence "$SAMPLES_PER_SEQUENCE" \
  --window-seconds "$WINDOW_SECONDS" \
  --fps "$FPS" \
  --seed "$SEED" \
  --ssim-clip -20 0 \
  --spectrum-clip -45 0 \
  "${SAMPLE_SEQUENCE_ARGS[@]}"

log "Doppler quantitative sampling complete"
