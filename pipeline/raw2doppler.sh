#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/quansj/miniforge3/envs/witwin/bin/python}"
TRAIN_JSON="${TRAIN_JSON:-$ROOT/datasets/Train_sp120_train_minus_val6.json}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/raw2doppler_sp120}"

KIND="${1:-${DATASET_KIND:-}}"
if [[ -z "$KIND" ]]; then
  echo "Usage: $0 GT|Sim1|Sim2" >&2
  exit 2
fi

GPU_IDS="${GPU_IDS:-0 1 2}"
BACKEND="${BACKEND:-torch}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-$WORKERS_PER_GPU}"
RAW_FRAME_START="${RAW_FRAME_START:-1}"
ROI_MARGIN="${ROI_MARGIN:-0.5}"
ROI_REDUCER="${ROI_REDUCER:-mean}"
OVERWRITE="${OVERWRITE:-0}"

case "${KIND,,}" in
  gt)
    DATASET_DIR="${DATASET_DIR:-$ROOT/datasets/GT_sequences}"
    OUT_DIR="${OUT_DIR:-$ROOT/results/Quantitative_analysis/sim1vsgt/doppler_spectrum/GT}"
    RANGEMAT_CORRECTION="${RANGEMAT_CORRECTION:-on}"
    PEAKVALMAT_CORRECTION="${PEAKVALMAT_CORRECTION:-on}"
    LOG_PREFIX="GT"
    ;;
  sim1)
    DATASET_DIR="${DATASET_DIR:-$ROOT/datasets/Sim1_sequences}"
    OUT_DIR="${OUT_DIR:-$ROOT/results/Quantitative_analysis/sim1vsgt/doppler_spectrum/Sim1}"
    RANGEMAT_CORRECTION="${RANGEMAT_CORRECTION:-off}"
    PEAKVALMAT_CORRECTION="${PEAKVALMAT_CORRECTION:-off}"
    LOG_PREFIX="Sim1"
    ;;
  sim2)
    DATASET_DIR="${DATASET_DIR:-$ROOT/datasets/Sim2_sequences}"
    OUT_DIR="${OUT_DIR:-$ROOT/results/Quantitative_analysis/sim2vsgt/doppler_spectrum/Sim2}"
    RANGEMAT_CORRECTION="${RANGEMAT_CORRECTION:-off}"
    PEAKVALMAT_CORRECTION="${PEAKVALMAT_CORRECTION:-off}"
    LOG_PREFIX="Sim2"
    ;;
  *)
    echo "Unknown dataset kind '$KIND'; expected GT, Sim1, or Sim2." >&2
    exit 2
    ;;
esac

if ! [[ "$WORKERS_PER_GPU" =~ ^[0-9]+$ ]] || (( WORKERS_PER_GPU < 1 )); then
  echo "WORKERS_PER_GPU must be a positive integer; got '$WORKERS_PER_GPU'" >&2
  exit 2
fi
if ! [[ "$MAX_IN_FLIGHT" =~ ^[0-9]+$ ]] || (( MAX_IN_FLIGHT < 1 )); then
  echo "MAX_IN_FLIGHT must be a positive integer; got '$MAX_IN_FLIGHT'" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

GPU_IDS="${GPU_IDS//,/ }"
read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_IDS produced no GPU IDs: '$GPU_IDS'" >&2
  exit 1
fi

if [[ -n "${SEQUENCES:-}" ]]; then
  read -r -a SEQS <<< "${SEQUENCES//,/ }"
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
fi

if [[ "${#SEQS[@]}" -lt 1 ]]; then
  echo "No sequences selected." >&2
  exit 1
fi

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

run_slot() {
  local slot_index="$1"
  local gpu="$2"
  local total_slots="${#GPUS[@]}"
  local overwrite_arg=()
  local seq
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_arg=(--overwrite)
  fi
  log "${LOG_PREFIX}: slot=$slot_index gpu=$gpu start"
  for seq_index in "${!SEQS[@]}"; do
    if (( seq_index % total_slots != slot_index )); then
      continue
    fi
    seq="${SEQS[$seq_index]}"
    if [[ "$OVERWRITE" != "1" && -f "$OUT_DIR/${seq}.npy" ]]; then
      log "${LOG_PREFIX} seq${seq}: skip existing spectrum"
      continue
    fi
    log "${LOG_PREFIX} seq${seq}: raw echo -> Doppler spectrum on gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/code/Echo_data_processing/raw_echo_to_doppler_spectrum.py" \
      --dataset-dir "$DATASET_DIR" \
      --sequence "$seq" \
      --train "$TRAIN_JSON" \
      --out-dir "$OUT_DIR" \
      --roi-margin "$ROI_MARGIN" \
      --backend "$BACKEND" \
      --gpu-device 0 \
      --workers "$WORKERS_PER_GPU" \
      --max-in-flight "$MAX_IN_FLIGHT" \
      --raw-frame-start "$RAW_FRAME_START" \
      --rangemat-correction "$RANGEMAT_CORRECTION" \
      --peakvalmat-correction "$PEAKVALMAT_CORRECTION" \
      --roi-reducer "$ROI_REDUCER" \
      "${overwrite_arg[@]}" \
      > "$LOG_DIR/${LOG_PREFIX}_seq${seq}_raw2doppler.log" 2>&1
  done
  log "${LOG_PREFIX}: slot=$slot_index gpu=$gpu complete"
}

log "${LOG_PREFIX}: raw echo -> Doppler spectrum start dataset=$DATASET_DIR out=$OUT_DIR gpus=${GPUS[*]} corrections=$RANGEMAT_CORRECTION/$PEAKVALMAT_CORRECTION"
pids=()
for slot_index in "${!GPUS[@]}"; do
  run_slot "$slot_index" "${GPUS[$slot_index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  exit 1
fi
log "${LOG_PREFIX}: raw echo -> Doppler spectrum complete"
