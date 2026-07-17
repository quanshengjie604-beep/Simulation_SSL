#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/quansj/miniforge3/envs/witwin/bin/python}"
TRAIN_JSON="${TRAIN_JSON:-$ROOT/datasets/Train_sp120_train_minus_val6.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/datasets/Sim1_sequences}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/sim1_gen_sp120}"

GPU_IDS="${GPU_IDS:-0 1 2}"
BACKEND="${BACKEND:-pytorch}"
RX_ORDER="${RX_ORDER:-rtpose_raw}"
IQ_SCALE="${IQ_SCALE:-auto}"
FILE_IDX="${FILE_IDX:-0000}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

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

label_path = Path(sys.argv[1])
labels = json.loads(label_path.read_text(encoding="utf-8"))
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

is_sim1_complete() {
  local seq="$1"
  "$PY" - "$TRAIN_JSON" "$OUTPUT_ROOT" "$seq" "$FILE_IDX" <<'PY'
import json
import sys
from pathlib import Path

train_json = Path(sys.argv[1])
output_root = Path(sys.argv[2])
seq = str(int(sys.argv[3]))
file_idx = sys.argv[4]
labels = json.loads(train_json.read_text(encoding="utf-8"))
if seq not in labels:
    raise SystemExit(1)
frames = []
for anns in labels[seq].values():
    for ann in anns or []:
        frames.append(int(ann["Radar_frameID"]))
if not frames:
    raise SystemExit(1)
max_frame = max(frames)
shape_bytes = 256 * 12 * 64 * 4 * 2 * 2
expected_data = max_frame * shape_bytes
expected_idx = 24 + 48 * max_frame
bin_dir = output_root / seq / "radar" / "bin"
for name in ("master", "slave1", "slave2", "slave3"):
    data_path = bin_dir / f"{name}_{file_idx}_data.bin"
    idx_path = bin_dir / f"{name}_{file_idx}_idx.bin"
    if not data_path.exists() or data_path.stat().st_size != expected_data:
        raise SystemExit(1)
    if not idx_path.exists() or idx_path.stat().st_size != expected_idx:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_sequence() {
  local seq="$1"
  local gpu="$2"
  local log_path="$LOG_DIR/seq${seq}_sim1_gen.log"
  if [[ "$OVERWRITE" != "1" ]] && is_sim1_complete "$seq"; then
    log "seq${seq}: skip complete Sim1 raw echo"
    return
  fi

  log "seq${seq}: generate Sim1 raw echo on gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/code/sim1_raw_echo_generation/simulate_rtpose_witwin.py" \
    --train-json "$TRAIN_JSON" \
    --sequence "$seq" \
    --output-root "$OUTPUT_ROOT" \
    --file-idx "$FILE_IDX" \
    --backend "$BACKEND" \
    --device cuda:0 \
    --rx-order "$RX_ORDER" \
    --iq-scale "$IQ_SCALE" \
    > "$log_path" 2>&1
}

run_slot() {
  local slot_index="$1"
  local gpu="$2"
  local total_slots="${#GPUS[@]}"
  log "slot $slot_index gpu=$gpu start"
  for seq_index in "${!SEQS[@]}"; do
    if (( seq_index % total_slots != slot_index )); then
      continue
    fi
    run_sequence "${SEQS[$seq_index]}" "$gpu"
  done
  log "slot $slot_index gpu=$gpu complete"
}

log "Sim1 raw echo generation start: sequences=${#SEQS[@]} gpus=${GPUS[*]} output=$OUTPUT_ROOT"
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
log "Sim1 raw echo generation complete"
