#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/bigdata/users/quansj/miniforge3/envs/witwin/bin/python}"
TRAIN_JSON="${TRAIN_JSON:-$ROOT/datasets/Train_sp120_train_minus_val6.json}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/smpl_models/smplx_compatible}"
FIT_OUT_DIR="${FIT_OUT_DIR:-$ROOT/results/SMPL_fit/sim2_sp120_sequence_fit}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/datasets/Sim2_sequences}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/sim2_gen_sp120}"

GPU_IDS="${GPU_IDS:-0 1 2}"
GENDER="${GENDER:-male}"
TAG="${TAG:-sim2_dense_v9}"
PRESET="${PRESET:-v9}"
ITERS_SCALE="${ITERS_SCALE:-1.0}"
MAKE_GIF="${MAKE_GIF:-0}"

BACKEND="${BACKEND:-dirichlet}"
RX_ORDER="${RX_ORDER:-attachment}"
IQ_SCALE="${IQ_SCALE:-auto}"
FILE_IDX="${FILE_IDX:-0000}"
START_RADAR_FRAME="${START_RADAR_FRAME:-1}"
MAX_RADAR_FRAME="${MAX_RADAR_FRAME:-}"
INTERP="${INTERP:-cubic}"
MAX_TRIANGLE_SCATTERERS="${MAX_TRIANGLE_SCATTERERS:-512}"
INTENSITY_THRESHOLD="${INTENSITY_THRESHOLD:-1e-14}"
RESOLUTION="${RESOLUTION:-96}"
EPSILON_R="${EPSILON_R:-5.0}"
SAVE_SCATTERERS="${SAVE_SCATTERERS:-0}"

OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FIT="${OVERWRITE_FIT:-$OVERWRITE}"
OVERWRITE_ECHO="${OVERWRITE_ECHO:-$OVERWRITE}"

mkdir -p "$FIT_OUT_DIR" "$OUTPUT_ROOT" "$LOG_DIR"

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

fit_npz_path() {
  local seq="$1"
  printf '%s/seq%s_joint_labels_temporal_%s_fit.npz' "$FIT_OUT_DIR" "$seq" "$TAG"
}

fit_metrics_path() {
  local seq="$1"
  printf '%s/seq%s_joint_labels_temporal_%s_metrics.json' "$FIT_OUT_DIR" "$seq" "$TAG"
}

sequence_max_radar_frame() {
  local seq="$1"
  "$PY" - "$TRAIN_JSON" "$seq" "$MAX_RADAR_FRAME" <<'PY'
import json
import sys
from pathlib import Path

train_json = Path(sys.argv[1])
seq = str(int(sys.argv[2]))
override = sys.argv[3].strip()
if override:
    print(int(override))
    raise SystemExit(0)
labels = json.loads(train_json.read_text(encoding="utf-8"))
if seq not in labels:
    raise SystemExit(f"Sequence {seq} not found in {train_json}")
frames = []
for anns in labels[seq].values():
    for ann in anns or []:
        frames.append(int(ann["Radar_frameID"]))
if not frames:
    raise SystemExit(f"Sequence {seq} has no radar frames")
print(max(frames))
PY
}

is_fit_complete() {
  local seq="$1"
  "$PY" - "$(fit_npz_path "$seq")" "$(fit_metrics_path "$seq")" <<'PY'
import json
import math
import sys
from pathlib import Path

npz_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
if not npz_path.exists() or not metrics_path.exists():
    raise SystemExit(1)
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if int(metrics.get("n_frames", 0)) <= 0 or int(metrics.get("n_keyframes", 0)) <= 0:
    raise SystemExit(1)
mpjpe = float(metrics.get("keyframe_mpjpe_m", metrics.get("mpjpe_m", "nan")))
if not math.isfinite(mpjpe):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

is_sim2_complete() {
  local seq="$1"
  local max_frame="$2"
  "$PY" - "$OUTPUT_ROOT" "$seq" "$FILE_IDX" "$START_RADAR_FRAME" "$max_frame" <<'PY'
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
seq = str(int(sys.argv[2]))
file_idx = sys.argv[3]
start_frame = int(sys.argv[4])
max_frame = int(sys.argv[5])
num_frames = max_frame - start_frame + 1
if num_frames <= 0:
    raise SystemExit(1)
shape_bytes = 256 * 12 * 64 * 4 * 2 * 2
expected_data = num_frames * shape_bytes
expected_idx = 24 + 48 * num_frames
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

run_fit_sequence() {
  local seq="$1"
  local gpu="$2"
  local log_path="$LOG_DIR/seq${seq}_smpl_fit.log"
  if [[ "$OVERWRITE_FIT" != "1" ]] && is_fit_complete "$seq"; then
    log "seq${seq}: skip complete SMPL fit"
    return
  fi

  log "seq${seq}: fit dense SMPL sequence on gpu=$gpu"
  fit_args=()
  if [[ "$MAKE_GIF" != "1" ]]; then
    fit_args+=(--no-gif)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/code/SMPL_fit/fit_smpl_sequence.py" \
    --train-json "$TRAIN_JSON" \
    --sequence "$seq" \
    --model-root "$MODEL_ROOT" \
    --gender "$GENDER" \
    --device cuda:0 \
    --out-dir "$FIT_OUT_DIR" \
    --tag "$TAG" \
    --preset "$PRESET" \
    --iters-scale "$ITERS_SCALE" \
    "${fit_args[@]}" \
    > "$log_path" 2>&1
}

run_echo_sequence() {
  local seq="$1"
  local gpu="$2"
  local max_frame
  max_frame="$(sequence_max_radar_frame "$seq")"
  local fit_npz
  fit_npz="$(fit_npz_path "$seq")"
  local log_path="$LOG_DIR/seq${seq}_sim2_echo.log"

  if [[ ! -f "$fit_npz" ]]; then
    echo "Missing SMPL fit for seq${seq}: $fit_npz" >&2
    return 1
  fi
  if [[ "$OVERWRITE_ECHO" != "1" ]] && is_sim2_complete "$seq" "$max_frame"; then
    log "seq${seq}: skip complete Sim2 raw echo"
    return
  fi

  log "seq${seq}: generate Sim2 raw echo on gpu=$gpu max_radar_frame=$max_frame"
  sim2_args=()
  if [[ "$SAVE_SCATTERERS" == "1" ]]; then
    sim2_args+=(--save-scatterers)
  fi
  if "$PY" "$ROOT/code/sim2/simulate_smpl_mesh_witwin_echo.py" --help 2>&1 | grep -q -- '--max-triangle-scatterers'; then
    sim2_args+=(--max-triangle-scatterers "$MAX_TRIANGLE_SCATTERERS")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/code/sim2/simulate_smpl_mesh_witwin_echo.py" \
    --fit-npz "$fit_npz" \
    --sequence "$seq" \
    --output-root "$OUTPUT_ROOT" \
    --file-idx "$FILE_IDX" \
    --backend "$BACKEND" \
    --device cuda:0 \
    --rx-order "$RX_ORDER" \
    --iq-scale "$IQ_SCALE" \
    --start-radar-frame "$START_RADAR_FRAME" \
    --max-radar-frame "$max_frame" \
    --interp "$INTERP" \
    --intensity-threshold "$INTENSITY_THRESHOLD" \
    --resolution "$RESOLUTION" \
    --epsilon-r "$EPSILON_R" \
    "${sim2_args[@]}" \
    > "$log_path" 2>&1
}

run_fit_slot() {
  local slot_index="$1"
  local gpu="$2"
  local total_slots="${#GPUS[@]}"
  log "fit slot $slot_index gpu=$gpu start"
  for seq_index in "${!SEQS[@]}"; do
    if (( seq_index % total_slots != slot_index )); then
      continue
    fi
    run_fit_sequence "${SEQS[$seq_index]}" "$gpu"
  done
  log "fit slot $slot_index gpu=$gpu complete"
}

run_echo_slot() {
  local slot_index="$1"
  local gpu="$2"
  local total_slots="${#GPUS[@]}"
  log "echo slot $slot_index gpu=$gpu start"
  for seq_index in "${!SEQS[@]}"; do
    if (( seq_index % total_slots != slot_index )); then
      continue
    fi
    run_echo_sequence "${SEQS[$seq_index]}" "$gpu"
  done
  log "echo slot $slot_index gpu=$gpu complete"
}

wait_for_phase() {
  local failed=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    exit 1
  fi
}

write_fit_summary() {
  "$PY" - "$ROOT" "$FIT_OUT_DIR" "$TAG" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics_dir = Path(sys.argv[2])
tag = sys.argv[3]
sys.path.insert(0, str(root / "code" / "SMPL_fit"))
from fit_smpl_sequence import summarize_fit_metrics

summarize_fit_metrics(
    metrics_dir,
    f"seq*_joint_labels_temporal_{tag}_metrics.json",
    metrics_dir / "mpjpe_summary.csv",
    metrics_dir / "mpjpe_summary.md",
)
PY
}

log "Sim2 pipeline start: sequences=${#SEQS[@]} gpus=${GPUS[*]}"
log "Phase 1/2 SMPL fit: train_json=$TRAIN_JSON fit_out=$FIT_OUT_DIR tag=$TAG"
pids=()
for slot_index in "${!GPUS[@]}"; do
  run_fit_slot "$slot_index" "${GPUS[$slot_index]}" &
  pids+=("$!")
done
wait_for_phase "${pids[@]}"
write_fit_summary
log "Phase 1/2 SMPL fit complete"

log "Phase 2/2 Sim2 raw echo: output=$OUTPUT_ROOT"
pids=()
for slot_index in "${!GPUS[@]}"; do
  run_echo_slot "$slot_index" "${GPUS[$slot_index]}" &
  pids+=("$!")
done
wait_for_phase "${pids[@]}"
log "Sim2 pipeline complete"
