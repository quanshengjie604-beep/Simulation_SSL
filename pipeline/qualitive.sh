#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PY:-}" ]]; then
  if [[ -x /bigdata/users/quansj/miniforge3/envs/witwin/bin/python ]]; then
    PY="/bigdata/users/quansj/miniforge3/envs/witwin/bin/python"
  elif [[ -x /home/quansj/miniforge3/envs/witwin/bin/python ]]; then
    PY="/home/quansj/miniforge3/envs/witwin/bin/python"
  else
    PY="$(command -v python3)"
  fi
fi
if [[ ! -x "$PY" ]]; then
  echo "Python executable not found or not executable: $PY" >&2
  exit 2
fi

TRAIN_JSON="${TRAIN_JSON:-$ROOT/datasets/Train_sp120_train_minus_val6.json}"
KEYPOINTS="${KEYPOINTS:-$ROOT/datasets/Keypoints_meta.txt}"
GT_ROOT="${GT_ROOT:-$ROOT/datasets/GT_sequences}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$ROOT/datasets/Sim1_sequences}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-Sim1}"
# Default sequences are present in Train_sp120_train_minus_val6.json and cover six activities:
# 1 Stand and Wave Hand; 4 Stand and Lift Leg; 8 Stand with Random Pose;
# 10 Walk; 15 Walk and Sit; 12 Walk and Wave Hand.
SEQUENCES="${SEQUENCES:-1 4 8 10 15 12}"

RADAR_POSE_OUT_DIR="${RADAR_POSE_OUT_DIR:-$ROOT/results/Qualitive_analysis/sim1vsgt/radar_pose_calibration}"
CFAR_OUT_DIR="${CFAR_OUT_DIR:-$ROOT/results/Qualitive_analysis/sim1vsgt/cfar_pose_pointcloud}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/qualitive_sim1vsgt}"

RUN_RADAR_POSE="${RUN_RADAR_POSE:-1}"
RUN_CFAR="${RUN_CFAR:-1}"
CALIB_MAX_FRAMES="${CALIB_MAX_FRAMES:-90}"
CALIB_WORKERS="${CALIB_WORKERS:-4}"
CFAR_MAX_FRAMES="${CFAR_MAX_FRAMES:-0}"
CFAR_JOBS="${CFAR_JOBS:-4}"
CFAR_X_RANGE="${CFAR_X_RANGE:-1.2 6.2}"
CFAR_Y_RANGE="${CFAR_Y_RANGE:--2.0 2.0}"
CFAR_Z_RANGE="${CFAR_Z_RANGE:--1.5 1.5}"
CFAR_ROI_Z_MARGIN="${CFAR_ROI_Z_MARGIN:-1.0}"
CFAR_LOCAL_MAX="${CFAR_LOCAL_MAX:-0}"
CFAR_PFA="${CFAR_PFA:-1e-7}"

mkdir -p "$RADAR_POSE_OUT_DIR" "$CFAR_OUT_DIR" "$LOG_DIR"

read -r -a SEQS <<< "${SEQUENCES//,/ }"
if [[ "${#SEQS[@]}" -lt 1 ]]; then
  echo "No sequences selected. Set SEQUENCES='1 4 8 10 15 12'." >&2
  exit 2
fi

if ! [[ "$CFAR_JOBS" =~ ^[0-9]+$ ]] || (( CFAR_JOBS < 1 )); then
  echo "CFAR_JOBS must be a positive integer; got '$CFAR_JOBS'" >&2
  exit 2
fi

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

frame_ids_for_sequence() {
  local seq="$1"
  "$PY" - "$TRAIN_JSON" "$seq" "$CFAR_MAX_FRAMES" <<'PY'
import json
import sys
from pathlib import Path

train_path = Path(sys.argv[1])
seq = str(int(sys.argv[2]))
max_frames = int(sys.argv[3])
labels = json.loads(train_path.read_text(encoding="utf-8"))
block = labels.get(seq)
if not isinstance(block, dict):
    sys.exit(0)

frames = []
seen = set()
for frame_key in sorted(block, key=lambda item: int(item) if str(item).isdigit() else str(item)):
    annotations = block[frame_key] or []
    if not annotations:
        continue
    radar_id = str(annotations[0].get("Radar_frameID", "")).zfill(6)
    if radar_id and radar_id not in seen:
        frames.append(radar_id)
        seen.add(radar_id)

if max_frames > 0 and len(frames) > max_frames:
    if max_frames == 1:
        frames = [frames[0]]
    else:
        indexes = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        frames = [frames[i] for i in indexes]

for frame in frames:
    print(frame)
PY
}

run_radar_pose_calibration() {
  local seq="$1"
  local gt_dir="$GT_ROOT/$seq/radar/npy_DZYX_complex"
  local candidate_dir="$CANDIDATE_ROOT/$seq/radar/npy_DZYX_complex"
  local out="$RADAR_POSE_OUT_DIR/sequence${seq}_GT_vs_${CANDIDATE_LABEL}_relative_echo_db.gif"
  local log_path="$LOG_DIR/sequence${seq}_radar_pose_calibration.log"

  if [[ ! -d "$gt_dir" ]]; then
    log "seq $seq radar-pose skip: missing GT complex dir $gt_dir"
    return 0
  fi
  if [[ ! -d "$candidate_dir" ]]; then
    log "seq $seq radar-pose skip: missing $CANDIDATE_LABEL complex dir $candidate_dir"
    return 0
  fi

  log "seq $seq radar-pose start -> $out"
  "$PY" "$ROOT/code/Qualitive_analysis/visualize_radar_pose_calibration.py" \
    --sequence "$seq" \
    --train "$TRAIN_JSON" \
    --keypoints "$KEYPOINTS" \
    --radar-dir "$gt_dir" \
    --compare-radar-dir "$candidate_dir" \
    --radar-label GT \
    --compare-label "$CANDIDATE_LABEL" \
    --out "$out" \
    --max-frames "$CALIB_MAX_FRAMES" \
    --workers "$CALIB_WORKERS" \
    > "$log_path" 2>&1
  log "seq $seq radar-pose complete"
}

run_cfar_frame() {
  local seq="$1"
  local frame="$2"
  local gt="$GT_ROOT/$seq/radar/npy_DZYX_mag_roi_f16_norm/$frame.npy"
  local candidate="$CANDIDATE_ROOT/$seq/radar/npy_DZYX_mag_roi_f16_norm/$frame.npy"
  local out_dir="$CFAR_OUT_DIR/sequence${seq}"
  local out="$out_dir/frame${frame}_${CANDIDATE_LABEL}_vs_GT.png"
  local log_path="$LOG_DIR/sequence${seq}_frame${frame}_cfar_pose_pointcloud.log"

  if [[ ! -f "$gt" ]]; then
    log "seq $seq frame $frame CFAR skip: missing GT roicache $gt"
    return 0
  fi
  if [[ ! -f "$candidate" ]]; then
    log "seq $seq frame $frame CFAR skip: missing $CANDIDATE_LABEL roicache $candidate"
    return 0
  fi

  mkdir -p "$out_dir"
  local cfar_extra_args=()
  if [[ "$CFAR_LOCAL_MAX" == "1" ]]; then
    cfar_extra_args+=(--local-max)
  fi
  "$PY" "$ROOT/code/Qualitive_analysis/visualize_cfar_pose_pointcloud.py" \
    --sequence "$seq" \
    --frame-id "$frame" \
    --train "$TRAIN_JSON" \
    --keypoints "$KEYPOINTS" \
    --gt "$gt" \
    --candidate "$candidate" \
    --candidate-label "$CANDIDATE_LABEL" \
    --x-range $CFAR_X_RANGE \
    --y-range $CFAR_Y_RANGE \
    --z-range $CFAR_Z_RANGE \
    --roi-z-margin "$CFAR_ROI_Z_MARGIN" \
    --pfa "$CFAR_PFA" \
    "${cfar_extra_args[@]}" \
    --out "$out" \
    > "$log_path" 2>&1
}

run_cfar_sequence() {
  local seq="$1"
  local frames=()
  mapfile -t frames < <(frame_ids_for_sequence "$seq")
  if [[ "${#frames[@]}" -lt 1 ]]; then
    log "seq $seq CFAR skip: no frames found in $TRAIN_JSON"
    return 0
  fi

  log "seq $seq CFAR start frames=${#frames[@]} jobs=$CFAR_JOBS"
  local pids=()
  local failed=0
  local index=0
  for frame in "${frames[@]}"; do
    run_cfar_frame "$seq" "$frame" &
    pids+=("$!")
    index=$((index + 1))
    if (( ${#pids[@]} >= CFAR_JOBS )); then
      local next=("${pids[0]}")
      if ! wait "${next[0]}"; then
        failed=1
      fi
      pids=("${pids[@]:1}")
    fi
    if (( index % 25 == 0 )); then
      log "seq $seq CFAR submitted $index/${#frames[@]}"
    fi
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "seq $seq CFAR failed; inspect $LOG_DIR" >&2
    return 1
  fi
  log "seq $seq CFAR complete"
}

log "qualitive pipeline start sequences=${SEQS[*]} train=$TRAIN_JSON candidate=$CANDIDATE_LABEL"
for seq in "${SEQS[@]}"; do
  if [[ "$RUN_RADAR_POSE" == "1" ]]; then
    run_radar_pose_calibration "$seq"
  fi
  if [[ "$RUN_CFAR" == "1" ]]; then
    run_cfar_sequence "$seq"
  fi
done
log "qualitive pipeline complete"
