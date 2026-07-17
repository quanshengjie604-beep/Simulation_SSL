#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNSTREAM_DIR="${DOWNSTREAM_DIR:-$ROOT/code/Downstream_analysis}"
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$ROOT/.venv_rtpose_cu121/bin/python" ]]; then
    PY="$ROOT/.venv_rtpose_cu121/bin/python"
  elif [[ -x /bigdata/users/quansj/miniforge3/envs/witwin/bin/python ]]; then
    PY="/bigdata/users/quansj/miniforge3/envs/witwin/bin/python"
  else
    PY="$(command -v python3)"
  fi
fi
if [[ ! -x "$PY" ]]; then
  echo "Python executable not found or not executable: $PY" >&2
  exit 2
fi

TRAIN_JSON="${TRAIN_JSON:-$ROOT/datasets/Train_sp120_train_minus_val6.json}"
VAL_JSON="${VAL_JSON:-$ROOT/datasets/Val_sp120_by_motion.json}"
TEST_JSON="${TEST_JSON:-$ROOT/datasets/Test_sp120_by_motion6.json}"
GT_ROOT="${GT_ROOT:-$ROOT/datasets/GT_sequences}"
SIM_ROOT="${SIM_ROOT:-$ROOT/datasets/Sim1_sequences}"
CACHE_ROOT="${CACHE_ROOT:-/ssdtemp/users/quansj/rtpose_cache_local}"
CACHE_DIR="${CACHE_DIR:-npy_DZYX_mag_roi_f16_norm}"

CONFIG="${CONFIG:-$DOWNSTREAM_DIR/configs/cruw_pose/hr3d_one_hm_doppler_sp120_replacement.py}"
WORK_BASE="${WORK_BASE:-$ROOT/work_dirs/replacement_experiment}"
RESULTS_BASE="${RESULTS_BASE:-$ROOT/results/replacement_experiment}"
PLAN_DIR="${PLAN_DIR:-$RESULTS_BASE/plan}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/replacement_experiment}"

RATIOS="${RATIOS:-25 50 75 100}"
REPLACE_BY="${REPLACE_BY:-sequence}"
SEED="${SEED:-20260715}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
GPUS="${GPUS:-0,1,2}"
NPROC="${NPROC:-3}"
COPY_JOBS="${COPY_JOBS:-8}"

PLAN_ONLY="${PLAN_ONLY:-0}"
PREPARE_GT_CACHE="${PREPARE_GT_CACHE:-1}"
RUN_GT_BASELINE="${RUN_GT_BASELINE:-1}"
RUN_REPLACEMENT="${RUN_REPLACEMENT:-1}"

mkdir -p "$WORK_BASE" "$RESULTS_BASE" "$PLAN_DIR" "$LOG_DIR"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file missing: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory missing: $path" >&2
    exit 2
  fi
}

copy_manifest() {
  local manifest="$1"
  local copied_log="$2"
  : > "$copied_log"
  if [[ ! -s "$manifest" ]]; then
    log "copy skip: empty manifest $manifest"
    return 0
  fi
  "$PY" - "$manifest" "$COPY_JOBS" "$copied_log" <<'PY'
import concurrent.futures
import shutil
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
workers = int(sys.argv[2])
copied_log = Path(sys.argv[3])

pairs = []
with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        src, dst = line.split("\t", 1)
        pairs.append((Path(src), Path(dst)))

def copy_one(pair):
    src, dst = pair
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)

with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    with copied_log.open("a", encoding="utf-8") as out:
        for dst in executor.map(copy_one, pairs, chunksize=64):
            out.write(dst + "\n")
print(f"copied {len(pairs)} files from {manifest}")
PY
}

verify_manifest() {
  local manifest="$1"
  "$PY" - "$manifest" <<'PY'
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
missing_src = []
missing_dst = []
count = 0
with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        src, dst = line.split("\t", 1)
        count += 1
        if not Path(src).exists():
            missing_src.append(src)
        if not Path(dst).exists():
            missing_dst.append(dst)

if missing_src or missing_dst:
    print(f"manifest={manifest} count={count}", file=sys.stderr)
    if missing_src:
        print("missing sources:", missing_src[:20], file=sys.stderr)
    if missing_dst:
        print("missing destinations:", missing_dst[:20], file=sys.stderr)
    raise SystemExit(1)
print(f"verified {count} files from {manifest}")
PY
}

write_plan() {
  log "writing replacement plan in $PLAN_DIR"
  local need_replacement=0
  if [[ "$RUN_REPLACEMENT" == "1" ]]; then
    need_replacement=1
  fi
  "$PY" - \
    "$TRAIN_JSON" "$VAL_JSON" "$TEST_JSON" \
    "$GT_ROOT" "$SIM_ROOT" "$CACHE_ROOT" "$CACHE_DIR" \
    "$PLAN_DIR" "$SEED" "$REPLACE_BY" "$RATIOS" "$need_replacement" <<'PY'
import json
import math
import random
import sys
from pathlib import Path

train_json, val_json, test_json = [Path(item) for item in sys.argv[1:4]]
gt_root, sim_root, cache_root = [Path(item) for item in sys.argv[4:7]]
cache_dir = sys.argv[7]
plan_dir = Path(sys.argv[8])
seed = int(sys.argv[9])
replace_by = sys.argv[10].lower()
ratios = [int(item) for item in sys.argv[11].replace(",", " ").split()]
need_replacement = bool(int(sys.argv[12]))

if replace_by not in {"sequence", "frame"}:
    raise SystemExit("-- REPLACE_BY must be sequence or frame")

label_files = {
    "train": train_json,
    "val": val_json,
    "test": test_json,
}

def sorted_items(block):
    return sorted(block.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))

def frames_from_json(path):
    labels = json.loads(path.read_text(encoding="utf-8"))
    out = []
    seen = set()
    for seq in sorted(labels, key=lambda item: int(item)):
        block = labels[seq]
        if not isinstance(block, dict):
            continue
        for frame_key, annotations in sorted_items(block):
            annotations = annotations or []
            if not annotations:
                continue
            radar_id = str(annotations[0].get("Radar_frameID", "")).zfill(6)
            key = (str(int(seq)), radar_id)
            if radar_id and key not in seen:
                out.append((str(int(seq)), radar_id, str(frame_key)))
                seen.add(key)
    return out

split_frames = {split: frames_from_json(path) for split, path in label_files.items()}
all_gt_keys = sorted(
    {(seq, radar_id) for frames in split_frames.values() for seq, radar_id, _frame_key in frames},
    key=lambda item: (int(item[0]), int(item[1])),
)

def cache_dst(seq, radar_id):
    return cache_root / "sequences" / seq / "radar" / cache_dir / f"{radar_id}.npy"

def gt_src(seq, radar_id):
    return gt_root / seq / "radar" / cache_dir / f"{radar_id}.npy"

def sim_src(seq, radar_id):
    return sim_root / seq / "radar" / cache_dir / f"{radar_id}.npy"

def write_manifest(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for src, dst in pairs:
            f.write(f"{src}\t{dst}\n")

gt_pairs = [(gt_src(seq, radar_id), cache_dst(seq, radar_id)) for seq, radar_id in all_gt_keys]
missing_gt = [str(src) for src, _dst in gt_pairs if not src.exists()]
if missing_gt:
    raise SystemExit("Missing GT ROI cache files:\n" + "\n".join(missing_gt[:50]))
write_manifest(plan_dir / "gt_union_manifest.tsv", gt_pairs)

train_keys = [(seq, radar_id) for seq, radar_id, _frame_key in split_frames["train"]]
if replace_by == "sequence":
    units = sorted({seq for seq, _radar_id in train_keys}, key=lambda item: int(item))
    missing_units = []
    for seq in units:
        seq_keys = [radar_id for s, radar_id in train_keys if s == seq]
        if not all(sim_src(seq, radar_id).exists() for radar_id in seq_keys):
            missing_units.append(seq)
    if missing_units and need_replacement:
        raise SystemExit("Missing complete Sim1 ROI cache for train sequences:\n" + " ".join(missing_units))
else:
    units = sorted(train_keys, key=lambda item: (int(item[0]), int(item[1])))
    missing_frames = [f"{seq}/{radar_id}" for seq, radar_id in units if not sim_src(seq, radar_id).exists()]
    if missing_frames and need_replacement:
        raise SystemExit("Missing Sim1 ROI cache for train frames:\n" + "\n".join(missing_frames[:50]))

rng = random.Random(seed)
order = list(units)
rng.shuffle(order)
plan = {
    "seed": seed,
    "replace_by": replace_by,
    "ratios": {},
    "label_files": {split: str(path) for split, path in label_files.items()},
    "gt_root": str(gt_root),
    "sim_root": str(sim_root),
    "cache_root": str(cache_root),
    "cache_dir": cache_dir,
    "split_frame_counts": {split: len(frames) for split, frames in split_frames.items()},
    "gt_union_frames": len(all_gt_keys),
    "num_units": len(units),
    "need_replacement": need_replacement,
}
if replace_by == "sequence":
    plan["unit_order"] = order
    plan["missing_sim1_units"] = missing_units
else:
    plan["unit_order"] = [f"{seq}/{radar_id}" for seq, radar_id in order]
    plan["missing_sim1_frames"] = missing_frames

if (replace_by == "sequence" and missing_units) or (replace_by == "frame" and missing_frames):
    (plan_dir / "missing_sim1.txt").write_text(
        "\n".join(missing_units if replace_by == "sequence" else missing_frames) + "\n",
        encoding="utf-8",
    )
    if not need_replacement:
        (plan_dir / "replacement_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(plan, indent=2))
        raise SystemExit(0)

previous = set()
for ratio in ratios:
    count = math.ceil(len(order) * ratio / 100.0)
    selected = order[:count]
    selected_set = set(selected)
    delta = [unit for unit in selected if unit not in previous]
    previous = selected_set

    selected_keys = []
    if replace_by == "sequence":
        selected_sequences = set(selected)
        selected_keys = [(seq, radar_id) for seq, radar_id in train_keys if seq in selected_sequences]
        selected_label = selected
        delta_label = delta
    else:
        selected_keys = selected
        selected_label = [f"{seq}/{radar_id}" for seq, radar_id in selected]
        delta_label = [f"{seq}/{radar_id}" for seq, radar_id in delta]

    pairs = [(sim_src(seq, radar_id), cache_dst(seq, radar_id)) for seq, radar_id in selected_keys]
    write_manifest(plan_dir / f"sim1_p{ratio}_manifest.tsv", pairs)
    (plan_dir / f"sim1_p{ratio}_selected_units.txt").write_text("\n".join(map(str, selected_label)) + "\n", encoding="utf-8")
    (plan_dir / f"sim1_p{ratio}_delta_units.txt").write_text("\n".join(map(str, delta_label)) + "\n", encoding="utf-8")
    plan["ratios"][str(ratio)] = {
        "ratio_percent": ratio,
        "num_selected_units": len(selected),
        "num_delta_units": len(delta),
        "num_replaced_train_frames": len(selected_keys),
        "manifest": str(plan_dir / f"sim1_p{ratio}_manifest.tsv"),
    }

(plan_dir / "replacement_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
print(json.dumps(plan, indent=2))
PY
}

prepare_gt_cache() {
  if [[ "${CONFIRM_SSD_OVERWRITE:-}" != "YES" ]]; then
    echo "[safety] This step removes and recreates all contents under $CACHE_ROOT." >&2
    echo "[safety] Re-run with CONFIRM_SSD_OVERWRITE=YES to actually overwrite SSD cache." >&2
    exit 2
  fi
  case "$CACHE_ROOT" in
    /ssdtemp/users/quansj/*) ;;
    *)
      if [[ "${ALLOW_NON_SSD_CACHE_ROOT:-}" != "YES" ]]; then
        echo "[safety] Refusing to remove non-standard CACHE_ROOT=$CACHE_ROOT" >&2
        echo "[safety] Set ALLOW_NON_SSD_CACHE_ROOT=YES only if this path is intentional." >&2
        exit 2
      fi
      ;;
  esac

  log "resetting SSD cache root at $CACHE_ROOT"
  rm -rf "$CACHE_ROOT"
  mkdir -p "$CACHE_ROOT/sequences"
  copy_manifest "$PLAN_DIR/gt_union_manifest.tsv" "$PLAN_DIR/gt_union_copied_files.txt"
  verify_manifest "$PLAN_DIR/gt_union_manifest.tsv"
}

sync_results() {
  local tag="$1"
  local work_dir="$WORK_BASE/$tag"
  local result_dir="$RESULTS_BASE/$tag"
  mkdir -p "$result_dir"
  find "$work_dir" \
    \( -name 'epoch_eval_summary.csv' -o -name 'epoch_motion_summary.csv' -o -name 'epoch_joint_summary.csv' -o -name '*_results.json' -o -name '*_seq_results.json' \) \
    -type f -print0 \
    | while IFS= read -r -d '' file; do
        rel="${file#$work_dir/}"
        mkdir -p "$result_dir/$(dirname "$rel")"
        cp -f "$file" "$result_dir/$rel"
      done
}

train_run() {
  local tag="$1"
  local work_dir="$WORK_BASE/$tag"
  local result_dir="$RESULTS_BASE/$tag"
  local log_path="$LOG_DIR/${tag}_train.log"
  mkdir -p "$work_dir" "$result_dir"
  log "training $tag for $TOTAL_EPOCHS epochs; work_dir=$work_dir"
  (
    cd "$DOWNSTREAM_DIR"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    export PYTHONPATH="$DOWNSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}"
    export RTPOSE_CACHE_ROOT="$CACHE_ROOT"
    export RTPOSE_TOTAL_EPOCHS="$TOTAL_EPOCHS"
    export RTPOSE_WORK_DIR="$work_dir"
    "$PY" -m torch.distributed.run \
      --nproc_per_node="$NPROC" \
      tools/train.py "$CONFIG" \
      --launcher pytorch \
      --work_dir "$work_dir"
  ) 2>&1 | tee "$log_path"
  sync_results "$tag"
  log "training $tag complete; results=$result_dir"
}

replace_and_train() {
  local ratio="$1"
  local tag="sim1_p${ratio}"
  local manifest="$PLAN_DIR/sim1_p${ratio}_manifest.tsv"
  local copied="$PLAN_DIR/sim1_p${ratio}_copied_files.txt"
  log "applying Sim1 replacement p${ratio}"
  copy_manifest "$manifest" "$copied"
  verify_manifest "$manifest"
  train_run "$tag"
}

require_file "$TRAIN_JSON"
require_file "$VAL_JSON"
require_file "$TEST_JSON"
require_file "$CONFIG"
require_dir "$GT_ROOT"
require_dir "$SIM_ROOT"

write_plan
if [[ "$PLAN_ONLY" == "1" ]]; then
  log "PLAN_ONLY=1; stop after writing $PLAN_DIR/replacement_plan.json"
  exit 0
fi

if [[ "$PREPARE_GT_CACHE" == "1" ]]; then
  prepare_gt_cache
fi

if [[ "$RUN_GT_BASELINE" == "1" ]]; then
  train_run "gt_baseline"
fi

if [[ "$RUN_REPLACEMENT" == "1" ]]; then
  for ratio in ${RATIOS//,/ }; do
    replace_and_train "$ratio"
  done
fi

log "replacement experiment complete"
