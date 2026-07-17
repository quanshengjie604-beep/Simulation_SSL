#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/bigdata/users/quansj/miniforge3/envs/witwin/bin/python}"

TRAIN_JSON="${TRAIN_JSON:-$ROOT/datasets/Train_sp120_train_minus_val6.json}"
FILEMETA_JSON="${FILEMETA_JSON:-$ROOT/datasets/filemeta.json}"
GT_ROOT="${GT_ROOT:-$ROOT/datasets/GT_sequences}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$ROOT/datasets/Sim1_sequences}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-Sim1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/Quantitative_analysis/sim1vsgt/RAE}"

"$PY" "$ROOT/code/Quantitative_analysis/batch_cfar_chamfer.py" \
  --train "$TRAIN_JSON" \
  --filemeta "$FILEMETA_JSON" \
  --gt-root "$GT_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --candidate-label "$CANDIDATE_LABEL" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
