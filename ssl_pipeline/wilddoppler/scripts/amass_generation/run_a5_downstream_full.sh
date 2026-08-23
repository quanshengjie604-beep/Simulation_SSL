#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-self_supervised_pretrain/checkpoints/contrastive_a5_dw_amass_equal/contrastive_mobilenet_v2_last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/a5_contrastive_downstream_full}"
TASKS="${TASKS:-MotionState VelocityRegression SingleHand}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0,cuda:2}"
MAX_JOBS="${MAX_JOBS:-2}"

export PYTHONPATH="$(dirname "${ROOT}"):${ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a DEVICES <<< "${DEVICE_LIST}"
if [[ "${#DEVICES[@]}" -eq 0 ]]; then
  echo "DEVICE_LIST is empty" >&2
  exit 1
fi

wait_for_slot() {
  while (( "$(jobs -pr | wc -l)" >= MAX_JOBS )); do
    wait -n
  done
}

run_one() {
  local task="$1"
  local cross_location="$2"
  local device="$3"
  local tag="cross_subject"
  local extra=()
  if [[ "${cross_location}" == "True" ]]; then
    tag="cross_location"
    extra+=(cross_location=True)
  fi
  echo "== A5 downstream full: ${task} ${tag} on ${device} =="
  "${PYTHON}" train.py method=contrastive task_name="${task}" \
    paths.ckpt="${PRETRAIN_CKPT}" \
    output_dir="${OUTPUT_DIR}/${task}_${tag}" \
    +device="${device}" \
    +knn.enabled=False \
    +linear_probe.variants.frozen_backbone_nonlinear_head.enabled=False \
    +linear_probe.variants.linear_probe.enabled=False \
    "${extra[@]}"
}

job_idx=0
for task in ${TASKS}; do
  wait_for_slot
  run_one "${task}" False "${DEVICES[$((job_idx % ${#DEVICES[@]}))]}" &
  job_idx=$((job_idx + 1))
  if [[ "${task}" != "SingleHand" ]]; then
    wait_for_slot
    run_one "${task}" True "${DEVICES[$((job_idx % ${#DEVICES[@]}))]}" &
    job_idx=$((job_idx + 1))
  fi
done

wait
