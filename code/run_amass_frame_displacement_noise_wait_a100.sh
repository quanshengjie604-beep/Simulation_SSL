#!/usr/bin/env bash
set -eu

cd /bigdata/users/quansj/datasets/Doppler/simulator

OUT_DIR="legacy/results/amass_smplx_micro_doppler/frame_displacement_noise"
LOG_DIR="${OUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
PYTHON="/bigdata/users/quansj/miniforge3/envs/witwin/bin/python"
GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-12000}"

mkdir -p "${LOG_DIR}"

while true; do
    FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX}" | tr -d ' ')"
    if [ "${FREE_MB}" -ge "${MIN_FREE_MB}" ]; then
        {
            date
            echo "Starting AMASS frame displacement noise experiment on GPU ${GPU_INDEX} with ${FREE_MB} MiB free"
            CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON}" code/run_amass_frame_displacement_noise.py \
                --device cuda:0 \
                --backend dirichlet \
                --smpl-device cpu \
                --noise-std-cm 0 0.5 1 2 3 \
                --out-dir "${OUT_DIR}" \
                --overwrite
            date
            echo "Finished"
        } >"${LOG_FILE}" 2>&1
        break
    fi
    {
        date
        echo "Waiting for GPU ${GPU_INDEX}: ${FREE_MB} MiB free, need ${MIN_FREE_MB} MiB"
    } >>"${LOG_FILE}"
    sleep 60
done
