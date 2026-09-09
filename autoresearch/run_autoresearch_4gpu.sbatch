#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hbm40g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH -t 01:00:00
#SBATCH -q premium
#SBATCH -J autoresearch

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/modules" && -d "${SLURM_SUBMIT_DIR}/autoresearch" ]]; then
        REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    elif [[ -f "${SLURM_SUBMIT_DIR}/../modules" && -d "${SLURM_SUBMIT_DIR}/../autoresearch" ]]; then
        REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    else
        echo "Could not infer repo root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
        exit 1
    fi
else
    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

OUTPUT_DIR="${REPO_ROOT}/autoresearch/output/latest"
mkdir -p "${OUTPUT_DIR}"
LOG_OUT="${OUTPUT_DIR}/autoresearch-${SLURM_JOB_ID}.log"
exec > >(tee -a "${LOG_OUT}") 2>&1

cd "${REPO_ROOT}"
source "${REPO_ROOT}/modules"

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Running on $(hostname)"
echo "Log: ${LOG_OUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "which python: $(which python)"

python -u -m autoresearch.run_loop \
    --config autoresearch/configs/default.yaml \
    --gpus 0,1,2,3

echo "Job ${SLURM_JOB_ID} finished at $(date)"
