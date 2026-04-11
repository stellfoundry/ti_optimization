#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -J qs-itg-transport

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/modules" && -d "${SLURM_SUBMIT_DIR}/phase2" ]]; then
        REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    elif [[ -f "${SLURM_SUBMIT_DIR}/../modules" && -d "${SLURM_SUBMIT_DIR}/../phase2" ]]; then
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

OUTPUT_DIR="${REPO_ROOT}/phase2/output"
mkdir -p "${OUTPUT_DIR}"

LOG_OUT="${OUTPUT_DIR}/qs-itg-transport-${SLURM_JOB_ID}.log"
exec > >(tee -a "${LOG_OUT}") 2>&1

cd "${REPO_ROOT}"
source "${REPO_ROOT}/modules"

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Running on $(hostname)"
echo "Log: ${LOG_OUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
echo "which python: $(which python)"

nvidia-smi -L

python -u phase2/transport_postprocessing.py

echo "Job ${SLURM_JOB_ID} finished at $(date)"
