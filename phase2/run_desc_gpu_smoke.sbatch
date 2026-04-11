#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -t 00:05:00
#SBATCH -q debug
#SBATCH -J desc-gpu-smoke

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

LOG_OUT="${OUTPUT_DIR}/desc-gpu-smoke-${SLURM_JOB_ID}.log"
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

python -u - <<'PY'
import os
import sys

print("python executable:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))
print("SLURM_STEP_GPUS:", os.environ.get("SLURM_STEP_GPUS"))

import torch

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch device 0:", torch.cuda.get_device_name(0))

from desc import set_device
import jax

set_device("gpu")
devices = jax.devices()
platforms = [device.platform for device in devices]
print("jax.default_backend():", jax.default_backend())
print("jax.devices():", devices)
print("jax platforms:", platforms)

if "gpu" not in platforms:
    raise SystemExit("desc.set_device('gpu') did not expose a GPU-backed JAX device")

print("DESC GPU smoke test passed")
PY

echo "Job ${SLURM_JOB_ID} finished at $(date)"
