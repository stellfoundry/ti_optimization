#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hbm40g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH -t 24:00:00
#SBATCH -q premium
#SBATCH -J autoresearch

set -euo pipefail

REPO_ROOT="/global/cfs/cdirs/m4505/rchurchi/ti_optimization"
PYTHON="/global/cfs/projectdirs/m4505/rchurchi/conda-envs/ti_optimization/bin/python"

OUTPUT_DIR="${REPO_ROOT}/autoresearch/output/latest"
mkdir -p "${OUTPUT_DIR}"
LOG_OUT="${OUTPUT_DIR}/autoresearch-${SLURM_JOB_ID}.log"
exec > >(tee -a "${LOG_OUT}") 2>&1

cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Running on $(hostname)"
echo "Log: ${LOG_OUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "which python: $(which python)"

"${PYTHON}" -u -m autoresearch.run_loop \
    --config autoresearch/configs/default.yaml \
    --gpus 0,1,2,3

echo "Job ${SLURM_JOB_ID} finished at $(date)"
