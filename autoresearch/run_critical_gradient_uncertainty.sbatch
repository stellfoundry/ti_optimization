#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -q premium
#SBATCH -J crit-grad-err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "${SLURM_SUBMIT_DIR}/autoresearch" && -d "${SLURM_SUBMIT_DIR}/phase3" ]]; then
        REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    elif [[ -d "${SLURM_SUBMIT_DIR}/../autoresearch" && -d "${SLURM_SUBMIT_DIR}/../phase3" ]]; then
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

CAMPAIGN_DIR="${CAMPAIGN_DIR:-${REPO_ROOT}/autoresearch/output/latest}"
SELECTION="${SELECTION:-best-score}"
ERROR_BAR="${ERROR_BAR:-std}"
OUTPUT_PATH="${OUTPUT_PATH:-${CAMPAIGN_DIR}/critical_gradient_by_generation_${SELECTION}_uncertainty.png}"
JSON_PATH="${JSON_PATH:-${OUTPUT_PATH%.png}.json}"

mkdir -p "${CAMPAIGN_DIR}"
LOG_OUT="${CAMPAIGN_DIR}/critical-gradient-uncertainty-${SLURM_JOB_ID}.log"
exec > >(tee -a "${LOG_OUT}") 2>&1

cd "${REPO_ROOT}"
module load python
conda activate ti_optimization

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Running on $(hostname)"
echo "Log: ${LOG_OUT}"
echo "Campaign: ${CAMPAIGN_DIR}"
echo "Selection: ${SELECTION}"
echo "Error bar: ${ERROR_BAR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "which python: $(which python)"

N_ENSEMBLE_ARGS=()
if [[ -n "${N_ENSEMBLE:-}" ]]; then
    N_ENSEMBLE_ARGS=(--n-ensemble "${N_ENSEMBLE}")
fi

RUN_ID_ARGS=()
if [[ -n "${RUN_ID:-}" ]]; then
    RUN_ID_ARGS=(--run-id "${RUN_ID}")
fi

srun python -u -m autoresearch.plot_critical_gradient_uncertainty \
    --campaign-dir "${CAMPAIGN_DIR}" \
    --selection "${SELECTION}" \
    --error "${ERROR_BAR}" \
    --device gpu \
    --output "${OUTPUT_PATH}" \
    --json-output "${JSON_PATH}" \
    "${N_ENSEMBLE_ARGS[@]}" \
    "${RUN_ID_ARGS[@]}"

echo "Wrote plot: ${OUTPUT_PATH}"
echo "Wrote data: ${JSON_PATH}"
echo "Job ${SLURM_JOB_ID} finished at $(date)"
