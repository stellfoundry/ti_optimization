#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hmb80g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH -q debug
#SBATCH -J ti_phase1_gx
#SBATCH -o ti_phase1_gx.%j.out
#SBATCH -e ti_phase1_gx.%j.err
#SBATCH --mail-user=jmlarson@anl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR=/global/cfs/cdirs/m4505/jmlarson/ti_optimization/phase1
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/env_ti_optimization.sh"

# Clear only CPU-forcing variables from login-node tests.
unset JAX_PLATFORM_NAME
unset JAX_PLATFORMS
unset JAX_PLUGINS

echo "Running on $(hostname)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "Python=$(which python3)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"

nvidia-smi || true

test -d "$TI_OPT_GX_DIR"
test -f "$TI_OPT_GX_INPUT_FILE"
test -f "$TI_OPT_GX_LAUNCH"

# srun python3 -u "${SCRIPT_DIR}/test_desc_gx_opt.py"
# srun python3 -u "${SCRIPT_DIR}/sample_desc_gx_params_db.py"
srun python3 -u "${SCRIPT_DIR}/sample_desc_gx_params_db_2.py"
