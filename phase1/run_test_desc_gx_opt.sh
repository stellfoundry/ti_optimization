#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hmb80g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH -q debug

#load all gx related stuff
# source gx.config
# module load python
# conda activate desc-gx
# module list

# python -u test_desc_gx_opt.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

module load python
conda activate desc-gx

export TI_OPT_GX_DIR="${TI_OPT_GX_DIR:-${REPO_ROOT}/gx}"
export TI_OPT_GX_EXEC="${TI_OPT_GX_EXEC:-gx}"
export TI_OPT_GX_INPUT_FILE="${TI_OPT_GX_INPUT_FILE:-${SCRIPT_DIR}/gx.in}"
export TI_OPT_GX_LAUNCH="${TI_OPT_GX_LAUNCH:-${SCRIPT_DIR}/gx-launch.sh}"

srun python3 -u "${SCRIPT_DIR}/test_desc_gx_opt.py"
