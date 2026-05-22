#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hmb80g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH -q regular
#SBATCH -J qs_gx_opt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

module load python
conda activate desc-gx

export TI_OPT_GX_DIR="${TI_OPT_GX_DIR:-${REPO_ROOT}/gx}"
export TI_OPT_GX_EXEC="${TI_OPT_GX_EXEC:-gx}"
export TI_OPT_GX_INPUT_FILE="${TI_OPT_GX_INPUT_FILE:-${SCRIPT_DIR}/gx.in}"
export TI_OPT_GX_LAUNCH="${TI_OPT_GX_LAUNCH:-${SCRIPT_DIR}/gx-launch.sh}"

# Keep JAX from preallocating GPU memory that GX needs.
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

srun python3 -u "${SCRIPT_DIR}/qs_gx_turbulence_opt.py"
