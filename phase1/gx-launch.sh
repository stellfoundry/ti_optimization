#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <gx-executable> [gx-args...]" >&2
  exit 2
fi

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /opt/cray/pe/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /opt/cray/pe/lmod/lmod/init/bash
  fi
fi

if ! command -v module >/dev/null 2>&1; then
  echo "module command not found; cannot prepare the GX runtime on Perlmutter." >&2
  exit 1
fi

module load PrgEnv-gnu/8.5.0
module load nvidia-mixed/24.5
module load cudatoolkit/12.4
module load cray-mpich/8.1.28
module load nccl
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load python >/dev/null 2>&1 || true

export GK_SYSTEM="${GK_SYSTEM:-perlmutter}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-0}"
export FI_CXI_RDZV_GET_MIN="${FI_CXI_RDZV_GET_MIN:-0}"
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD="${FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD:-16777216}"

exec "$@"
