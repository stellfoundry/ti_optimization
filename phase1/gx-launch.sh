#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <gx-executable> [gx-args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GX_CONFIG="${TI_OPT_GX_CONFIG:-${SCRIPT_DIR}/gx.config}"

if [[ ! -f "${GX_CONFIG}" ]]; then
  echo "GX config not found at ${GX_CONFIG}" >&2
  exit 1
fi

if [[ -z "${GX_PATH:-}" && "$1" == */* ]]; then
  export GX_PATH="$(cd "$(dirname "$1")" && pwd)"
fi

# The parent DESC/JAX process runs with GPUs hidden so it cannot claim an
# exclusive-process context. Restore Slurm's original device list for GX only.
if [[ -n "${TI_OPT_GX_CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="${TI_OPT_GX_CUDA_VISIBLE_DEVICES}"
fi

# shellcheck disable=SC1090
source "${GX_CONFIG}"

exec "$@"
