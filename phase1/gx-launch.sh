#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <gx-executable> [gx-args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GX_CONFIG="${SCRIPT_DIR}/gx.config"

if [[ ! -f "${GX_CONFIG}" ]]; then
  echo "GX config not found at ${GX_CONFIG}" >&2
  exit 1
fi

if [[ -z "${GX_PATH:-}" && "$1" == */* ]]; then
  export GX_PATH="$(cd "$(dirname "$1")" && pwd)"
fi

# shellcheck disable=SC1090
source "${GX_CONFIG}"

exec "$@"
