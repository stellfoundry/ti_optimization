#!/usr/bin/env bash
set -euo pipefail

# install.sh: Perlmutter-focused dependency setup for the phase0 demo.
#
# Assumes you already created and activated your conda environment.
# Usage:
#   ./install.sh [--desc-dir ./DESC] [--skip-torch]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_TORCH=1
DESC_REPO_URL="https://github.com/PlasmaControl/DESC.git"
DESC_BRANCH="cj/itg-proxy"
DESC_DIR="${SCRIPT_DIR}/DESC"
GX_GEOMETRY_PATH="${SCRIPT_DIR}/gx_geometry"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Assumes your conda environment is already activated.

Options:
  --desc-dir PATH        Location to clone/update DESC (default: ${DESC_DIR})
  --skip-torch           Do not install PyTorch
  -h, --help             Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --desc-dir)
      DESC_DIR="${2:-}"
      shift 2
      ;;
    --skip-torch)
      INSTALL_TORCH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "==> Target system: Perlmutter"
if command -v module >/dev/null 2>&1; then
  # Use Python module if available (typical on NERSC systems).
  module load python >/dev/null 2>&1 || true
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python not found in PATH. Activate your conda env first." >&2
  exit 1
fi

if ! command -v pip >/dev/null 2>&1; then
  echo "Error: pip not found in PATH. Activate your conda env first." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git command not found." >&2
  exit 1
fi

echo "==> Using python: $(command -v python)"
echo "==> Using pip: $(command -v pip)"

echo "==> Upgrading core packaging tools"
pip install --upgrade pip setuptools wheel

echo "==> Ensuring DESC checkout is present at ${DESC_DIR}"
if [[ -d "${DESC_DIR}/.git" ]]; then
  git -C "${DESC_DIR}" fetch origin
else
  rm -rf "${DESC_DIR}"
  git clone "${DESC_REPO_URL}" "${DESC_DIR}"
fi

echo "==> Checking out DESC branch: ${DESC_BRANCH}"
git -C "${DESC_DIR}" checkout "${DESC_BRANCH}"
git -C "${DESC_DIR}" pull --ff-only origin "${DESC_BRANCH}"

echo "==> Installing DESC from: ${DESC_DIR}"
pip install -e "${DESC_DIR}"

if [[ -d "${GX_GEOMETRY_PATH}" ]]; then
  echo "==> Installing gx_geometry from: ${GX_GEOMETRY_PATH}"
  pip install -e "${GX_GEOMETRY_PATH}"
else
  echo "==> WARNING: gx_geometry folder not found at ${GX_GEOMETRY_PATH}. Skipping."
fi

if [[ ${INSTALL_TORCH} -eq 1 ]]; then
  echo "==> Installing optional dependency: torch"
  pip install torch
else
  echo "==> Skipping torch installation"
fi

echo
echo "Installation complete."
