#!/usr/bin/env bash

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  IS_SOURCED=1
fi

# When sourced, do not mutate caller shell options.
if [[ ${IS_SOURCED} -eq 0 ]]; then
  set -euo pipefail
fi

# install.sh: Perlmutter-focused dependency setup for ti_optimization.
#
# Assumes you already created and activated your conda environment.
# Usage:
#   ./install.sh [--desc-dir ./DESC] [--python-version 3.12] [--install-torch] [--recreate-env] [--debug-pause]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_TORCH=1
RECREATE_ENV=0
DEBUG_PAUSE=0
CONDA_PYTHON_VERSION="3.12"
DESC_REPO_URL="https://github.com/byoungj/DESC/"
DESC_BRANCH="cj/itg-proxy"
DESC_DIR="${SCRIPT_DIR}/DESC"
GX_GEOMETRY_PATH="${SCRIPT_DIR}/gx_geometry"
if [[ -d "${SCRIPT_DIR}/phase0/gx_geometry" ]]; then
  GX_GEOMETRY_PATH="${SCRIPT_DIR}/phase0/gx_geometry"
fi
UTILS_PY="${SCRIPT_DIR}/utils.py"
MODEL_DIR="${SCRIPT_DIR}/data/neural_networks/cyclic_invariant_models"

abort_install() {
  local reason="${1:-Unknown failure}"
  echo
  echo "======================================================================="
  echo "====================== INSTALLATION ABORTED ============================"
  echo "Reason: ${reason}"
  echo "======================================================================="
  echo
}

stop_script() {
  local code="${1:-1}"
  return "${code}" 2>/dev/null || exit "${code}"
}

fail_install() {
  local reason="${1:-Unknown failure}"
  abort_install "${reason}"
  stop_script 1
}

# Debug helper: pause after major stages when DEBUG_PAUSE=1.
pause_step() {
  local label="${1:-Continue}"
  if [[ "${DEBUG_PAUSE}" == "1" ]]; then
    if [[ -t 0 ]]; then
      read -r -p "[debug pause] ${label}. Press Enter to continue..." _
    else
      echo "[debug pause] ${label} (no interactive TTY, continuing)"
    fi
  fi
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Assumes your conda environment is already activated.

Options:
  --desc-dir PATH        Location to clone/update DESC (default: ${DESC_DIR})
  --python-version VER   Python version for conda env (default: ${CONDA_PYTHON_VERSION})
  --install-torch        Install PyTorch (kept for compatibility; default: on)
  --skip-torch           Skip PyTorch install
  --recreate-env         Remove and recreate existing conda env
  --debug-pause          Pause at major steps for interactive debugging
  -h, --help             Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --desc-dir)
      DESC_DIR="${2:-}"
      shift 2
      ;;
    --python-version)
      CONDA_PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --skip-torch)
      INSTALL_TORCH=0
      shift
      ;;
    --install-torch)
      INSTALL_TORCH=1
      shift
      ;;
    --recreate-env)
      RECREATE_ENV=1
      shift
      ;;
    --debug-pause)
      DEBUG_PAUSE=1
      shift
      ;;
    -h|--help)
      usage
      stop_script 0
      ;;
    *)
      usage
      fail_install "Unknown argument: $1"
      ;;
  esac
done

pause_step "Argument parsing complete"

echo "==> Target system: Perlmutter"
if command -v module >/dev/null 2>&1; then
  # Avoid stale CUDA module state interfering with jax[cuda12] wheels.
  module unload cudatoolkit >/dev/null 2>&1 || true
  # Use Python module if available (typical on NERSC systems).
  module load python >/dev/null 2>&1 || true

  if ! command -v conda >/dev/null 2>&1; then
    fail_install "conda not found after loading python module."
  fi

  # Initialize conda for this non-interactive shell.
  eval "$(conda shell.bash hook)"

  if conda env list | awk '{print $1}' | grep -Fxq "ti_optimization"; then
    if [[ ${RECREATE_ENV} -eq 0 && -t 0 ]]; then
      read -r -p "==> Conda env 'ti_optimization' exists. Delete and recreate it? [y/N] " recreate_answer
      case "${recreate_answer}" in
        y|Y|yes|YES|Yes)
          RECREATE_ENV=1
          ;;
      esac
    fi

    if [[ ${RECREATE_ENV} -eq 1 ]]; then
      conda env remove -n ti_optimization -y
      conda create -y -n ti_optimization "python=${CONDA_PYTHON_VERSION}"
    else
      if [[ -t 0 ]]; then
        echo "==> Reusing existing conda env: ti_optimization"
      else
        echo "==> Reusing existing conda env: ti_optimization (non-interactive shell; use --recreate-env to rebuild)"
      fi
    fi
  else
    conda create -y -n ti_optimization "python=${CONDA_PYTHON_VERSION}"
  fi

  conda activate ti_optimization
fi

ACTIVE_PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if python - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] < (3, 13) else 1)
PY
then
  :
else
  fail_install "Detected Python ${ACTIVE_PY_MM}. Use Python <3.13 (recommended: 3.12) for DESC/JAX compatibility. Recreate env with --python-version 3.12."
fi

pause_step "Python/conda environment setup complete"

if ! command -v python >/dev/null 2>&1; then
  fail_install "python not found in PATH. Activate your conda env first."
fi

if ! command -v pip >/dev/null 2>&1; then
  fail_install "pip not found in PATH. Activate your conda env first."
fi

if ! command -v git >/dev/null 2>&1; then
  fail_install "git command not found."
fi

pause_step "Command checks complete"

echo "==> Using python: $(command -v python)"
echo "==> Using pip: $(command -v pip)"

echo "==> Upgrading core packaging tools"
pip install --upgrade pip setuptools wheel
pause_step "Packaging tool upgrade complete"

echo "==> Ensuring DESC checkout is present at ${DESC_DIR}"
if [[ -d "${DESC_DIR}/.git" ]]; then
  git -C "${DESC_DIR}" fetch origin
else
  rm -rf "${DESC_DIR}"
  git clone "${DESC_REPO_URL}" "${DESC_DIR}"
fi
pause_step "DESC checkout/update complete"

echo "==> Checking out DESC branch: ${DESC_BRANCH}"
git -C "${DESC_DIR}" checkout "${DESC_BRANCH}"
git -C "${DESC_DIR}" pull --ff-only origin "${DESC_BRANCH}"
pause_step "DESC branch checkout complete"

echo "==> Installing DESC from: ${DESC_DIR}"
pip install --no-cache-dir -r "${DESC_DIR}/devtools/dev-requirements.txt"
pip install --no-cache-dir --editable "${DESC_DIR}"
pip install --no-cache-dir "jax[cuda12]==0.6.2"
pause_step "DESC install complete"

if [[ -d "${GX_GEOMETRY_PATH}" ]]; then
  echo "==> Installing gx_geometry from: ${GX_GEOMETRY_PATH}"
  pip install -e "${GX_GEOMETRY_PATH}"
else
  echo "==> WARNING: gx_geometry folder not found at ${GX_GEOMETRY_PATH}. Skipping."
fi
pause_step "gx_geometry step complete"

if [[ ${INSTALL_TORCH} -eq 1 ]]; then
  echo "==> Installing dependency: torch"
  pip install torch
else
  echo "==> Skipping torch installation"
fi
pause_step "torch step complete"

echo "==> Installing helper dependency: gdown"
pip install --no-cache-dir gdown
pause_step "gdown install complete"

echo "==> Downloading NN model assets"
if [[ ! -f "${UTILS_PY}" ]]; then
  fail_install "utils.py not found at ${UTILS_PY}"
fi
python "${UTILS_PY}" --model-dir "${MODEL_DIR}"
pause_step "NN model asset download complete"

echo
echo "Installation complete."
