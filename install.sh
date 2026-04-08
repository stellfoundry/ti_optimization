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
# Creates or reuses the ti_optimization conda environment, installs DESC/JAX,
# builds GX on Perlmutter, and downloads the model assets.
# Usage:
#   ./install.sh [--desc-dir ./DESC] [--gx-dir ./gx] [--python-version 3.12] [--install-torch] [--recreate-env] [--debug-pause]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="ti_optimization"
INSTALL_TORCH=1
INSTALL_GX=1
RECREATE_ENV=0
DEBUG_PAUSE=0
CONDA_PYTHON_VERSION="3.12"
DESC_REPO_URL="https://github.com/rmchurch/DESC.git"
DESC_BRANCH="rmchurch/ti-opt"
DESC_DIR="${SCRIPT_DIR}/DESC"
GX_REPO_URL="https://bitbucket.org/gyrokinetics/gx.git"
GX_BRANCH="gx"
GX_DIR="${SCRIPT_DIR}/gx"
GX_JOBS=8
GX_SYSTEM="perlmutter"
GX_CONFIG="${SCRIPT_DIR}/gx.config"
GX_LAUNCH_SCRIPT="${SCRIPT_DIR}/phase1/gx-launch.sh"
if [[ -f "${SCRIPT_DIR}/gx-launch.sh" ]]; then
  GX_LAUNCH_SCRIPT="${SCRIPT_DIR}/gx-launch.sh"
fi
GX_GEOMETRY_REPO_URL="https://github.com/landreman/gx_geometry.git"
GX_GEOMETRY_BRANCH="main"
GX_GEOMETRY_PATH="${SCRIPT_DIR}/gx_geometry"
UTILS_PY="${SCRIPT_DIR}/utils.py"
MODEL_DIR="${SCRIPT_DIR}/data/neural_networks/cyclic_invariant_models"
GX_PYTHON_DEPS=(netCDF4)

abort_install() {
  local reason="${1:-Unknown failure}"
  echo
  echo "======================================================================="
  echo "====================== INSTALLATION ABORTED ============================"
  echo "Reason: ${reason}"
  echo "======================================================================="
  echo
}

fail_install() {
  local reason="${1:-Unknown failure}"
  abort_install "${reason}"
  return 1
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

Creates or reuses the ${ENV_NAME} conda environment.

Options:
  --desc-dir PATH        Location to clone/update DESC (default: ${DESC_DIR})
  --gx-dir PATH          Location to clone/update GX (default: ${GX_DIR})
  --gx-branch NAME       GX branch to clone/build (default: ${GX_BRANCH})
  --gx-repo-url URL      GX repository URL (default: ${GX_REPO_URL})
  --gx-jobs N            Parallel jobs for GX build (default: ${GX_JOBS})
  --skip-gx             Skip GX clone/build
  --python-version VER   Python version for conda env (default: ${CONDA_PYTHON_VERSION})
  --install-torch        Install PyTorch (kept for compatibility; default: on)
  --skip-torch           Skip PyTorch install
  --recreate-env         Remove and recreate existing conda env
  --debug-pause          Pause at major steps for interactive debugging
  -h, --help             Show this help message
USAGE
}

ensure_module_cmd() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f /opt/cray/pe/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /opt/cray/pe/lmod/lmod/init/bash
  fi
  if ! command -v module >/dev/null 2>&1; then
    fail_install "module command not found. This installer expects the Perlmutter module stack."
  fi
}

load_python_module_stack() {
  ensure_module_cmd
  module unload cudatoolkit >/dev/null 2>&1 || true
  module load python >/dev/null 2>&1 || true
}

ensure_gx_config() {
  if [[ ! -f "${GX_CONFIG}" ]]; then
    fail_install "GX config not found at ${GX_CONFIG}"
    return 1
  fi
}

build_gx_in_subprocess() {
  local gx_build_dir

  ensure_gx_config || return 1
  gx_build_dir="$(cd "${GX_DIR}" && pwd)"

  if ! GX_CONFIG="${GX_CONFIG}" \
    GX_JOBS="${GX_JOBS}" \
    GX_PATH="${gx_build_dir}" \
    GK_SYSTEM="${GX_SYSTEM}" \
    bash <<'EOF'
set -euo pipefail
source "${GX_CONFIG}"
echo "==> GX module stack:"
module --redirect list 2>/dev/null || module -t list 2>&1 || true
cd "${GX_PATH}"
make clean >/dev/null 2>&1 || true
make -j "${GX_JOBS}"
EOF
  then
    fail_install "GX build failed. The GX source was cloned to ${GX_DIR} for inspection."
    return 1
  fi
}

ensure_repo_checkout() {
  local repo_url="$1"
  local branch="$2"
  local dest="$3"
  local label="$4"

  echo "==> Ensuring ${label} checkout is present at ${dest}"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" fetch origin "${branch}"
  else
    rm -rf "${dest}"
    git clone --branch "${branch}" "${repo_url}" "${dest}"
  fi
  pause_step "${label} checkout/update complete"

  echo "==> Checking out ${label} branch: ${branch}"
  git -C "${dest}" checkout "${branch}"
  git -C "${dest}" pull --ff-only origin "${branch}"
  pause_step "${label} branch checkout complete"
}

build_gx() {
  if [[ ${INSTALL_GX} -eq 0 ]]; then
    echo "==> Skipping GX installation"
    return 0
  fi

  ensure_repo_checkout "${GX_REPO_URL}" "${GX_BRANCH}" "${GX_DIR}" "GX"

  echo "==> Installing GX Python-side helpers into ${ENV_NAME}"
  if python - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] < (3, 11) else 1)
PY
  then
    pip install --no-cache-dir "${GX_PYTHON_DEPS[@]}" tomli
  else
    pip install --no-cache-dir "${GX_PYTHON_DEPS[@]}"
  fi
  pause_step "GX Python dependency install complete"

  echo "==> Building GX from source for ${GX_SYSTEM}"
  build_gx_in_subprocess
  pause_step "GX build complete"

  if [[ ! -x "${GX_DIR}/gx" ]]; then
    fail_install "GX build finished without producing ${GX_DIR}/gx"
    return 1
  fi

  chmod +x "${GX_LAUNCH_SCRIPT}" >/dev/null 2>&1 || true
}

main() {
  local old_shell_opts=""

  if [[ ${IS_SOURCED} -eq 1 ]]; then
    old_shell_opts="$(set +o)"
    trap 'eval "$old_shell_opts"' RETURN
  fi
  set -euo pipefail

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --desc-dir)
        DESC_DIR="${2:-}"
        shift 2
        ;;
      --gx-dir)
        GX_DIR="${2:-}"
        shift 2
        ;;
      --gx-branch)
        GX_BRANCH="${2:-}"
        shift 2
        ;;
      --gx-repo-url)
        GX_REPO_URL="${2:-}"
        shift 2
        ;;
      --gx-jobs)
        GX_JOBS="${2:-}"
        shift 2
        ;;
      --skip-gx)
        INSTALL_GX=0
        shift
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
        return 0
        ;;
      *)
        usage
        fail_install "Unknown argument: $1"
        return 1
        ;;
    esac
  done

  pause_step "Argument parsing complete"

  echo "==> Target system: Perlmutter"
  load_python_module_stack

  if ! command -v conda >/dev/null 2>&1; then
    fail_install "conda not found after loading the Perlmutter python module."
    return 1
  fi

  # Initialize conda for this non-interactive shell.
  eval "$(conda shell.bash hook)"

  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    if [[ ${RECREATE_ENV} -eq 0 && -t 0 ]]; then
      read -r -p "==> Conda env '${ENV_NAME}' exists. Delete and recreate it? [y/N] " recreate_answer
      case "${recreate_answer}" in
        y|Y|yes|YES|Yes)
          RECREATE_ENV=1
          ;;
      esac
    fi

    if [[ ${RECREATE_ENV} -eq 1 ]]; then
      conda env remove -n "${ENV_NAME}" -y
      conda create -y -n "${ENV_NAME}" "python=${CONDA_PYTHON_VERSION}"
    else
      if [[ -t 0 ]]; then
        echo "==> Reusing existing conda env: ${ENV_NAME}"
      else
        echo "==> Reusing existing conda env: ${ENV_NAME} (non-interactive shell; use --recreate-env to rebuild)"
      fi
    fi
  else
    conda create -y -n "${ENV_NAME}" "python=${CONDA_PYTHON_VERSION}"
  fi

  conda activate "${ENV_NAME}"

  ACTIVE_PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if python - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] < (3, 13) else 1)
PY
  then
    :
  else
    fail_install "Detected Python ${ACTIVE_PY_MM}. Use Python <3.13 (recommended: 3.12) for DESC/JAX compatibility. Recreate env with --python-version 3.12."
    return 1
  fi

  pause_step "Python/conda environment setup complete"

  if ! command -v python >/dev/null 2>&1; then
    fail_install "python not found in PATH. Activate your conda env first."
    return 1
  fi

  if ! command -v pip >/dev/null 2>&1; then
    fail_install "pip not found in PATH. Activate your conda env first."
    return 1
  fi

  if ! command -v git >/dev/null 2>&1; then
    fail_install "git command not found."
    return 1
  fi

  pause_step "Command checks complete"

  echo "==> Using python: $(command -v python)"
  echo "==> Using pip: $(command -v pip)"

  echo "==> Upgrading core packaging tools"
  pip install --upgrade pip setuptools wheel
  pause_step "Packaging tool upgrade complete"

  ensure_repo_checkout "${DESC_REPO_URL}" "${DESC_BRANCH}" "${DESC_DIR}" "DESC"

  echo "==> Installing DESC from: ${DESC_DIR}"
  pip install --no-cache-dir -r "${DESC_DIR}/devtools/dev-requirements.txt"
  pip install --no-cache-dir --editable "${DESC_DIR}"
  pip install --no-cache-dir "jax[cuda12]==0.6.2"
  pause_step "DESC install complete"

  echo "==> NOTE: JAX is installed with pip-managed CUDA wheels"
  echo "    Keep 'cudatoolkit' unloaded in the parent Python/JAX shell."
  echo "    If JAX sees the wrong CUDA libraries at runtime, also run: unset LD_LIBRARY_PATH"
  pause_step "JAX runtime note emitted"

  build_gx

  ensure_repo_checkout "${GX_GEOMETRY_REPO_URL}" "${GX_GEOMETRY_BRANCH}" "${GX_GEOMETRY_PATH}" "gx_geometry"

  echo "==> Installing gx_geometry from: ${GX_GEOMETRY_PATH}"
  pip install -e "${GX_GEOMETRY_PATH}"
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
    return 1
  fi
  python "${UTILS_PY}" --model-dir "${MODEL_DIR}"
  pause_step "NN model asset download complete"

  echo
  echo "Installation complete."
  if [[ ${INSTALL_GX} -eq 1 ]]; then
    echo
    echo "GX was cloned to: ${GX_DIR}"
    echo "GX launcher wrapper: ${GX_LAUNCH_SCRIPT}"
    echo "For DESC+GX runs on Perlmutter, keep JAX in the parent process and use:"
    echo "  launch_cmd=['${GX_LAUNCH_SCRIPT}']"
    echo "  exec='${GX_DIR}/gx'"
  fi
}

main "$@"
main_status=$?
if [[ ${IS_SOURCED} -eq 1 ]]; then
  return "${main_status}"
else
  exit "${main_status}"
fi
