#!/bin/bash
#SBATCH -A m4505_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -q premium
#SBATCH -J qs-itg-profile

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/modules" && -d "${SLURM_SUBMIT_DIR}/phase2" ]]; then
        REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    elif [[ -f "${SLURM_SUBMIT_DIR}/../modules" && -d "${SLURM_SUBMIT_DIR}/../phase2" ]]; then
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
SCRIPT_DIR="${REPO_ROOT}/phase2"
OUTPUT_DIR="${REPO_ROOT}/phase2/output"
mkdir -p "${OUTPUT_DIR}"

LOG_OUT="${OUTPUT_DIR}/qs-itg-profile-${SLURM_JOB_ID}.log"
PROFILE_OUT="${OUTPUT_DIR}/qs_turbulence_opt_multi-${SLURM_JOB_ID}.cprof"
SUMMARY_OUT="${OUTPUT_DIR}/qs_turbulence_opt_multi-${SLURM_JOB_ID}.pstats.txt"

exec > >(tee -a "${LOG_OUT}") 2>&1

cd "${REPO_ROOT}"
source "${REPO_ROOT}/modules"

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Running on $(hostname)"
echo "Log: ${LOG_OUT}"
echo "Profile: ${PROFILE_OUT}"
echo "Summary: ${SUMMARY_OUT}"

# Uncomment these if JAX/XLA still hits command-buffer OOM on a dedicated node.
# export XLA_FLAGS=--xla_gpu_enable_command_buffer=
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

/usr/bin/time -p srun python -u -m cProfile -o "${PROFILE_OUT}" \
    "${REPO_ROOT}/phase2/qs_turbulence_opt_multi.py"

export PROFILE_OUT SUMMARY_OUT
python - <<'PY'
import os
import pstats

profile_out = os.environ["PROFILE_OUT"]
summary_out = os.environ["SUMMARY_OUT"]

with open(summary_out, "w", encoding="utf-8") as f:
    stats = pstats.Stats(profile_out, stream=f)
    f.write("Top cumulative functions\n")
    f.write("=" * 80 + "\n")
    stats.sort_stats("cumulative").print_stats(60)

    f.write("\nRepo-local cumulative hotspots\n")
    f.write("=" * 80 + "\n")
    repo_items = []
    for func, stat in stats.stats.items():
        cc, nc, tt, ct, callers = stat
        filename, line, name = func
        if "/global/cfs/" in filename or filename.startswith(("phase2/", "DESC/", "DESCgx/")):
            repo_items.append((ct, tt, nc, filename, line, name))
    repo_items.sort(reverse=True)
    for ct, tt, nc, filename, line, name in repo_items[:40]:
        f.write(
            f"{ct:9.3f} cum {tt:9.3f} tot {nc:8d} calls  "
            f"{filename}:{line}({name})\n"
        )

    f.write("\nTurbulence-specific cumulative hotspots\n")
    f.write("=" * 80 + "\n")
    turb_items = []
    for func, stat in stats.stats.items():
        cc, nc, tt, ct, callers = stat
        filename, line, name = func
        if "turbulence" in filename or "NNITGProxy" in name or "_turbulence" in filename:
            turb_items.append((ct, tt, nc, filename, line, name))
    turb_items.sort(reverse=True)
    for ct, tt, nc, filename, line, name in turb_items[:40]:
        f.write(
            f"{ct:9.3f} cum {tt:9.3f} tot {nc:8d} calls  "
            f"{filename}:{line}({name})\n"
        )
PY

echo "Job ${SLURM_JOB_ID} finished at $(date)"
