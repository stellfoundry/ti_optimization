#!/bin/bash

# Repository locations
export TI_OPT_ROOT=/global/cfs/cdirs/m4505/jmlarson/ti_optimization
export TI_OPT_PHASE1=${TI_OPT_ROOT}/phase1

# GX installation
export TI_OPT_GX_DIR=/global/cfs/projectdirs/m4505/rchurchi/gx
export TI_OPT_GX_EXEC=gx
export TI_OPT_GX_INPUT_FILE=${TI_OPT_PHASE1}/gx.in
export TI_OPT_GX_LAUNCH=${TI_OPT_PHASE1}/gx-launch.sh

export TI_OPT_NN_DATA_DIR=/global/cfs/cdirs/m4505/rchurchi/ti_optimization/data/neural_networks

# Conda environment
module load python

conda activate /global/cfs/projectdirs/m4505/rchurchi/conda-envs/ti_optimization

# Useful diagnostics
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "TI_OPT_ROOT=${TI_OPT_ROOT}"
echo "TI_OPT_GX_DIR=${TI_OPT_GX_DIR}"
echo "TI_OPT_GX_INPUT_FILE=${TI_OPT_GX_INPUT_FILE}"
echo "TI_OPT_GX_LAUNCH=${TI_OPT_GX_LAUNCH}"
