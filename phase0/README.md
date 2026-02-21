## ITG Proxy Demo

## Setup

## Automated install (Perlmutter)

Use the helper script to install dependencies (assumes your conda env is already created and activated):

```bash
conda activate <your-existing-env>
cd phase0
./install.sh
```

Optional flags:
- `--desc-dir <path>` to choose where DESC is cloned/updated (default: `phase0/DESC`)
- `--skip-torch` to skip optional PyTorch install

1. **Install DESC** with the `cj/itg-proxy` branch (manual alternative):
   ```bash
   git clone https://github.com/PlasmaControl/DESC.git
   cd DESC
   git checkout cj/itg-proxy
   pip install -e .
   ```

2. **Install gx_geometry** (included in this folder):
   ```bash
   pip install -e gx_geometry
   ```

3. **PyTorch** (optional, for script 02 PyTorch comparison):
   ```bash
   pip install torch
   ```

##Data (included)

All data is pre-populated:

`data/neural_networks/cyclic_invariant_models/` - 507 CNN model weights
`data/neural_networks/deephyper_exp/results.csv` - DeepHyper hyperparameters
`data/data_for_ML/20241210-01-assembleFluxTubeTensor_allConfigs_filtered.pkl` - flux-tube geometry tensor (N, 96, 7)
`data/data_for_ML/20241210-01-GX_results_for_fixed_gradients_allConfigs_filtered.pkl` - GX Q values (fixed gradients)
`data/data_for_ML/20241210-01-GX_results_for_random_gradients_allConfigs_filtered.pkl` - GX Q values (random gradients)

## Scripts

`00_compare_gx_signals.py` - Validates that DESC's 7 GX geometric coefficients match the reference implementation across 6 equilibria. Produces a 7-panel comparison plot.
`01_validate_plumbing.py` - Confirms ITGProxy and NNITGProxy produce finite, positive values. Tests `return_signals`, `return_per_alpha`, `return_std`. Verifies JAX differentiability.
`02_compare_pytorch_vs_desc.py` - Loads pre-computed flux-tube tensor and GX results. Runs both PyTorch CyclicInvariantNet and DESC ensemble on the test set, comparing R^2 scores and per-sample predictions.
`03_qs_turbulence_opt.py` - Runs combined NNITGProxy optimization. Saves equilibria as .h5 and plots before/after flux surfaces.
`04_multi_alpha_robustness.py` - Evaluates NNITGProxy at 8 field lines across [0, pi] for both equilibria. Shows per-alpha Q with ensemble error bars.
`05_temperature_gradient_scan.py` - Scans a/LT from 0.5 to 10.0 at fixed a/Ln=1.0. Plots Q vs temperature gradient for initial and optimized equilibria.

## Folder structure

```
demo/
    data/
        neural_networks/
            cyclic_invariant_models/   (507 model .pth files)
            deephyper_exp/results.csv
        data_for_ML/
            20241210-01-assembleFluxTubeTensor_allConfigs_filtered.pkl
            20241210-01-GX_results_for_fixed_gradients_allConfigs_filtered.pkl
            20241210-01-GX_results_for_random_gradients_allConfigs_filtered.pkl
    gx_geometry/                  
    output/                       
    00_compare_gx_signals.py
    01_validate_plumbing.py
    02_compare_pytorch_vs_desc.py
    03_qs_turbulence_opt.py
    04_multi_alpha_robustness.py
    05_temperature_gradient_scan.py
    README.md
```
