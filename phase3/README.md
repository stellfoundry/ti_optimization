# Phase3

Phase3 optimizes the Landreman-Paul `precise_QA` equilibrium using a scan-derived
critical gradient instead of a single fixed-gradient NNITGProxy evaluation.

## What changed

- The optimization objective now sweeps `a/LT` over a fixed grid for each `rho`
  and `alpha`, computes the ensemble-mean GX neural-network heat flux, and extracts
  a smooth critical gradient `a/LT_crit`.
- The DESC objective that is actually minimized is the deficit
  `a/LT_max - a/LT_crit`, so the default target of zero means pushing the onset of
  turbulence to the largest scanned gradient.
- This implementation is local to `phase3` and does not patch
  `DESC/desc/objectives`. The new objective class lives in
  `phase3/critical_gradient_objective.py`.

## Default assumptions

- `a/LT` scan grid: `np.linspace(0.5, 10.0, 20)`
- Critical-flux threshold: `Q = 1.0`
- Threshold extraction: smooth first-threshold crossing in log-`Q`
- Multiple `alpha` samples are aggregated with a mean, matching the existing
  `NNITGProxy` convention of returning one value per `rho`
- Optimization uses `n_ensemble=1`; postprocessing uses `n_ensemble=3`

## Outputs

Running `phase3/qs_critical_gradient_opt.py` writes to `phase3/output`:

- `initial.h5`
- `final.h5`
- `03_flux_surface_comparison.png`
- `04_critical_gradient_scan.png`
- `05_critical_gradient_vs_rho.png`
- `critical_gradient_summary.txt`

Submit on Perlmutter with:

```bash
sbatch phase3/run_qs_critical_gradient_opt.sbatch
```
