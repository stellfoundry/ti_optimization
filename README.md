# ti_optimization
Optimizing for ion temperature using GX AI surrogate

## INSTALLATION
Currently simply run `source install.sh`. This will create the conda env, install the needed repos, build GX on Perlmutter, and download the neural network model weights. The installer now checks out DESC only from `https://github.com/rmchurch/DESC.git` on the `rmchurch/ti-opt` branch. That integration branch combines `byoungj/DESC:cj/itg-proxy` with `PlasmaControl/DESC:yge/gx`.

The installer keeps JAX on pip-managed CUDA wheels and builds GX against the Perlmutter CUDA/MPI/NetCDF modules in a separate shell sourced from `gx.config`. That split only works cleanly if GX is launched in its own subprocess environment. The repo includes [`phase1/gx-launch.sh`](/global/cfs/projectdirs/m4505/rchurchi/ti_optimization/phase1/gx-launch.sh) for that purpose. If you use the DESC GX external objective, pass:

```python
launch_cmd=["/path/to/ti_optimization/phase1/gx-launch.sh"]
```

with `exec` pointing at the built GX executable, typically `/path/to/ti_optimization/gx/gx`.

<!-- Currently don't need this till later phases. 
Follow instructions for installing the AI GX version of T3D https://bitbucket.org/rmc256/t3d/src/ml/README.md -->

## PLAN
Use as a starting point the optimization of the reactor scale stellarator from the Landreman-Paul precise QA stellarator paper, the example found in DESC: `https://github.com/PlasmaControl/DESC/blob/master/desc/examples/reactor_QA.py`

### Phase0: Fixed gradient scale lengths, fixed locations
- Idea: With fixed locations and fixed gradient scale lengths (same as Matt's paper, fprim = 0.9, tprim = 3.0), calculate heat fluxes, and simply average for a global optimization target, along with other targets/constraints from the normal DESC optimization
- Implementation by Byoungchan Jang (byoungj@umd.edu) in the `phase0` folder. See README.md in `phase0` folder for more details.

### Phase1: 
