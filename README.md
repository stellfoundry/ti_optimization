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

### Phase0: Fixed gradient scale lengths, single location
- Idea: Optimize MHD equilibrium using as a optimization target the turbulent ion heat flux calculated using the GX neural network surrogate at a single, fixed location (s=0.5) and fixed gradient scale lengths (same as Matt's paper, fprim = 0.9, tprim = 3.0), along with other targets/constraints from the normal DESC optimization
- Implementation by Byoungchan Jang (byoungj@umd.edu) in the `phase0` folder. See README.md in `phase0` folder for more details.

### Phase1: Multi-fidelity optimization (WORK-IN-PROGRESS)
- Idea: Multi-fidelity optimization, using the GX neural network surrogate for low-fidelity and the actual GX simulation for high-fidelity when needed. 

### Phase2: Fixed gradient scale lengths, multiple locations
- Idea: Optimize MHD equilibrium using as an optimization target the turbulent ion heat flux calculated using the GX neural network surrogate at multiple radial and field line angles (alpha) and fixed gradient scale lengths (same as Matt's paper, fprim = 0.9, tprim = 3.0), along with other targets/constraints from the normal DESC optimization.

### Phase3: Critical gradient
- Idea: Optimize MHD equilibrium using as an optimization target the turbulent ion heat flux calculated using the GX neural network surrogate at multiple radial and field line angles (alpha), but now using a critical gradient model, where the gradient scale length is swept in order to find the correspoonding critical gradient, and optimzizing  along with other targets/constraints from the normal DESC optimization.