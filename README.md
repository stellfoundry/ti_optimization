# ti_optimization
Optimizing for ion temperature using GX AI surrogate

## INSTALLATION
Currently simply run `source install.sh`. This will create the conda env and install the needed repos including downloading the custom DESC branch and the neural network model weights. 

<!-- Currently don't need this till later phases. 
Follow instructions for installing the AI GX version of T3D https://bitbucket.org/rmc256/t3d/src/ml/README.md -->

## PLAN
Use as a starting point the optimization of the reactor scale stellarator from the Landreman-Paul precise QA stellarator paper, the example found in DESC: `https://github.com/PlasmaControl/DESC/blob/master/desc/examples/reactor_QA.py`

### Phase0: Fixed gradient scale lengths, fixed locations
- Idea: With fixed locations and fixed gradient scale lengths (same as Matt's paper, fprim = 0.9, tprim = 3.0), calculate heat fluxes, and simply average for a global optimization target, along with other targets/constraints from the normal DESC optimization
- Implementation by Byoungchan Jang (byoungj@umd.edu) in the `phase0` folder. See README.md in `phase0` folder for more details.

### Phase1: 