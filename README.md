#  Variational augmentation of Gaussian continuum basis sets for calculating atomic higher harmonic generation spectra

Contains python scripts, ipython notebooks and data from our study on using the TDCI approach with Gaussian basis sets for simulating higher-harmonic generation (HHG).

## basissets
The optimized basis sets for calculating HHG spectra in psi4 format (as `.gbs` files). 

## data
```
data
├── hdata
├── qprop
├── sfa
└── tdci
```
- All the hamiltonians are saved as `.npz` files in the folder `hdata`.
- The `qprop` folder contains all the data of SAE simulations using Qprop(v3.2), `sfa` folder contains the mathematica notebooks for SFA calculations and, the `tdci` folder contains all the data from TDCI simulations for H-, He and Li+ for different laser parameters.  
- In the `*tdprop.txt` files that containe the time propagation data from TDCI simulations, the data is saved in a column-wise format with observables stored in the following order (left to right):
    + time (in fs)
    + norm $\langle\Psi(t)|\Psi(t)\rangle$
    + correlation function  $\langle\Psi(t)|\Psi(0)\rangle$
    + $\langle\Psi(t)|\hat{x}|\Psi(t)\rangle$
    + $\langle\Psi(t)|\hat{y}|\Psi(t)\rangle$
    + $\langle\Psi(t)|\hat{z}|\Psi(t)\rangle$
    + $\langle\Psi(t)|\hat{H}_{0}|\Psi(t)\rangle$
    


## scripts
All the python scripts used to optimize basissets, run TDCI simulations and analyse the data are provided here.

## notebooks
The visualizations of our data from TDCI simulations as ipython notebooks.

NOTE:
While necessary comments and citations have been provided where needed, please do write to us if you find something is missing somewhere.
