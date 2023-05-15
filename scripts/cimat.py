#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
import os 
import sys
import numpy as np
import endyn
import psi4
from threadpoolctl import threadpool_limits
from time import perf_counter 
import argparse
import textwrap

# parse the input parameters
parser = argparse.ArgumentParser(
      #formatter_class=argparse.RawDescriptionHelpFormatter,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('-nthr', '--numthreads', type=int, 
    default=2,
    help="Number for processor thread that can be allocated for this job.")

parser.add_argument('-m', '--molfile', type=str, 
    help="path to the file with input geometry in xyz format.")

parser.add_argument('-b', '--basisfile', type=str, 
    help="path to the .gbs file with exponents and coefficients of basis sets in psi4 format.")

parser.add_argument('-s', '--scratchdir', type=str, 
    default=os.getcwd(),
    help="path to the scratch directory to be used for psi4 computations.")
args = parser.parse_args()
basisfile = args.basisfile
molfile = args.molfile
ncore = args.numthreads
scratchdir = args.scratchdir
# setting some environment variables
os.environ['OMP_NUM_THREADS'] = str(ncore)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncore)
os.environ["MKL_NUM_THREADS"] = str(ncore)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncore)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncore)
os.environ["PSI_SCRATCH"] = scratchdir
# reading basisset
file = open(basisfile, 'r')
basisset = ''
for line in file.readlines():
      if line[0] != '!':
            basisset += line 
file.close()
basis_dict = {'basstring': basisset}

# Lets intiate AOint() a subclass of psi4utils()
basis = 'custom_basis'
with threadpool_limits(limits=ncore, user_api='blas'):
	mol = endyn.utils.molecule(basis, molfile, wd='./', ncore=ncore, psi4mem='210 Gb', 
	numpymem=210, custom_basis=True, basis_dict=basis_dict, 
	store_wfn=False, properties=['dipoles'], 
      psi4options={'puream': False, 'scf_type': 'direct', 's_tolerance':1e-8})


print('Begining CISD calculation .....\n')
start = perf_counter()
from endyn.configint.rcisd import CISD
with threadpool_limits(limits=ncore, user_api='blas'):
      cisd = CISD(mol,options={'doubles':True}, ncore=ncore)#, active_space=[1,10])
      ndim = sum(cisd.num_csfs)
      print(cisd.num_csfs, ndim)
      cisd.save_hcisd()
stop = perf_counter()
print('Completed generating Hamiltonian matrix of size ({ndim:d},{ndim:d}) in {time:3.2f} seconds\n'.format(ndim=ndim, time=(stop-start)))
# Calculating and saving dipoles in CSF basis
print('Completed the task and saved data in {time:3.2f}\n'.format(time=(stop-start)))
print('Calculating Dipoles in CSF basis\n')
start = perf_counter()
cisd.save_dpx()
cisd.save_dpy()
cisd.save_dpz()
stop = perf_counter()
print('Completed generating Dipole matrices in {time:3.2f}\n'.format(time=(stop-start)))

# print CISD or CIS groud-state energy for a reference 
start = perf_counter()
print('Diagonalizing HCISD matrix.....\n')
HCISD = np.load('cimat.npz')['HCISD']
e_endyn = cisd.energy(HCISD)
psi4.core.clean()
if cisd.options['doubles']:
    e_psi4 = psi4.energy('CISD')
    print("endyn CISD E0: {e:16.16f}".format(e=e_endyn))
    print("psi4 CISD E0: {e:16.16f}".format(e=e_psi4))
else:
    cisd_psi4 = psi4.energy('scf')
    print("endyn CIS E0: {e:16.16f}".format(e=e_endyn))
    print("psi4 CIS E0: {e:16.16f}".format(e=e_psi4))
print("dE : {dE:1.2E}\n".format(dE=abs(e_psi4 - e_endyn)))
# Getting all the eigenvalues and eigen vectors
start = perf_counter()
with threadpool_limits(limits=ncore, user_api='blas'):
      vals, vecs = cisd.get_eigen(HCISD)
del HCISD
np.savez('hdata.npz', eigvals=vals, eigvecs=vecs, scf_energy=mol.scf_energy, mo_eps=mol.mo_eps[0], csfs=cisd.csfs, num_csfs=cisd.num_csfs)
del vals, vecs
stop = perf_counter()
print('Completed Diagonalization task and saved data in {time:3.2f}\n'.format(time=(stop-start)))
