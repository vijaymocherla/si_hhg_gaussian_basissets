#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
# Python script to run TDCI simulations with heuristic lifetimes to calculate HHG spectra.
# To be run as :
# $ python tdci.py $ARGS
#
import os 
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
from endyn.lib import SplitOperator, RungeKutta, CrankNicholson
from endyn.utils import excite
from endyn.utils.units import fs_to_au, watt_per_cm2_to_au,wavelength_to_freq
from threadpoolctl import threadpool_limits
from time import perf_counter
import argparse
import textwrap

# parse the input parameters
parser = argparse.ArgumentParser(#formatter_class=argparse.RawDescriptionHelpFormatter,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                )
parser.add_argument('-i', '--intensity', type=float, 
    help="Peak pulse intensity of the laser in W/cm^2")
parser.add_argument('-wl', '--wavelength', type=float,
    default=800,
    help="Wavelength of the lase in nm")
parser.add_argument('-n', '--nopcyc', type=float,
    default=10,
    help="No. of optical cycles of the  laser pulse")
parser.add_argument('-a', '--atom' , type=str,
    # default='He',
    help=textwrap.dedent("""Atom or molecule option for calculating lifetimes using ionization potential.\n
		    Following are the available options and their IPs: \n
		    ```
            'H-' : 0.02771614, #  0.754195 eV (Source: Lykke et. al. (1991). Phys. Rev. A, 43(11), 6104.
		    'H2' : 0.56689375, # 15.425930 eV (Source: NIST Data base)
		    'He' : 0.90357185, # 24.587387 eV (Source: NIST Data base)
		    'Li+': 2.77972859, # 75.640097 eV (Source: NIST Data base)
		    'Ne' : 0.79248402, # 21.564540 eV (Source: NIST Data base)
  		    'F-' : 0.12499539, #  3.401290 eV (Source: Chem. Rev. 2002, 102, 231−282)
		    'Na+': 1.73774162, # 47.286380 eV (Source: NIST Data base)
		    'H2O': 0.46381424, # 12.621000 eV (Source: NIST Data base)
		    'NH3': 0.37006651, # 10.070000 eV (Source: NIST Data base)
		    'CH4': 0.46341000, # 12.610000 eV (Source: NIST Data base)
            ```
            """))   
parser.add_argument('-m', '--method', type=str, 
    default="RK4",
    help=textwrap.dedent("""The keyword to choose time propagation method.
		Following are the currently available TDSE propagators.
		```
		   RK4 : 4th order Runge-Kutta
		   SO : Split-Operator 
		   CN : Crank-Nicholson
		```
        """))
parser.add_argument('-lt', '--use_lifetime', type=int, 
    default=1,
    help="Switch for lifetime models, use 1 to apply heuristic lifetime models.")
parser.add_argument('-nthr', '--numthreads', type=int, 
    default=2,
    help="Number for processor thread that can be allocated for this job.")
parser.add_argument('-p', '--pulse', type=str,
    default="sinesqr",
    help="Pulse envelope shape." )
args = parser.parse_args()
# input args
wd = '..'
I = float(args.intensity)
nopcyc = float(args.nopcyc)
atom = args.atom
method = args.method
use_lifetime = int(args.use_lifetime)
ncore = str(args.numthreads)
wl_nm = args.wavelength
pulse = args.pulse
os.environ["OMP_NUM_THREADS"] = ncore 
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore 
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore

# Started loading matrices
start= perf_counter()
print('Loading matrices .....\n')
cimatfile = os.path.join(wd, 'cimat.npz')
hdatafile = os.path.join(wd, 'hdata.npz')
hdata = np.load(hdatafile)
eigvals, eigvecs = hdata['eigvals'], hdata['eigvecs']
scf_energy = hdata['scf_energy']
mo_eps= hdata['mo_eps']
csfs = hdata['csfs']
del hdata
HCISD = np.load(cimatfile)['HCISD']
csf_dpx = np.load(os.path.join(wd, 'dpx.npz'))['csf_dpx']
csf_dpy = np.load(os.path.join(wd, 'dpy.npz'))['csf_dpy']
csf_dpz = np.load(os.path.join(wd, 'dpz.npz'))['csf_dpz']
stop = perf_counter()
print('Completed loading matrices in {time:3.6f} seconds.\n'.format(time=stop-start))
# defining extra ops
ops_list = [csf_dpx, csf_dpy, csf_dpz, HCISD]
headers = "{op1:>20}\t{op2:>20}\t{op3:>20}\t{op4:>20}\t".format(op1="dpx", op2="dpy", op3="dpz", op4="hcisd") 

# Laser params
ti = 0.0
tf = 80.0*fs_to_au
dt = 0.0001*fs_to_au
E0 = watt_per_cm2_to_au(I)
w0 = wavelength_to_freq(wl_nm*1e-9)
phase = 0.0
print_nstep = 100
field_params = (0, 0, 1)
time_params = (ti, tf, dt, print_nstep)
time_array = np.arange(ti, tf, dt)
tp = tf/2.0
laser_params = E0, w0, tp, nopcyc, phase
field_x = np.zeros(time_array.shape[0])
field_y = np.zeros(time_array.shape[0])
if pulse=='sinesqr':
   field_z = np.array([excite.sinesqr_pulse(t, laser_params) for t in time_array])
elif pulse=='gaussian':
   field_z = np.array([excite.gaussian_pulse(t, laser_params) for t in time_array])
elif pulse=='trapezoidal':
   field_z = np.array([excite.trapezoidal_pulse(t, laser_params) for t in time_array])

# Calculating Complex Lifetimes
if use_lifetime == 1:
    # Heuristic Lifetime model
    from endyn.tdci.lifetime import heuristic
    ionp = {
    'H-' : 0.02771614, #  0.754195 eV (Source: Lykke et. al. (1991). Phys. Rev. A, 43(11), 6104.
    'H2' : 0.56689375, # 15.425930 eV (Source: NIST Data base)
    'He' : 0.90357185, # 24.587387 eV (Source: NIST Data base)
    'Li+': 2.77972859, # 75.640097 eV (Source: NIST Data base)
    'Ne' : 0.79248402, # 21.564540 eV (Source: NIST Data base)
    'F-' : 0.12499539, #  3.401290 eV (Source: Chem. Rev. 2002, 102, 231−282)
    'Na+': 1.73774162, # 47.286380 eV (Source: NIST Data base)
    'H2O': 0.46381424, # 12.621000 eV (Source: NIST Data base)
    'NH3': 0.37006651, # 10.070000 eV (Source: NIST Data base)
    'CH4': 0.46341000, # 12.610000 eV (Source: NIST Data base)
    }
    model = heuristic(eigvals, eigvecs, csfs, mo_eps)
    IP = ionp[atom]
    ip_params = E0, w0, IP
    with threadpool_limits(limits=int(ncore), user_api='blas'):
        cmplx_vals = model.cmplx_energies(ip_params, use_d2=True)
else:
    cmplx_vals = eigvals
# defining the intial state
psi0 = np.array(eigvecs[:,0],dtype=np.cdouble)
# starting the propagation
outfilename = 'tdprop.txt'
if use_lifetime:
    outfilename = 'lt_' + method.lower()+'_'+outfilename
else:
    outfilename = method.lower()+'_'+outfilename
start = perf_counter()
if method == "SO":
    SplitOperator.runPropagator(cmplx_vals, eigvecs, psi0, 
                                time_params, field_params, 
                                csf_dpx, csf_dpy, csf_dpz,
                                field_x, field_y, field_z,
                                ops_list, headers, outfilename)
elif method == "RK4":
    RungeKutta.runPropagator(cmplx_vals, eigvecs, psi0,
                             time_params, field_params,
                             csf_dpx, csf_dpy, csf_dpz,
                             field_x, field_y, field_z,
                             ops_list, headers, outfilename)
elif method == "CN":
    CrankNicholson.runPropagator(cmplx_vals, eigvecs, psi0,
                                 time_params, field_params,
                                 csf_dpx, csf_dpy, csf_dpz,
                                 field_x, field_y, field_z,
                                 ops_list, headers, outfilename)
else:
    raise Exception("Given method does not exist, please choose one of the following: SO, RK4 or CN.")
stop = perf_counter()
lt_cond = ""
if use_lifetime:
    lt_cond = "with lifetime models"
else:
    lt_cond = "without lifetime models"
print("Time taken for {method:s} {lt_cond:s}  {time:3.2f} seconds".format(method=method, lt_cond=lt_cond, time=(stop-start)))
