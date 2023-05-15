#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
# sample input 
# $ python -u ../basis_opt.py -cb aug-cc-pvqz -m he.xyz -N 3 -l 3 -c 0 -a He -nthr 22 &> he_aqz_330.out &
# 
import os
import sys
import numpy as np
import basis_set_exchange as bse
import psi4
from scipy import optimize
from time import perf_counter
import argparse
import textwrap


def gen_init_params(N, lmax, c):
    """Generates initial guess parameters
    """
    num_bass = np.zeros(lmax+1, dtype=int)
    for i_l in range(lmax+1):
        num_bass[i_l] = N - i_l*c
    a = [0.1*i_l + ibas+0.1  for i_l in range(lmax+1) for ibas in range(1, num_bass[i_l]+1)]
    b = [0.1*i_l + ibas+0.1  for i_l in range(lmax+1) for ibas in range(1, num_bass[i_l]+1)]
    params = np.array(a+b)
    return params, num_bass


def gen_hybrid_basstring(params, num_bass, lmax, c=0):
    """Generates hybrid basis strings
    """
    Nparams = len(params)
    Nvals = int((Nparams)/2)
    a = params[0:Nvals]
    b = params[Nvals:Nparams]
    orb = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    zeff = 1.7
    hybrid_basstring = ''
    k = 0
    for i_l in range(lmax+1):
        for ibas in range(1,num_bass[i_l]+1):
            hybrid_basstring += "{orb}    {nbl:<2d}   {Ci:3.2f}\n".format(orb=orb[i_l], nbl=1, Ci=1.00)
            alpha = (zeff / (2*(a[k]*ibas + b[k])) )**2
            hybrid_basstring += "\t  {exp:<15.6E} \t \t {coeff:<8.6E}\n".format(exp=alpha, coeff=1.0).replace("E", "D")
            k += 1
    return hybrid_basstring


def set_basis(basis_dict):
    def basisspec_psi4_yo__mybasis(mol, role):
        basstrings = {}
        mol.set_basis_all_atoms("ALLATOMS",)
        basstrings['allatoms'] = basis_dict['basstring']
        return basstrings
    psi4.qcdb.libmintsbasisset.basishorde['USERDEF'] = basisspec_psi4_yo__mybasis
    psi4.set_options({'basis': 'userdef'})


def set_psi4(ncore=22, memory='256 Gb', outfile='psi4_output.dat', psi4options={}):
    """Sets core parameters for psi4
    """
    psi4.core.set_output_file(outfile, False)  # psi4 output
    psi4.core.set_num_threads(ncore)
    psi4.set_memory(memory)
    options_dict = {
        'puream': False,          # 'True' to use spherical basis sets
        'scf_type': 'pk',         # run density fitted
        'maxiter': 150,           # maximum iterations for SCF
        'reference': 'rhf',       # restricted hartree-fock
        'guess': 'core',          # initial guess for SCF
        'freeze_core': False,     # for larger calculations freeze core
        'e_convergence': 1e-8,    # for tighter energy convergence
        'd_convergence': 1e-10,   # for tighter density convergence
        'ints_tolerance': 1e-12,  # for tighter eri threshold
    }
    # setting any user provided psi4options
    options_dict.update(psi4options)
    psi4.set_options(options_dict)
    return 0


def get_energy(params, mol, basis_dict, atom, num_bass, lmax, c, basis_outfile='userdef.txt'):
    """Computes the ground state energy for a given hybrid basis.
    """
    hybrid_basstring = gen_hybrid_basstring(params, num_bass, lmax, c)
    basstring = ''
    fobj = open(basis_outfile, 'wb')
    for elem in basis_dict.keys():
        if elem==atom:
            basstring += basis_dict[elem] + hybrid_basstring+'****\n'
            fobj.write((basis_dict[elem] + '! hybrid functions\n' + hybrid_basstring+'****\n').encode("utf-8"))
        else:
            basstring += basis_dict[elem] + '****\n'
            fobj.write((basis_dict[elem] + '****\n').encode("utf-8"))
    if atom not in basis_dict.keys():
        basstring += 'H1     0\n' + hybrid_basstring + '****\n'
        fobj.write(('H1     0\n' + hybrid_basstring + '****\n').encode("utf-8"))
    fobj.close()
    set_basis({'basstring': basstring})
    try:
        energy, wfn = psi4.energy('scf', return_wfn=True, molecule=mol)
    except:
        energy = 0.0
    psi4.core.clean()
    print('Energy: {energy:.16f} [Eh]'.format(energy=energy))
    return energy


def energy_minimizer(params, atom, core_basis, lmax, num_bass, c, 
                     molstr, ncore=4, memory='256 Gb',
                     basis_outfile='userdef.txt', 
                     psi4_outfile='psi4_output.dat',
                     psi4options={}, conv_options={}, 
                     opt_method='Nelder-Mead', 
                     add_field=False, field_strength=(0.0, 0.0, 0.0)):
    if add_field:
        psi4options['perturb_h'] = True
        psi4options['perturb_with'] = 'dipole'
        fx, fy, fz = field_strength
        psi4options['perturb_dipole'] = [fx, fy, fz]
    set_psi4(ncore, memory, psi4_outfile, psi4options)
    mol = psi4.geometry(molstr)
    atoms_list = list(mol.to_dict()['elem'])
    if 'Gh' in atoms_list:
        atoms_list.remove('Gh')
    corebasis_dict = {}
    for element in atoms_list:
        print(element)
        corebasis_dict[element] = bse.get_basis(core_basis, fmt='gaussian94', header=False, elements=[element]).split('****')[0]
    options = ({'maxiter':15000, 'disp': True})
    if opt_method == 'BFGS':
        options['gtol'] = 1e-5
    elif opt_method == 'Nelder-Mead':
        # options['fatol'] = 1e-7
        options['adaptive'] = True
    options.update(conv_options)
    res = optimize.minimize(get_energy, params,jac='3-point',
                            args=(mol, corebasis_dict, atom, 
                            num_bass, lmax, c, basis_outfile),
                            method=opt_method, options=options)
    params_final = res.x
    energy_final = res.fun
    return energy_final, params_final


if __name__ == '__main__':
    # parse the input parameters
    parser = argparse.ArgumentParser(
        #formatter_class=argparse.RawDescriptionHelpFormatter,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )    
    parser.add_argument('-m', '--molfile', type=str, 
    help="Path to the file with input geometry in xyz format.")
    parser.add_argument('-cb', '--corebasis', type=str,
    help="Core basisset to which K-functions are to be added.")
    parser.add_argument('-a', '--atom', type=str,
    help="Atom 'X' on which to place the K-functions.")
    parser.add_argument('-N', '--N', type=int,
    default=4,
    help="Number of K-functions to add on the atom 'X' for every l upto lmax.")
    parser.add_argument('-l', '--lmax', type=int,
    default=4,
    help="Highest angular momentum of K-functions to be added.")
    parser.add_argument('-c', '--c', type=int,
    default=0,
    help="Parameter to reduce no. of K-functions. N-l-c functions are added for a given (N,l,c)")
    parser.add_argument('-fb', '--field_bool', type=int,
    default=0,
    help="Switch to turn ON/OFF the static electric field")
    parser.add_argument('-fx', '--fx', type=float,
    default=0.0, 
    help="Strength of the static electric field along x (in a.u.)")
    parser.add_argument('-fy', '--fy', type=float,
    default=0.0, 
    help="Strength of the static electric field along y (in a.u.)")
    parser.add_argument('-fz', '--fz', type=float,
    default=0.0, 
    help="Strength of the static electric field along z (in a.u.)")
    parser.add_argument('-nthr', '--numthreads', type=int, 
    default=2,
    help="Number for processor thread that can be allocated for this job.")
    args = parser.parse_args()
    ncore = args.numthreads
    # setting some environment variables
    os.environ['OMP_NUM_THREADS'] = str(ncore)
    os.environ["OPENBLAS_NUM_THREADS"] = str(ncore)
    os.environ["MKL_NUM_THREADS"] = str(ncore)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncore)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncore)
    # reading molfile into molstr
    molfile = args.molfile        
    with open(molfile, 'r') as input_file:
        molstr = input_file.read()
    # params input
    N = args.N
    lmax = args.lmax
    c = args.c
    # atom params
    atom = args.atom
    core_basis = args.corebasis
    molname = os.path.splitext(os.path.basename(molfile))[0]
    jobname = '{mol}_{N}{l}{c}'.format(mol=molname, N=N,l=lmax,c=c)
    psi4_outfile = jobname+'_emin_psi4.out'
    basis_outfile = jobname+'_emin.gbs'
    #field params
    add_field = args.field_bool
    Fx = args.fx
    Fy = args.fy
    Fz = args.fz
    field_strength=(Fx, Fy, Fz)    
    print('Input Parameters:')
    print('-----------------')
    print((' N : {N:2d}\n' +
        ' c : {c:2d}\n' +
        ' lmax : {lmax:2d}\n' +
        ' atom : {atom}\n' +
        ' core basis : {core_basis}\n' +
        ' field : {field}\n' +
        ' Fx : {Fx:3.4f}\n' +
        ' Fy : {Fy:3.4f}\n' +
        ' Fz : {Fz:3.4f}\n').format(lmax=lmax, N=N, c=c, atom=atom,
                                    core_basis=core_basis,
                                    field=add_field, Fx=Fx, Fy=Fy, Fz=Fz))
    # Running the minimizer
    print('Generating initial parameters \n')
    params, num_bass = gen_init_params(N, lmax, c)
    print('Begining the optimization process \n')
    psi4options = {'scf_type':'pk', 
                   # 's_orthogonalization':'symmetric',
           'guess': 'core',
           's_tolerance':1e-7,
           's_cholesky_tolerance':1e-8,
}
    start = perf_counter()
    energy_final, params_final = energy_minimizer(params, atom, core_basis,
                       lmax, num_bass, c, molstr=molstr, 
               ncore=ncore,opt_method='BFGS',
                       psi4_outfile=psi4_outfile, psi4options=psi4options,
             basis_outfile=basis_outfile, add_field=add_field,field_strength=field_strength)
    stop = perf_counter()
    np.save(jobname+'_emin', params_final)
    print('Optimization process ran for {t: 6.4f} seconds'.format(t=(stop-start)))
    print('Energy: {E:.12f} [Eh]'.format(E=energy_final))