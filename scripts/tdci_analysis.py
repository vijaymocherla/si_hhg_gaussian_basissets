#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
# 

import os
import numpy as np
from hhg_analysis import *
from plotparams import *
# Some constant internal varaibles
_path2images='../simulating_hhg_with_gaussian_basis_sets/images/'


def get_tdprop_data(filename):
    prop_data = np.loadtxt(filename, skiprows=1)
    time_fs = prop_data[:, 0]
    norm = prop_data[:, 1]
    autocorr = prop_data[:, 2]
    X, Y, Z = prop_data[:, 3], prop_data[:, 4], prop_data[:, 5]
    energy = prop_data[:, 6]
    dipoles = (X, Y, Z)
    # Fx, Fy, Fz = prop_data[:, 7], prop_data[:, 8], prop_data[:, 9]
    # fields = (Fx, Fy, Fz)
    return (time_fs, norm, autocorr, dipoles, energy)

def get_tdci_data(molpath, run, method='rk4'):
    jobs = [(method+'_tdprop.txt'), ('lt_'+method+'_tdprop.txt')]
    wd = os.path.join(molpath,run)
    mol = molpath.split('/')[-1]
    data_dict = {}
    atom = mol.split('_')[0]
    ip = {
    'h':   0.02771614,  #  0.754195 eV (Source: Lykke et. al. (1991). Phys. Rev.A, 43(11), 6104.
    'h2':  0.56689375, # 15.425930 eV (Source: NIST Data base)
    'he':  0.90357185, # 24.587387 eV (Source: NIST Data base)
    'li':  2.77972859,  # 75.640097 eV (Source: NIST Data base)
    'ne':  0.79248402, # 21.564540 eV (Source: NIST Data base)
    'f':   0.12499539,  #  3.401290 eV (Source: Chem. Rev. 2002, 102, 231−282)
    'na':  1.73774162,  # 47.286380 eV (Source: NIST Data base)
    'h2o': 0.46381424, # 12.621000 eV (Source: NIST Data base)
    'nh3': 0.37006651, # 10.070000 eV (Source: NIST Data base)
    'ch4': 0.46341000, # 12.610000 eV (Source: NIST Data base)
    }   
    run_info = run.split('_')
    intensity = run_info[1]
    sigmafs = float(run_info[2][:-2])/2
    wavelength = float(run_info[3][:-2])
    w0 = wavelength_to_freq(wavelength)
    E0 = calc_E0(float(intensity))
    Up = (E0/(2 * w0))**2
    e_cutoff = ip[atom] + (3.17*Up)
    info_dict = {'mol': mol,
                 'run': run,
                 'atom': atom,
                 'ip': ip[atom],
                 'e_cutoff': e_cutoff,
                 'intensity': intensity, 
                 'sigmafs': sigmafs, 
                 'wavelength': wavelength, 
                 'w0': w0, 
                 'E0': E0, 
                 'Up': Up,
                }
    data_dict['info'] = info_dict
    data_dict['jobs'] = jobs
    for filename in jobs:
        time_fs, norm, autocorr, dipoles, energy = get_tdprop_data(os.path.join(wd,filename))
        dt = time_fs[2] - time_fs[1]
        freq, *Glist = calc_Gobs(time_fs, dipoles[2], dt_fs=dt, return_moments=True)
        
        hhg_dict = {}
        hhg_vals = Glist
        hhg_keys = ['hhg_dip', 'hhg_vel', 'hhg_acc']
        hhg_dict = {hhg_keys[i]: hhg_vals[i] for i in range(3)}
        hhg_dict['time_fs'] = time_fs
        hhg_dict['dip'] = dipoles[2]
        hhg_dict['ho'] = freq/w0
        hhg_dict['energy'] = energy
        hhg_dict['autocorr'] = autocorr
        hhg_dict['norm'] = norm
        data_dict[filename] = hhg_dict
    return data_dict

def plot_dipole_dyn(data, method='rk4', savefig=True, imgtype='png'):    
    data_info = data['info']
    mol = data_info['mol'] 
    run = data_info['run'] 
    atom = data_info['atom']
    ip = data_info['ip']
    w0 = data_info['w0']
    intensity = data_info['intensity']
    e_cutoff = data_info['e_cutoff']
    fig = plt.figure(figsize=(8,6))
    data_info = data['info']
    data_mt = data[data['jobs'][0]]
    data_lt_mt = data[data['jobs'][1]]
    plt.plot(data_mt['time_fs'], data_mt['dip'], '--', label=method.upper()+' w/o $\Gamma_k$')
    plt.plot(data_lt_mt['time_fs'], data_lt_mt['dip'], label=method.upper()+' w/ lifetime')
    plt.legend()
    plt.title(data_info['mol'])
    plt.xlabel('time (fs)')
    plt.ylabel('dipole (a.u.)')
    plt.title(mol+' I$_0$ =   {I:} W/cm$^2$'.format(I=intensity))
    fig.tight_layout()
    if savefig:
        plt.savefig('{mol}_{run}_dipole.{imgtype}'.format(mol=data_info['mol'],run=data_info['run'],imgtype=imgtype),  dpi=1000, format=imgtype)
    plt.show()


def plot_hhg(data, xlim=[0,120], ylim=[-20,5], savefig=True, imgtype='png'):
    data_info = data['info']
    mol = data_info['mol'] 
    run = data_info['run'] 
    atom = data_info['atom']
    ip = data_info['ip']
    w0 = data_info['w0']
    intensity = data_info['intensity']
    e_cutoff = data_info['e_cutoff']
    data_mt = data[data['jobs'][0]]
    data_lt_mt = data[data['jobs'][1]]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(data_mt['ho'], np.log10(data_mt['hhg_vel']), '-', color='tomato', label='w/o $\Gamma_k$')
    ax.plot(data_lt_mt['ho'], np.log10(data_lt_mt['hhg_vel']), '-', color='blue', alpha=0.5, label=' w/ lifetime')
    ax.axvline(ip/w0, color='grey', linestyle='dotted', linewidth=1.0)
    ax.axvline(e_cutoff/w0, color='grey', linestyle='dashed', linewidth=0.85, alpha=0.6)
    # setting custom x and y limits
    #xlim[1] = e_cutoff/w0 + 45
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])
    # rendering and saving the plot
    ax.set_xlabel('Harmonic orders ($N=\omega / \omega_{0} $)')
    ax.set_ylabel('$\log_{10}$(I$_{\mathbf{HHG}}$) ')
    ax.set_title('HHG plots for {mol:} I$_0$ =   {I:} W/cm$^2$'.format(mol=mol,I=intensity))
    ax.text(0.850, 0.90,
    ('$\cdots\cdots$ \ $I_p$ \ \ \  : {ionp: 3.4f} E$_h$ \n'+
     '$--$ \ E$_\mathbf{cut}$ : {e_cutoff:3.4f} E$_h$\n').format(cut='{cut}',ionp=ip, e_cutoff=e_cutoff),
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, fontsize=14)
    ax.legend( bbox_to_anchor=[0.85, 0.80], 
           loc='center', ncol=1, fontsize=14)
    fig.tight_layout()
    if savefig:
        plt.savefig('{mol}_{run}_hhg.{imgtype}'.format(mol=mol,run=run,imgtype=imgtype),  dpi=1000, format=imgtype)
    plt.show()


def plot_energy(data, method='rk4',savefig=True, imgtype='png'):
    data_info = data['info']
    mol = data_info['mol'] 
    run = data_info['run'] 
    atom = data_info['atom']
    ip = data_info['ip']
    w0 = data_info['w0']
    intensity = data_info['intensity']
    e_cutoff = data_info['e_cutoff']
    fig, ax1 = plt.subplots(figsize=(8,6))
    data_mt = data[data['jobs'][0]]
    data_lt_mt = data[data['jobs'][1]]
    ax2 = ax1.twinx()
    ln1, = ax1.plot(data_mt['time_fs'], data_mt['energy']-data_mt['energy'][0], label=method, color='tomato')
    ln2, = ax2.plot(data_lt_mt['time_fs'], data_lt_mt['energy']-data_lt_mt['energy'][0], label='lt'+method, color='mediumblue')
    ax1.set_xlabel('time (fs)')
    ax1.set_ylabel('$\Delta$ E (au)')
    ax2.set_ylabel('$\Delta$ E (au)')
    ax2.yaxis.label.set_color(ln2.get_color())
    ax2.spines["right"].set_edgecolor(ln2.get_color())
    ax2.tick_params(axis='y', colors=ln2.get_color())
    lns = [ln1] + [ln2]
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels,loc='center', bbox_to_anchor=[0.75, .75])
    ax1.set_title(mol+' I$_0$ =   {I:} W/cm$^2$'.format(I=intensity))
    fig.tight_layout()
    plt.show()
    if savefig:
        fig.savefig('{mol}_{run}_energy.{imgtype}'.format(mol=mol,run=run,imgtype=imgtype), dpi=1000, format=imgtype)


def plot_autocorr(data, method='rk4', savefig=True, imgtype='png'):
    data_info = data['info']
    mol = data_info['mol'] 
    run = data_info['run'] 
    atom = data_info['atom']
    ip = data_info['ip']
    w0 = data_info['w0']
    intensity = data_info['intensity']
    e_cutoff = data_info['e_cutoff']
    data_mt = data[data['jobs'][0]]
    data_lt_mt = data[data['jobs'][1]]
    fig, (ax1,ax2) = plt.subplots(1,2 , figsize=(12,6))
    ax1.plot(data_mt['time_fs'], data_mt['autocorr'], label='autocorr '+method, color='tomato')
    ax1.plot(data_lt_mt['time_fs'], data_lt_mt['autocorr'], label='autocorr lt'+method, color='mediumblue')
    ax2.plot(data_mt['time_fs'], data_mt['norm'], '-.', label='norm '+method, color='tomato')
    ax2.plot(data_lt_mt['time_fs'], data_lt_mt['norm'], '-.', label='norm lt'+method, color='mediumblue')
    ax1.set_xlabel('time (fs)')
    ax2.set_xlabel('time (fs)')
    ax1.set_ylabel('$|\langle \Psi_{t} | \Psi_{0} \\rangle|^2$')
    ax2.set_ylabel('$|\langle \Psi_{t} | \Psi_{t} \\rangle|^2$')
    ax1.legend(loc='center', bbox_to_anchor=[0.75, .75])
    ax2.legend(loc='center', bbox_to_anchor=[0.75, .75])
    y_fmt = mpl.ticker.FormatStrFormatter('%6.4f')
    ax1.yaxis.set_major_formatter(y_fmt)
    fig.suptitle(mol+' I$_0$ =   {I:} W/cm$^2$  \n \n Autocorrelation and Norm plots'.format(I=intensity))
    fig.tight_layout()
    plt.show()
    if savefig:
        fig.savefig('{mol}_{run}_autocorr.{imgtype}'.format(mol=mol,run=run,imgtype=imgtype), dpi=1000, format=imgtype)


def plot_data(mol, run, xlim=[0,120], ylim=[], savefig=False, method='rk4'):
    # getting tdci data
    data = get_tdci_data(mol, run, method)
    # plotting the data
    # uncomment the following lines for other data
    plot_energy(data, savefig=savefig)
    plot_autocorr(data, savefig=savefig)
    plot_dipole_dyn(data, savefig=savefig)
    plot_hhg(data, xlim, ylim, savefig=savefig)


def plot_hhg_pulse_comp(datadir, pulses, basis, 
                        method='rk4', xlim=[-0.2,60], ylim=[-18,-2], legend_pos=[0.65, 0.79],
                        savefig=True, imgtype='pdf', figsize=(22, 7), fontsize=19, return_data=False):
    """Plots and compares the HHG spectra of different pulse-envelope shapes.
    """
    job = datadir.split('/')[-1]
    for pulse in pulses:
        data = get_tdci_data(datadir, pulse)
        tdprop_file = '{method}_tdprop.txt'.format(method=method) 
        lifetime_tdprop_file = 'lt_{method}_tdprop.txt'.format(method=method) 
        if pulse.split('_')[0] == 'zs':
            zs = data[tdprop_file]
            zs_lt = data[lifetime_tdprop_file]
        elif pulse.split('_')[0] == 'zg':
            zg = data[tdprop_file]
            zg_lt = data[lifetime_tdprop_file]
        elif pulse.split('_')[0] == 'zt':
            zt = data[tdprop_file]
            zt_lt = data[lifetime_tdprop_file]
        else:
            raise Exception("Data not found!")
    # getting run parameters
    mol = data['info']['mol'] 
    run = data['info']['run'] 
    atom = data['info']['atom']
    ip = data['info']['ip']
    w0 = data['info']['w0']
    intensity = data['info']['intensity']
    e_cutoff = data['info']['e_cutoff']
    # Plotting the data
    fig, axs = plt.subplots(1, 3, figsize=(20,6), facecolor='white')
    # cos$^{2}$
    axs[0].plot(zs_lt['ho'], np.log10(zs_lt['hhg_vel']), '-', color='blue', label='cos$^2$ {basis}'.format(basis=basis))
    axs[0].plot(zs['ho'], np.log10(zs['hhg_vel']), '-', color='orchid',  
                alpha=0.65, label='cos$^2$ {basis} w/o $\Gamma_k$'.format(basis=basis))
    # Gaussian
    axs[1].plot(zg_lt['ho'], np.log10(zg_lt['hhg_vel']), '-', color='crimson', label='Gauss. {basis}'.format(basis=basis))
    axs[1].plot(zg['ho'], np.log10(zg['hhg_vel']), '-', color='orange', 
                alpha=0.65, label='Gauss. {basis} w/o $\Gamma_k$'.format(basis=basis))
    # Trapezoidal
    axs[2].plot(zt_lt['ho'], np.log10(zt_lt['hhg_vel']), '-', color='green', label='Trap. {basis}'.format(basis=basis))
    axs[2].plot(zt['ho'], np.log10(zt['hhg_vel']), '-', color='deepskyblue', 
                alpha=0.65, label='Trap. {basis} w/o $\Gamma_k$'.format(basis=basis))
    # Setting custom x and y limits
    for ax in axs:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', width=0.9, length=8, direction='out')
        ax.tick_params(which='minor', width=.3, length=4, direction='out')
        ax.tick_params(axis='both', labelsize=fontsize+2)
        ax.axvline(ip/w0, color='darkgray', linestyle='dotted', linewidth=1.0)
        ax.axvline(e_cutoff/w0, color='darkgray', linestyle='dashed', linewidth=0.85, alpha=0.6)
        ax.set_xlabel('Harmonic orders \ ($ N=\omega / \omega_{0} $)', fontsize=fontsize+4)
        ax.set_ylabel('$\log_{10}$(I$_{\mathbf{HHG}}$) ', fontsize=fontsize+4)
        ax.text(0.710, 0.9, ('$\cdots\cdots$ \ $I_p$ \ \ \  : {ionp: 3.4f} E$_h$ \n'+
            '$--$ \ E$_\mathbf{cut}$ : {e_cutoff:3.4f} E$_h$\n').format(cut='{cut}',ionp=ip, e_cutoff=e_cutoff),
            horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, fontsize=fontsize-3)
        ax.legend(bbox_to_anchor=legend_pos, loc='center', ncol=1, fontsize=fontsize-3.5, facecolor='white')
    plt.suptitle('Helium HHG plots for I$_0$ =   {I:} W/cm$^2$'.format(I=intensity), fontsize=fontsize+4)
    fig.tight_layout()
    if savefig:
        filename = '{path2images}{job}_{intensity}_pulse_comp_hhg.{imgtype}'.format(path2images=_path2images,job=job, intensity=intensity,imgtype=imgtype)
        plt.savefig(filename, dpi=1000, format=imgtype)
    if return_data:
        return data, fig, axs


def plot_hhg_basis_comp(basissets, pulse, label, 
                        atom_dir='helium', method='rk4', use_lifetime=True, 
                        xlim = [-0.2, 100], ylim = [-28,24], figsize=[7.2,6], fontsize=20,
                        savefig=True, imgtype='pdf', 
                        legend_pos=[0.8, 0.5], return_data=False):
    """Plots and compares the HHG spectra of different basissets.
    """
    datadir = lambda job : os.path.join(os.path.realpath('../'),'data/tdci/{atom}/{job}'.format(atom=atom_dir, job=job))
    tdprop_file = '{method}_tdprop.txt'.format(method=method) 
    lifetime_tdprop_file = 'lt_{method}_tdprop.txt'.format(method=method) 
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    colors = ['red','green','blue','orange','darkorchid']
    ishift = len(basissets)*10
    i = 0 # iterval for color
    for basis in basissets:
        data = get_tdci_data(datadir(basissets[basis]), pulse)
        tdci = data[tdprop_file]
        tdci_lt = data[lifetime_tdprop_file]
        if use_lifetime:
            ax.plot(tdci_lt['ho'], np.log10(tdci_lt['hhg_vel'])+ishift, '-', color=colors[i], label=basis)
        else:
            ax.plot(tdci['ho'], np.log10(tdci['hhg_vel'])+ishift, '-', color=colors[i], label='{basis} w/o $\Gamma_k$'.format(basis=basis))
        ishift -= 10
        i += 1
    # getting run parameters
    atom = data['info']['atom']
    ip = data['info']['ip']
    w0 = data['info']['w0']
    intensity = data['info']['intensity']
    e_cutoff = data['info']['e_cutoff']
    # setting plot parameters to render and save figure
    ax.axvline(ip/w0, color='grey', linestyle='dotted', linewidth=1.5)
    ax.axvline(e_cutoff/w0, color='grey', linestyle='dashed', linewidth=1.25, alpha=0.6)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=0.9, length=8, direction='out')
    ax.tick_params(which='minor', width=.3, length=4, direction='out')
    ax.tick_params(axis='both', labelsize=fontsize+2)
    ax.set_xlabel('Harmonic orders \ ($ N=\omega / \omega_{0} $)', fontsize=fontsize+4)
    ax.set_ylabel('$\log_{10}$(I$_{\mathbf{HHG}}$) ',fontsize=fontsize+4)
    ax.set_title('I$_0$ =   {I:} W/cm$^2$'.format(I=intensity), fontsize=fontsize+4)
    ax.text(0.75, 0.85, ('$\cdots\cdots$ \ $I_p$ \ \ \  : {ionp: 3.4f} E$_h$ \n'+
        '$--$ \ E$_\mathbf{cut}$ : {e_cutoff:3.4f} E$_h$\n').format(cut='{cut}',ionp=ip, e_cutoff=e_cutoff),
        horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, fontsize=fontsize-1)
    ax.legend(bbox_to_anchor=legend_pos, loc='center', ncol=1, fontsize=fontsize-2, facecolor='white')
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=6)
    fig.tight_layout()
    if savefig:
        filename = '{path2images}{atom}_{label}_{intensity}_hhg_comp.{imgtype}'.format(path2images=_path2images,atom=atom, label=label, intensity=intensity,imgtype=imgtype)
        plt.savefig(filename, dpi=1000, format=imgtype)
    if return_data:
        return data, fig, ax


def plot_comp_lifetimes(jobs, pulse, labels, atom_dir='helium', method='rk4', 
                        xlim=[-0.2,25], ylim=[-24,6], figsize=(7.64,6.4), fontsize=20,
                        savefig=True, imgtype='pdf', 
                        plot_label='lifetime', colors= ['crimson','darkblue'], 
                        subcolors = ['orange','orchid'], legend_pos=[0.78,.75],
                        return_data=False):
    datadir = lambda job : os.path.join(os.path.realpath('../'),'data/tdci/{atom}/{job}'.format(atom=atom_dir, job=job))
    tdprop_file = '{method}_tdprop.txt'.format(method=method) 
    lifetime_tdprop_file = 'lt_{method}_tdprop.txt'.format(method=method) 
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    i = 0 # iterval for color
    for job,label in zip(jobs,labels):
        ishift = i*10
        data = get_tdci_data(datadir(job), pulse)
        tdci = data[tdprop_file]
        tdci_lt = data[lifetime_tdprop_file]
        ax.plot(tdci_lt['ho'], np.log10(tdci_lt['hhg_vel'])+ishift, '-', color=colors[i], label=label)
        ax.plot(tdci['ho'], np.log10(tdci['hhg_vel'])+ishift, '-', color=subcolors[i], label='{label} w/o $\Gamma_k$'.format(label=label), alpha=0.65)
        i += 1
    # getting run parameters
    atom = data['info']['atom']
    ip = data['info']['ip']
    w0 = data['info']['w0']
    intensity = data['info']['intensity']
    e_cutoff = data['info']['e_cutoff']
    # setting plot parameters to render and save figure
    ax.axvline(ip/w0, color='grey', linestyle='dotted', linewidth=1.5)
    ax.axvline(e_cutoff/w0, color='grey', linestyle='dashed', linewidth=1.25, alpha=0.6)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=0.9, length=8, direction='out')
    ax.tick_params(which='minor', width=.3, length=4, direction='out')
    ax.tick_params(axis='both', labelsize=fontsize+2)
    ax.set_xlabel('Harmonic orders \ ($ N=\omega / \omega_{0} $)', fontsize=fontsize+4)
    ax.set_ylabel('$\log_{10}$(I$_{\mathbf{HHG}}$) ', fontsize=fontsize+4)
    ax.set_title('I$_0$ = {I:} W/cm$^2$'.format(I=intensity), fontsize=fontsize+4)
    ax.text(0.7, 0.9, ('$\cdots\cdots$ \ $I_p$ \ \ \  : {ionp: 3.4f} E$_h$ \n'+
        '$--$ \ E$_\mathbf{cut}$ : {e_cutoff:3.4f} E$_h$\n').format(cut='{cut}',ionp=ip, e_cutoff=e_cutoff),
        horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, fontsize=fontsize-1)
    ax.legend(bbox_to_anchor=legend_pos, loc='center', ncol=1, fontsize=fontsize-2, facecolor='white')
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=6)
    fig.tight_layout()
    if savefig:
        filename = '{path2images}{atom}_{label}_{intensity}_hhg_comp.{imgtype}'.format(path2images=_path2images,atom=atom, label=plot_label, intensity=intensity,imgtype=imgtype)
        plt.savefig(filename, dpi=1000, format=imgtype)
    if return_data:
        return data, fig, ax
