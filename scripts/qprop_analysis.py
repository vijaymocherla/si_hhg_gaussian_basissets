#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
import numpy as np
from hhg_analysis import *
from tdci_analysis import get_tdci_data
from plotparams import *

# Some constant internal varaibles
_path2images='../simulating_hhg_with_gaussian_basis_sets/images/'

def get_qprop_data(path2file):
    filename = 'real_prop_obser.dat'
    data = np.genfromtxt(path2file+filename)
    time_fs  = data[:, 0]/fs_to_au
    energy = data[:, 1]
    autocorr = data[:, 2]
    norm = data[:, 3]
    dpz = data[:,4]
    acc_real = data[:, 5]
    acc_imag = data[:,6]
    return (time_fs, energy, autocorr, norm, dpz, acc_real, acc_imag)
    
def read_spectra(filename, params=(0.0569, 20, 400)):
    w0, n, npc = params
    file = open(filename, 'r')
    data = file.readlines()
    file.close()
    spectra = np.array([float(i.replace('*^','E')) for i in data])
    sigma = np.pi*n/(w0) 
    dt = 2*sigma/(n*npc)
    freq = 2*np.pi*np.fft.fftfreq(n*npc,dt) 
    freq = freq[freq>=0]
    return freq, spectra


def plot_sae(sae_path, tdci_path, tdci_job, basis, text_pos=[0.75, 0.7], legend_pos=[0.75, 0.87],
            xlim = [-.10,90], ylim = [-20,-0], figsize=[7.2, 6], 
            savefig=True, return_data=False, fontsize=19,
            plot_label='qprop', imgtype='pdf'):
    # filename = sae_path+'real_prop_obser.dat'
    (time_fs, energy, autocorr, norm, dpz, acc_real, acc_imag) = get_qprop_data(sae_path)
    freq, *glist = calc_Gobs(time_fs, dpz, return_moments=True)
    hhg_qprop = [glist[1][i] for i in range(len(glist[1]))]
    # hhg_qprop = [1/freq[i]**2 * glist[0][i] for i in range(len(glist[1]))]
    tdci_data = get_tdci_data(tdci_path, tdci_job)
    freq_tdci, hhg_tdci = tdci_data['lt_rk4_tdprop.txt']['ho'], tdci_data['lt_rk4_tdprop.txt']['hhg_vel']
    data_info = tdci_data['info']
    fig, ax = plt.subplots(figsize=figsize)
    atom = data_info['atom']
    w0 = data_info['w0']
    ip = data_info['ip']
    intensity = data_info['intensity']
    ecut = data_info['e_cutoff']
    ax.plot(freq_tdci, np.log10(hhg_tdci), label='TD-CIS/{basis}'.format(basis=basis), linewidth=1.0)
    ax.plot(freq/w0, np.log10(hhg_qprop), label='SAE', linewidth=1.75, alpha=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Harmonic orders \ ($ N=\omega / \omega_{0} $)', fontsize=fontsize+4)
    ax.set_ylabel('$\log_{10}$(I$_{\mathbf{HHG}}$) ',fontsize=fontsize+4)
    # atom_labels = {'he':'Helium', 'h':'H$^-$', 'li':'Li$^+$'}
    # ax.set_title('{atom} HHG  for I$_0$ =  {I:2.1e} W/cm$^2$'.format(I=float(intensity),atom=atom_labels[atom]), fontsize=fontsize+4)
    ax.axvline(ip/w0, color='grey', linestyle='dotted', linewidth=1.0)
    ax.axvline(ecut/w0, color='grey', linestyle='dashed', linewidth=0.85, alpha=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=0.9, length=8, direction='out')
    ax.tick_params(which='minor', width=.3, length=4, direction='out')
    ax.tick_params(axis='both', labelsize=fontsize+2)
    ax.text(*text_pos, ('$\cdots\cdots$ \ $I_p$ \ \ \  : {ionp: 3.4f} E$_h$ \n'+
            '$--$ \ E$_\mathbf{cut}$ : {e_cutoff:3.4f} E$_h$\n').format(cut='{cut}',ionp=ip, e_cutoff=ecut),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes, fontsize=fontsize)
    ax.legend(bbox_to_anchor=legend_pos, loc='center', ncol=1, fontsize=fontsize-2, facecolor='white')
    fig.tight_layout()
    if savefig:
        filename = '{path2images}{atom}_{label}_{intensity}_hhg_comp.{imgtype}'.format(path2images=_path2images,atom=atom, label=plot_label, intensity=intensity,imgtype=imgtype)
        plt.savefig(filename, dpi=1000, format=imgtype)
    data = {'freq': freq, 'hhg': hhg_qprop}
    if return_data:
        return data, fig, ax
