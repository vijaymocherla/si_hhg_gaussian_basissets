#!/usr/bin/env python
#
# Author: Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
import numpy as np
from hhg_analysis import *
from plotparams import *

# Some constant internal varaibles
_path2images='../plots/'

def read_dipole(filename):
    dipoles = []
    file = open(filename, 'r')
    EOF = 0
    while not EOF:
        line = file.readline()
        if line == '':
            EOF = 1
            break
        dps = [complex(di.replace('*^', 'E').replace('*I', 'j').replace(' ','')) 
             for di in line.split('\n')[0][1:-1].split(',')]
        dipoles.append(dps)
    file.close()
    dipoles = np.array(dipoles)
    return dipoles

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
    # freq = 2*np.pi*np.fft.fftfreq(len(time), dt)


def get_sfa_hhg_spectra(filename, params=(0.0569, 20, 500)):
    w0, n, npc = params
    dipoles = read_dipole(filename)
    dpz = dipoles[:-1,0].real
    fs_to_au = 41.341374575751
    sigma = np.pi*n/(w0) 
    dt = 2*sigma/(n*npc)
    time = np.linspace(0, 2.0*sigma, int(2*sigma/dt)+1, endpoint=True)
    time_fs = time/fs_to_au
    freq, *Glist = calc_Gobs(time_fs, dpz, return_moments=True, dt_fs=(dt/fs_to_au))
    return freq, Glist

def plot_hhg_sfa_comp(spectrum_file, tdci_data,  sfa_params=[0.0569, 20, 400],  
                 colors=['orangered','green', 'blue'], 
                  figsize=[8.19,6.8], xlim=[-.05, 80], ylim=[-21,2], 
                  fontsize=20, linewidths=[1.25, 1.25, 0.95],
                  return_data=False, include_without_lifetime=False,
                  legend_pos=[0.75, 0.90], text_pos=[0.750, 0.7],
                  savefig=True, imgtype='pdf', print_title=False):
    w0, n, npc = sfa_params
    freq, spectra = read_spectra(spectrum_file, sfa_params)
    tdci_lt = tdci_data['lt_rk4_tdprop.txt']
    e_cutoff = tdci_data['info']['e_cutoff']
    ip = tdci_data['info']['ip']
    w0 = tdci_data['info']['w0']
    atom = tdci_data['info']['atom']
    atom_title = {'he': 'He', 'h': 'H$^-$', 'li':'Li$^+$'}
    intensity = tdci_data['info']['intensity']
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq/w0, np.log10(spectra), color=colors[0], label='SFA'.format(npc=npc), linewidth=linewidths[0])
    ax.plot(tdci_lt['ho'], np.log10(tdci_lt['hhg_vel']), color=colors[1], linewidth=linewidths[1], label='TD-CIS/aQZ+440', alpha=0.7)
    if include_without_lifetime:
        tdci_wo_lt = tdci_data['rk4_tdprop.txt']
        ax.plot(tdci_wo_lt['ho'], np.log10(tdci_wo_lt['hhg_vel']), color=colors[2], linewidth=linewidths[2], alpha=0.50, label='{label} w/o $\Gamma_k$'.format(label='TD-CIS/aQZ+440'))
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Harmonic order (N)', fontsize=fontsize+4)
    plt.ylabel('log$_{10}$(I$_\mathbf{HHG}$)', fontsize=fontsize+4)
    if print_title:
        plt.title(atom_title[atom]+' HHG spectra for I$_0$ = {I:} W/cm$^2$'.format(I=intensity))
    ax.axvline(ip/w0, color='grey', linestyle='dotted', linewidth=1.0)
    ax.axvline(e_cutoff/w0, color='grey', linestyle='dashed', linewidth=0.85, alpha=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=0.9, length=8, direction='out')
    ax.tick_params(which='minor', width=.3, length=4, direction='out')
    ax.tick_params(axis='both', labelsize=fontsize+2)
    ax.text(*text_pos,
        ('$\cdots\cdots$ \ $I_p$ \ \ \  : {ionp: 3.4f} E$_h$ \n'+
        '$--$ \ E$_\mathbf{cut}$ : {e_cutoff:3.4f} E$_h$\n').format(cut='{cut}',ionp=ip, e_cutoff=e_cutoff),
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax.transAxes, fontsize=fontsize-2)
    ax.legend(bbox_to_anchor=legend_pos, loc='center',fontsize=fontsize-3)
    fig.tight_layout()
    if savefig:
        filename = '{path2images}{atom}_{intensity}_sfa_comp_hhg.{imgtype}'.format(path2images=_path2images,atom=atom, intensity=intensity,imgtype=imgtype)
        plt.savefig(filename, dpi=1000, format=imgtype)
    data = {'freq': freq,  'hhg_spectra': spectra, }
    if return_data:
        return data, fig, ax

def plot_rbsfa_comp(dipole_file, spectrum_file, info,
                    atom_title='',
                    params=(0.0569, 20, 400),
                    xlim=[0, 80], ylim=[-25, 3],
                ):
    w0, n, npc = params
    freq_dip, Glist = get_sfa_hhg_spectra(dipole_file, params)
    freq, spectra = read_spectra(spectrum_file, params)
    fig, ax = plt.subplots()
    ax.plot(freq_dip/w0, np.log10(Glist[0]), label='norm$^2$ FT of $\mathbf{makeDipoleList()}$')
    ax.plot(freq/w0, np.log10(spectra), label='output of $\mathbf{getSpectrum()}$', linewidth=1.0, alpha=0.85)
    intensity = info['intensity']
    e_cutoff = info['e_cutoff']
    ip = info['ip']
    w0 = info['w0']
    plt.title(atom_title+' HHG spectra for I$_0$ = {I:} W/cm$^2$'.format(I=intensity))
    ax.legend()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Harmonic order (N)')
    ax.set_ylabel('log$_{10}$(I$_\mathbf{HHG}$)')
    ax.axvline(ip/w0, color='grey', linestyle='dotted', linewidth=1.0)
    ax.axvline(e_cutoff/w0, color='grey', linestyle='dashed', linewidth=0.85, alpha=0.6)
    ax.text(0.750, 0.7,
        ('$\cdots\cdots$ \ $I_p$ \ \ \  : {ionp: 3.4f} E$_h$ \n'+
        '$--$ \ E$_\mathbf{cut}$ : {e_cutoff:3.4f} E$_h$\n').format(cut='{cut}',ionp=ip, e_cutoff=e_cutoff),
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax.transAxes, fontsize=11)
    plt.legend(bbox_to_anchor=[0.7, 0.90], loc='center',fontsize=11)
    fig.tight_layout()
    data = {'freq': freq, 'freq_dip': freq_dip, 'hhg_spectra': spectra, 
            'hhg_dip': Glist[0], 'hhg_vel': Glist[1], 'hhg_acc': Glist[2]}
    return data, fig, ax    
