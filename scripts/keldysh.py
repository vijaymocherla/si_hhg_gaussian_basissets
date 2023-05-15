#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com> 
#
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib as mpl
c = const.value('speed of light in vacuum')
fs_to_au=41.341374575751
from plotparams import *

# Some constant internal varaibles
_path2images='../simulating_hhg_with_gaussian_basis_sets/images/'

ip = {
    'H-'   :  0.02771614, #   0.754195 eV (Source: Lykke et. al. (1991). Phys. Rev. A, 43(11), 6104.
    'H2'   :  0.56689375, #  15.425930 eV (Source: NIST Data base)
    'He'   :  0.90357185, #  24.587387 eV (Source: NIST Data base)
    'Li+'  :  2.77972859, #  75.640097 eV (Source: NIST Data base)
    'Be2+' :  5.65559404, # 153.896205 eV (Source: NIST Data base)
    'Ne'   :  0.79248402, #  21.564540 eV (Source: NIST Data base)
    'F-'   :  0.12499539, #   3.401290 eV (Source: Chem. Rev. 2002, 102, 231−282)
    'Na+'  :  1.73774162, #  47.286380 eV (Source: NIST Data base)
    'H2O'  :  0.46381424, #  12.621000 eV (Source: NIST Data base)
    'NH3'  :  0.37006651, #  10.070000 eV (Source: NIST Data base)
    'CH4'  :  0.46341000, #  12.610000 eV (Source: NIST Data base)
}   

def watt_per_cm2_to_au(I):
        eps0 = const.value('vacuum electric permittivity')
        c = const.value('speed of light in vacuum')
        I = I*1e4 # I in W/cm-2 * 1e4 
        E0 = np.sqrt(I/(0.5*eps0*c)) /const.value('atomic unit of electric field')
        # print('E0 : {E0:10.16f}'.format(E0=E0))
        return E0

def freq_to_wavelength(freq):
    wavelength = 2*np.pi*c/freq * 1/(1e15*fs_to_au)
    return wavelength

def wavelength_to_freq(wavelength):
    freq = 2*np.pi*c/wavelength * 1/(1e15*fs_to_au)
    return freq


def keldysh(intensity, wavelength_nm, atom):
    Ip = ip[atom]
    E0 = watt_per_cm2_to_au(intensity)
    w = wavelength_to_freq(wavelength_nm*1e-9)
    Up = E0**2 / (4*w**2)
    Rmax = 2*E0/ (w**2)
    Ecutoff = ip[atom] + 3.17*Up
    gamma = np.sqrt(Ip/(2*Up))
    data_dict={
        'intensity':intensity,
        'wavelength_nm':wavelength_nm,
        'gamma':gamma,
        'Ecutoff':Ecutoff,
        'Up':Up,
        'Rmax':Rmax,}
    return data_dict

def plot_keldysh(atom, I_lim=[1e9, 5e14], wln_lim=[400,2300],title='', plotname='', savefig=True, imgtype='pdf', fontsize=19):
    Imin, Imax = I_lim
    wln_min, wln_max = wln_lim 
    I = np.logspace(np.log10(Imin), np.log10(Imax), 1000, endpoint=True)
    wlen = np.linspace(wln_min, wln_max, 1000) 
    wavelengths, intensities = np.meshgrid(wlen, I)
    gamma = keldysh(intensities, wavelengths, atom)['gamma']
    fig, ax = plt.subplots(figsize=(8,6), facecolor='white')
    if title=='':
        title=atom
    # ax.set_title(title + r' Keldysh Parameter Plot', fontsize=fontsize+4)
    ax.set_xlabel(r'$\lambda$ (nm)', fontsize=fontsize+4)
    ax.set_ylabel(r'I$_{0}$ (W/cm$^{2}$)', fontsize=fontsize+4)
    ax.tick_params(axis='both', labelsize=fontsize+2)
    cmap = cm.Spectral
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=6)
    # cbar = fig.colorbar(cs)
    levels = [0.60, 0.65, 0.675, 0.71, 0.725,0.75, 0.80, 0.825, 0.85, 0.875, 0.9 , 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1 ,1.15, 1.20, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
    cs = ax.contourf(wavelengths, (intensities), gamma, levels=levels, cmap=cmap, extend='both')
    cbar = fig.colorbar(cs, spacing='proportional')
    cbar.ax.tick_params(labelsize=fontsize) 
    cbar.ax.set_title('$\mathbf{\gamma}$', y=1.05, fontsize=fontsize+4)
    n = 7
    # ax.yaxis.set_major_locator(plt.LogLocator(5))
    # ax.set_yscale('log')
    ax.yaxis.set_ticks([I[i*int(1000/n)] for i in range(2,n+1)])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1E'))
    fig.tight_layout()
    if savefig:
        fig.savefig('{path2images}{plotname}_keldysh.{imgtype}'.format(path2images=_path2images,plotname=plotname,imgtype=imgtype),dpi=1000, format=imgtype)