#!/usr/bin/env
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>

import numpy as np
import scipy.constants as const 
c = const.value('speed of light in vacuum')
# Some constant internal varaibles
fs_to_au = 41.341374575751


def freq_to_wavelength(freq):
    wavelength = 2*np.pi*c/freq * 1/(1e15*fs_to_au)
    return wavelength


def wavelength_to_freq(wavelength_nm):
    freq = 2*np.pi*c/(wavelength_nm*1e-9) * 1/(1e15*fs_to_au)
    return freq


def calc_E0(I):
    eps0 = const.value('vacuum electric permittivity')
    I = I*1e4 # I in W/cm-2 * 1e4 
    E0 = np.sqrt(I/(0.5*eps0*c)) /const.value('atomic unit of electric field')
    # print('E0 : {E0:10.16f}'.format(E0=E0))
    return E0



def calc_moments(time_fs, observable):
    time = time_fs * fs_to_au
    velocity = np.gradient(observable, time)
    acceleration = np.gradient(velocity, time)
    return velocity, acceleration


def calc_fft(time_fs, observable, dt_fs=1e-4, return_moments=False):
    dt_fs = time_fs[2]-time_fs[1]
    dt = dt_fs * fs_to_au
    time = time_fs * fs_to_au
    freq = 2*np.pi*np.fft.fftfreq(len(time), dt)
    obs_fft = np.fft.fft(observable)
    if return_moments:
        velocity, acceleration = calc_moments(time_fs, observable)
        obs_vel_fft = np.fft.fft(velocity)
        obs_acc_fft = np.fft.fft(acceleration)
        return freq, obs_fft, obs_vel_fft, obs_acc_fft
    else:
        return freq, obs_fft


def calc_Gobs(time_fs, observable, return_moments=False, dt_fs=1e-4):
    freq, *obs_list = calc_fft(time_fs, observable, dt_fs=dt_fs, return_moments=return_moments)
    G_obs = abs(1/((time_fs[-1]-time_fs[0])*fs_to_au) * obs_list[0])**2
    n = int(freq.shape[0]/2)
    if return_moments:
        G_obs_vel = abs(1/((time_fs[-1]-time_fs[0])*fs_to_au) * obs_list[1])**2
        G_obs_acc = abs(1/((time_fs[-1]-time_fs[0])*fs_to_au) * obs_list[2])**2
        return freq[:n], G_obs[:n], G_obs_vel[:n], G_obs_acc[:n]
    else:
        return freq[:n], G_obs[:n]


