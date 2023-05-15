#!/usr/bin/env python
#
#
# tweaking matplotlib 
import matplotlib as mpl
mpl.rc('axes', facecolor='none')
mpl.rc('figure', facecolor='white')
mpl.rc('savefig', facecolor='none')
mpl.rc('text', usetex=True)
mpl.rc('font', family='sans-serif', serif='Sans')
mpl.rc('font', size=12)
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Some constant internal varaibles
_path2images='../simulating_hhg_with_gaussian_basis_sets/images/'
