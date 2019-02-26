"""
Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2018 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Generic imports
import sys
import os
import time

# Scientific imports
import scipy
import pandas as pd
import numpy as np

# Plot imports
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Cloud simulation imports
import cloudsim
import stochalloc
import vecfunc

from cloudsim import sim_data
from cloudsim import job
from cloudsim import dataset as ds, stats

from stochalloc import data
from stochalloc import sim
from stochalloc import results

# vecfunc.loader.force_compile(True)


def init_plot(font_size=13, paper=False):
    # mpl.rc('font',family='Times New Roman', size=font_size)
    # plt.rc('font', family='serif', serif='Times New Roman', weight='normal', size=font_size)
    if paper:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=font_size)
    else:
        plt.rc('font', size=font_size)


def save_fig(fname, lgd=None, **kwargs):
    filepath = os.path.join('../figures', "%s.pdf" % fname)
    if lgd is None:
        bbox_extra_artists = None
    elif type(lgd) in (list, tuple):
        bbox_extra_artists = lgd
    else:
        bbox_extra_artists = (lgd,)
    plt.savefig(filepath, format='pdf', bbox_extra_artists=bbox_extra_artists,
                bbox_inches='tight', dpi=300, **kwargs)
