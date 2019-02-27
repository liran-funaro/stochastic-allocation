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
import math
import numpy as np

from cloudsim import dataset
from stochalloc.data import plot, produce
from stochalloc.data.gen import generate_init_data, generate_data


def generate(sd):
    generate_init_data(sd)
    generate_data(sd)


eps = np.finfo(np.float32).eps

folder_format = ("{ndim}d-"
                 "{performance[shape]}-"
                 "{n}p-"
                 "{resources[short-name]}")


metadata_1d_real = {
    'n': 2**10,
    'ndim': 1,
    'prefix': 'realdata',
    'resources': {
        'name': ['CPU'],
        'short-name': ['CPU'],
        'size': [64.],
        'size-unit': ['CPUs'],
        'granularity': [1 / 64],
    },
    'valuation': {
        'method': 'uniform',
        'wealth-dist': ['lomax', math.log(5, 4)],
        'size': 2**10,
    },
    'performance': {
        'step-func-size': 3,
        'shape': 'increasing',
    },
    'load': {
        'real-load': True,
        'max-days': 100,
        'over-the-top': True,
        'enrich-method': 'beta',
        'rounds': (24, 60 * 5),  # 24 h. 5 times per minute (each 12 seconds)
        'anchor-resource': 0,
    },
}
real_increasing_1d = dataset.alter_dataset(metadata_1d_real, folder_format, generator_func=generate)
real_concave_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('performance', 'shape'), 'concave'))
real_linear_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('performance', 'shape'), 'linear'))

real_p08_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('valuation', 'wealth-dist', 1), 0.8),
                                    prefix='realdata-p08')
real_p13_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('valuation', 'wealth-dist', 1), 1.3),
                                    prefix='realdata-p13')

real_discrete_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('load', 'enrich-method'), 'discrete'),
                                         prefix='realdata-discrete')
real_beta05_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('load', 'enrich-method'), ('beta', 0.5)),
                                       prefix='realdata-beta05')
real_beta10_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('load', 'enrich-method'), ('beta', 10)),
                                       prefix='realdata-beta10')
real_beta50_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('load', 'enrich-method'), ('beta', 50)),
                                       prefix='realdata-beta50')

real_nooverthetop_1d = dataset.alter_dataset(real_increasing_1d, folder_format, (('load', 'over-the-top'), False),
                                             prefix='realdata-nooverthetop')

beta_load_1d = dataset.alter_dataset(real_increasing_1d, folder_format, perfix='lomax', load={
        'real-load': False,
        'max-wealth-correlation': ('beta', (5, 5, -1, 2), (-1, 1)),
        'max-dist': ('weibull_min', (0.7459332547040183, 0, 1.3332197558174346 / 64), (1 / 128, 1)),
        'mean-dist': ('weibull_min', 'lambda x,a,b,c: a * np.exp(-b * x) + c',
                      ((0.9401380240103668, 0.10157238857262471, 0.623958148197448), 0,
                       (-0.6325148139842581, 0.5100329263777716, 0.6226713765042841)), (1 / 256, 64)),
        'density-dist': ('beta', (1, 15, 0, 128))
    })


metadata_2d = {
    'n': 1024,
    'ndim': 2,
    'prefix': 'lomax',
    'resources': {
        'name': ['CPU', 'Bandwidth'],
        'short-name': ['CPU', 'BW'],
        'size': [64., 256.],
        'size-unit': ['CPUs', 'GB/s'],
        'granularity': [1 / 64, 256 / 4096],
    },
    'valuation': {
        'method': 'uniform',
        'wealth-dist': ['lomax', (math.log(5, 4), 0, 1), (0, 100)],
        'val-ratio-dist': ['norm', (0.735, 0.255), (0+eps, 1-eps)],
        'size': 1024,
    },
    'performance': {
        'step-func-size': 512,
    },
    'load': {
        'rounds': (24, 60 * 5),  # 24 h. 5 times per minute (each 12 seconds)
        'anchor-resource': 0,

        'max-wealth-correlation': ('beta', (5, 5, -1, 2), (-1, 1)),
        'max-dist': ('weibull_min', (0.7459332547040183, 0, 1.3332197558174346 / 64), (1 / 128, 1)),
        'mean-dist': ('weibull_min', 'lambda x,a,b,c: a * np.exp(-b * x) + c',
                      ((0.9401380240103668, 0.10157238857262471, 0.623958148197448), 0,
                       (-0.6325148139842581, 0.5100329263777716, 0.6226713765042841)), (1 / 256, 64)),
        'density-dist': ('beta', (1, 15, 0, 128))
    },
}
