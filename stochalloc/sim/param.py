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
import os
import numpy as np
from cloudsim import dataset


def get_simulation_key(sim_func, sim_name, sim_param=None, sub_name=None, set_reserved_param=True):
    if type(sim_param) in (tuple, list):
        sim_param = "-".join(map(str, sim_param))

    ret = []
    if sim_name == 'reserved' or sim_param == 0:
        # Ignore sim_param and sub_name
        sim_name = 'reserved'
        if set_reserved_param:
            sim_param = 0

    for val in (sub_name, sim_name, sim_param, sim_func):
        if val is None:
            continue

        val = dataset.convert_to_nice_filename(val)
        val = val.replace(os.path.sep, ":")
        ret.append(val)

    return tuple(ret)


SIM_TYPE_BUNDLE = 'bundle'
SIM_TYPE_SERVER = 'server'


def get_bundle_simulation_key(sim_name, sim_param=None, sub_name=None):
    return get_simulation_key(SIM_TYPE_BUNDLE, sim_name, sim_param, sub_name)


def get_server_simulation_key(sim_name, sim_param=None, sub_name=None):
    return get_simulation_key(SIM_TYPE_SERVER, sim_name, sim_param, sub_name)


def get_density_allocs(density=6):
    count = 20 * density + 1
    u = np.linspace(-10, 10, count)
    return 2**u


def get_reserved_allocs(ds_obj):
    base_bundle = np.concatenate([[0], np.geomspace(1 / 1024, 1 / 2, 10)])
    return [base_bundle * s for s in ds_obj.metadata['resources']['size']]


def get_shares_allocs(ds_obj):
    return get_reserved_allocs(ds_obj)


def get_shares_limit(ds_obj):
    return get_reserved_allocs(ds_obj)


def get_shares_cdf_size(ds_obj):
    d = ds_obj.metadata['valuation']['size']
    res_size = ds_obj.metadata['resources']['size']
    res_gran = ds_obj.metadata['resources']['granularity']

    return [(d - 1) * np.ceil((r / g) / (d - 1)).astype(int) + 1 for r, g in zip(res_size, res_gran)]
