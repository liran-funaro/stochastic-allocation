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
from stochalloc.stochalloclib import alloc_methods
from stochalloc.sim import param


#########################################################################################################
# Reserved Only Alloc Workers
#########################################################################################################

def select_bundle_reserved_only_worker(_ds_obj, _index, sd, reserved_allocs, reserved_unit_price,
                                       overcommit_factor=1):
    bundle = alloc_methods.select_alloc_reserved(sd, reserved_allocs, reserved_unit_price)
    bundle_stats = alloc_methods.calc_alloc_stats(sd, bundle, reserved_allocs, overcommit_factor=overcommit_factor)
    return {
        'input': {
            'reserved-unit-price': reserved_unit_price,
            'reserved-allocs': reserved_allocs,
        },
        'bundle': bundle,
        'bundle-stats': bundle_stats,
    }


#########################################################################################################
# Stochastic Alloc Workers
#########################################################################################################


def select_bundle_shares_worker(ds_obj, index, sd, sim_type, learn_iter, sub_name, *args, change_count=128, **kwargs):
    prev_sim_key = param.get_bundle_simulation_key(sim_type, learn_iter - 1, sub_name)
    prev_ret = ds_obj.read_result_file(prev_sim_key, index)
    prev_bundle = prev_ret['bundle']

    bundle = alloc_methods.select_alloc_shares(sd, *args, **kwargs)
    seed = alloc_methods.select_alloc_partial_update(sd, prev_bundle, bundle, change_count=change_count)

    reserved_allocs, shares_allocs = args
    bundle_stats = alloc_methods.calc_alloc_stats(sd, bundle, reserved_allocs, shares_allocs)
    return {
        'input': {
            'reserved-allocs': reserved_allocs,
            'shares-allocs': shares_allocs,
            'kwargs': kwargs,
        },
        'param': {
            'seed': seed,
        },
        'bundle': bundle,
        'bundle-stats': bundle_stats,
    }


def select_bundle_shares_2_worker(ds_obj, index, sd, sim_name, learn_iter, sub_name, *args, change_count=128,
                                  multi_kwargs={}):
    prev_sim_key = param.get_bundle_simulation_key(sim_name, learn_iter - 1, sub_name)
    prev_ret = ds_obj.read_result_file(prev_sim_key, index)
    prev_bundle = prev_ret['bundle']

    reserved_allocs, shares_allocs = args
    bundle_options = [alloc_methods.select_alloc_shares(sd, *args, **kwargs) for kwargs in multi_kwargs]
    bundle = alloc_methods.select_alloc_best_of(*bundle_options)
    seed = alloc_methods.select_alloc_partial_update(sd, prev_bundle, bundle, change_count=change_count)

    bundle_stats = alloc_methods.calc_alloc_stats(sd, bundle, reserved_allocs, shares_allocs)
    return {
        'input': {
            'reserved-allocs': reserved_allocs,
            'shares-allocs': shares_allocs,
            'multi-kwargs': multi_kwargs,
        },
        'param': {
            'seed': seed,
        },
        'bundle': bundle,
        'bundle-stats': bundle_stats,
    }


#########################################################################################################
# Server Simulation Alloc Workers
#########################################################################################################


def server_simulation_worker(ds_obj, index, sd, sim_name, sim_param=None, sub_name=None, **kwargs):
    prev_sim_key = param.get_bundle_simulation_key(sim_name, sim_param, sub_name)
    bundle_result = ds_obj.read_result_file(prev_sim_key, index)
    return alloc_methods.machine_allocation(sd, bundle_result['bundle'], **kwargs)
