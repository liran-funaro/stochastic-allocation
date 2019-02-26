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
from cloudsim import dataset
from stochalloc.results import analyze
from stochalloc.sim import batch, workers, param
from stochalloc.stochalloclib.alloc_methods import Bundle


#########################################################################################################
# Reserved Only Alloc Simulation Starters
#########################################################################################################


def select_bundle_reserved_only_simulation(ds_obj: dataset.DataSet,
                                           sim_name='reserved', sim_param=None, sub_name=None,
                                           overcommit_factor=1, start_job=True, skip_existing=False):
    reserved_allocs = param.get_reserved_allocs(ds_obj)
    reserved_unit_price = [1 for _ in reserved_allocs]

    args = reserved_allocs, reserved_unit_price
    kwargs = dict(overcommit_factor=overcommit_factor)
    sim_key = param.get_bundle_simulation_key(sim_name, sim_param, sub_name)
    cur_sim = ds_obj.create_job(sim_key, workers.select_bundle_reserved_only_worker,
                                max_workers=12, start_job=start_job, skip_existing=skip_existing,
                                args=args, kwargs=kwargs)
    ds_obj.clear_cache()
    return cur_sim


#########################################################################################################
# Burst Alloc Workers
#########################################################################################################


def select_bundle_burst_via_shares_simulation(ds_obj: dataset.DataSet, sim_name, sim_param=1, sub_name=None,
                                              shares_unit_price=3, start_job=True, **kwargs):
    return select_bundle_shares_simulation(ds_obj, sim_name, sim_param, sub_name, shares_unit_price, have_limit=False,
                                           start_job=start_job, is_burst=True, **kwargs)


#########################################################################################################
# Stochastic Alloc Workers
#########################################################################################################

def select_bundle_shares_simulation(ds_obj: dataset.DataSet, sim_name, sim_param=1, sub_name=None,
                                    shares_unit_price=3, have_limit=True, start_job=True,
                                    is_burst=False, **kwargs):
    reserved_allocs = param.get_reserved_allocs(ds_obj)
    reserved_unit_price = [1 for _ in reserved_allocs]

    shares_allocs = param.get_shares_allocs(ds_obj)
    shares_unit_price = [shares_unit_price for _ in shares_allocs]

    shares_limit = param.get_shares_limit(ds_obj)
    shares_info = analyze.get_share_cdf(ds_obj, sim_name, sim_param - 1, sub_name, result_type='both',
                                        shares_allocs=shares_allocs, shares_limit=shares_limit, method='hist')
    _, _, shares_cdf_unlimited, shares_cdf_limited = shares_info

    if is_burst:
        bundle_type = Bundle.BURST
        have_limit = False
        kwargs = dict(reserve_equal_shares=True, reserved_as_average=True, **kwargs)
    elif have_limit:
        bundle_type = Bundle.SHARES_LIMITED
    else:
        bundle_type = Bundle.SHARES_UNLIMITED

    if have_limit:
        shares_cdf = shares_cdf_limited
    else:
        shares_cdf = shares_cdf_unlimited
        shares_limit = None

    args = (sim_name, sim_param, sub_name, reserved_allocs, shares_allocs)
    kwargs = dict(
        reserved_unit_price=reserved_unit_price,
        shares_unit_price=shares_unit_price,
        shares_limit=shares_limit,
        shares_alloc_cdf_list=shares_cdf,
        bundle_type=bundle_type,
        **kwargs
    )
    sim_key = param.get_bundle_simulation_key(sim_name, sim_param, sub_name)
    cur_sim = ds_obj.create_job(sim_key, workers.select_bundle_shares_worker,
                                max_workers=12, start_job=start_job, args=args, kwargs=kwargs)
    ds_obj.clear_cache()
    return cur_sim


def select_bundle_shares_and_burst_simulation(ds_obj: dataset.DataSet, sim_name, sim_param=1, sub_name=None,
                                              shares_unit_price=3, shares_have_limit=True,
                                              burst_unit_price=3, start_job=True, **kwargs):
    reserved_allocs = param.get_reserved_allocs(ds_obj)
    reserved_unit_price = [1 for _ in reserved_allocs]

    shares_allocs = param.get_shares_allocs(ds_obj)
    shares_unit_price = [shares_unit_price for _ in shares_allocs]
    burst_unit_price = [burst_unit_price for _ in shares_allocs]

    shares_limit = param.get_shares_limit(ds_obj)
    shares_info = analyze.get_share_cdf(ds_obj, sim_name, sim_param - 1, sub_name, result_type='both',
                                        shares_allocs=shares_allocs, shares_limit=shares_limit, method='hist')
    _, _, shares_cdf_unlimited, shares_cdf_limited = shares_info

    burst_kwargs = dict(
        shares_unit_price=burst_unit_price,
        shares_limit=None,
        shares_alloc_cdf_list=shares_cdf_unlimited,
        bundle_type=Bundle.BURST,
        reserve_equal_shares=True,
        reserved_as_average=True,
        **kwargs
    )

    shares_kwargs = dict(
        shares_unit_price=shares_unit_price,
        shares_limit=shares_limit if shares_have_limit else None,
        shares_alloc_cdf_list=shares_cdf_limited if shares_have_limit else shares_cdf_unlimited,
        bundle_type=Bundle.SHARES_LIMITED if shares_have_limit else Bundle.SHARES_UNLIMITED,
        **kwargs
    )

    multi_kwargs = [dict(a, reserved_unit_price=reserved_unit_price) for a in (burst_kwargs, shares_kwargs)]
    args = (sim_name, sim_param, sub_name, reserved_allocs, shares_allocs)
    kwargs = dict(multi_kwargs=multi_kwargs)
    sim_key = param.get_bundle_simulation_key(sim_name, sim_param, sub_name)
    cur_sim = ds_obj.create_job(sim_key, workers.select_bundle_shares_2_worker,
                                max_workers=12, start_job=start_job, args=args, kwargs=kwargs)
    ds_obj.clear_cache()
    return cur_sim


#########################################################################################################
# Server Simulation Alloc Workers
#########################################################################################################


def server_simulation(ds_obj: dataset.DataSet,
                      sim_name, sim_param=None, sub_name=None, start_job=True, skip_existing=False, **kwargs):
    sim_key = param.get_server_simulation_key(sim_name, sim_param, sub_name)
    args = (sim_name, sim_param, sub_name)
    cur_sim = ds_obj.create_job(sim_key, workers.server_simulation_worker, max_workers=12, start_job=start_job,
                                skip_existing=skip_existing, args=args, kwargs=kwargs)
    ds_obj.clear_cache()
    return cur_sim
