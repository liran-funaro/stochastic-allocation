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
import numpy as np
import itertools

from cloudsim import dataset
from stochalloc import sim, results


def all_simulations(ds_obj: dataset.DataSet, iter_end=None, run_reserved=True, test_best=False,
                    sim_name_prefix=None, sub_name=None, **test_parameters):
    if run_reserved:
        ds_obj.add_batch_job(sim.select_bundle_reserved_only_simulation, skip_existing=True)
        ds_obj.add_batch_job(sim.server_simulation,
                             sim_name='reserved', max_groups_per_iteration=10, repeat_iterations=64,
                             cover_all_players=True, skip_existing=True)

    if isinstance(sim_name_prefix, str):
        prefix = sim_name_prefix.strip('- ')
        prefix = "%s-" % prefix
    else:
        prefix = ''

    if test_best:
        test_limit_share_prices = np.array([0.5])
    else:
        test_limit_share_prices = np.array([0.15, 0.5, 0.6, 0.7, 0.9])
    multi_iterative_simulation(ds_obj, 'shares', prefix+'shares-limit-{shares_unit_price:g}', sub_name,
                               iter_end=iter_end,
                               base_select_alloc_kwargs=dict(
                                   have_limit=True
                               ),
                               update_select_alloc_kwargs=dict(
                                   shares_unit_price=test_limit_share_prices,
                                   **test_parameters
                               ),
                               server_sim_kwargs=dict(max_groups_per_iteration=1,
                                                      repeat_iterations=16,
                                                      cover_all_players=False),
                               )

    if test_best:
        test_burst_prices = np.array([2])
    else:
        test_burst_prices = np.array([2, 3, 4])
    multi_iterative_simulation(ds_obj, 'burst', prefix+'burst-via-shares-{shares_unit_price:g}', sub_name,
                               iter_end=iter_end,
                               base_select_alloc_kwargs=dict(
                               ),
                               update_select_alloc_kwargs=dict(
                                   shares_unit_price=test_burst_prices,
                                   **test_parameters
                               ),
                               server_sim_kwargs=dict(max_groups_per_iteration=1,
                                                      repeat_iterations=16,
                                                      cover_all_players=False),
                               )
    if test_best:
        test_unlimit_share_prices = np.array([5])
    else:
        test_unlimit_share_prices = np.array([3, 5, 8])
    multi_iterative_simulation(ds_obj, 'shares', prefix+'shares-unlimit-{shares_unit_price:g}', sub_name,
                               iter_end=iter_end,
                               base_select_alloc_kwargs=dict(
                                   have_limit=False,
                               ),
                               update_select_alloc_kwargs=dict(
                                   shares_unit_price=test_unlimit_share_prices,
                                   **test_parameters
                               ),
                               server_sim_kwargs=dict(max_groups_per_iteration=1,
                                                      repeat_iterations=16,
                                                      cover_all_players=False),
                               )

    if test_best:
        test_limit_share_prices = np.array([0.5, 0.6])
        test_burst_prices = np.array([2, 3])
    multi_iterative_simulation(ds_obj, 'shares-and-burst',
                               prefix+'shares-limit-and-burst-{shares_unit_price:g}-{burst_unit_price:g}',
                               sub_name,
                               iter_end=iter_end,
                               base_select_alloc_kwargs=dict(
                                   shares_have_limit=True
                               ),
                               update_select_alloc_kwargs=dict(
                                   shares_unit_price=test_limit_share_prices,
                                   burst_unit_price=test_burst_prices,
                                   **test_parameters
                               ),
                               server_sim_kwargs=dict(max_groups_per_iteration=1,
                                                      repeat_iterations=16,
                                                      cover_all_players=False),
                               )


def test_change_count_simulations(ds_obj: dataset.DataSet, iter_end=None):
    return all_simulations(ds_obj, iter_end, test_best=True, sim_name_prefix='change-{change_count:g}',
                           change_count=[128, 192, 256, 320, 384, 448, 512])


def iterative_simulation(ds_obj: dataset.DataSet, sim_type,
                         sim_name, sub_name=None, iter_start=1, iter_count=5, iter_end=None,
                         select_alloc_kwargs=None, server_sim_kwargs=None):
    select_alloc_kwargs = select_alloc_kwargs or {}
    server_sim_kwargs = server_sim_kwargs or {}

    iter_start, iter_type = results.find_sim_next_iteration(ds_obj, sim_name, sub_name, iter_start)
    if iter_end is None:
        iter_end = iter_start + iter_count - 1
    if iter_end < iter_start:
        print(sim_name, "- Iterations already exists: %s to %s" % (iter_start, iter_end))
        return
    print(sim_name, "- Adding iterations: %s to %s" % (iter_start, iter_end))

    select_bundle_func = {
        'shares': sim.select_bundle_shares_simulation,
        'burst': sim.select_bundle_burst_via_shares_simulation,
        'shares-and-burst': sim.select_bundle_shares_and_burst_simulation,
    }.get(sim_type, None)
    if select_bundle_func is None:
        raise ValueError(f"Unknown sim type: {sim_type}")

    if iter_type == 'server':
        ds_obj.add_batch_job(sim.server_simulation,
                             sim_name=sim_name, sim_param=iter_start, sub_name=sub_name,
                             **server_sim_kwargs)
        iter_start += 1

    for i in range(iter_start, iter_end + 1):
        ds_obj.add_batch_job(select_bundle_func,
                             sim_name=sim_name, sim_param=i, sub_name=sub_name,
                             **select_alloc_kwargs)

        ds_obj.add_batch_job(sim.server_simulation,
                             sim_name=sim_name, sim_param=i, sub_name=sub_name,
                             **server_sim_kwargs)


def multi_iterative_simulation(ds_obj, sim_type, sim_name_fmt, sub_name=None,
                               iter_start=1, iter_count=5, iter_end=None,
                               base_select_alloc_kwargs=None, update_select_alloc_kwargs=None, server_sim_kwargs=None):
    base_select_alloc_kwargs = base_select_alloc_kwargs or {}
    update_select_alloc_kwargs = update_select_alloc_kwargs or {}

    keys, value_lists = zip(*update_select_alloc_kwargs.items())
    for values in itertools.product(*value_lists):
        select_alloc_kwargs = base_select_alloc_kwargs.copy()
        select_alloc_kwargs.update(zip(keys, values))
        cur_sim_name = sim_name_fmt.format(**select_alloc_kwargs)
        iterative_simulation(ds_obj, sim_type, cur_sim_name, sub_name,
                             iter_start, iter_count, iter_end,
                             select_alloc_kwargs=select_alloc_kwargs, server_sim_kwargs=server_sim_kwargs)
