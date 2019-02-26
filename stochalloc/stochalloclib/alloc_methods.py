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
import itertools
import numpy as np

from cloudsim import stats
from stochalloc import stochalloclib
from stochalloc.data import produce

eps = np.finfo(np.float32).eps
max_flt = np.finfo(np.float32).max / 2


class Bundle:
    RESERVED = 1,
    SHARES_LIMITED = 2,
    SHARES_UNLIMITED = 3,
    BURST = 4


def optimize_select_alloc(bundle, valuation, limit_alloc, bundle_mesh, unit_price):
    val_shape = valuation.shape[1:]
    cur_bundle_price = np.sum([b * u for b, u in zip(bundle_mesh, unit_price)], axis=0)
    profits = valuation - cur_bundle_price

    max_profit_arg = np.array([np.argmax(r) for r in profits])
    max_profit_ind = np.array([np.unravel_index(r, val_shape) for r in max_profit_arg])
    max_profit = profits[(np.arange(len(profits)), *zip(*max_profit_ind))]
    active_players = np.where(max_profit > eps)[0]

    cur_arg = max_profit_arg[active_players]
    cur_ind = max_profit_ind[active_players]
    bundle['active'][active_players] = True
    bundle['ind'][active_players] = cur_ind
    bundle['profit'][active_players] = max_profit[active_players]
    bundle['price'][active_players] = cur_bundle_price.take(cur_arg)
    bundle['val'][active_players] = valuation[(active_players, *zip(*cur_ind))]
    bundle['limit'][active_players] = limit_alloc[(active_players, *zip(*cur_ind))]


def convert_select_index_to_allocation(bundle_ind, bundles):
    """ Bundle selection (change index to actual allocation) """
    axis = [slice(None)] * (len(bundle_ind.shape) - 1)
    bundle_select = np.array([np.take([*b, -1], bundle_ind[(*axis, i)]) for i, b in enumerate(bundles)], dtype=float)
    return np.moveaxis(bundle_select, 0, -1)


def select_alloc_reserved(sd, reserved_allocs, reserved_unit_price):
    """
    Returns the reserved allocation of each player given a set of bundles and unit prices.
    """
    n = sd.n
    ndim = sd.ndim
    total_resources = np.array(sd.meta['resources']['size'])

    bundles = reserved_allocs
    bundle_mesh = np.meshgrid(*bundles, sparse=False, indexing='ij')
    for m in bundle_mesh:
        # Must choose a bundle of something
        m[(0,) * len(bundles)] = max_flt

    bundle = {
        'ind': np.full((n, len(bundles)), -1, dtype=int),
        'limit': np.full((n, ndim), total_resources, dtype=float),
        'active': np.full(n, False, dtype=bool),
        'val': np.zeros(n, dtype=float),
        'profit': np.zeros(n, dtype=float),
        'price': np.zeros(n, dtype=float),
        'type': np.full(n, Bundle.RESERVED, dtype=np.uint32)
    }

    valuation = produce.calc_valuation_for_reserved(sd, reserved_allocs)
    limit_alloc = np.full(valuation.shape + (ndim,), total_resources, dtype=valuation.dtype)
    optimize_select_alloc(bundle, valuation, limit_alloc, bundle_mesh, reserved_unit_price)

    bundle['select'] = convert_select_index_to_allocation(bundle['ind'], bundles)
    bundle['limit'] = bundle['select'].copy()
    return bundle


def select_alloc_shares(sd, reserved_allocs, shares_allocs, reserved_unit_price, shares_unit_price,
                        shares_limit=None, shares_alloc_cdf_list=None, reserve_equal_shares=False,
                        reserved_as_average=False, bundle_type: int = Bundle.SHARES_LIMITED):
    """
    Returns the reserved+shares allocation of each player given a set of bundles and unit prices.
    """
    n = sd.n
    ndim = sd.ndim
    total_resources = np.array(sd.meta['resources']['size'])

    bundles = (*reserved_allocs, *shares_allocs)

    if shares_limit is not None:
        if len(shares_limit) != ndim:
            sd.log("Shares limit length does not match ndim: %d != %d" % (len(shares_limit), ndim))
            shares_limit = None
        elif any([len(p) != len(s) for p, s in zip(shares_allocs, shares_limit)]):
            sd.log("Shares limit length does not match shares portions: %s != %s" % (
                (*map(len, shares_allocs)), (*map(len, shares_limit))))
            shares_limit = None

    if shares_limit is None:
        bundles_unit = bundles
    else:
        bundles_unit = (*reserved_allocs, *shares_limit)
    bundle_mesh = np.meshgrid(*bundles_unit, sparse=False, indexing='ij')

    for m in bundle_mesh:
        # Must choose a bundle of something
        m[(0,) * len(bundles)] = max_flt
    if reserve_equal_shares is True:
        bundle_mask = np.full([len(b) for b in bundles_unit], False, dtype=bool)
        for ind in itertools.product(*[range(len(a)) for a in reserved_allocs]):
            # Can choose reserved without shares
            bundle_mask[ind + (0,) * ndim] = True
            # But if choose shares, require: reserved=share index
            bundle_mask[ind*2] = True
        for m in bundle_mesh:
            # Must choose a bundle of something
            m[~bundle_mask] = max_flt

    bundle = {
        'ind': np.full((n, len(bundles)), -1, dtype=int),
        'limit': np.full((n, ndim), total_resources, dtype=float),
        'active': np.full(n, False, dtype=bool),
        'val': np.zeros(n, dtype=float),
        'profit': np.zeros(n, dtype=float),
        'price': np.zeros(n, dtype=float),
        'type': np.full(n, bundle_type, dtype=np.uint32)
    }

    # Valuation (each iteration have different shares CDF)
    valuation, limit_alloc = produce.calc_valuation_for_bundles_and_alloc_cdf(sd, reserved_allocs,
                                                                              shares_alloc_cdf_list,
                                                                              reserved_as_average=reserved_as_average)
    # Bundle unit price
    bundle_unit_price = (*reserved_unit_price, *shares_unit_price)
    optimize_select_alloc(bundle, valuation, limit_alloc, bundle_mesh, bundle_unit_price)
    bundle['select'] = convert_select_index_to_allocation(bundle['ind'], bundles)

    # Bundle limit
    bundle_ind, bundle_select = bundle['ind'], bundle['select']
    axis = [slice(None)] * (len(bundle_ind.shape) - 1)
    if shares_limit is None:  # Shares limit was not specified (unlimited)
        # noinspection PyTypeChecker
        bundle_limit = bundle_select[(*axis, slice(None, ndim))].copy()
        bundle_limit[bundle_ind[(*axis, slice(ndim, None))] > 0] = total_resources
        # bundle_limit = np.full((n, ndim), total_resources, dtype=float)
    else:  # Add specified shares limit to the selected bundle
        bundle_limit = np.array([np.take([*b, 0], bundle_ind[(*axis, i + ndim)]) for i, b in enumerate(shares_limit)],
                                dtype=float)
        bundle_limit = np.moveaxis(bundle_limit, 0, -1)
        bundle_limit += bundle_select[(*axis, slice(None, ndim))]
        # noinspection PyTypeChecker
        bundle_limit = np.minimum(np.full(bundle_limit.shape, total_resources), bundle_limit)

    bundle['limit'] = np.minimum(bundle_limit, bundle['limit'])
    return bundle


def select_alloc_partial_update(sd, prev_bundle, new_bundle, change_count=128):
    n = sd.n

    # Get prev bundle iteration and set the new bundle iteration.
    prev_bundle_iter = prev_bundle.get('iter')
    if prev_bundle_iter is None:
        prev_bundle_iter = np.zeros(n, dtype=np.uint32)
        prev_bundle['iter'] = prev_bundle_iter

    new_bundle_iter = np.max(prev_bundle_iter) + 1
    new_bundle['iter'] = np.full(n, new_bundle_iter, dtype=np.uint32)

    # Fit prev bundle to the size of the new bundle
    if prev_bundle['ind'].shape[-1] < new_bundle['ind'].shape[-1]:
        prev_bundle_ind = np.zeros((n, new_bundle['ind'].shape[-1]), dtype=int)
        prev_bundle_select = np.zeros((n, new_bundle['select'].shape[-1]), dtype=float)
        prev_bundle_ind[:, :prev_bundle['ind'].shape[-1]] = prev_bundle['ind']
        prev_bundle_select[:, :prev_bundle['select'].shape[-1]] = prev_bundle['select']
        prev_bundle['ind'] = prev_bundle_ind
        prev_bundle['select'] = prev_bundle_select
        if 'type' not in prev_bundle:
            prev_bundle['type'] = np.full(n, Bundle.RESERVED, dtype=np.uint32)

    # Select which players will update their bundle (+inactive players).
    updating_players = ~prev_bundle['active']

    seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    updating_p = 1.2 ** (new_bundle_iter - prev_bundle_iter).astype(float)
    updating_p /= np.sum(updating_p)
    updating_players_ind = np.random.choice(a=range(n), size=change_count, replace=False, p=updating_p)
    updating_players[updating_players_ind] = True

    # Taking the information of the non updating players.
    for k in new_bundle:
        new_bundle[k][~updating_players] = prev_bundle[k][~updating_players]

    return seed


def select_alloc_best_of(*bundle_options):
    n_bundles = len(bundle_options)
    if n_bundles == 0:
        return
    elif n_bundles == 1:
        return bundle_options[0]

    joint_bundles = {k: np.array([b[k] for b in bundle_options]) for k in bundle_options[0]}
    best_of = np.argmax(joint_bundles['profit'], axis=0)
    n = len(best_of)
    return {k: v[best_of, range(n)] for k, v in joint_bundles.items()}


def calc_alloc_stats(sd, bundle, reserved_allocs, shares_allocs=None, max_servers=128, niter=1024,
                     overcommit_factor=1):
    ndim = sd.ndim
    total_resources = np.array(sd.meta['resources']['size']).astype(float)

    bundle_select, bundle_active, bundle_price = bundle['select'], bundle['active'], bundle['price']
    bundle_ind = bundle['ind']

    bundle_ndim = bundle_select.shape[-1]
    shares_allocation = bundle_ndim > ndim

    # Allocation Statistics
    alloc_stats = {
        'reserved-hist': [],
    }
    if shares_allocation:
        alloc_stats.update({
            'shares-hist': [],
            'reserved-shares-hist-cloud': [],
        })

    reserved_bins = list(map(stats.calc_hist_bin_edges, reserved_allocs))
    if shares_allocation:
        shares_bins = list(map(stats.calc_hist_bin_edges, shares_allocs))
    else:
        shares_bins = None

    # No need to allocate players to servers to calculate active count
    alloc_stats['active-player-count'] = bundle_active.sum()
    alloc_stats['host-revenue'] = bundle_price.sum()

    reserved = bundle_select[bundle_active, :ndim]
    shares = bundle_select[bundle_active, ndim:] if shares_allocation else None
    alloc_res = stochalloclib.allocate_players_to_servers(total_resources * overcommit_factor, reserved, shares,
                                                          max_servers=max_servers, niter=niter,
                                                          return_groups_count=0)
    alloc_count, res_allocated, active, _ = alloc_res
    alloc_stats['allocated-count'] = alloc_count
    alloc_stats['active-servers'] = active

    res_allocated[:, :ndim] /= total_resources
    alloc_stats['resources-allocated'] = res_allocated

    # Calculate the reserved and shares histogram
    for d in range(sd.ndim):
        reserved_hist = np.histogram(reserved[:, d], bins=reserved_bins[d])[0]
        alloc_stats['reserved-hist'].append(reserved_hist)
        if shares_allocation:
            shares_hist = np.histogram(shares[:, d], bins=shares_bins[d])[0]
            alloc_stats['shares-hist'].append(shares_hist)

            bundles_ind_bins = [np.arange(len(l) + 1) - 0.5 for l in (reserved_allocs[d], shares_allocs[d])]
            h, _, _ = np.histogram2d(bundle_ind[:, d], bundle_ind[:, sd.ndim+d], bins=bundles_ind_bins)
            alloc_stats['reserved-shares-hist-cloud'].append(h)

    # Calculate reserved 2d hist (cloud)
    if ndim == 2:
        h, _, _ = np.histogram2d(reserved[:, 0], reserved[:, 1], bins=reserved_bins)
        alloc_stats['reserved-hist-cloud'] = h

    return alloc_stats


def machine_allocation(sd, bundle, load=None, max_servers=128, repeat_iterations=16, max_groups_per_iteration=1,
                       overcommit_factor=1, force_cfs=False, force_no_limit=False, cover_all_players=False):
    """
    Return statistics about packed players per server
    """
    n = sd.n
    ndim = sd.ndim
    cdf_size = 512
    total_resources = np.array(sd.meta['resources']['size']).astype(float)
    usage_bins_e = np.linspace(0, 1, cdf_size+1)
    usage_bins_e[0] = -max_flt
    usage_bins_e[-1] = max_flt

    cdf_x = np.linspace(0, 1, cdf_size)
    cdf_x[0] = -max_flt
    cdf_x[-1] = max_flt

    shares_allocation = bundle['select'].shape[-1] > ndim
    repeat_iterations = np.maximum(repeat_iterations, 1)
    save_alloc = repeat_iterations == 1

    bundle_iter = np.max(bundle.get('iter', 0))

    if load is None:
        load = produce.get_load(sd, bundle_iter)

    servers_stats = {
        'mean-utilization': np.zeros(ndim, dtype=float),
        'max-utilization': np.zeros(ndim, dtype=float),
        'utilization-cdf': np.zeros((ndim, cdf_size), dtype=float),

        'reserved-mean-utilization': np.zeros(ndim, dtype=float),
        'reserved-max-utilization': np.zeros(ndim, dtype=float),
        'reserved-utilization-cdf': np.zeros((ndim, cdf_size), dtype=float),

        'utilization-ratio-cdf': np.zeros(cdf_size, dtype=float),
        'alloc-realization-cdf': np.zeros(cdf_size, dtype=float),
        'reserve-usage-hist': np.zeros((ndim, cdf_size, cdf_size), dtype=float),
    }
    s = servers_stats

    players_stats = {
        'max-value': np.zeros(n, dtype=float),
        'expected-value': np.zeros(n, dtype=float),
        'effective-value': np.zeros(n, dtype=float),
        'expected-price': np.zeros(n, dtype=float),
        'expected-mean-usage': np.zeros((n, ndim), dtype=float),
        'effective-mean-usage': np.zeros((n, ndim), dtype=float),
        'alloc-realization-portion': np.zeros(n, dtype=float),
    }
    p = players_stats

    alloc_data = []

    # For each combination of unit prices
    cur_selected = bundle['select']
    cur_val = bundle['val']
    cur_bundle_limit = bundle['limit']
    cur_bundle_price = bundle['price']

    active_players = bundle['active']
    active_players_count = active_players.sum()
    if active_players_count == 0:
        all_groups = []
    else:
        all_groups = np.full((repeat_iterations, n), -1, dtype=np.int32)
        reserved = cur_selected[active_players, :ndim]
        shares = cur_selected[active_players, ndim:] if shares_allocation else None
        alloc_res = stochalloclib.allocate_players_to_servers(total_resources * overcommit_factor, reserved, shares,
                                                              max_servers=max_servers, niter=repeat_iterations,
                                                              return_groups_count=repeat_iterations)
        _, _, _, groups = alloc_res
        all_groups[:, active_players] = groups
    play_count = np.zeros(n, dtype=np.uint32)
    played_iterations = 0

    # Iterate all orders
    for groups in all_groups:
        if cover_all_players and np.sum(play_count > 0) == n:
            break

        # Don't include the last group
        groups_count = np.minimum(max_groups_per_iteration, groups.max())
        for g in range(groups_count):
            sd.log("\rGroup %2d/%2d        " % (g+1, groups_count), end="")
            players = np.where(groups == g)[0]
            play_count[players] += 1
            played_iterations += 1

            # Get players' reserved and shares
            cur_reserved = cur_selected[players, :ndim]
            cur_shares = cur_selected[players, ndim:] if shares_allocation else None
            if not force_no_limit:
                cur_limit = cur_bundle_limit[players]
            else:
                cur_limit = None

            # Calculate players count, allocates resources, estimated SW and host's revenues
            max_value = produce.calc_max_value(sd, players, load=load)
            p['max-value'][players] += max_value

            expected_value = cur_val[players]
            p['expected-value'][players] += expected_value

            p['expected-price'][players] += cur_bundle_price[players]
            p['expected-mean-usage'][players] = cur_reserved

            # Run allocation algorithm
            alloc, required_resources = fixed_and_shares_allocation(sd, cur_reserved, cur_shares, cur_limit,
                                                                    players=players, load=load, force_cfs=force_cfs,
                                                                    iteration=bundle_iter)
            if save_alloc:
                # noinspection PyTypeChecker
                alloc_data.append((players, alloc, required_resources, cur_reserved, cur_shares, cur_limit))

            value = produce.calc_value_for_alloc(sd, players, alloc)
            p['effective-value'][players] += value

            p['effective-mean-usage'][players] = alloc.mean(axis=-2)

            have_requirements = required_resources > eps
            alloc_realization = alloc[have_requirements]/required_resources[have_requirements]
            _, cdf_y = stats.cdf_from_sample(alloc_realization, cdf_x=cdf_x)
            s['alloc-realization-cdf'] += cdf_y
            p['alloc-realization-portion'][players] += [np.mean(a[m] / r[m]) for m, a, r in
                                                        zip(have_requirements, alloc, required_resources)]

            usage = alloc.sum(axis=0) / total_resources
            s['mean-utilization'] += usage.mean(axis=0)
            s['max-utilization'] = np.maximum(usage.max(axis=0), s['max-utilization'])
            utilization_cdf = s['utilization-cdf']
            for cdf_dim in range(sd.ndim):
                cdf_x, cdf_y = stats.cdf_from_sample(usage[:, cdf_dim], cdf_x=cdf_x)
                utilization_cdf[cdf_dim] += cdf_y

            if ndim > 1:
                usage_ratio = np.min(usage, axis=-1) / np.max(usage, axis=-1)
                cdf_x, cdf_y = stats.cdf_from_sample(usage_ratio, cdf_x=cdf_x)
                s['utilization-ratio-cdf'] += cdf_y

            reserved_alloc = np.minimum(np.moveaxis(alloc, 0, 1), cur_reserved)
            reserved_usage = reserved_alloc.sum(axis=1) / total_resources

            s['reserved-mean-utilization'] += reserved_usage.mean(axis=0)
            s['reserved-max-utilization'] = np.maximum(reserved_usage.max(axis=0), s['reserved-max-utilization'])
            reserved_utilization_cdf = s['reserved-utilization-cdf']
            for cdf_dim in range(sd.ndim):
                cdf_x, cdf_y = stats.cdf_from_sample(reserved_usage[:, cdf_dim], cdf_x=cdf_x)
                reserved_utilization_cdf[cdf_dim] += cdf_y

            for d in range(sd.ndim):
                h, _, _ = np.histogram2d(reserved_usage[:, d], usage[:, d], bins=usage_bins_e)
                s['reserve-usage-hist'][d] += h.astype(float)

            # End group allocation
        # End all group allocation (a single unit price)
    # End unit price all iterations

    m = play_count > 1
    for stats_key in p:
        if type(p[stats_key]) not in (list, tuple):
            p[stats_key][m] = (p[stats_key][m].T / play_count[m]).T
        else:
            for stats_cur_ind in range(len(p[stats_key])):
                t = p[stats_key][stats_cur_ind][m]
                p[stats_key][stats_cur_ind][m] = (t.T / play_count[m]).T

    for stats_key in s:
        if stats_key in ['reserve-usage-hist', 'max-utilization', 'reserved-max-utilization']:
            continue
        if type(s[stats_key]) not in (list, tuple):
            s[stats_key] /= played_iterations
        else:
            for stats_cur_ind in range(len(s[stats_key])):
                s[stats_key][stats_cur_ind] /= played_iterations

    return {
        'server-stats': servers_stats,
        'player-stats': players_stats,
        'alloc-data': alloc_data,
    }


def fixed_and_shares_allocation(sd, reserved: np.ndarray, shares: np.ndarray=None, limit: np.ndarray=None,
                                players=(), load=None, force_cfs=False, iteration: int=None):
    """
    Returns each player's resource usage for a fixed allocation
    Return shape: required_resources.shape
    """
    if load is None:
        load = produce.get_load(sd, iteration)

    # Required resources shape: n, load-rounds, resource-count
    required_resources = np.array([produce.get_alloc_for_perf(sd, load[i], i) for i in players]).astype(float)

    if shares is None and not force_cfs:
        alloc = [np.minimum(req, res) for req, res in zip(required_resources, reserved)]
        return np.array(alloc), required_resources

    total_resources = np.array(sd.meta['resources']['size']).astype(float)
    alloc_step_funcs = [produce.get_alloc_step_func(sd, i) for i in players]
    load_rounds = required_resources.shape[1]

    if limit is None:
        # noinspection PyTypeChecker
        limit = np.full_like(reserved, total_resources)
    if shares is None:
        shares = reserved

    # Ret shape: load-rounds, n, resource-count
    ret = [stochalloclib.cfs(total_resources, 2 ** 8, required_resources[:, l, :], reserved, shares, limit,
                             alloc_step_funcs) for l in range(load_rounds)]
    return np.moveaxis(np.array(ret), 0, 1), required_resources
