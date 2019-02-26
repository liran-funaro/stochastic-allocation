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
import gc
import itertools
import functools
import numpy as np

import vecfunc
from cloudsim import stats


#####################################################################
# Performance Getters
#####################################################################

def get_alloc_for_perf(sd, perf_value, k):
    """ Get the allocation for a given performance """
    perf_x = sd.data['perf-x']
    return np.moveaxis(np.array([np.interp(perf_value, perf_x, a) for a in sd.data['perf'][k]]), 0, -1)


def _get_value_for_alloc_transpose(sd, allocs, k, value_x):
    return np.array([np.interp(a, p, value_x) for p, a in zip(sd.data['perf'][k], allocs)]).min(axis=0)


def _get_perf_for_alloc_transpose(sd, allocs, k):
    return _get_value_for_alloc_transpose(sd, allocs, k, sd.data['perf-x'])


def get_perf_for_alloc(sd, allocs, k):
    return _get_perf_for_alloc_transpose(sd, np.moveaxis(allocs, -1, 0), k)


def get_alloc_step_func(sd, k=0):
    return sd.data['step-func'][k]


#####################################################################
# Valuation Getters
#####################################################################

def get_val(sd, i, x, val_key='val-imm-perf'):
    perf_x = sd.data['perf-x']
    val = sd.data[val_key][i]
    return np.interp(x, perf_x, val)


#####################################################################
# Load Getters and Statistics
#####################################################################

def get_single_load(sd, p, iteration: int = None):
    if iteration is None:
        iteration = 0

    load_n_days = sd.data['load-n-days'][p]
    return sd.data['load', p, iteration % load_n_days]


def get_load(sd, iteration: int = None):
    if iteration is None:
        iteration = 0

    return [sd.data['load', i, iteration % n_days] for i, n_days in enumerate(sd.data['load-n-days'])]


def get_full_load(sd, p):
    load_n_days = sd.data['load-n-days'][p]
    full_load = []
    for d in range(load_n_days):
        full_load.extend(sd.data['load', p, d])
    return full_load


def get_cdf_size(sd, val_size_factor=None):
    val_size = sd.meta['valuation']['size']
    if val_size_factor is None:
        val_size_factor = 16
    return ((val_size - 1) * val_size_factor) + 1


def get_load_samples_per_day(sd):
    samples_per_day = sd.dist_data.get('load-samples-per-day', None)
    if isinstance(samples_per_day, int):
        return samples_per_day
    else:
        return np.prod(sd.meta['load']['rounds'])


def read_single_load_cdf(sd, p, cdf_size=None):
    if cdf_size is None:
        cdf_size = get_cdf_size(sd)

    return sd.data.get(['load-cdf', cdf_size, p], None)


def read_load_cdf(sd, cdf_size=None):
    if cdf_size is None:
        cdf_size = get_cdf_size(sd)
    return [read_single_load_cdf(sd, p, cdf_size) for p in range(sd.n)]


def calc_load_cdf(sd, load, cdf_size=None):
    """
    Calculate the load CDF of each of the players
    """
    if cdf_size is None:
        cdf_size = get_cdf_size(sd)

    return [stats.cdf_from_sample(l, cdf_size, x_limits=(0, 1))[1] for l in load]


def calc_single_load_cdf(sd, load, cdf_size=None):
    """
    Calculate the load CDF of a single player
    """
    if cdf_size is None:
        cdf_size = get_cdf_size(sd)

    return stats.cdf_from_sample(load, cdf_size, x_limits=(0, 1))[1]


#####################################################################
# Performance/Load Statistics
#####################################################################


def add_reserved_to_alloc_sample(sd, alloc_sample, reserved=None):
    """
    Add the player's reserved allocation to the allocation sample
    """
    res_size = sd.meta['resources']['size']
    if reserved is not None:
        return [[np.minimum(a + r, s) for a, s, r in zip(alloc_sample, res_size, pr)] for pr in reserved]
    else:
        return [alloc_sample] * sd.n


def calc_perf_for_alloc_product(sd, alloc_sample):
    """
    Calculate the performance for all the combinations of the allocation sample.
    """
    alloc_mesh = (np.meshgrid(*a, sparse=False, indexing='ij') for a in alloc_sample)
    return [_get_perf_for_alloc_transpose(sd, a, p) for p, a in enumerate(alloc_mesh)]


def calc_perf_for_alloc_sample(sd, alloc_sample):
    """
    Calculate the performance for the allocation sample.
    """
    return [get_perf_for_alloc(sd, a, p) for p, a in enumerate(alloc_sample)]


def calc_value_for_perf(sd, players, res_perf):
    """
    Calculate the performance for the allocation sample.
    """
    val_progress = sd.data['val-progress']
    val_imm_perf = sd.data['val-imm-perf']
    perf_x = sd.data['perf-x']

    alloc_imm_val = [np.interp(p, perf_x, val_imm_perf[i]).mean() for i, p in zip(players, res_perf)]

    mean_load = sd.data['mean-load']
    alloc_progress = [np.mean(p) / mean_load[i] for i, p in zip(players, res_perf)]
    alloc_progress_val = [np.interp(p, perf_x, val_progress[i]) for i, p in zip(players, alloc_progress)]

    # Imm -> Value per round
    # Progress -> Value for perf / load
    return np.add(alloc_imm_val, alloc_progress_val)


def calc_value_for_alloc(sd, players, alloc):
    """
    Calculate the performance for the allocation sample.
    """
    res_perf = [get_perf_for_alloc(sd, a, i) for i, a in zip(players, alloc)]
    return calc_value_for_perf(sd, players, res_perf)


def calc_max_value(sd, players=None, load=None):
    if load is None:
        load = get_load(sd)
    if players is None:
        players = range(sd.n)
    return calc_value_for_perf(sd, players, [load[i] for i in players])


def get_max_value(sd, players=None):
    if players is None:
        players = range(sd.n)
    return sd.dist_data['wealth'][players]


def get_max_value_error(sd, players=None):
    effective_max_value = calc_max_value(sd, players)
    real_max_value = get_max_value(sd, players)
    e = (real_max_value - effective_max_value)
    return np.sqrt((e * e).sum())


def calc_perf_cdf_for_alloc_product(sd, alloc_sample):
    """
    Calculate an approximation of the performance CDF.
    Method: Testing all combination of the allocation sample.
    """
    val_size = sd.meta['valuation']['size']
    perf_sample = calc_perf_for_alloc_product(sd, alloc_sample)
    return [stats.cdf_from_sample(p, val_size, x_limits=(0, 1))[1] for p in perf_sample]


def calc_perf_cdf_for_alloc_random(sd, alloc_sample):
    """
    Calculate an approximation of the performance CDF on the basis of an allocation sample
    Method: Testing random combination of the allocation sample.
    """
    val_size = sd.meta['valuation']['size']
    rand_alloc_sample = [[np.random.permutation(ai) for ai in a] for a in alloc_sample]
    perf_sample = calc_perf_for_alloc_sample(sd, rand_alloc_sample)
    return [stats.cdf_from_sample(p, val_size, x_limits=(0, 1))[1] for p in perf_sample]


def iter_alloc_stats_list(alloc_stats_list):
    """
    Iterate over all the combination of allocation statistics.
    Yields: index, [alloc1_stat, alloc2_stat, ...]
    """
    for t in itertools.product(*[enumerate(m) for m in alloc_stats_list]):
        yield zip(*t)


def iter_perf_cdf_for_alloc_sample_list(sd, alloc_sample_list, reserved=None, perf_cdf_method='random'):
    """
    Iterate over the performance CDF on the basis of the allocation sample list.
    Yields: index, performance-CDF
    """
    if perf_cdf_method == 'random':
        method_fn = calc_perf_cdf_for_alloc_random
    elif perf_cdf_method == 'product':
        method_fn = calc_perf_cdf_for_alloc_product
    else:
        raise ValueError("No such perf CDF method: %s" % perf_cdf_method)

    for ind, alloc_sample in iter_alloc_stats_list(alloc_sample_list):
        reserved_sample = add_reserved_to_alloc_sample(sd, alloc_sample, reserved)
        yield ind, method_fn(sd, reserved_sample)


#####################################################################
# Statistical Adjustment of the Valuation
#####################################################################
"""
Calculating maximal performance valuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We want the LV: valuation for a provided load CDF
According to the law of total probability:
   LV(x) = int{f(l)*V(min{l,x}) dl} (0<l<1)
   LV(x) = S(x)*V(x) + int{f(l)*V(l) dl} (0<l<x)


Method 1 (continuous integral) - May be subjective to numerical errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Notice f=F'
   LV(x) = S(x)*V(x) + F(x)*V(x) - F(0)*V(0) - int{F(l)*V'(l) dl} (0<l<x)
Notice F(0)*F(0)==0 and S+F==1
   LV(x) = V(x) - int{F(l)*V'(l) dl} (0<l<x))
We discretized it as follows:
   LV[i] = V[i] - sum{F[l]*V'[l]*dl} (0<=l<=i)
   LV    = V - cumsum{F*V'}*dl

Method 2 (discretize sum) - More numerically stable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using the law of total probability:
And using the assumption that the load and valuation are uniform between two samples.
   LV[i] = S[i]*V[i] + sum{Pr(l-1<load<l)*(V(l-1)+V(l)/2} (0<l<i)
Which yields:
   LV[i] = S[i]*V[i] + sum{(F[l]-F[l-1])*(V(l)+V(l-1)/2} (0<l<i)
   LV = S*V + cumsum(diff(F)*mid(V))

Calculating valuation given maximal performance valuation and CDF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We want the valuation for a provided max performance CDF given LV
According to the law of total probability:
   V = int{int{f_P(p)*f_L(l)*V(min{p,l}) dl dp}} (0<l,p<1)
   V = int{f_P(p) * int{f_L(l)*V(min{p,l}) dl} dp} (0<l,p<1)
   V = int{f_P(p) * LV(p) dp} (0<p<1)
We discretized it as follows:
   V = sum{Pr(l-1<perf<l)*(V_max_load(l-1)+V_max_load(l)/2} (0<l<i)
   V = sum(diff(F_p)*mid(V_max_load))
"""

#####################################################################
# Method 2 for Statistical Adjustment of the Valuation
#####################################################################


def calc_max_perf_valuation_for_load_match(sd, load_cdf=None, load=None, val_key='val-imm-perf'):
    """
    Calculate V_max_load(x) = int{f(l)*V(min{l,x}) dl} (0<l<1)
    Method 2 (discretize sum) - More numerically stable
    """
    if load_cdf is None:
        load_cdf = calc_load_cdf(sd, load=load)

    val = sd.data[val_key]
    sf = stats.cdf_to_sf(load_cdf)
    pr_load = np.diff(load_cdf, axis=-1)
    val_mean = stats.vec_mean(val, axis=-1)

    max_load_val = sf * val
    axis = [slice(None)] * max_load_val.ndim
    axis[-1] = slice(1, None)
    max_load_val[axis] += np.cumsum(pr_load * val_mean, axis=-1)
    return max_load_val


def calc_max_perf_valuation_for_load(sd, val_key='val-imm-perf', val=None):
    """
    Calculate V_max_load(x) = int{f(l)*V(min{l,x}) dl} (0<l<1)
    Method 2 (discretize sum) - More numerically stable
    """
    if val is None:
        val = sd.data[val_key]

    res = []
    if type(val) in [list, tuple]:
        for p, v in enumerate(val):
            cdf = read_single_load_cdf(sd, p)
            res.append(vecfunc.expected_value_cumsum(v, cdf))
    else:
        for p in range(sd.n):
            cdf = read_single_load_cdf(sd, p)
            res.append(vecfunc.expected_value_cumsum(val, cdf))

    return np.array(res)


def calc_max_perf_valuation_for_load_2(sd, val_key='val-imm-perf'):
    """
    Calculate V_max_load(x) = int{f(l)*V(min{l,x}) dl} (0<l<1)
    Method 2 (discretize sum) - More numerically stable
    """
    load_cdf = read_load_cdf(sd, cdf_size=get_cdf_size(sd, 1))

    perf_x = sd.data['perf-x']
    cdf_len = len(load_cdf[0])
    cdf_perf_x = np.linspace(0, 1, cdf_len)

    val = sd.data[val_key]

    if type(val) in [list, tuple]:
        val = np.array([np.interp(cdf_perf_x, perf_x, v) for v in sd.data[val_key]])
    else:
        val = np.interp(cdf_perf_x, perf_x, val)

    sf = stats.cdf_to_sf(load_cdf)
    pr_load = np.diff(load_cdf, axis=-1)
    val_mean = stats.vec_mean(val, axis=-1)

    max_load_val = sf * val
    axis = [slice(None)] * max_load_val.ndim
    axis[-1] = slice(1, None)
    max_load_val[axis] += np.cumsum(pr_load * val_mean, axis=-1)
    return max_load_val


#####################################################################
# Statistical Adjustment of the Valuation: Performance/Allocation
#####################################################################


def calc_separated_valuation_for_reserved(sd, reserved_allocs, max_perf_imm_val=None, max_perf_avg_perf=None):
    """
    Calculate all player's valuation for a group of n dim allocation's bundles.
    """
    max_perf_imm_val, max_perf_avg_perf = get_adjusted_valuations(sd, max_perf_imm_val, max_perf_avg_perf)

    alloc_mesh = np.meshgrid(*reserved_allocs, sparse=False, indexing='ij')

    max_alloc_imm_val = [_get_value_for_alloc_transpose(sd, alloc_mesh, i, v) for i, v in enumerate(max_perf_imm_val)]
    max_alloc_avg_perf = [_get_value_for_alloc_transpose(sd, alloc_mesh, i, v) for i, v in enumerate(max_perf_avg_perf)]

    mean_load = sd.data['mean-load']
    max_alloc_progress = [np.minimum(ep / el, 1) for ep, el in zip(max_alloc_avg_perf, mean_load)]

    val_progress = sd.data['val-progress']
    perf_x = sd.data['perf-x']
    max_alloc_progress_val = [np.interp(p, perf_x, vp) for vp, p in zip(val_progress, max_alloc_progress)]
    return max_alloc_imm_val, max_alloc_progress_val


def calc_valuation_for_reserved(sd, reserved_allocs, max_perf_imm_val=None, max_perf_avg_perf=None):
    """
    Calculate all player's valuation for a group of n dim allocation's bundles.
    """
    ret = calc_separated_valuation_for_reserved(sd, reserved_allocs, max_perf_imm_val, max_perf_avg_perf)
    max_alloc_imm_val, max_alloc_progress_val = ret
    return np.add(max_alloc_imm_val, max_alloc_progress_val)


def calc_valuation_for_average_alloc(sd, total_resources, average_allocs, granularity,
                                     max_perf_imm_val=None, max_perf_avg_perf=None,
                                     safe_factor=1):
    """
    Calculate all player's valuation for a group of n dim allocation's bundles.
    """
    n = sd.n
    ndim = sd.ndim
    max_perf_imm_val, max_perf_avg_perf = get_adjusted_valuations(sd, max_perf_imm_val, max_perf_avg_perf)

    allocs = [np.linspace(0, r, granularity) for r in total_resources]
    alloc_mesh = np.meshgrid(*allocs, sparse=False, indexing='ij')

    max_alloc_imm_val = [_get_value_for_alloc_transpose(sd, alloc_mesh, i, v) for i, v in enumerate(max_perf_imm_val)]
    max_alloc_avg_perf = [_get_value_for_alloc_transpose(sd, alloc_mesh, i, v) for i, v in enumerate(max_perf_avg_perf)]

    mean_load = sd.data['mean-load']
    max_alloc_progress = [np.minimum(ep / el, 1) for ep, el in zip(max_alloc_avg_perf, mean_load)]

    val_progress = sd.data['val-progress']
    perf_x = sd.data['perf-x']
    max_alloc_progress_val = [np.interp(p, perf_x, vp) for vp, p in zip(val_progress, max_alloc_progress)]
    val = np.add(max_alloc_imm_val, max_alloc_progress_val)

    max_perf_avg_alloc = [calc_max_perf_valuation_for_load(sd, val=list(sd.data['perf'][:, d])) for d in range(ndim)]
    # shape = ndim, n, val_size
    max_alloc_avg_alloc = [
        [_get_value_for_alloc_transpose(sd, alloc_mesh, i, v) for i, v in enumerate(d)] for d in max_perf_avg_alloc]
    # shape = ndim, n, *(granularity,) * ndim
    max_alloc_avg_alloc = np.moveaxis(np.array(max_alloc_avg_alloc), 0, -1)
    # max_alloc_avg_perf.shape = n, *(granularity,) * ndim
    # max_alloc_avg_alloc = np.array([get_alloc_for_perf(sd, v, i) for i, v in enumerate(max_alloc_avg_perf)])
    # max_alloc_avg_alloc.shape = n, *(granularity,) * ndim, ndim

    val_max_alloc = np.zeros((n, *map(len, average_allocs), ndim), dtype=float)
    # max_alloc_ind.shape = n, *(average_count, ) * ndim, ndim
    # max_alloc.shape = n, *(average_count, ) * ndim, ndim

    average_val = np.zeros((n, *map(len, average_allocs)))
    for ind, avg in iter_alloc_stats_list(average_allocs):
        best = ((max_alloc_avg_alloc - np.multiply(avg, safe_factor))**2).sum(axis=-1)
        best_ind = [np.unravel_index(b.argmin(), b.shape) for b in best]
        val_max_alloc[np.s_[:, ] + ind] = [[a[i] for a, i in zip(allocs, b)] for b in best_ind]
        average_val[np.s_[:, ] + ind] = [v[b] for b, v in zip(best_ind, val)]

    return average_val, val_max_alloc


def calc_max_alloc_valuation_for_load(sd, max_perf_val=None, val_key='val-imm-perf'):
    """
    Calculate the valuation for a maximal allocation.
    """
    if max_perf_val is None:
        max_perf_val = calc_max_perf_valuation_for_load(sd, val_key=val_key)
    perf = sd.data['perf']
    val_x = sd.data['perf-x']
    return [np.interp(p, val_x, v) for p, v in zip(perf, max_perf_val)]


def calc_valuation_for_alloc_sample_list_via_simulation(sd, alloc_sample_list, players=None, load=None, reserved=None,
                                                        iteration: int = None):
    """
    Calculate all player's valuation for a group of n dim allocation's example sample.
    """
    if load is None:
        load = get_load(sd, iteration)
    if players is None:
        players = range(sd.n)

    v = np.zeros((len(players), *map(len, alloc_sample_list)))

    imm_val = sd.data['val-imm-perf']
    prg_val = sd.data['val-progress']
    val_x = sd.data['perf-x']

    mean_load = [sd.data['mean-load'][p] for p in players]

    for ind, alloc_sample in iter_alloc_stats_list(alloc_sample_list):
        player_alloc_sample = add_reserved_to_alloc_sample(sd, alloc_sample, reserved)
        perf_limit = [get_perf_for_alloc(sd, np.array(a).T, i) for i, a in zip(players, player_alloc_sample)]
        perf = [np.minimum(load[i], p) for i, p in zip(players, perf_limit)]
        progress = np.minimum(np.mean(perf, axis=1) / mean_load, 1)
        v[(slice(None), *ind)]  = [np.interp(p, val_x, imm_val[i]).mean() for i, p in zip(players, perf)]
        v[(slice(None), *ind)] += [np.interp(p, val_x, prg_val[i]) for i, p in zip(players, progress)]

    return v


def calc_valuation_for_alloc_sample_list(sd, alloc_sample_list,
                                         reserved=None, perf_cdf_method='random'):
    """
    Calculate all player's valuation for a group of n dim allocation's statistics.
    Method: Calculate the maximal performance CDF and calculate its minimum with the load CDF.
    """
    load_cdf = read_load_cdf(sd)

    imm_val = sd.data['val-imm-perf']
    avg_val = sd.data['val-avg-perf']
    val_x = sd.data['perf-x']
    v = np.zeros((sd.n, *[len(m) for m in alloc_sample_list]))

    mean_load = sd.data['mean-load']

    for ind, max_perf_cdf in iter_perf_cdf_for_alloc_sample_list(sd, alloc_sample_list, reserved, perf_cdf_method):
        actual_perf_cdf = stats.cdf_minimum(max_perf_cdf, load_cdf)
        expected_perf = stats.expected_via_cdf(actual_perf_cdf, min_val=0, max_val=1, axis=-1)
        progress = np.minimum(expected_perf / mean_load, 1)
        v[(slice(None), *ind)]  = [vecfunc.expected_value(iv, [cdf]) for iv, cdf in zip(imm_val, actual_perf_cdf)]
        v[(slice(None), *ind)] += [np.interp(p, val_x, av) for av, p in zip(avg_val, progress)]

    return v


def calc_valuation_for_alloc_sample_list_via_max_perf_val(sd, alloc_sample_list, reserved=None,
                                                          perf_cdf_method='random'):
    """
    Calculate all player's valuation for a group of n dim allocation's statistics.
    Method: Calculate the maximal performance CDF and find the expected value using the max performance valuation.
    """
    max_perf_imm_val, max_perf_avg_perf = get_adjusted_valuations(sd)
    avg_val = sd.data['val-avg-perf']
    val_x = sd.data['perf-x']

    v = np.zeros((sd.n, *[len(m) for m in alloc_sample_list]))

    mean_load = sd.data['mean-load']

    for ind, max_perf_cdf in iter_perf_cdf_for_alloc_sample_list(sd, alloc_sample_list, reserved, perf_cdf_method):
        expected_perf = np.array(
            [vecfunc.expected_value(mpp, [cdf]) for mpp, cdf in zip(max_perf_avg_perf, max_perf_cdf)])
        progress = np.minimum(expected_perf / mean_load, 1)
        v[(slice(None), *ind)] = [vecfunc.expected_value(mpv, [cdf]) for mpv, cdf in
                                  zip(max_perf_imm_val, max_perf_cdf)]
        v[(slice(None), *ind)] += [np.interp(p, val_x, av) for av, p in zip(avg_val, progress)]

    return v


def _calc_single_expected_value_for_alloc_cdf_list(sd, alloc_value, alloc_cdf_list, reserved):
    """
    Calculate a single player's expected value for a group of n dim allocation's CDF.
    """
    v = np.zeros(list(map(len, alloc_cdf_list)))
    res_size = sd.meta['resources']['size']

    for ind, alloc_cdf in iter_alloc_stats_list(alloc_cdf_list):
        alloc_cdf = [stats.cdf_shift(cdf, res, max_val=s) for cdf, res, s in zip(alloc_cdf, reserved, res_size)]
        v[ind] = vecfunc.expected_value(alloc_value, alloc_cdf)

    return v


def get_adjusted_valuations(sd, max_perf_imm_val=None, max_perf_avg_perf=None):
    if max_perf_imm_val is not None and max_perf_avg_perf is not None:
        return max_perf_imm_val, max_perf_avg_perf

    if max_perf_imm_val is None:
        max_perf_imm_val = sd.data.get('max-perf-imm-val', None)
        if max_perf_imm_val is None:
            max_perf_imm_val = calc_max_perf_valuation_for_load(sd, val_key='val-imm-perf')
            sd.data['max-perf-imm-val'] = max_perf_imm_val

    if max_perf_avg_perf is None:
        max_perf_avg_perf = sd.data.get('max-perf-avg-perf', None)
        if max_perf_avg_perf is None:
            max_perf_avg_perf = calc_max_perf_valuation_for_load(sd, val_key='perf-x')
            sd.data['max-perf-avg-perf'] = max_perf_avg_perf

    gc.collect()
    return max_perf_imm_val, max_perf_avg_perf


def _calc_single_expected_1d_value_for_alloc_cdf_list(sd, k, val, alloc_cdf_list, reserved):
    """
    Calculate a single player's expected value for a group of n dim allocation's CDF.
    """
    v = np.zeros(list(map(len, alloc_cdf_list)))
    res_size = sd.meta['resources']['size']
    cdf_size = [len(a[0]) for a in alloc_cdf_list]
    perf = sd.data['perf'][k]
    eps = np.finfo(np.float32).eps

    resource_x = [np.linspace(pr+eps, r + pr + eps, cdf_s) for r, pr, cdf_s in zip(res_size, reserved, cdf_size)]

    for ind, alloc_cdf in iter_alloc_stats_list(alloc_cdf_list):
        alloc_cdf = [np.interp(p, rx, a, left=0) for p, a, rx in zip(perf, alloc_cdf, resource_x)]
        cdf = alloc_cdf[0]
        for a in alloc_cdf[1:]:
            cdf = stats.cdf_minimum(cdf, a)
        v[ind] = vecfunc.expected_value(val, [cdf])

    return v


def _calc_multi_expected_1d_value_for_alloc_cdf_list(sd, k, vals, alloc_cdf_list, reserved, reserved_as_average=False):
    """
    Calculate a single player's expected value for a group of n dim allocation's CDF.
    """
    ndim = sd.ndim
    res_size = sd.meta['resources']['size']
    val_size = sd.meta['valuation']['size']

    v = [np.zeros((*map(len, alloc_cdf_list),)) for _ in vals]
    limit_alloc = np.full((*map(len, alloc_cdf_list), ndim), res_size, dtype=float)
    cdf_size = [len(a[0]) for a in alloc_cdf_list]
    perf = sd.data['perf'][k]
    perf_x = sd.data['perf-x']
    eps = np.finfo(np.float32).eps

    resource_x = [np.linspace(pr+eps, r + pr + eps, cdf_s) for r, pr, cdf_s in zip(res_size, reserved, cdf_size)]
    coeff = np.empty(val_size, dtype=np.float64)

    load_cdf = read_single_load_cdf(sd, k, cdf_size=perf.shape[1])

    for ind, alloc_cdf in iter_alloc_stats_list(alloc_cdf_list):
        max_pref_cdf_lst = (np.interp(p, rx, a, left=0) for p, a, rx in zip(perf, alloc_cdf, resource_x))
        max_perf_cdf = functools.reduce(stats.cdf_minimum, max_pref_cdf_lst)
        max_perf_cdf[-1] = 1

        # Ugly workaround: ii > 0 means it is not reserved allocation
        if reserved_as_average and any(ii > 0 for ii in ind):
            expected_perf_cdf = stats.cdf_minimum(load_cdf, max_perf_cdf)
            max_perf_exp_alloc = [vecfunc.expected_value_cumsum(p, expected_perf_cdf) for p in perf]
            limit_perf = np.min([np.interp(r, v, perf_x) for r, v in zip(reserved, max_perf_exp_alloc)]) + eps
            limit_alloc[ind] = np.maximum([np.interp(limit_perf, perf_x, p) for p in perf], [1e-10]*ndim)
            c = int(np.ceil(limit_perf * (len(max_perf_cdf)-1)))
            max_perf_cdf[c:] = 1

        coeff[:-1], coeff[-1] = max_perf_cdf[1:], 2 - max_perf_cdf[-1]
        coeff[0] += max_perf_cdf[0]
        coeff[1:] -= max_perf_cdf[:-1]
        for i, vv in enumerate(vals):
            v[i][ind] = np.vdot(vv, coeff)

    for vv in v:
        vv /= 2

    return (*v, limit_alloc)


def calc_valuation_for_alloc_cdf_list(sd, alloc_cdf_list, reserved=None, players=None,
                                      max_perf_imm_val=None, max_perf_avg_perf=None, reserved_as_average=False):
    """
    Calculate all player's valuation for a group of n dim allocation's CDF.
    """
    if players is None:
        players = range(sd.n)
    if reserved is None:
        reserved = np.zeros((len(players), sd.ndim))

    max_perf_imm_val, max_perf_avg_perf = get_adjusted_valuations(sd, max_perf_imm_val, max_perf_avg_perf)

    imm_v, expected_perf, limit_alloc = zip(
        *(_calc_multi_expected_1d_value_for_alloc_cdf_list(sd, i, (max_perf_imm_val[i], max_perf_avg_perf[i]),
                                                           alloc_cdf_list, res, reserved_as_average=reserved_as_average)
          for i, res in zip(players, reserved)))

    mean_load = sd.data['mean-load']
    val_progress = sd.data['val-progress']
    val_x = sd.data['perf-x']

    expected_progress = (np.minimum(ep / mean_load[i], 1) for i, ep in zip(players, expected_perf))
    prg_v = [np.interp(ep, val_x, val_progress[i]) for i, ep in zip(players, expected_progress)]
    return np.add(imm_v, prg_v), np.array(limit_alloc)


def calc_valuation_for_bundles_and_alloc_cdf(sd, bundles_list, alloc_cdf_list, reserved_as_average=False):
    """
    Calculate all player's valuation for a group of n dim allocation's CDF and bundles.
    """
    n = sd.n
    ndim = sd.ndim

    v = np.zeros([n, *map(len, [*bundles_list, *alloc_cdf_list])], dtype=float)
    limit_alloc = np.zeros([n, *map(len, [*bundles_list, *alloc_cdf_list]), ndim], dtype=float)

    max_perf_imm_val, max_perf_avg_perf = get_adjusted_valuations(sd)
    max_alloc = sd.data['max-alloc']

    for t in itertools.product(*map(enumerate, bundles_list)):
        ind, bundle = zip(*t)

        players = np.full(n, True, dtype=bool)
        for r, i in enumerate(ind):
            if i > 0:
                players &= stats.greater_or_close(max_alloc[:, r], bundles_list[r][i-1])

        active_count = np.sum(players)
        sd.log("\rBundle index: %s - active: %s            " % (ind, active_count), end="")

        if active_count == 0:
            continue

        reserved = np.full((active_count, sd.ndim), bundle)
        v_res = calc_valuation_for_alloc_cdf_list(sd, alloc_cdf_list, reserved=reserved, players=np.where(players)[0],
                                                  max_perf_imm_val=max_perf_imm_val,
                                                  max_perf_avg_perf=max_perf_avg_perf,
                                                  reserved_as_average=reserved_as_average)
        alloc_cdf_val, alloc_cdf_limit = v_res
        v[(players, *ind)] = alloc_cdf_val
        limit_alloc[(players, *ind)] = alloc_cdf_limit

    return v, limit_alloc


def calc_valuation_for_bundles_and_alloc_cdf_via_simulation(sd, bundles_list, alloc_cdf_list, load=None):
    """
    Calculate all player's valuation for a group of n dim allocation's CDF and bundles.
    """
    n = sd.n

    v = np.zeros([n, *map(len, [*bundles_list, *alloc_cdf_list])])
    res_size = sd.meta['resources']['size']

    alloc_sample_list = [[stats.sample_from_cdf(np.linspace(0, r, len(cdf)), cdf, 7200) for cdf in share_cdf] for
                         r, share_cdf in zip(res_size, alloc_cdf_list)]

    max_alloc = sd.data['max-alloc']

    for t in itertools.product(*map(enumerate, bundles_list)):
        ind, bundle = zip(*t)

        players = np.full(n, True, dtype=bool)
        for r, i in enumerate(ind):
            if i > 0:
                players &= stats.greater_or_close(max_alloc[:, r], bundles_list[r][i-1])

        active_count = np.sum(players)
        sd.log("\rBundle index: %s - active: %s            " % (ind, active_count), end="")

        if active_count == 0:
            continue

        players = np.where(players)[0]
        reserved = np.full((active_count, sd.ndim), bundle)
        alloc_cdf_val = calc_valuation_for_alloc_sample_list_via_simulation(sd, alloc_sample_list,
                                                                            players=players,
                                                                            load=load,
                                                                            reserved=reserved)
        v[(players, *ind)] = alloc_cdf_val

    return v
