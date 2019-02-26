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

from cloudsim import stats
from stochalloc import results
from stochalloc.sim import param


#####################################################################################################
# Azure
#####################################################################################################

AZURE_BINS = [1., 2., 4., 8., 16.]
AZURE_HIST = [4.9906121214619169e+01, 2.4839417867111738e+01, 1.1351015286276914e+01, 1.3887257065986283e+01,
              1.6188566005898398e-02]

AZURE_BINS_MAX = [1/16, 1/8, 1/4, 1/2, 1., 2., 4., 8., 16.]
# MAX
AZURE_HIST_MAX = [0.12176818735309942, 0.056636366210743916, 0.23382728335578562,
                  0.506895477586158, 49.09240223147283, 24.895728404351292,
                  15.96053254366057, 9.055750411625018, 0.0764590943845043]
# P95
AZURE_HIST_P95 = [4.5810729355033155e+00, 7.4440412474564202e+00, 2.3868587448572157e+01,
                  2.2211569191434965e+01, 1.9081196321872561e+01, 1.3094327867923994e+01,
                  6.3606684709394763e+00, 3.3431637883256267e+00, 1.5372727971487634e-02]


def compare_unified_results_to_azure(unified_result):
    azure_bins_e = np.array([0, 1.5, 3, 6, 12, 100.])
    azure_hist = np.array(AZURE_HIST) / 100
    unified_result['azure-bins'] = np.array(AZURE_BINS)
    unified_result['azure-hist'] = azure_hist

    reserved_hist_x = unified_result['input']['reserved-allocs'][0]
    reserved_hist_y = unified_result['reserved-hist'][0]
    bins_l = np.searchsorted(reserved_hist_x, azure_bins_e)
    s = np.array([np.sum(reserved_hist_y[:, bins_l[i - 1]:bins_l[i]], axis=-1) for i in range(1, len(bins_l))])
    player_count = np.sum(s, axis=0)
    s /= player_count
    hist = np.moveaxis(s, 0, -1)
    unified_result['reserved-hist-azure-bins'] = hist
    diff = 1 + np.abs(hist[:, :-1] - azure_hist[:-1]) / azure_hist[:-1]
    err = np.sqrt(np.sum(diff ** 2, axis=-1))
    unified_result['azure-err'] = err

    player_count = 10000
    server_count = np.sum(azure_hist * player_count * np.array(AZURE_BINS)) / 64
    player_per_server = player_count / server_count
    unified_result['azure-player-per-server'] = player_per_server


#####################################################################################################
# Summery Helpers
#####################################################################################################


def concatenate_active(data, axis=-1, eps=np.finfo(np.float32).eps, ignore_count=1):
    if axis < 0:
        axis = data.ndim + axis
    active = np.maximum(np.sum(data > eps, axis=axis) - ignore_count, 1)
    Ni, Nk = data.shape[:axis], data.shape[axis + 1:]
    out = np.zeros(np.sum(active))
    ind = 0
    for ii in np.ndindex(Ni):
        for kk in np.ndindex(Nk):
            cur_active = active[ii + kk]
            next_ind = ind + cur_active
            out[ind:next_ind] = data[ii + np.s_[:cur_active, ] + kk]
            ind = next_ind
    return out


def get_mean_alloc_count(alloc_count, axis=-1, eps=np.finfo(np.float32).eps, ignore_count=1):
    if axis < 0:
        axis = alloc_count.ndim + axis
    active = np.maximum(np.sum(alloc_count > eps, axis=axis) - ignore_count, 1)
    Ni, Nk = alloc_count.shape[:axis], alloc_count.shape[axis + 1:]
    out = np.zeros((*Ni, *Nk))
    for ii in np.ndindex(Ni):
        for kk in np.ndindex(Nk):
            out[ii + np.s_[..., ] + kk] = alloc_count[ii + np.s_[:active[ii + kk], ] + kk].mean()
    return out


def get_always_active_mean_alloc_count(active, alloc_count, axis=-1, ignore_count=1):
    if axis < 0:
        axis = alloc_count.ndim + axis
    eps = np.finfo(np.float32).eps
    always_active = np.maximum(np.sum(active > (1-eps), axis=-1) - ignore_count, 1)
    Ni, Nk = alloc_count.shape[:axis], alloc_count.shape[axis + 1:]
    out = np.zeros((*Ni, *Nk))
    for ii in np.ndindex(Ni):
        for kk in np.ndindex(Nk):
            out[ii + np.s_[..., ] + kk] = alloc_count[ii + np.s_[:always_active[ii], ] + kk].mean()
    return out


def concatenate_always_active(active, data, axis=-1, ignore_count=1):
    if axis < 0:
        axis = data.ndim + axis
    eps = np.finfo(np.float32).eps
    always_active = np.maximum(np.sum(active > (1 - eps), axis=-1) - ignore_count, 1)
    Ni, Nk = data.shape[:axis], data.shape[axis + 1:]
    out = np.zeros(np.sum(always_active))
    ind = 0
    for ii in np.ndindex(Ni):
        for kk in np.ndindex(Nk):
            cur_active = always_active[ii]
            next_ind = ind + cur_active
            out[ind:next_ind] = data[ii + np.s_[:cur_active, ] + kk]
            ind = next_ind
    return out

#####################################################################################################
# Shares
#####################################################################################################


def get_share_cdf(ds_obj, sim_name, sim_param=None, sub_name=None, result_type='limited',
                  shares_allocs=None, total_shares=None, shares_limit=None, method='hist'):
    methods = {
        'util': get_share_via_util,
        'hist': get_share_via_histogram,
        'alloc': get_share_via_alloc
    }
    func = methods.get(method, get_share_via_histogram)
    return func(ds_obj, sim_name, sim_param, sub_name, result_type=result_type, shares_allocs=shares_allocs,
                total_shares=total_shares, shares_limit=shares_limit)


#####################################################################################################
# Shares via actual allocation (alloc-data)
#####################################################################################################

def get_share_via_alloc(ds_obj, sim_name, sim_param=None, sub_name=None, only_reserved=False):
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                       require_server_sim_results=True)
    unit_prices = unified_result['unit-prices']
    shares_portions = unified_result['shares-portions']
    res_size = ds_obj.metadata['resources']['size']
    ndim = ds_obj.metadata['ndim']

    cdf_size = param.get_shares_cdf_size(ds_obj)
    cdf_x_vals = [np.linspace(0, r, s) for r, s in zip(res_size, cdf_size)]

    result_shape = [len(up) for up in unit_prices]
    shares_cdf_all_up = [np.zeros([*result_shape, len(p), s], dtype=float) for p, s in zip(shares_portions, cdf_size)]

    for a in unified_result['server-stats'][0]['alloc-data']:
        ind, players, alloc, required_resources, reserved, shares = a

        usage = alloc.sum(axis=0)
        reserved_alloc = np.moveaxis(np.minimum(np.moveaxis(alloc, 0, 1), reserved), 1, 0)
        reserved_usage = reserved_alloc.sum(axis=0)

        for d in range(ndim):
            reserved_share = res_size[d] - reserved_usage[:, d]
            total_unused = res_size[d] - usage[:, d]

            for i, p in enumerate(shares_portions[d]):
                if np.isclose(p, 0):
                    shares_cdf_all_up[d][(*ind, i)] = np.ones(cdf_size[d])
                    continue
                share = p * reserved_share
                if not only_reserved:
                    share = np.maximum(share, total_unused)
                cdf_x, cdf_y = stats.cdf_from_sample(share, cdf_x=cdf_x_vals[d])
                shares_cdf_all_up[d][(*ind, i)] = cdf_y

    return shares_cdf_all_up


#####################################################################################################
# Shares via utilization CDF
#####################################################################################################


def get_share_via_util(ds_obj, sim_name, sim_param=None, sub_name=None, result_type='limited',
                       shares_allocs=None, total_shares=None, shares_limit=None):
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                       require_server_sim_results=True)
    res_input = unified_result['input']
    res_size = res_input['res-size']
    ndim = res_input['ndim']
    util_cdf = unified_result['utilization-cdf']
    reserved_util_cdf = unified_result['reserved-utilization-cdf']
    un_util_cdf = 1 - np.flip(util_cdf, axis=-1)
    reserved_un_util_cdf = 1 - np.flip(reserved_util_cdf, axis=-1)
    eps = np.finfo(np.float32).eps

    if shares_allocs is None or len(shares_allocs) == 0:
        shares_allocs = res_input.get('shares-allocs', None)
    if shares_allocs is None or len(shares_allocs) == 0:
        shares_allocs = param.get_shares_allocs(ds_obj)

    cdf_size = param.get_shares_cdf_size(ds_obj)
    cdf_x_vals = [np.linspace(0, r, len(s)) for r, s in zip(res_size, un_util_cdf)]
    shares_cdf_x_vals = [np.linspace(0, r, s) for r, s in zip(res_size, cdf_size)]
    shares_cdf = [np.zeros([len(p), s], dtype=float) for p, s in zip(shares_allocs, cdf_size)]

    if total_shares is None:
        if unified_result['resources-allocated'].shape[-1] > ndim:
            total_shares = get_always_active_mean_alloc_count(unified_result['active-servers'],
                                                              unified_result['resources-allocated'][:, ndim:], axis=-2)
        else:
            total_shares = [0] * ndim

    for d in range(ndim):
        if total_shares[d] > eps:
            shares_portions = shares_allocs[d] / total_shares[d]
        else:
            shares_portions = shares_allocs[d] / np.max(shares_allocs[d])

        for p_ind, p in enumerate(shares_portions):
            if np.isclose(p, 0):
                shares_cdf[d][p_ind] = 1
                continue
            cur_reserved_un_util_cdf = np.interp(cdf_x_vals[d], cdf_x_vals[d]*p, reserved_un_util_cdf[d])
            ret_cdf = stats.cdf_maximum(cur_reserved_un_util_cdf, un_util_cdf[d])
            shares_cdf[d][p_ind] = np.interp(shares_cdf_x_vals[d], cdf_x_vals[d], ret_cdf)

    if result_type == 'unlimited':
        return shares_allocs, shares_cdf

    if shares_limit is None or len(shares_limit) == 0:
        shares_limit = res_input.get('shares-limit', None)
    if shares_limit is None or len(shares_limit) == 0:
        shares_limit = param.get_shares_limit(ds_obj)

    shares_cdf_limited = get_limited_shares(ds_obj, shares_cdf, shares_limit)

    if result_type == 'limited':
        return shares_allocs, shares_limit, shares_cdf_limited
    else:
        return shares_allocs, shares_limit, shares_cdf, shares_cdf_limited


#####################################################################################################
# Shares via utilization histogram
#####################################################################################################


def get_share_via_histogram(ds_obj, sim_name, sim_param=None, sub_name=None, result_type='limited',
                            shares_allocs=None, total_shares=None, shares_limit=None):
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                       require_server_sim_results=True, load_hist=True)
    res_input = unified_result['input']
    res_size = res_input['res-size']
    ndim = res_input['ndim']
    full_h = unified_result['reserve-usage-hist']

    if shares_allocs is None or len(shares_allocs) == 0:
        shares_allocs = res_input.get('shares-allocs', None)
    if shares_allocs is None or len(shares_allocs) == 0:
        shares_allocs = param.get_shares_allocs(ds_obj)

    cdf_size = param.get_shares_cdf_size(ds_obj)
    cdf_x_vals = [np.linspace(0, r, s) for r, s in zip(res_size, cdf_size)]
    cdf_x_edges = [stats.calc_hist_bin_edges(c) for c in cdf_x_vals]
    shares_cdf = [np.zeros([len(p), s], dtype=float) for p, s in zip(shares_allocs, cdf_size)]

    eps = np.finfo(np.float32).eps

    if total_shares is None:
        if unified_result['resources-allocated'].shape[-1] > ndim:
            total_shares = get_always_active_mean_alloc_count(unified_result['active-servers'],
                                                              unified_result['resources-allocated'][:, ndim:], axis=-2)
        else:
            total_shares = [0] * ndim

    for d in range(ndim):
        if total_shares[d] > eps:
            shares_portions = shares_allocs[d] / total_shares[d]
        else:
            shares_portions = shares_allocs[d] / np.sum(shares_allocs[d])

        cur_h = full_h[d]
        usage_bins = [np.linspace(0, res_size[d], s + 1) for s in cur_h.shape]
        usage_centers = [(c[1:] + c[:-1]) / 2 for c in usage_bins]

        m = cur_h > eps
        h = cur_h[m]

        used_reserved, used_total = np.meshgrid(*usage_centers, indexing='ij')

        for p_ind, p in enumerate(shares_portions):
            if np.isclose(p, 0):
                shares_cdf[d][p_ind] = 1
                continue
            share = np.maximum((res_size[d] - used_reserved)*p, res_size[d] - used_total)
            hist, e = np.histogram(share[m], bins=cdf_x_edges[d], weights=h)

            f = hist > 0
            f[0] = True
            # f[-1] = True
            cdf_y = np.cumsum(hist[f])
            cdf_y[0] = 0
            if cdf_y[-1] < eps:
                continue
            cdf_y /= cdf_y[-1]
            shares_cdf[d][p_ind] = np.interp(cdf_x_vals[d], cdf_x_vals[d][f], cdf_y)

    if result_type == 'unlimited':
        return shares_allocs, shares_cdf

    if shares_limit is None or len(shares_limit) == 0:
        shares_limit = res_input.get('shares-limit', None)
    if shares_limit is None or len(shares_limit) == 0:
        shares_limit = param.get_shares_limit(ds_obj)

    shares_cdf_limited = get_limited_shares(ds_obj, shares_cdf, shares_limit)

    if result_type == 'limited':
        return shares_allocs, shares_limit, shares_cdf_limited
    else:
        return shares_allocs, shares_limit, shares_cdf, shares_cdf_limited


def get_limited_shares(ds_obj, shares_cdf, shares_limit):
    ndim = ds_obj.metadata['ndim']
    res_size = ds_obj.metadata['resources']['size']

    shares_cdf_limited = [s.copy() for s in shares_cdf]
    for d in range(ndim):
        for i, l in enumerate(shares_limit[d]):
            c = int((l / res_size[d]) * shares_cdf_limited[d].shape[-1])
            shares_cdf_limited[d][i, c:] = 1

    return shares_cdf_limited


##############################################################################################################
# LOSS functions
##############################################################################################################

def shares_pack_learn_loss(ds_obj, sim_name, sub_name=None, iter_count=None):
    res = []
    cdfs = []
    iter_range = results.get_sub_name_parameters(ds_obj, sim_name, sub_name, require_server_sim_results=True)
    iter_range = iter_range[-iter_count]
    for i in iter_range:
        unified_result = results.read_mean_unified_results(ds_obj, sim_name, i, sub_name,
                                                           require_server_sim_results=True, load_hist=True)
        res.append(unified_result)

        shares_cdf = get_share_via_histogram(ds_obj, sim_name, i, sub_name, 'limited')
        cdfs.append(shares_cdf)

    relevant_fields = ['resources-allocated', 'allocated-count', 'reserved-hist', 'shares-hist', 'reserve-usage-hist']
    cmp_data = list(map(lambda r: list(map(r.get, relevant_fields)), res))

    x_shape = (len(cmp_data),) * 2
    pack_loss = np.full(x_shape, np.nan, dtype=float)

    for i, j in itertools.product(*[range(l) for l in x_shape]):
        if i < j:
            continue
        loss = 0
        for k, a, b in zip(relevant_fields, cmp_data[i], cmp_data[j]):
            try:
                y = (a - b)  # /np.max(b)
                loss += (y * y).sum()
            except Exception as e1:
                try:
                    for aa, bb in zip(a, b):
                        y = (aa - bb)  # /np.max(bb)
                        loss += (y * y).sum()
                except Exception as e2:
                    print("[ERROR]", (i, j), k, e1, e2)

        pack_loss[i, j] = np.sqrt(loss)

    ret_shares_loss = np.full(x_shape, np.nan, dtype=float)

    for i, j in itertools.product(*[range(l) for l in x_shape]):
        if i < j:
            continue
        loss = 0
        for a, b in zip(cdfs[i], cdfs[j]):
            try:
                y = a - b
                loss += (y * y).sum()
            except Exception as e:
                print("[ERROR]", (i, j), "CDFS", e)
        ret_shares_loss[i, j] = np.sqrt(loss)

    return iter_range, pack_loss, ret_shares_loss


def shares_loss(ds_obj, sim_name, sub_name=None, iter_count=None):
    cdfs = []
    iter_range = results.get_sub_name_parameters(ds_obj, sim_name, sub_name, require_server_sim_results=True)
    iter_range = iter_range[-iter_count]
    for i in iter_range:
        shares_cdf = get_share_via_histogram(ds_obj, sim_name, i, sub_name, 'limited')
        cdfs.append(shares_cdf)

    ndim = len(cdfs[0])
    case_count = len(cdfs[0][0])
    x_shape = (len(cdfs),) * 2
    ret_shares_mse = np.full((case_count, ndim, *x_shape), np.nan, dtype=float)
    ret_shares_mean_err = np.full((case_count, ndim, *x_shape), np.nan, dtype=float)
    ret_shares_max_err = np.full((case_count, ndim, *x_shape), np.nan, dtype=float)

    for i, j in itertools.product(*[range(l) for l in x_shape]):
        if i > j:
            continue
        # For each resource CDFs
        for d, (a, b) in enumerate(zip(cdfs[i], cdfs[j])):
            # For each unit price
            for c in range(case_count):
                try:
                    y = np.abs(a[c] - b[c])
                    ret_shares_mse[c, d, i, j] = (y * y).mean()
                    ret_shares_mean_err[c, d, i, j] = y.mean()
                    ret_shares_max_err[c, d, i, j] = y.max()
                except Exception as e:
                    print("[ERROR]", (i, j), "CDFS", e)

    return iter_range, ret_shares_mse, ret_shares_mean_err, ret_shares_max_err
