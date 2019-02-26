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
import time
import numpy as np

import vecfunc
from cloudsim import stats, azure
from cloudsim.sim_data import SimulationData

import matplotlib.pylab as plt

from stochalloc.data import produce


#####################################################################
# Helper functions
#####################################################################

def get_method(method_data):
    if type(method_data) in (list, tuple):
        return method_data[0], method_data[1:]
    else:
        return method_data, None


def get_method_str(method_name, method_params):
    if not method_params:
        return method_name
    else:
        return "%s%s" % (method_name, list(method_params))


#####################################################################
# Generating Simulation Data
#####################################################################

def generate_init_data(sd: SimulationData):
    sd.init_seed()

    sd.log("Generating initial data (seed: %d)..." % sd.seed)

    generate_distributions(sd)
    generate_init_performance(sd)
    generate_init_load(sd)
    generate_init_valuation(sd)

    return sd


def generate_data(sd: SimulationData):
    if 'generate-time' in sd.meta:
        sd.log("Data already generated.")
        return
    sd.log("Generating data...")

    t1 = time.time()
    generate_performance(sd)
    generate_load(sd)
    generate_valuation(sd)
    generate_time = time.time() - t1

    sd.meta['generate-time'] = generate_time
    sd.log("Generation time: %.2f seconds." % generate_time)
    sd.save()


def generate_distributions(sd: SimulationData):
    """
    Generates distribution:
     - wealth,  valuations, performance, load
    """
    n = sd.n
    dist_data = sd.dist_data
    eps = np.finfo('float32').eps
    sim_index = sd.meta['index']

    bundles = np.array([0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64], dtype=float)
    dist_data['bundles'] = bundles
    dist_data['azure-bundles'] = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=float)

    sd.log("Generating distributions: ", end="")

    # Performance
    sd.log("performance", end="")
    dist_data['perf-freq'] = np.random.randint(4, 16, n)

    # Collect Azure Data
    sd.log(", collect azure data", end="")
    is_real_load = sd.meta['load'].setdefault('real-load', False)
    full_azure_data = azure.read_azure_data()
    if is_real_load:
        vm_ids = azure.get_available_vm_ids()
        assigned_ids = vm_ids[(sim_index*n):((sim_index+1)*n)]
        players_azure_data = full_azure_data.loc[full_azure_data['vm id'].isin(assigned_ids)]
        dist_data['azure-players-ids'] = list(players_azure_data['vm id'])

        vm_category = list(players_azure_data['vm category'])
        cores = np.array(players_azure_data['vm virtual core count'])
        mean_cpu = np.array(players_azure_data['avg cpu'], dtype=float) * cores / 100.
        max_cpu = np.array(players_azure_data['max cpu'], dtype=float) * cores / 100.
    else:
        run_time = np.array(full_azure_data['timestamp vm deleted'] - full_azure_data['timestamp vm created'])
        relevant_players = np.where(run_time > 60)[0]
        azure_players = np.random.choice(relevant_players, n, replace=False)
        dist_data['azure-players'] = azure_players
        all_cores = full_azure_data['vm virtual core count']
        all_avg_cpu = full_azure_data['avg cpu'] * all_cores / 100.
        all_max_cpu = full_azure_data['max cpu'] * all_cores / 100.

        vm_category = full_azure_data['vm category'][azure_players]
        mean_cpu = np.array(all_avg_cpu)[azure_players]
        max_cpu = np.array(all_max_cpu)[azure_players]
        cores = np.array(all_cores)[azure_players]
        # max_mem = np.array(full_azure_data['vm memory (gb)'])[azure_players]

    # Valuation
    sd.log(", valuation", end="")
    cat = {'interactive': 0, 'unkown': 0.5, 'delay-insensitive': 1}
    val_ratio_mean = np.array(list(map(lambda p: cat[p.lower()], vm_category)))
    dist_data['val-ratio'] = stats.trunc_norm(0, 1, val_ratio_mean, 0.25, len(val_ratio_mean))

    # Load
    sd.log(", load", end="")
    dist_data['mean-load'] = mean_cpu
    dist_data['max-load'] = max_cpu
    dist_data['selected-cores'] = cores

    # rounds = sd.meta['load']['rounds']
    max_bundle_ind = get_relevant_bundle_ind(bundles, max_cpu, one_over=False)
    sd.internal['selected-bundle'] = np.array(bundles[max_bundle_ind].copy(), dtype=float)

    if not is_real_load:
        load_density = stats.scipy_dist_ppf(np.random.uniform(eps, 1 - eps, n), sd.meta['load']['density-dist'])

        anchor = sd.meta['load']['anchor-resource']
        anchor_size = sd.meta['resources']['size'][anchor]
        reevaluate_mean = [
            stats.search_rising_func(stats.get_beta_trunc_mean, m, 0, anchor_size, d, anchor_size, c,
                                     full_output=False)
            for m, d, c in zip(mean_cpu, load_density, cores)]

        dist_data['load'] = np.array(list(zip(reevaluate_mean, load_density)))

    # Wealth: Indicate the expected income of the player
    sd.log(", wealth.")

    dist, dist_param = stats.get_scipy_dist(sd.meta['valuation']['wealth-dist'])
    player_min_ppf = dist.cdf(cores*1.1, *dist_param) + eps

    wealth_uniform = np.random.uniform(0 + eps, 1 - eps, n)
    wealth_uniform *= 1-player_min_ppf
    wealth_uniform += player_min_ppf
    dist_data['wealth-uniform'] = wealth_uniform
    dist_data['wealth'] = stats.scipy_dist_ppf(wealth_uniform, sd.meta['valuation']['wealth-dist'])


#####################################################################
# Performance
#####################################################################

def generate_init_performance(sd: SimulationData):
    dist_data = sd.dist_data

    sd.log("Generating performance: initial func.")

    perf_freq = dist_data['perf-freq']
    perf_shape = sd.meta['performance'].setdefault('shape', 'concave' if sd.ndim == 1 else 'increasing')
    if perf_shape == 'concave':
        if sd.ndim == 1:
            init_perf = [vecfunc.rand.init_sample_1d_concave() for _ in perf_freq]
        else:
            raise ValueError("Don't be here")
    elif perf_shape == 'increasing':
        init_perf = [vecfunc.rand.init_sample_nd_uniform(frequency=f, increasing=True, force_edge=True,
                                                         ndim=sd.ndim) for f in perf_freq]
    else:  # 'linear'
        init_perf = [([0, 1], [0, 1]) for _ in perf_freq]

    sd.init_data['perf'] = init_perf


def generate_performance(sd: SimulationData):
    val_size = sd.meta['valuation']['size']

    sd.log("Generating performance: full funcs.")

    perf_x = np.linspace(0, 1, val_size)
    sd.data['perf-x'] = perf_x

    init_perf = sd.init_data['perf']
    if sd.meta['performance']['shape'] in ['concave', 'increasing']:
        perf = [vecfunc.vecinterp.refine_chaikin_corner_cutting_xy(*p) for p in init_perf]
        # Resample refined function to create smaller representation
        sd.internal['perf'] = np.array([[np.interp(perf_x, p[-1], a) for a in p[:-1]] for p in perf])
    else:
        sd.internal['perf'] = np.array([[np.linspace(a[0], a[1], val_size) for a in p[:-1]] for p in init_perf])


#####################################################################
# Load
#####################################################################

def generate_init_load(_sd: SimulationData):
    pass


def generate_real_load(sd: SimulationData):
    total_rounds = np.prod(sd.meta['load']['rounds'])
    dist_data = sd.dist_data

    core_eps = 0.1
    azure_interval = 60*5
    load_interval = 12
    repeat_sample = int(azure_interval / load_interval)
    azure_samples_per_day = int(total_rounds / repeat_sample)
    dist_data['load-samples-per-day'] = azure_samples_per_day * repeat_sample

    max_days = sd.meta['load'].setdefault('max-days', 1)

    anchor = sd.meta['load']['anchor-resource']
    anchor_size = sd.meta['resources']['size'][anchor]
    selected_bundle = sd.internal['selected-bundle']
    azure_players_ids = dist_data['azure-players-ids']

    enrich_method = sd.meta['load'].setdefault('enrich-method', 'beta')
    if type(enrich_method) in (list, tuple):
        enrich_method, density = enrich_method
    else:
        density = None
    enrich_method, default_density = dict(
        beta=(vecfunc.rand.beta_min_max_mean, 1),
        discrete=(vecfunc.rand.min_max_avg_dist, 0.1),
    ).get(enrich_method)

    if density is None:
        density = default_density

    over_the_top = sd.meta['load'].setdefault('over-the-top', True)

    sd.log("Generating load, ", end="")

    anchor_usage_list = []
    for i, (vm_id, n_cores) in enumerate(zip(azure_players_ids, selected_bundle)):
        sd.log("\rLoad for client: %s                  " % i, end="")
        u = azure.get_vm_id_cpu_data(vm_id)
        u = u.values[:, 1:] * (n_cores / 100)
        if over_the_top:
            u[u[:, 1] > (n_cores-core_eps), 1] = anchor_size

        max_azure_samples = min(max_days, int(len(u) / azure_samples_per_day)) * azure_samples_per_day

        cur_load = np.array([], dtype=float)
        for u_min, u_max, u_avg in u[:max_azure_samples]:
            if np.isclose(u_max, u_min):
                new_load = np.full(repeat_sample, u_avg, dtype=float)
            else:
                new_load = enrich_method(u_min, u_max, u_avg, density, size=repeat_sample)
            cur_load = np.append(cur_load, new_load)

        # selected_load_inds = np.sort(np.random.choice(np.arange(len(cur_load)), total_rounds, replace=False))
        # anchor_usage_list.append(cur_load[selected_load_inds])
        anchor_usage_list.append(cur_load)
    sd.log("")

    max_usage = [np.max(b) for b in anchor_usage_list]
    return anchor_usage_list, max_usage


def generate_load_from_dist(sd: SimulationData):
    total_rounds = np.prod(sd.meta['load']['rounds'])
    dist_data = sd.dist_data

    # Density Reduce Factor
    f = 0.95

    anchor = sd.meta['load']['anchor-resource']
    anchor_size = sd.meta['resources']['size'][anchor]
    bundles = dist_data['bundles']
    selected_bundle = sd.internal['selected-bundle']
    selected_bundle_ind = np.array([np.abs(bundles - b).argmin() for b in selected_bundle], dtype=np.uint32)
    sd.log("Generating load, ", end="")

    load_data = dist_data['load']
    anchor_usage_list = [vecfunc.rand.beta_mean_max(m, d, anchor_size, size=total_rounds) for m, d in load_data]
    anchor_usage_list = np.array(anchor_usage_list)

    while True:
        max_usage = np.max(anchor_usage_list, axis=-1)
        too_low = np.where(max_usage < bundles[selected_bundle_ind - 1])[0]
        if len(too_low) == 0:
            break
        load_data[too_low, 1] *= f
        anchor_usage_list[too_low] = [vecfunc.rand.beta_mean_max(m, d, anchor_size, size=total_rounds) for m, d in
                                      load_data[too_low]]

    return anchor_usage_list, max_usage


def get_relevant_bundle_ind(all_bundles, usage, one_over=True):
    if one_over:
        eps = 1e-12
        all_bundles = all_bundles - eps
    return np.minimum(np.searchsorted(all_bundles, usage), len(all_bundles)-1)


def generate_load(sd: SimulationData):
    is_real_load = sd.meta['load'].setdefault('real-load', False)
    if is_real_load:
        anchor_usage_list, max_usage = generate_real_load(sd)
    else:
        anchor_usage_list, max_usage = generate_load_from_dist(sd)

    sd.log("Adjusting perf func to max usage, ", end="")
    anchor = sd.meta['load']['anchor-resource']
    all_bundles = sd.dist_data['bundles'].copy()
    max_bundle_ind = get_relevant_bundle_ind(all_bundles, max_usage)
    max_bundle = all_bundles[max_bundle_ind].astype(float)
    perf = sd.internal['perf']
    perf[:, anchor] = (perf[:, anchor].T * max_bundle).T
    sd.data['perf'] = perf

    # Converting the load to performance units from the anchor resource
    sd.log("converting load.", )
    perf_x = sd.data['perf-x']
    load = [np.interp(b, p[anchor], perf_x) for b, p in zip(anchor_usage_list, sd.data['perf'])]
    sd.internal['load'] = load

    # Write load per day
    samples_per_day = produce.get_load_samples_per_day(sd)
    load_n_days = [int(len(l)/samples_per_day) for l in load]
    sd.data['load-n-days'] = load_n_days
    for player_index, (player_load, player_days) in enumerate(zip(load, load_n_days)):
        for day_index in range(player_days):
            s = day_index * samples_per_day
            e = s + samples_per_day
            day_load = player_load[s:e]
            sd.data['load', player_index, day_index] = day_load

    # Resample refined function to create the step function
    n = sd.n
    step_size = sd.meta['performance']['step-func-size']
    step_x = np.linspace(0, 1, step_size + 1)[1:]
    sd.data['step-x'] = step_x
    sd.data['step-func'] = [produce.get_alloc_for_perf(sd, step_x, i) for i in range(n)]

    # Calculate mean/max load
    sd.data['mean-load'] = [l.mean() for l in load]
    max_load = [l.max() for l in load]
    sd.data['max-load'] = max_load
    percentile_99_load = [np.percentile(l, 99.9) for l in load]
    sd.data['percentile-99-load'] = percentile_99_load

    eps = np.finfo('float32').eps
    sd.data['max-alloc'] = np.array([produce.get_alloc_for_perf(sd, l, i) for i, l in enumerate(max_load)]) + eps
    sd.data['percentile-99-alloc'] = np.array(
        [produce.get_alloc_for_perf(sd, l, i) for i, l in enumerate(percentile_99_load)]) + eps

    # Calculate load CDF
    for cdf_size in [produce.get_cdf_size(sd, val_size_factor=f) for f in (1, None)]:
        for player_index, player_load in enumerate(load):
            player_cdf = produce.calc_single_load_cdf(sd, player_load, cdf_size=cdf_size)
            sd.data['load-cdf', cdf_size, player_index] = player_cdf


#####################################################################
# Valuation
#####################################################################

def generate_init_valuation(_sd: SimulationData):
    pass


def generate_valuation(sd: SimulationData):
    sd.log("Generating valuation, ", end="")
    dist_data = sd.dist_data

    sd.data['max-perf-avg-perf'] = produce.calc_max_perf_valuation_for_load(sd, val_key='perf-x')

    sd.log("full funcs", end="")
    val_imm_perf = []
    val_progress = []
    for i in range(sd.n):
        imm_val, prog_val = generate_valuation_from_load(sd, i)
        val_imm_perf.append(imm_val)
        val_progress.append(prog_val)
    sd.data['val-imm-perf'] = val_imm_perf
    sd.data['val-progress'] = val_progress

    sd.log(", max-perf", end="")
    sd.data['max-perf-imm-val'] = produce.calc_max_perf_valuation_for_load(sd, val_key='val-imm-perf')

    sd.dist_data['selected-bundle'] = sd.internal['selected-bundle']

    if sd.verbose:
        sd.log(", validating.")

        anchor = sd.meta['load']['anchor-resource']
        # reserved = np.array([dist_data['azure-bundles']])
        reserved = np.array([dist_data['bundles'].copy()])
        max_alloc_imm_val, max_alloc_progress_val = produce.calc_separated_valuation_for_reserved(sd, reserved)
        reserved_val = np.add(max_alloc_imm_val, max_alloc_progress_val)
        val_shape = reserved_val.shape[1:]
        max_profit_arg = np.array([np.argmax(v - reserved) for v in reserved_val])
        max_profit_ind = np.array([np.unravel_index(r, val_shape)[anchor] for r in max_profit_arg])
        req_bundle_ind = np.array([np.abs(reserved - r).argmin() for r in sd.internal['selected-bundle']])
        neq = np.where(max_profit_ind != req_bundle_ind)[0]
        if len(neq) > 0:
            print("%s Mismatch is bundle" % len(neq))
            for i in neq:
                print("[%d] Best: %.3g != Req: %.3g" % (i, reserved[0, max_profit_ind[i]], reserved[0, req_bundle_ind[i]]))
                print('Reserved Val:', dict(zip(reserved[0], reserved_val[i])))
                print('Imm Val:', dict(zip(reserved[0], max_alloc_imm_val[i])))
                print('Prog Val:', dict(zip(reserved[0], max_alloc_progress_val[i])))
                generate_valuation_from_load(sd, i, graphic_output=True)


def generate_valuation_from_load(sd: SimulationData, k, graphic_output=False):
    anchor = sd.meta['load']['anchor-resource']
    dist_data = sd.dist_data

    req_bundle = sd.internal['selected-bundle'][k]
    assert not np.isclose(req_bundle, 0), "[%s] Req bundle: %s" % (k, req_bundle)

    wealth = dist_data['wealth'][k]
    progress_budget_ratio = dist_data['val-ratio'][k]
    imm_perf_budget_ratio = 1 - progress_budget_ratio
    progress_budget = wealth * progress_budget_ratio
    imm_perf_budget = wealth * imm_perf_budget_ratio

    eps = 1e-12
    imm_eps = eps * imm_perf_budget_ratio
    prog_eps = eps * progress_budget_ratio
    # all_bundles = np.array(dist_data['azure-bundles'], dtype=float)
    all_bundles = dist_data['bundles'].copy()

    max_alloc = sd.data['max-alloc'][k, anchor]
    percentile_99_alloc = sd.data['percentile-99-alloc'][k, anchor]
    is_real_load = sd.meta['load'].get('real-load', False)

    if is_real_load:
        max_bundle_ind = get_relevant_bundle_ind(all_bundles, max_alloc)
        bundle_percentile_ind = get_relevant_bundle_ind(all_bundles, percentile_99_alloc)
        assert max_bundle_ind >= bundle_percentile_ind, f"max alloc: {max_alloc} - " \
                                                        f"p alloc {percentile_99_alloc} - " \
                                                        f"req bundle: {req_bundle} - " \
                                                        f"max-bundle-ind: {max_bundle_ind} - " \
                                                        f"p-bundle-ind: {bundle_percentile_ind}"
        all_bundles = all_bundles[:bundle_percentile_ind+1]
        # Take the bundle that fits 99.9% of the client's load
        if all_bundles[bundle_percentile_ind] < req_bundle:
            prev_req_bundle = req_bundle
            req_bundle = all_bundles[bundle_percentile_ind]
            sd.log(k, "%.2f" % max_alloc, all_bundles[-1], prev_req_bundle, req_bundle)
            sd.internal['selected-bundle'][k] = req_bundle
    else:
        max_bundle_ind = get_relevant_bundle_ind(all_bundles, max_alloc)
        all_bundles = all_bundles[:max_bundle_ind + 1]

    req_bundle_ind = np.abs(all_bundles - req_bundle).argmin()
    prev_bundles = [b for b in all_bundles if b < req_bundle - eps]
    next_bundles = [b for b in all_bundles if b > req_bundle + eps]
    assert len(prev_bundles) > 0, f"max alloc: {max_alloc} - req bundle: {req_bundle} - " \
                                  f"max-bundle-ind: {max_bundle_ind}"
    # assert len(next_bundles) > 0, "max alloc: %s - req bundle: %s" % (max_alloc, req_bundle)

    load = sd.internal.get('load', None)
    if load is None:
        load = produce.get_full_load(sd, k)
    else:
        load = load[k]
    alloc_load = produce.get_alloc_for_perf(sd, load, k)[:, anchor]
    cdf_size = produce.get_cdf_size(sd)
    alloc_x, alloc_cdf_y = stats.cdf_from_sample(alloc_load, cdf_size, x_limits=(0, all_bundles[-1]))
    # if graphic_output:
    #     plt.plot(alloc_x, alloc_cdf_y, label='Alloc ECDF')
    #     plt.xlabel("CPU Usage")
    #     plt.show()

    def get_left_i(input_x, input_b):
        after_i_diff = input_x - input_b
        after_i_diff[after_i_diff > 0] = after_i_diff.min()-1
        return after_i_diff.argmax()

    prev_bundle_i = get_left_i(alloc_x, prev_bundles[-1])
    req_bundle_i = get_left_i(alloc_x, req_bundle) + 1

    # val - bundle < wealth - req_bundle
    # val = wealth - req_bundle + bundle
    upper_x = all_bundles + (wealth - req_bundle)
    upper_x_lim = upper_x * 0.8
    upper_x_lim[req_bundle_ind:] = upper_x[req_bundle_ind] + (all_bundles[req_bundle_ind:] - all_bundles[req_bundle_ind]) * 0.5

    imm_upper_x_lim = upper_x_lim * imm_perf_budget_ratio

    alloc_imm_val = alloc_x.copy()
    alloc_imm_max_val = vecfunc.expected_value_cumsum(alloc_imm_val, alloc_cdf_y)

    def set_imm_val(updated_val):
        alloc_imm_val[:] = updated_val
        alloc_imm_max_val[:] = vecfunc.expected_value_cumsum(alloc_imm_val, alloc_cdf_y)

    # (1) Fit to wealth
    u1 = imm_perf_budget / np.interp(req_bundle, alloc_x, alloc_imm_max_val)
    set_imm_val(alloc_imm_val * u1)

    # (2) Reduce previous bundles
    if len(prev_bundles) > 0:
        prev_bundles_max_val = np.interp(prev_bundles, alloc_x, alloc_imm_max_val)
        prev_bundles_imm_upper_x_lim = imm_upper_x_lim[:req_bundle_ind]
        over_limit = (prev_bundles_max_val - prev_bundles_imm_upper_x_lim) > imm_eps
        if np.sum(over_limit) > 0:
            u2 = np.min(prev_bundles_imm_upper_x_lim[over_limit] / prev_bundles_max_val[over_limit])
            set_imm_val(alloc_imm_val * u2)

            # (3) Increase required bundle to wealth (if reduced before)
            prev_bundle_max_val = np.interp(prev_bundles[-1], alloc_x, alloc_imm_max_val)
            req_bundle_max_val = np.interp(req_bundle, alloc_x, alloc_imm_max_val)
            u3 = (imm_perf_budget - prev_bundle_max_val) / (req_bundle_max_val - prev_bundle_max_val)
            prev_bundle_alloc_imm_val = np.interp(prev_bundles[-1], alloc_x, alloc_imm_val)
            alloc_imm_val[prev_bundle_i:] = prev_bundle_alloc_imm_val * (1-u3) + alloc_imm_val[prev_bundle_i:] * u3
            set_imm_val(alloc_imm_val)

    # (4) Reduce following bundles
    if len(next_bundles) > 0:
        next_bundles_max_val = np.interp(next_bundles, alloc_x, alloc_imm_max_val)
        next_bundles_imm_upper_x_lim = imm_upper_x_lim[req_bundle_ind+1:]
        over_limit = (next_bundles_max_val - next_bundles_imm_upper_x_lim) > imm_eps
        if np.sum(over_limit) > 0:
            req_bundle_max_val = np.interp(req_bundle, alloc_x, alloc_imm_max_val)
            u4 = np.min((next_bundles_imm_upper_x_lim[over_limit] - req_bundle_max_val) / (
                        next_bundles_max_val[over_limit] - req_bundle_max_val))
            req_bundle_alloc_imm_val = np.interp(req_bundle, alloc_x, alloc_imm_val)
            alloc_imm_val[req_bundle_i:] = req_bundle_alloc_imm_val * (1-u4) + alloc_imm_val[req_bundle_i:] * u4
            set_imm_val(alloc_imm_val)

    perf_x = sd.data['perf-x']
    x_load = np.interp(alloc_x, sd.data['perf'][k, anchor], perf_x)
    imm_val = np.interp(perf_x, x_load, alloc_imm_val)

    if graphic_output:
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 2, 1)

        fake_x_ticks = np.arange(len(all_bundles))
        fake_alloc_x = np.interp(alloc_x, all_bundles, fake_x_ticks)
        fake_req_bundle = np.interp(req_bundle, all_bundles, fake_x_ticks)

        plt.xticks(fake_x_ticks, list(map('{:.3g}'.format, all_bundles)))
        plt.grid(linestyle=":")
        plt.xlabel("Bundle (CPUs)")
        plt.ylabel("Value ($)")
        plt.title("Imm Valuation")

        plt.plot(fake_alloc_x, alloc_imm_max_val, label="Imm Val")

        alloc_unit_price = alloc_x * imm_perf_budget_ratio
        plt.plot(fake_alloc_x, alloc_unit_price, color='grey', linestyle='--', label='Price')
        plt.plot(fake_x_ticks, upper_x * imm_perf_budget_ratio, color='red', linewidth=1, label='Upper Limit')
        plt.plot(fake_x_ticks, imm_upper_x_lim, color='red', linewidth=1, linestyle=':', label='Enforced Limit')

        d = alloc_imm_max_val - alloc_unit_price
        i = d.argmax()
        plt.gca().axvline(x=fake_alloc_x[i], linestyle='--', color='green', linewidth=1, alpha=0.7, label='Optimal')
        print("Optimal (total)  :", alloc_x[i], "- val:", alloc_imm_max_val[i], "- profit:", d[i])

        bundles_imm_val = np.interp(all_bundles, alloc_x, alloc_imm_max_val)
        d = bundles_imm_val - (all_bundles * imm_perf_budget_ratio)
        i = d.argmax()
        plt.gca().axvline(x=fake_x_ticks[i], linestyle='--', color='blue', linewidth=1, alpha=0.7, label='Optimal Bundle')
        print("Optimal (bundle) :", all_bundles[i], "- val:", bundles_imm_val[i], "- profit:", d[i],
              "- next-val:", bundles_imm_val[min(i+1, len(bundles_imm_val)-1)])

        plt.gca().axvline(x=fake_req_bundle, linestyle='--', color='black', linewidth=1, alpha=0.7, label='Target')
        plt.gca().axhline(y=imm_perf_budget, linestyle='--', color='black', linewidth=1, alpha=0.7)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=4)

    # Start progress val
    mean_load = sd.data['mean-load'][k]
    prog_upper_x_lim = upper_x_lim * progress_budget_ratio

    alloc_avg_perf = np.interp(alloc_x, sd.data['perf'][k, anchor], sd.data['max-perf-avg-perf'][k])

    # (1) Fit to wealth
    u1 = progress_budget / np.interp(req_bundle, alloc_x, alloc_avg_perf)
    alloc_prog_val = alloc_avg_perf * u1

    # (2) Reduce previous bundles
    if len(prev_bundles) > 0:
        prev_bundles_prog_val = np.interp(prev_bundles, alloc_x, alloc_prog_val)
        prev_bundles_prog_upper_x_lim = prog_upper_x_lim[:req_bundle_ind]
        over_limit = (prev_bundles_prog_val - prev_bundles_prog_upper_x_lim) > prog_eps
        if np.sum(over_limit) > 0:
            u2 = np.min(prev_bundles_prog_upper_x_lim[over_limit] / prev_bundles_prog_val[over_limit])
            alloc_prog_val *= u2

            # (3) Increase required bundle to wealth (if reduced before)
            prev_bundle_prog_val = np.interp(prev_bundles[-1], alloc_x, alloc_prog_val)
            req_bundle_prog_val = np.interp(req_bundle, alloc_x, alloc_prog_val)
            u3 = (progress_budget - prev_bundle_prog_val) / (req_bundle_prog_val - prev_bundle_prog_val)
            alloc_prog_val[prev_bundle_i:] = prev_bundle_prog_val * (1-u3) + alloc_prog_val[prev_bundle_i:] * u3

    # (4) Reduce following bundles
    if len(next_bundles) > 0:
        next_bundles_prog_val = np.interp(next_bundles, alloc_x, alloc_prog_val)
        next_bundles_prog_upper_x_lim = prog_upper_x_lim[req_bundle_ind+1:]
        over_limit = (next_bundles_prog_val - next_bundles_prog_upper_x_lim) > prog_eps
        if np.sum(over_limit) > 0:
            req_bundle_prog_val = np.interp(req_bundle, alloc_x, alloc_prog_val)
            u4 = (next_bundles_prog_upper_x_lim[over_limit] - req_bundle_prog_val) / (
                        next_bundles_prog_val[over_limit] - req_bundle_prog_val)
            u4 = np.min(u4)
            alloc_prog_val[req_bundle_i:] = req_bundle_prog_val * (1-u4) + alloc_prog_val[req_bundle_i:] * u4

    prog_val = np.interp(perf_x, alloc_avg_perf / mean_load, alloc_prog_val)

    if graphic_output:
        plt.subplot(1, 2, 2)
        plt.plot(fake_alloc_x, alloc_prog_val, label="Prog Val")

        alloc_unit_price = alloc_x * progress_budget_ratio
        plt.plot(fake_alloc_x, alloc_unit_price, color='grey', linestyle='--', label='Price')
        plt.plot(fake_x_ticks, upper_x * progress_budget_ratio, color='red', linewidth=1, label='Upper Limit')
        plt.plot(fake_x_ticks, prog_upper_x_lim, color='red', linewidth=1, linestyle=':', label='Enforced Limit')

        d = alloc_prog_val - alloc_unit_price
        i = d.argmax()
        plt.gca().axvline(x=fake_alloc_x[i], linestyle='--', color='green', linewidth=1, alpha=0.7, label='Optimal')
        print("Optimal (total) :", alloc_x[i], "- val:", alloc_prog_val[i], "- profit:", d[i])

        bundles_prog_val = np.interp(all_bundles, alloc_x, alloc_prog_val)
        d = bundles_prog_val - (all_bundles * progress_budget_ratio)
        i = d.argmax()
        plt.gca().axvline(x=fake_x_ticks[i], linestyle='--', color='blue', linewidth=1, alpha=0.7, label='Optimal Bundle')
        print("Optimal (bundle) :", all_bundles[i], "- val:", bundles_prog_val[i], "- profit:", d[i],
              "- next-val:", bundles_prog_val[min(i+1, len(bundles_prog_val)-1)])

        plt.gca().axvline(x=fake_req_bundle, linestyle='--', color='black', linewidth=1, alpha=0.7, label='Target')
        plt.gca().axhline(y=progress_budget, linestyle='--', color='black', linewidth=1, alpha=0.7)

        plt.xticks(fake_x_ticks, list(map("{:.3g}".format, all_bundles)))
        plt.xlabel("Bundle (CPUs)")
        plt.ylabel("Value ($)")
        plt.title("Prog Valuation")
        plt.grid(linestyle=":")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)

        plt.show()

    return imm_val, prog_val
