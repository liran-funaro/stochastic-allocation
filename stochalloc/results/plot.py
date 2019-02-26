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
import tabulate

from cloudsim import stats
from stochalloc import results
from stochalloc.results import analyze
import itertools
import vecfunc

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


#####################################################################################################
# Style Helpers
#####################################################################################################

def fraction_to_text(a, tex=False):
    if not isinstance(a, float):
        return str(a)
    d, n = a.as_integer_ratio()
    if n == 1:
        return str(d)
    if d > 1e7 or n > 1e7:
        return str(a)

    if tex:
        return r"$\frac{%s}{%s}$" % (d, n)
    else:
        return "%s/%s" % (d, n)


def format_grid_and_legend():
    plt.grid(linestyle=":", linewidth=1)
    ax = plt.gca()
    lines_count = len(ax.lines)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ncol = max(1, int(lines_count / 10))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol)


def set_axis_params(main_ax, label, color, pos, ax_space, set_main_ax=False):
    if set_main_ax:
        ax = main_ax
    else:
        ax = main_ax.twinx()
    ax.set_ylabel(label)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)
    if not set_main_ax:
        ax.spines["right"].set_position(("axes", pos))
        pos += ax_space
    return ax, pos


#####################################################################################################
# Unified Results summary
#####################################################################################################


def plot_select_bundle_summary(ds_obj, sim_name, sim_param=None, sub_name=None, compare_with_reserved=False):
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name)
    if compare_with_reserved:
        reserved_only_unified_result = results.read_mean_unified_results(ds_obj, 'reserved')
    else:
        reserved_only_unified_result = None

    eps = np.finfo(np.float32).eps

    lines = []
    for k in ('active-player-count', 'allocated-count', 'active-servers', 'always-active-servers',
              'host-revenue', ('resources-allocated', 0), ('resources-allocated', 1), 'effective-value-mean-ratio'):
        name = k
        ind = 0
        if k == 'always-active-servers':
            k = 'active-servers'
        if type(k) == tuple:
            name = "-".join(map(str, k))
            ind = k[1]
            k = k[0]
        if k not in unified_result:
            continue
        cur_line = [name, unified_result[k]]
        if compare_with_reserved:
            cur_line.append(reserved_only_unified_result[k])

        for i in range(1, len(cur_line)):
            if name == 'allocated-count':
                cur_line[i] = analyze.get_always_active_mean_alloc_count(unified_result['active-servers'], cur_line[i])
            elif name == 'active-servers':
                cur_line[i] = np.sum(cur_line[i], axis=-1)
            elif name == 'always-active-servers':
                cur_line[i] = np.sum(cur_line[i] > (1-eps), axis=-1)
            elif k == 'resources-allocated':
                cur_line[i] = analyze.get_always_active_mean_alloc_count(unified_result['active-servers'], cur_line[i],
                                                                         axis=-2)
                if i == 2 and ind > 0:
                    cur_line[i] = 1
                else:
                    cur_line[i] = cur_line[i][ind]
        if compare_with_reserved:
            cur_line.append(cur_line[1]/cur_line[2])

        lines.append(cur_line)

    headers = "Key", "Value"
    if compare_with_reserved:
        headers += "Reserved Value", "Factor"

    print(tabulate.tabulate(lines, headers=headers, tablefmt='pipe', numalign='decimal'))


def plot_select_alloc_summary_compare(ds_obj, sim_name, sim_param=None, sub_name=None, req_server_price=64):
    reserved_only_unified_result = results.read_mean_unified_results(ds_obj, 'reserved')
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                       require_server_sim_results=False)
    server_price = unified_result['input']['server-price']
    reserved_only_server_price = reserved_only_unified_result['input']['server-price']

    up_ind = [np.abs(server_price - req_server_price).argmin()]
    res_up_ind = [np.abs(reserved_only_server_price - req_server_price).argmin()]

    active_count = unified_result['active-player-count'][up_ind]
    reserved_only_active_count = reserved_only_unified_result['active-player-count'][res_up_ind]

    allocated_count = unified_result['allocated-count'][up_ind]
    sel_allocated_count = unified_result['selected-allocated-count'][up_ind]
    reserved_only_allocated_count = reserved_only_unified_result['allocated-count'][res_up_ind]

    host_revenue = unified_result['host-revenue'][up_ind]
    sel_host_revenue = unified_result['selected-host-revenue'][up_ind]
    reserved_only_host_revenue = reserved_only_unified_result['host-revenue'][res_up_ind]

    plt.figure(figsize=(8, 4))
    fake_x = np.arange(len(res_up_ind)).astype(float)
    width = 0.17

    mean_alloc_count = analyze.get_mean_alloc_count(allocated_count, eps=1)
    sel_mean_alloc_count = analyze.get_mean_alloc_count(sel_allocated_count, eps=1)
    res_mean_alloc_count = analyze.get_mean_alloc_count(reserved_only_allocated_count, eps=1)

    text_space = 2 * 0.01

    def plot_norm_data(x, d, r, label):
        y = d / r
        plt.bar(x, y, width=width, label=label)
        for i, fv, v1, v2 in zip(x, y, d, r):
            plt.text(i, fv + text_space, "%.1f/%.1f" % (v1, v2), color='blue', fontweight='bold',
                     ha='center', va='bottom', rotation=90)
        x += width

    plot_norm_data(fake_x, active_count, reserved_only_active_count, 'Active Count')
    plot_norm_data(fake_x, sel_mean_alloc_count, res_mean_alloc_count, '(Original) Players in Server')
    plot_norm_data(fake_x, mean_alloc_count, res_mean_alloc_count, 'Players in Server')
    plot_norm_data(fake_x, sel_host_revenue, reserved_only_host_revenue, '(Original) Host Revenue')
    plot_norm_data(fake_x, host_revenue, reserved_only_host_revenue, 'Host Revenue')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.tight_layout()
    plt.grid(linestyle=":", linewidth=1)
    ylim = plt.gca().get_ylim()
    plt.ylim((0, ylim[1]+1))
    plt.title("Compare Social Welfare")
    plt.show()


def plot_allocation_utilization_summary(ds_obj, sim_name, sim_param=None, sub_name=None):
    try:
        unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                           require_server_sim_results=True)
    except:
        return
    res_names = unified_result['input']['res-names']
    ndim = ds_obj.metadata['ndim']

    util_cdf = unified_result['utilization-cdf']
    res_util_cdf = unified_result['reserved-utilization-cdf']

    mean_util = unified_result.get('mean-utilization', unified_result.get('utilization', [0]*ndim))
    res_mean_util = unified_result.get('reserved-mean-utilization',
                                       unified_result.get('reserved-utilization', [0] * ndim))

    max_util = unified_result.get('max-utilization', [1]*ndim)
    res_max_util = unified_result.get('reserved-max-utilization', [1] * ndim)

    util_ratio_cdf = unified_result.get('utilization-ratio-cdf', None)
    alloc_realization_cdf = unified_result.get('alloc-realization-cdf', None)
    alloc_realization_portion = unified_result.get('alloc-realization-portion', None)

    cols = 2
    lines_count = int(np.ceil(((ndim * 2) + 1) / cols))
    cur_plot = 1

    plt.figure(figsize=(16, 3 * lines_count))

    if any(a is not None for a in [util_cdf, res_util_cdf]):
        for d in range(ndim):
            plt.subplot(lines_count, cols, cur_plot)
            cur_plot += 1
            x = np.linspace(0, 1, len(util_cdf[d]))
            plt.plot(x, util_cdf[d], label="Total")
            print("Mean utilization: %g - Max utilization: %g" % (mean_util[d], max_util[d]))
            cur_x = np.interp(mean_util[d], util_cdf[d], x)
            plt.plot([cur_x], [mean_util[d]], marker='*')

            x = np.linspace(0, 1, len(res_util_cdf[d]))
            plt.plot(x, res_util_cdf[d], label="Reserved",
                     linestyle='-.')
            print("Reserved Mean utilization: %g - Max utilization: %g" % (res_mean_util[d], res_max_util[d]))
            cur_x = np.interp(res_mean_util[d], res_util_cdf[d], x)
            plt.plot([cur_x], [res_mean_util[d]], marker='*')

            plt.title("%s Utilization" % res_names[d])
            plt.grid(linestyle=":", linewidth=1)
            plt.xlabel("Portion of %s" % res_names[d])
            plt.ylabel("CDF")
            plt.legend()

    if alloc_realization_cdf is not None and alloc_realization_portion is not None:
        plt.subplot(lines_count, cols, cur_plot)
        cur_plot += 1
        x = np.linspace(0, 1, len(alloc_realization_cdf))
        plt.plot(x, alloc_realization_cdf, label="Joint Alloc. Realization")

        m = unified_result['max-value'] > np.finfo(np.float32).eps
        cdf_x, cdf_y = stats.cdf_from_sample(np.minimum(alloc_realization_portion[m].flatten(), 1))
        plt.plot(cdf_x, cdf_y, label="Player's Average Realization")

        plt.title("Allocation Realization (%)")
        plt.grid(linestyle=":", linewidth=1)
        plt.legend()

    if util_ratio_cdf is not None and ndim > 1:
        plt.subplot(lines_count, cols, cur_plot)
        cur_plot += 1
        x = np.linspace(0, 1, len(util_ratio_cdf))
        plt.plot(x, util_ratio_cdf, label="Min used / Max used")
        plt.title("Utilization Ratio")
        plt.grid(linestyle=":", linewidth=1)

    plt.show()


def plot_allocation_sw_summary(ds_obj, sim_name, sim_param=None, sub_name=None):
    try:
        unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                           require_server_sim_results=True)
    except:
        return

    max_value = unified_result.get('max-value', None)
    expected_value = unified_result.get('expected-value', None)
    effective_value = unified_result.get('effective-value', None)

    width = 0.3
    if any(a is not None for a in [max_value, expected_value, effective_value]):
        plt.figure(figsize=(8, 3))
        plt.bar([0], [max_value.sum()], label='Max Social Welfare', width=width)
        plt.bar([width], [expected_value.sum()], label='Expected Social Welfare', width=width)
        plt.bar([2 * width], [effective_value.sum()], label='Effective Social Welfare', width=width)

        plt.xticks([])
        plt.title("Social Welfare")
        plt.xlabel("Server Price ($)")
        plt.ylabel("Social Welfare ($)")
        format_grid_and_legend()
        plt.show()


def plot_valuation_hist(ds_obj, sim_name, sim_param=None, sub_name=None, color=None, alpha=0.5):
    try:
        unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                           require_server_sim_results=True)
    except:
        return

    ratios = unified_result['effective-value-ratio']
    # max_ratio = np.max(ratios) * 1.1
    hist_size = 32
    hist_bin_e = np.linspace(0 - 1e-18, 10, hist_size)
    val_ratio_hist, _ = np.histogram(ratios, bins=hist_bin_e)
    hist_bin_e[0] = 0
    hist_bin_e[-1] = 2 * hist_bin_e[-2] - hist_bin_e[-3]

    val_ratio_hist = val_ratio_hist.astype(float)
    val_ratio_hist /= np.sum(val_ratio_hist)
    bins = stats.calc_hist_bin_centers(hist_bin_e)
    width = (bins[-1] - bins[0])/len(bins)

    plt.bar(bins, val_ratio_hist, width=width, edgecolor='black', alpha=alpha,
            label='%s %s %s' % (sim_name, sim_param, sub_name), color=color)
    # ax = plt.gca()
    # ax.set_yscale('log')
    # plt.show()


def plot_valuation_violin(ds_obj, sim_name, sim_param=None, sub_name=None):
    try:
        unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                           require_server_sim_results=True)
    except:
        return
    val_ratios = unified_result['effective-value-ratio']
    alloc_server_price = unified_result['input'].get('alloc-server-price', None)
    plt.violinplot(val_ratios, alloc_server_price, showmeans=True, showextrema=False, showmedians=True)


def plot_bundle_select_hist(ds_obj, sim_name, sim_param=None, sub_name=None):
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                       require_server_sim_results=False)
    res_names = unified_result['input']['res-names']
    ndim = unified_result['input']['ndim']
    shares_hist = unified_result.get('shares-hist', None)

    reserved_allocs = np.round(unified_result['input']['reserved-allocs'], 5)

    res_alloc = unified_result.get('resources-allocated', None)
    active = unified_result.get('active-servers', None)
    if all(a is not None for a in (res_alloc, active)):
        res_alloc = analyze.get_always_active_mean_alloc_count(active, res_alloc, axis=-2)

    nplots = ndim
    if shares_hist is not None:
        nplots *= 2
    ncols = 2
    nrows = int(nplots + 1 / ncols)
    plot_ind = 1
    plt.figure(figsize=(16, 4 * nrows))

    for d in range(ndim):
        reserved_hist_x = reserved_allocs[d]
        reserved_hist_fake_x = np.arange(len(reserved_hist_x))
        reserved_hist_y = unified_result['reserved-hist'][d]
        player_count = np.sum(reserved_hist_y)

        plt.subplot(ncols, nrows, plot_ind)
        plot_ind += 1
        plt.bar(reserved_hist_fake_x, reserved_hist_y)
        text_space = reserved_hist_y.max() * 0.01
        for i, fv, v in zip(reserved_hist_fake_x, reserved_hist_y, reserved_hist_y):
            plt.text(i, max(v, fv) + text_space, "%.1f%%" % (100 * v / player_count),
                     color='blue', fontweight='bold', ha='center')
        plt.xticks(reserved_hist_fake_x, map(fraction_to_text, reserved_hist_x))
        total_needed_resource = np.sum(reserved_hist_x * reserved_hist_y)
        # noinspection PyStringFormat
        title = "Selected resource for %s - Total need: %.2f" % (res_names[d], total_needed_resource)
        if res_alloc is not None:
            title += " - Total allocated: %.2f" % res_alloc[d]
        plt.title(title)

        if sim_name == 'reserved':
            inds = [np.abs(reserved_hist_x - a).argmin() for a in analyze.AZURE_BINS_MAX]
            plt.bar(reserved_hist_fake_x[inds], np.array(analyze.AZURE_HIST_MAX) * player_count / 100, color='black',
                    width=0.5, alpha=0.5)

    if shares_hist is None:
        return

    shares_allocs = np.round(unified_result['input']['shares-allocs'], 5)
    shares_limit = unified_result['input']['shares-limit']
    if shares_limit is not None:
        shares_limit = np.round(shares_limit, 5)

    for d in range(ndim):
        shares_hist_x = np.array(shares_allocs[d])
        shares_hist_fake_x = np.arange(len(shares_hist_x))
        shares_hist_y = unified_result['shares-hist'][d]
        player_count = np.sum(shares_hist_y)

        plt.subplot(ncols, nrows, plot_ind)
        plot_ind += 1

        plt.bar(shares_hist_fake_x, shares_hist_y)
        text_space = shares_hist_y.max() * 0.01
        for i, fv, v in zip(shares_hist_fake_x, shares_hist_y, shares_hist_y):
            plt.text(i, max(fv, v)+text_space, "%.2f%%" % (100 * v / player_count),
                     color='blue', fontweight='bold', ha='center')
        if shares_limit is not None:
            x_ticks = list(
                map("\n".join, zip(map("{:g}".format, shares_hist_x), map(fraction_to_text, shares_limit[d]))))
        else:
            x_ticks = list(map(fraction_to_text, shares_hist_x))

        plt.xticks(shares_hist_fake_x, x_ticks)
        plt.title("Select shares for %s- Total allocated: %.2f" % (res_names[d], res_alloc[ndim+d]))


def plot_mean_unified_results(ds_obj, sim_name, sim_param=None, sub_name=None, compare_with_reserved=False):
    plot_select_bundle_summary(ds_obj, sim_name, sim_param, sub_name, compare_with_reserved=compare_with_reserved)
    plot_bundle_select_hist(ds_obj, sim_name, sim_param, sub_name)
    plt.show()
    plot_allocation_utilization_summary(ds_obj, sim_name, sim_param, sub_name)
    plot_allocation_sw_summary(ds_obj, sim_name, sim_param, sub_name)
    plot_valuation_hist(ds_obj, sim_name, sim_param, sub_name, alpha=0.7)
    if compare_with_reserved:
        plot_valuation_hist(ds_obj, 'reserved', alpha=0.5, color='white')
    plot_share(ds_obj, sim_name, sim_param, sub_name, method='hist')


#####################################################################################################
# Shares Unified Results summary
#####################################################################################################


def plot_shares_iter_summary(ds_obj, sim_name, sub_name=None, clean_output=False, show_active=False,
                             mean_over_last_iterations=20):
    show_active &= not clean_output
    reserved_only_unified_result = results.read_mean_unified_results(ds_obj, 'reserved',
                                                                     require_server_sim_results=True)
    all_iter_result = results.read_all_param_unified_results(ds_obj, sim_name, sub_name=sub_name)
    iter_len = len(all_iter_result['input']['range'])
    if iter_len < 2:
        return
    iter_range = np.arange(iter_len)
    iter_ticks = np.array(all_iter_result['input']['range'])

    res_names = all_iter_result['input']['res-names']
    ndim = ds_obj.metadata['ndim']

    active_count = all_iter_result['active-player-count']
    allocated_count = all_iter_result['allocated-count']
    host_revenue = all_iter_result['host-revenue']
    res_alloc = all_iter_result['resources-allocated']
    active_servers = all_iter_result['active-servers']
    util_cdf = all_iter_result['utilization-cdf']
    if not clean_output:
        effective_val = all_iter_result['effective-value-mean-ratio']
    else:
        effective_val = None

    reserved_only_active_count = reserved_only_unified_result['active-player-count']
    reserved_only_allocated_count = reserved_only_unified_result['allocated-count']
    reserved_only_host_revenue = reserved_only_unified_result['host-revenue']
    reserved_only_active_servers = reserved_only_unified_result['active-servers']

    plt.figure(figsize=(16, 4))
    main_ax = plt.gca()
    main_ax.set_xlabel('Iteration')
    if len(iter_range) < 10:
        main_ax.set_xticks(iter_range)

    twin_ax_pos = 1
    ax_space = 0.07
    lines = []

    # Active count
    if show_active:
        ax_players, twin_ax_pos = set_axis_params(main_ax, 'Players (active)', 'blue', twin_ax_pos, ax_space,
                                                  set_main_ax=True)
        lines.extend(ax_players.plot(iter_range, active_count, color='blue', label='Avg. Active Players'))
        lines.append(ax_players.axhline(y=reserved_only_active_count, linestyle='--', linewidth=1, alpha=0.7,
                                        color='blue', label="Reserved Active Players"))

    # Allocated Servers count
    ax_alloc_players, twin_ax_pos = set_axis_params(main_ax, 'Players (allocated)', 'red', twin_ax_pos, ax_space,
                                                    set_main_ax=not show_active)
    ax_servers, twin_ax_pos = set_axis_params(main_ax, 'Servers', 'pink', twin_ax_pos, ax_space)

    mean_alloc_count = analyze.get_always_active_mean_alloc_count(active_servers, allocated_count)
    res_mean_alloc_count = analyze.get_always_active_mean_alloc_count(reserved_only_active_servers,
                                                                      reserved_only_allocated_count)

    eps = np.finfo(np.float32).eps
    all_iter_active_server_count = np.sum(active_servers, axis=-1)
    res_active_server_count = np.sum(reserved_only_active_servers, axis=-1)
    always_active_server_count = np.sum(active_servers > 1 - eps, axis=-1)

    lines.extend(ax_alloc_players.plot(iter_range, mean_alloc_count, color='red', label='Players in Server'))
    lines.extend(ax_servers.plot(iter_range, all_iter_active_server_count, color='pink',
                                 label='Servers Count'))
    lines.append(ax_alloc_players.axhline(y=res_mean_alloc_count, linestyle='--', linewidth=1, alpha=0.7, color='red',
                                          label='Reserved Players in Server'))
    lines.append(ax_servers.axhline(y=res_active_server_count, linestyle='--', linewidth=1, alpha=0.7,
                                    color='pink', label='Reserved Servers Count'))
    ax_alloc_players.set_ylim((0, None))

    all_iter_mean_alloc_count = np.mean(mean_alloc_count[-mean_over_last_iterations:])
    all_iter_mean_active_server_count = np.mean(all_iter_active_server_count[-mean_over_last_iterations:])
    print("Mean allocated players: %.3f (X%.3f)" % (all_iter_mean_alloc_count,
                                                    all_iter_mean_alloc_count/res_mean_alloc_count))
    print("Mean servers count: %.3f (X%.3f)" % (all_iter_mean_active_server_count,
                                                all_iter_mean_active_server_count/res_active_server_count))
    print("Mean always active servers count: %.3f" % np.mean(always_active_server_count[-mean_over_last_iterations:]))

    # Host revenue
    ax_rev, twin_ax_pos = set_axis_params(main_ax, 'Revenue Factor (%)', 'green', twin_ax_pos, ax_space)
    ax_rev.set_ylim(None, 1.1)
    lines.extend(ax_rev.plot(iter_range, host_revenue / reserved_only_host_revenue,
                             color='green', label='Host Revenue'))
    lines.append(ax_rev.axhline(y=1, linestyle='--', linewidth=1, alpha=0.7, color='green',
                                label='Reserved Host Revenue'))
    all_iter_host_revenue = np.mean(host_revenue[-mean_over_last_iterations:])
    print("Mean host revenue: %.3f (X%.3f)" % (all_iter_host_revenue,
                                               all_iter_host_revenue / reserved_only_host_revenue))

    # Resource allocation
    if not clean_output:
        res_colors = plt.cm.Set2(np.linspace(0, 1, 8, endpoint=False) + 1 / 16)
        mean_res_alloc = analyze.get_always_active_mean_alloc_count(active_servers, res_alloc, axis=-2)

        ax_val, twin_ax_pos = set_axis_params(main_ax, 'Valuation Factor (%)', '#444444', twin_ax_pos, ax_space)
        # Valuation Factor
        lines.extend(ax_val.plot(iter_range, effective_val, color='black', linestyle="-", label='Effective Val'))
        ax_val.set_ylim((0, None))

        ax_alloc, twin_ax_pos = set_axis_params(main_ax, 'Allocation Factor (%)', 'orange', twin_ax_pos, ax_space)
        ax_alloc.set_ylim(0, 1.1)
        # ax_alloc.grid(True)
        for d in range(ndim):
            lines.extend(ax_alloc.plot(iter_range, mean_res_alloc[:, d], color=res_colors[d], linestyle="-",
                                       label='%s Allocated' % res_names[d]))

        if mean_res_alloc.shape[-1] > ndim:
            ax_alloc_shares, twin_ax_pos = set_axis_params(main_ax, 'Shares Allocation', 'orange', twin_ax_pos, ax_space)
            for d in range(ndim):
                lines.extend(ax_alloc_shares.plot(iter_range, mean_res_alloc[:, d+ndim], color=res_colors[d],
                                                  linestyle="--", label='%s Shares Allocated' % res_names[d]))

    # Utilization
    ax_util, twin_ax_pos = set_axis_params(main_ax, 'Utilization (%)', 'black', twin_ax_pos, ax_space)
    ax_util.set_ylim(0, 1.1)
    for d in range(ndim):
        percentiles = [0.25, 0.5, 0.75]
        util_y = np.array([np.interp(percentiles, cdf[d], np.linspace(0, 1, len(cdf[d]))) for cdf in util_cdf])
        for p, u in zip(percentiles, util_y.T):
            lines.extend(ax_util.plot(iter_range, u, label="%s %g Utilization" % (res_names[d], p), color='grey'))

    plt.xlim((iter_range[0], iter_range[-1]))
    main_ax.xaxis.set_minor_locator(MultipleLocator(1))
    if len(iter_range) > 10:
        major_ticks = [iter_range[0]] + list(iter_range[4::5])
    else:
        major_ticks = iter_range
    if major_ticks[-1] != iter_range[-1]:
        if iter_range[-1] - major_ticks[-1] > 2:
            major_ticks += [iter_range[-1]]
        else:
            major_ticks[-1] = iter_range[-1]
    main_ax.set_xticks(major_ticks, minor=False)
    main_ax.set_xticklabels([iter_ticks[t] if (0 <= t < len(iter_ticks)) else '' for t in major_ticks], minor=False)

    legend_cols = 5
    upper_legend_space = 1.15 + 0.1 * int((len(lines) + 1) / legend_cols)
    plt.legend(lines, [l.get_label() for l in lines], loc='upper center',
               bbox_to_anchor=(0.5, upper_legend_space), ncol=legend_cols)
    plt.tight_layout()
    plt.grid(linestyle=":", linewidth=1)
    plt.show()


#####################################################################################################
# Share CDF
#####################################################################################################


def plot_share(ds_obj, sim_name, sim_param=None, sub_name=None, show_zero=False, log_scale=True,
               shares_allocs=None, total_shares=None, method='hist'):
    try:
        unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                           require_server_sim_results=True)
    except:
        return
    res_input = unified_result['input']

    res_names = res_input['res-names']
    res_size = res_input['res-size']

    ret = analyze.get_share_cdf(ds_obj, sim_name, sim_param, sub_name, result_type='both', shares_allocs=shares_allocs,
                                total_shares=total_shares, method=method)
    shares_allocs, shares_limit, shares_cdf, shares_cdf_limited = ret
    shares_allocs = np.round(shares_allocs, 5)
    shares_limit = np.round(shares_limit, 5)

    ndim = len(shares_cdf)
    cols = 2
    lines_count = int(np.ceil(ndim / cols))
    cur_plot = 1

    plt_func = plt.plot if not log_scale else plt.semilogx

    plt.figure(figsize=(16, 3*lines_count))

    for d in range(ndim):
        plt.subplot(lines_count, cols, cur_plot)
        cur_plot += 1

        for p_ind, p in enumerate(shares_allocs[d]):
            if np.isclose(p, 0) and not show_zero:
                continue
            cdf_y = shares_cdf[d][p_ind]
            cdf_x = np.linspace(0, res_size, len(cdf_y))
            plt_func(cdf_x, cdf_y, linestyle=':', alpha=0.5, linewidth=1)

        plt.gca().set_prop_cycle(None)

        cur_shares_limit = shares_limit[d]
        if cur_shares_limit.ndim > 1:
            cur_shares_limit = cur_shares_limit

        for p_ind, (p, l) in enumerate(zip(shares_allocs[d], cur_shares_limit)):
            if np.isclose(p, 0) and not show_zero:
                continue
            cdf_y = shares_cdf_limited[d][p_ind]
            cdf_x = np.linspace(0, res_size, len(cdf_y))
            plt_func(cdf_x, cdf_y, label="%s (%s)" % (fraction_to_text(p), fraction_to_text(l)),
                     linestyle='-', alpha=0.8, linewidth=1)

        plt.title("%s Share Alloc" % res_names[d])
        plt.grid(linestyle=":", linewidth=1)
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    plt.tight_layout()


##############################################################################################################
# LOSS functions
##############################################################################################################

def plot_shares_loss(iter_range, shares_cdf_loss):
    plot_count = np.prod(shares_cdf_loss.shape[:2])
    cols = 2 if plot_count > 1 else 1
    lines_count = int(np.ceil(plot_count / cols))
    cur_plot = 1

    plt.figure(figsize=(16, 6*lines_count))
    d_ticks = [np.arange(*iter_range) for _ in shares_cdf_loss.shape[2:]]

    for c, d in itertools.product(*map(range, shares_cdf_loss.shape[:2])):
        plt.subplot(lines_count, cols, cur_plot)
        cur_plot += 1
        vecfunc.plot(shares_cdf_loss[c, d], d_ticks=d_ticks, fmt='.1e')
        plt.title("%d:%d Shares CDF LOSS" % (c, d))


def plot_loss(iter_range, share_pack_loss, shares_cdf_loss):
    d_ticks = [np.arange(*iter_range) for _ in share_pack_loss.shape]
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    vecfunc.plot(share_pack_loss, d_ticks=d_ticks, fmt='.0f')
    plt.title("Pack LOSS")

    plt.subplot(1, 2, 2)
    vecfunc.plot(shares_cdf_loss, d_ticks=d_ticks, fmt='.0f')
    plt.title("Shares CDF LOSS")


#####################################################################################################
# Per unit price analyze
#####################################################################################################


def plot_bundle_select_hist_azure(ds_obj, sim_name, sim_param=None, sub_name=None):
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                       require_server_sim_results=False)
    res_names = unified_result['input']['res-names']

    plt.figure(figsize=(16, 4))

    reserved_hist_x = analyze.AZURE_BINS
    reserved_hist_fake_x = np.arange(len(reserved_hist_x))

    reserved_hist_y = unified_result['reserved-hist'][0]
    player_count = np.sum(reserved_hist_y)

    plt.bar(reserved_hist_fake_x, reserved_hist_y)
    text_space = reserved_hist_y.max() * 0.01
    for i, fv, v in zip(reserved_hist_fake_x, reserved_hist_y, reserved_hist_y):
        plt.text(i, max(v, fv) + text_space, "%.1f%%" % (100 * v / player_count),
                 color='blue', fontweight='bold', ha='center')

    reserved_hist_y = analyze.AZURE_HIST
    player_count = np.sum(reserved_hist_y)

    plt.bar(reserved_hist_fake_x+0.5, reserved_hist_y)
    text_space = reserved_hist_y.max() * 0.01
    for i, fv, v in zip(reserved_hist_fake_x, reserved_hist_y, reserved_hist_y):
        plt.text(i, max(v, fv) + text_space, "%.1f%%" % (100 * v / player_count),
                 color='orange', fontweight='bold', ha='center')

    plt.xticks(reserved_hist_fake_x, map('{:.3f}'.format, reserved_hist_x))

    # noinspection PyStringFormat
    plt.title("Selected resource for %s " % res_names[0])


def plot_alloc_histogram(ds_obj, sim_name, sim_param=None, sub_name=None, d=0):
    unified_result = results.read_mean_unified_results(ds_obj, sim_name, sim_param, sub_name,
                                                       require_server_sim_results=True, load_hist=True)

    h = unified_result['reserve-usage-hist'][d]
    d_vals = [np.linspace(0, 1, s + 1) for s in h.shape]
    d_vals = [(d[1:]+d[:-1])/2 for d in d_vals]
    vecfunc.plot(h, d_vals=d_vals, d_keys=['Reserved Util.', 'Total Util.'], annotate=False, mask_zero=True)
