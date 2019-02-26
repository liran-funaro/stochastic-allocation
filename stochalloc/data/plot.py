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

from matplotlib import pylab as plt
from matplotlib.colors import LogNorm

from stochalloc.data import produce


#####################################################################
# Wealth
#####################################################################

def plot_wealth(sd):
    dist_data = sd.dist_data
    x = np.arange(sd.n)

    wealth = dist_data['wealth']

    sorted_wealth = np.sort(wealth)
    plt.plot(x, wealth)
    plt.plot(x, sorted_wealth)
    plt.xlabel('Player')
    plt.ylabel('Wealth')
    plt.show()


def plot_wealth_max_usage(sd):
    dist_data = sd.dist_data
    res_size = sd.meta['resources']['size']
    res_name = sd.meta['resources']['name']
    x = np.arange(sd.n)

    wealth = dist_data['wealth']
    wealth_uniform = dist_data['wealth-uniform']
    max_usage = dist_data['max-usage']
    max_usage_uniform = dist_data.get('max-usage-uniform', None)

    a = np.argsort(wealth_uniform)

    ax_w = plt.gca()
    ax_u = plt.twinx(ax_w)

    ax_w.plot(x, wealth[a], color='black', label='Wealth', alpha=0.7, linestyle="--", linewidth=2)
    ax_u.plot(x, wealth_uniform[a], color='black', label='Wealth Uniform', alpha=0.7, linestyle=":", linewidth=1)

    colors = 'red', 'green', 'blue'
    for i, (r, n) in enumerate(zip(res_size, res_name)):
        ax_u.plot(x, max_usage[a, i] / r, color=colors[i], label='%s Max' % n, linewidth=1)
        if max_usage_uniform is not None:
            ax_u.plot(x, max_usage_uniform[i, a], color=colors[i], label='%s Uniform' % n, alpha=0.7, linestyle=":",
                      linewidth=1)
    plt.legend()


#####################################################################
# Performance
#####################################################################

def plot_perf(sd, k=0, show_initial=True, color='black'):
    res_names = sd.meta['resources']['name']
    perf_x = sd.data['perf-x']
    init_perf = sd.init_data['perf'][k]
    perf = sd.data['perf'][k]

    if sd.ndim == 2:
        ax = plt.gca(projection='3d')
        if show_initial:
            ax.plot(*init_perf, label='Initial Perf.', linestyle=":", alpha=0.7, color='green')
        ax.plot(*perf, perf_x, label='Refined Perf', color=color)
        ax.set_xlabel(res_names[0])
        ax.set_ylabel(res_names[1])
        ax.set_zlabel("Performance")
    elif sd.ndim == 1:
        if show_initial:
            plt.plot(np.array(init_perf[0])*perf[0][-1], init_perf[1], label='Initial Perf.',
                     linestyle=":", alpha=0.7, color='green')
        plt.plot(*perf, perf_x, label='Refined Perf', color=color)
        plt.xlabel(res_names[0])
        plt.ylabel("Performance")


def plot_step_func(sd, k=0):
    res_names = sd.meta['resources']['name']
    step_func = sd.data['step-func'][k].T
    if sd.ndim == 2:
        plt.scatter(*step_func, s=1, c=sd.data['step-x'])
        plt.xlabel(res_names[0])
        plt.ylabel(res_names[1])
    elif sd.ndim == 1:
        plt.plot(*step_func, sd.data['step-x'], marker='o')
        plt.xlabel(res_names[0])
        plt.ylabel("Performance")


def plot_many_perf(sd, players=None, range_start=None, range_count=None, show_initial=True):
    if players is None:
        players = range(range_start, range_start + range_count)
    else:
        range_count = len(players)

    rows = np.ceil(range_count / 4)
    plt.figure(figsize=(16, 4 * rows))
    t = 1
    for i in players:
        if sd.ndim == 2:
            plt.subplot(rows, 4, t, projection='3d')
        elif sd.ndim == 1:
            plt.subplot(rows, 4, t)
        t += 1
        plt.title('Player: %s' % i)
        plot_perf(sd, i, show_initial=show_initial)


def plot_many_step_func(sd, players=None, range_start=None, range_count=None):
    if players is None:
        players = range(range_start, range_start + range_count)
    else:
        range_count = len(players)

    rows = np.ceil(range_count / 4)
    plt.figure(figsize=(16, 4 * rows))
    t = 1
    for i in players:
        plt.subplot(rows, 4, t)
        t += 1
        plt.title('Player: %s' % i)
        plot_step_func(sd, i)
    plt.tight_layout()


#####################################################################
# Load
#####################################################################


def plot_load_heatmap(sd, k):
    load = produce.get_load(sd)[k] * 100

    time_splits = 64
    load_splits = 128
    load_bins = np.array_split(load, time_splits)
    rounds_per_bin = len(load) / time_splits
    bins_x = np.linspace(0, 24, len(load_bins))
    bins_y = np.linspace(0, np.max(load), load_splits)
    bins = np.zeros((len(bins_x), len(bins_y)-1))
    for i, b in enumerate(load_bins):
        bins[i] = np.histogram(b, bins_y)[0]

    bins = np.ma.masked_where(bins < 1, bins).astype(float) * 100. / rounds_per_bin
    cmap = plt.cm.Purples
    cmap.set_bad(color='white')

    heatmap = plt.pcolor(bins_x, bins_y, np.transpose(bins), cmap=cmap,
                         norm=LogNorm(vmin=bins.min(), vmax=100))

    mean_bin = [b.mean() for b in load_bins]
    plt.plot(bins_x, mean_bin, 'green', label='Local Mean', linewidth=2, alpha=0.6)

    # x = np.linspace(0, 1, len(base_load))
    # plt.plot(x, base_load, 'r-', label='base load', linewidth=2, linestyle="--", alpha=0.9)
    plt.xlabel('Time of Day (hour)')
    plt.ylabel('Required Performance')

    plt.xticks(np.arange(0, 25, 6))
    plt.legend()
    return heatmap


def plot_load_resource_heatmap(sd, k):
    load = produce.get_load(sd)[k]
    load = np.array(produce.get_alloc_for_perf(sd, load, k))[:, 0].astype(float)

    time_splits = 64
    load_splits = 128
    load_bins = np.array_split(load, time_splits)
    rounds_per_bin = len(load) / time_splits
    bins_x = np.linspace(0, 24, len(load_bins))
    bins_y = np.linspace(0, np.max(load), load_splits)
    bins = np.zeros((len(bins_x), len(bins_y)-1))
    for i, b in enumerate(load_bins):
        bins[i] = np.histogram(b, bins_y)[0]

    bins = np.ma.masked_where(bins < 1, bins).astype(float) * 100. / rounds_per_bin
    cmap = plt.cm.Purples
    cmap.set_bad(color='white')

    heatmap = plt.pcolor(bins_x, bins_y, np.transpose(bins), cmap=cmap,
                         norm=LogNorm(vmin=bins.min(), vmax=100))

    mean_bin = [b.mean() for b in load_bins]
    plt.plot(bins_x, mean_bin, 'green', label='Local Mean', linewidth=2, alpha=0.6)

    plt.xlabel('Time of Day (hour)')
    plt.ylabel('Required Resource')

    plt.xticks(np.arange(0, 25, 6))
    plt.legend()
    return heatmap


def plot_many_load(sd, players=None, range_start=None, range_count=None, resource=False):
    if players is None:
        players = range(range_start, range_start + range_count)
    else:
        range_count = len(players)

    rows = np.ceil(range_count / 4)
    plt.figure(figsize=(16, 5 * rows))
    t = 1

    # base_load = sd.dist_data['base-load']
    for i in players:
        plt.subplot(rows, 4, t)
        t += 1
        plt.title('[%s] %.2f %s %s' % (i, sd.dist_data['mean-load'][i], sd.dist_data['selected-bundle'][i],
                                       sd.dist_data['selected-cores'][i]))
        if resource:
            plot_load_resource_heatmap(sd, i)
        else:
            plot_load_heatmap(sd, i)


def plot_load_cdf(sd, p):
    load_cdf = produce.read_load_cdf(sd)[p]
    x = np.linspace(0, 100, len(load_cdf))
    plt.plot(x, load_cdf, color='black', linewidth=2)
    plt.xlabel('Load')
    plt.ylabel('Cumulative Probability')


def plot_many_load_cdf(sd, players=None, range_start=None, range_count=None):
    if players is None:
        players = range(range_start, range_start + range_count)
    else:
        range_count = len(players)

    load_cdf = produce.read_load_cdf(sd)

    h = int(np.ceil(range_count / 4))
    plt.figure(figsize=(16, 5 * h))
    t = 1
    for i in players:
        plt.subplot(h, 4, t)
        t += 1
        plt.title('Player: %s' % i)
        x = np.linspace(0, 100, len(load_cdf[i]))
        plt.plot(x, load_cdf[i])
        plt.xlabel('Load')
        plt.ylabel('Cumulative Probability')


#####################################################################
# Valuation
#####################################################################

def plot_val(sd, i, show_ratio=True):
    perf_x = sd.data['perf-x']
    p = sd.dist_data['val-ratio'][i]
    imm_label = "Immediate"
    prog_label = "Long Term"
    if show_ratio:
        imm_label += " (%.2f)" % p
        prog_label += " (%.2f)" % (1-p)
    plt.plot(perf_x*100, sd.data['val-imm-perf'][i], label=imm_label,
             color='black', linestyle='-', linewidth=2)
    plt.plot(perf_x*100, sd.data['val-progress'][i], label=prog_label,
             color='grey', linestyle='--', linewidth=2)
    plt.xlabel('Required Performance')
    plt.ylabel('Valuation')
    plt.legend()


def plot_many_val(sd, players=None, range_start=None, range_count=None):
    if players is None:
        players = range(range_start, range_start + range_count)
    else:
        range_count = len(players)

    rows = np.ceil(range_count / 4)
    plt.figure(figsize=(16, 4 * rows))
    t = 1
    for i in players:
        plt.subplot(rows, 4, t)
        t += 1
        plt.title('[%s]' % i)
        plot_val(sd, i)
