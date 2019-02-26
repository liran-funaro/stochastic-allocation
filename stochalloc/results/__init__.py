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
import re
import gc
import numpy as np
import itertools

from cloudsim.dataset import DataSet
from cloudsim.results import get_result_array
from stochalloc.results import analyze, plot
from stochalloc.sim import param


#####################################################################################################
# Read Unified Results
#####################################################################################################


def read_results(ds_obj: DataSet, sim_name, sim_param=None, sub_name=None, require_server_sim_results=False):
    """
    Returns the raw results data
    """
    # Read bundle results
    bundle_sim_key = param.get_bundle_simulation_key(sim_name, sim_param, sub_name)
    bundle_results = ds_obj.read_results(bundle_sim_key, return_index=False)
    assert len(bundle_results) > 0

    # Read server results
    server_sim_key = param.get_server_simulation_key(sim_name, sim_param, sub_name)
    try:
        server_results = ds_obj.read_results(server_sim_key, return_index=False)
        assert len(server_results) > 0
    except:
        server_results = None
        if require_server_sim_results:
            raise

    return bundle_results, server_results


def concatenate_simulation_results(ds_obj: DataSet, bundle_results, server_results=None,
                                   sim_name=None, sim_param=None, sub_name=None, load_hist=False):
    """
    Concatenate results such that for every result key with shape of (a,b,c), the result will have a shape of
    (a,b,c,results-count)
    """
    result = {
        'input': {
            'sim-name': sim_name,
            'sim-param': sim_param,
            'sub-name': sub_name
        }
    }

    # Set input
    result['input'].update(bundle_results[0]['input'])
    result['input']['res-names'] = ds_obj.metadata['resources']['name']
    result['input']['res-size'] = ds_obj.metadata['resources']['size']
    result['input']['ndim'] = ds_obj.metadata['ndim']

    players_keys = []
    result['input']['player-keys'] = players_keys

    # Parse bundle data
    for k, v in bundle_results[0]['bundle-stats'].items():
        if type(v) in (list, tuple):
            result[k] = [get_result_array(bundle_results, 'bundle-stats', k, i) for i in range(len(v))]
        else:
            result[k] = get_result_array(bundle_results, 'bundle-stats', k)

    for k, v in bundle_results[0]['bundle'].items():
        bundle_k = "bundle-%s" % k
        players_keys.append(bundle_k)
        result[bundle_k] = get_result_array(bundle_results, 'bundle', k)

    if server_results is None:
        return result

    for k in server_results[0]['server-stats'].keys():
        if not load_hist and k == 'reserve-usage-hist':
            continue
        v = server_results[0]['server-stats'][k]
        if type(v) in (list, tuple):
            result[k] = [get_result_array(server_results, 'server-stats', k, i) for i in range(len(v))]
        else:
            result[k] = get_result_array(server_results, 'server-stats', k)

    for k, v in server_results[0]['player-stats'].items():
        players_keys.append(k)
        result[k] = get_result_array(server_results, 'player-stats', k)

    cur_expected_val = result['expected-value']
    cur_effective_val = result['effective-value']
    m = cur_expected_val > np.finfo(np.float32).eps
    all_val_ratios = cur_effective_val[m] / cur_expected_val[m]
    result['effective-value-ratio'] = all_val_ratios
    result['effective-value-mean-ratio'] = np.mean(all_val_ratios)
    players_keys.extend(['effective-value-ratio', 'effective-value-mean-ratio'])

    return result


def convert_to_mean_simulation_results(result):
    """
    Average the results such that for every result key with shape of (a,b,c), the result will have the same shape with
    a value that is average over all the results.
    """
    player_keys = result['input']['player-keys']
    sum_keys = ('reserve-usage-hist', )
    for k, v in result.items():
        if k in ('input', *player_keys):
            continue
        if k in sum_keys:
            result[k] = sum(v)
        elif type(v) in (list, tuple):
            result[k] = [sum(vv)/len(vv) for vv in v]
        else:
            result[k] = sum(v) / len(v)
    return result


def read_unified_results(ds_obj: DataSet, sim_name, sim_param=None, sub_name=None, require_server_sim_results=False,
                         load_hist=False):
    try:
        bundle_results, server_results = read_results(ds_obj, sim_name, sim_param, sub_name, require_server_sim_results)
    except Exception as e:
        if sub_name is None or sim_param is not None:
            raise e
        sim_param = get_sub_name_parameters(ds_obj, sim_name, sub_name=sub_name,
                                            require_server_sim_results=require_server_sim_results)
        if len(sim_param) == 0:
            sim_param = 0
        else:
            sim_param = sim_param[-1]
        print("Reading parameter:", sim_param, "instead of None")
        bundle_results, server_results = read_results(ds_obj, sim_name, sim_param, sub_name,
                                                      require_server_sim_results)

    return concatenate_simulation_results(ds_obj, bundle_results, server_results, sim_name, sim_param, sub_name,
                                          load_hist)


def read_mean_unified_results(ds_obj: DataSet, sim_name, sim_param=None, sub_name=None,
                              require_server_sim_results=False, load_hist=False):
    result = read_unified_results(ds_obj, sim_name, sim_param, sub_name, require_server_sim_results, load_hist)
    return convert_to_mean_simulation_results(result)


def get_sub_name_parameters(ds_obj: DataSet, sim_name, sub_name=None, require_server_sim_results=False):
    sim_name = param.get_simulation_key(None, sim_name, None, sub_name, set_reserved_param=False)
    sim_store = ds_obj.result_store.get_child(sim_name)
    res_params = sim_store.keys()

    num_regexp = re.compile(r'(\d*[.]?\d+(?:[eE][+-]?\d+)?)')
    res_params = sorted(res_params, key=lambda x: (*map(float, num_regexp.findall(x)), res_params))

    def is_valid_param(k):
        if not sim_store[k].exists(param.SIM_TYPE_BUNDLE):
            return False
        if require_server_sim_results and not sim_store[k].exists(param.SIM_TYPE_SERVER):
            return False
        return True

    return list(filter(is_valid_param, res_params))


#####################################################################################################
# Shares Unified Results
#####################################################################################################


def find_sim_next_iteration(ds_obj: DataSet, sim_name, sub_name=None, iter_start=1):
    sim_name = param.get_simulation_key(None, sim_name, None, sub_name)
    sim_store = ds_obj.result_store.get_child(sim_name)
    if sim_store.empty():
        return iter_start, param.SIM_TYPE_BUNDLE
    all_sim_keys = set(sim_store.keys[:, :])

    for cur_iter in itertools.count(start=iter_start):
        bundle_key = str(cur_iter), param.SIM_TYPE_BUNDLE
        if bundle_key not in all_sim_keys:
            return cur_iter, param.SIM_TYPE_BUNDLE
        server_key = str(cur_iter), param.SIM_TYPE_SERVER
        if server_key not in all_sim_keys:
            return cur_iter, param.SIM_TYPE_SERVER

    return 0


def read_all_param_unified_results_raw(ds_obj: DataSet, sim_name, sub_name=None, iter_count=0, iter_start=None,
                                       mean=True, load_hist=False):
    iter_range = get_sub_name_parameters(ds_obj, sim_name, sub_name, require_server_sim_results=True)
    if iter_start is None:
        iter_range = iter_range[-iter_count:]
    else:
        iter_range = iter_range[iter_start:iter_start+iter_count]
    iter_len = len(iter_range)
    if iter_len == 0:
        return
    if ds_obj.verbose:
        print(sub_name, "- Iteration len:", iter_len)

    all_res = [None] * iter_len
    for ind, i in enumerate(iter_range):
        if mean:
            result = read_mean_unified_results(ds_obj, sim_name, i, sub_name, require_server_sim_results=True)
        else:
            result = read_unified_results(ds_obj, sim_name, i, sub_name, require_server_sim_results=True,
                                          load_hist=load_hist)
        all_res[ind] = result

    return all_res, iter_range


def read_all_param_unified_results(ds_obj: DataSet, sim_name, sub_name=None, iter_count=0, iter_start=None, mean=True,
                                   load_hist=False):
    all_res, iter_range = read_all_param_unified_results_raw(ds_obj, sim_name, sub_name, iter_count, iter_start, mean,
                                                             load_hist)

    result_input = all_res[-1]['input']
    for r in all_res:
        del r['input']

    result_input['range'] = iter_range
    ret = dict(input=result_input)

    for k, v in list(all_res[-1].items()):
        if k not in all_res[0]:
            continue
        joint_res = [r.pop(k) for r in all_res]
        if type(v) in (list, tuple):
            ret[k] = [np.array([r[i] for r in joint_res]) for i in range(len(v))]
        else:
            ret[k] = np.array(joint_res)

    gc.collect()
    return ret
