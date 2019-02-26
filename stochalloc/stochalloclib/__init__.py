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
import numpy as np
from stochalloc.stochalloclib import loader


def cfs(total_resources,     # ndarray[np.float32_t, ndim=1]
        epochs_count,        # int
        required_resources,  # ndarray[np.float32_t, ndim=2]
        reserved,            # ndarray[np.float32_t, ndim=2]
        shares,              # ndarray[np.float32_t, ndim=2]
        limit,               # ndarray[np.float32_t, ndim=2]
        alloc_step_funcs     # [ndarray[np.float32_t, ndim=2] for each player]
        ):
    """
    Return alloc: ndarray[np.float32_t, ndim=2]
    """
    assert total_resources.ndim == 1
    ndim = total_resources.shape[0]

    assert all(a.ndim == 2 for a in [required_resources, reserved, shares, limit])
    assert required_resources.shape[1] == ndim
    assert all(a.shape == required_resources.shape for a in [required_resources, reserved, shares, limit])
    n = required_resources.shape[0]

    assert len(alloc_step_funcs) == n
    for asf in alloc_step_funcs:
        assert asf.shape[1] == ndim
    alloc_step_len = np.array([asf.shape[0] for asf in alloc_step_funcs], dtype=np.uint32, order='C')

    total_resources = np.require(total_resources, dtype=np.float32, requirements=loader.read_req)
    resources_epoch = np.require(total_resources / epochs_count, dtype=np.float32, requirements=loader.read_req)
    required_resources = np.require(required_resources, dtype=np.float32, requirements=loader.read_req)
    reserved = np.require(reserved, dtype=np.float32, requirements=loader.read_req)
    shares = np.require(shares, dtype=np.float32, requirements=loader.read_req)
    limit = np.require(limit, dtype=np.float32, requirements=loader.read_req)

    alloc_step_funcs = np.require(np.concatenate(alloc_step_funcs, axis=0), dtype=np.float32,
                                  requirements=loader.read_req)
    alloc_step_len = np.require(alloc_step_len, dtype=np.uint32, requirements=loader.read_req)

    ret_alloc = np.zeros_like(required_resources, dtype=np.float32, order='C')
    ret_alloc = np.require(ret_alloc, dtype=np.float32, requirements=loader.write_req)

    lib = loader.load_lib()
    lib.cfs(ndim, n, total_resources, resources_epoch, required_resources, reserved, shares, limit,
            alloc_step_funcs, alloc_step_len, ret_alloc)
    return ret_alloc


def allocate_players_to_servers(
        reserved_total_resources,  # ndarray[np.float32_t, ndim=1]
        reserved,                  # ndarray[np.float32_t, ndim=2]
        shares,                    # ndarray[np.float32_t, ndim=2]
        max_servers=64,            # int
        niter=1024,                # int
        return_groups_count=0,     # int
        ):
    assert reserved_total_resources.ndim == 1
    assert reserved.ndim == 2
    if shares is not None:
        assert reserved.shape == shares.shape

    n = reserved.shape[0]
    ndim = reserved.shape[1]
    assert ndim == reserved_total_resources.shape[0]

    total_resources = np.require(reserved_total_resources, dtype=np.float32, requirements=loader.read_req)

    reserved = np.require(reserved, dtype=np.float32, requirements=loader.read_req)
    if shares is None:
        shares = np.zeros_like(reserved, dtype=np.float32, order='C')
    shares = np.require(shares, dtype=np.float32, requirements=loader.read_req)

    ret_allocated_count = np.zeros(max_servers, dtype=np.float32, order='C')
    ret_allocated_count = np.require(ret_allocated_count, dtype=np.float32, requirements=loader.write_req)

    ret_allocated_resources = np.zeros((max_servers, ndim * 2), dtype=np.float32, order='C')
    ret_allocated_resources = np.require(ret_allocated_resources, dtype=np.float32, requirements=loader.write_req)

    ret_active = np.zeros(max_servers, dtype=np.uint32, order='C')
    ret_active = np.require(ret_active, dtype=np.uint32, requirements=loader.write_req)

    ret_groups = np.zeros((return_groups_count, n), dtype=np.int32, order='C')
    ret_groups = np.require(ret_groups, dtype=np.int32, requirements=loader.write_req)

    seed = int.from_bytes(os.urandom(4), byteorder="big")

    lib = loader.load_lib()
    lib.allocate_players_to_servers(seed, ndim, n, max_servers, niter, return_groups_count,
                                    total_resources, reserved, shares,
                                    ret_allocated_count, ret_allocated_resources, ret_active, ret_groups)
    return ret_allocated_count, ret_allocated_resources, ret_active.astype(np.float) / niter, ret_groups
