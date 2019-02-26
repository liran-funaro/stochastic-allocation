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
import subprocess
import numpy as np
import os
import sys
import ctypes


"""
Read/write requirements from the numpy array.
 - ‘C_CONTIGUOUS’ (‘C’) - ensure a C-contiguous array
 - ‘ALIGNED’ (‘A’)      - ensure a data-type aligned array
 - ‘OWNDATA’ (‘O’)      - ensure an array that owns its own data
 - ‘WRITEABLE’ (‘W’)    - ensure a writable array

See numpy.require documentation for more information.
"""
read_req  = ('C', 'A', 'O')
write_req = (*read_req, 'W')


module_name = "stochalloclib.so"
debug_module_name = "stochalloclib_debug.so"
subpath_options = "stochalloclib/bin", "stochalloclib", "bin", "."


__lib__ = []
__force_compile__ = False


def force_compile(force=True):
    global __force_compile__
    __force_compile__ = force


def locate_lib_path(fname):
    """ Locate a file in the optional sub-folders"""
    curpath = os.path.dirname(os.path.abspath(__file__))

    while curpath != '/':
        for subpath in subpath_options:
            file_path = os.path.join(curpath, subpath, fname)
            if os.path.isfile(file_path):
                return file_path
        curpath = os.path.normpath(os.path.join(curpath, '..'))

    return None


def locate_dll(debug=False):
    """ Locate the module's DLL file """
    return locate_lib_path(module_name if not debug else debug_module_name)


def make_lib(debug=False):
    """ Compile the module to specific parameters """
    make_file_path = locate_lib_path("makefile")
    cwd = os.path.dirname(make_file_path)

    print("Building lib module (%s)" % ('debug' if debug else 'release'), file=sys.stderr)
    ret = subprocess.run("make" if not debug else "make debug",
                         shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.stderr:
        err = str(ret.stderr, 'utf-8')
        print(err, file=sys.stderr)
        raise RuntimeError("Could not compile module: %s" % err)


def load_lib(debug=False):
    """ Loads and initialize a module library, or use already loaded module """
    if len(__lib__) > 0:
        return __lib__[0]

    if __force_compile__:
        make_lib(debug)

    dll_path = locate_dll(debug)
    if dll_path is None:
        make_lib(debug)
        dll_path = locate_dll(debug)
        if dll_path is None:
            raise RuntimeError("No module was compiled.")
    lib = ctypes.cdll.LoadLibrary(dll_path)

    # Init types
    ndarray_1d_f = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags=read_req)
    ndarray_1d_ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags=read_req)
    ndarray_2d_i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags=read_req)
    ndarray_2d_f = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=read_req)
    ndarray_2d_f_w = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=write_req)

    # Init functions
    lib.cfs.argtypes = (
        ctypes.c_uint32, ctypes.c_uint32,  # ndim, n
        ndarray_1d_f, ndarray_1d_f,        # total_resources, resources_epoch
        ndarray_2d_f, ndarray_2d_f, ndarray_2d_f, ndarray_2d_f,  # required_resources, reserved, shares, limit
        ndarray_2d_f, ndarray_1d_ui,       # alloc_step_funcs, alloc_step_len
        ndarray_2d_f_w,                    # ret_alloc
    )

    lib.allocate_players_to_servers.argtypes = (
        *[ctypes.c_uint32] * 6,  # seed, ndim, n, max_servers, niter, return_groups_count
        ndarray_1d_f, ndarray_2d_f, ndarray_2d_f,  # total_resources, reserved, shares
        ndarray_1d_f, ndarray_2d_f, ndarray_1d_ui, ndarray_2d_i,  # return: allocated_count, allocated_resources,
                                                                  #         active_count, groups
    )

    __lib__.append(lib)
    return lib
