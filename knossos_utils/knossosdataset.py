# coding=utf-8
################################################################################
#  This file provides a class representation of a KNOSSOS-dataset for reading
#  and writing raw and overlay data.
#
#  (C) Copyright 2015 - now
#  Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V.
#
#  knossosdataset.py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 2 of
#  the License as published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  For further information feel free to contact
#  Sven.Dorkenwald@mpimf-heidelberg.mpg.de
#
#
################################################################################

################################################################################
#
# IMPORTANT NOTE to avoid confusions:
# KNOSSOS uses a 1-based coordinate system, but all functions in this file are
# 0-based. One should take this into account when reading coordinates from
# KNOSSOS for writing or reading data.
#
################################################################################

from __future__ import absolute_import, division, print_function
# builtins is either provided by Python 3 or by the "future" module for Python 2
# (http://python-future.org/)
from builtins import range, map, zip, filter, round, next, input, bytes, hex, \
    oct, chr, int
from functools import reduce

try:
    import cPickle as pickle
except ImportError:
    import pickle
import glob
import h5py
from io import BytesIO
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
try:
    from knossos_utils import mergelist_tools
except ImportError as e:
    print('mergelist_tools not available, using slow python fallback. '
          'Try to build the cython version of it.\n' + str(e))
    from knossos_utils import mergelist_tools_fallback as mergelist_tools
import numpy as np
import re
import scipy.misc
import scipy.ndimage
import shutil
import sys
import time
import requests
import os
import zipfile
import collections
from threading import Lock
import traceback
from PIL import Image

module_wide = {"init": False, "noprint": False, "snappy": None, "fadvise": None}


def our_glob(s):
    l = []
    for g in glob.glob(s):
        l.append(g.replace(os.path.sep, "/"))
    return l


def _print(s):
    global module_wide
    if not module_wide["noprint"]:
        print(s)
    return


def _set_noprint(noprint):
    global module_wide
    module_wide["noprint"] = noprint
    return


def _stdout(s):
    global module_wide
    if not module_wide["noprint"]:
        sys.stdout.write(s)
        sys.stdout.flush()
    return


def _as_shapearray(x, dim=3):
    """ Creates a np.ndarray that represents a shape.

    This is used to enable different forms of passing cube_shape parameters.
    For example, all of the following expressions are equal:
        np.array([128, 128, 128])
        _as_shapearray(np.array([128, 128, 128]))
        _as_shapearray([128, 128, 128])
        _as_shapearray((128, 128, 128))
        _as_shapearray(128)

    :param x: int or iterable
        If this is a number, the result is an array repeating it `dim` times.
        If this is an iterable, the result is a corresponding np.ndarray.
    :param dim: int
        Number of elements that the shape array should have.
    :return: np.ndarray
        Shape array
    """
    try:
        array = np.fromiter(x, dtype=np.int, count=dim)
    except TypeError:
        array = np.full(dim, x, dtype=np.int)
    return array


def moduleInit():
    global module_wide
    if module_wide["init"]:
        return
    module_wide["init"] = True
    try:
        import snappy
        module_wide["snappy"] = snappy
        assert hasattr(module_wide["snappy"], "decompress"), \
            "Snappy does not contain method 'decompress'. You probably have " \
            "to install 'python-snappy', instead of 'snappy'."
    except ImportError:
        _print("snappy is not available - you won't be able to write/read "
               "overlaycubes and k.zips. Reference for snappy: "
               "https://pypi.python.org/pypi/python-snappy/")
    try:
        import fadvise
        module_wide["fadvise"] = fadvise
    except ImportError:
        pass
    return


def get_first_block(dim, offset, cube_shape):
    """ Helper for iterating over cubes """
    cube_shape = _as_shapearray(cube_shape)
    return int(np.floor(offset[dim] / cube_shape[dim]))


def get_last_block(dim, size, offset, cube_shape):
    """ Helper for iterating over cubes """
    cube_shape = _as_shapearray(cube_shape)
    return int(np.floor((offset[dim]+size[dim]-1) / cube_shape[dim]))


def cut_matrix(data, offset_start, offset_end, cube_shape, start, end):
    """ Helper for cutting matrices extracted from cubes to a required size """
    cube_shape = _as_shapearray(cube_shape)

    cut_start = np.array(offset_start, dtype=np.int)
    number_cubes = np.array(end) - np.array(start)
    cut_end = np.array(number_cubes * cube_shape - offset_end, dtype=np.int)

    return data[cut_start[0]: cut_end[0],
                cut_start[1]: cut_end[1],
                cut_start[2]: cut_end[2]]


def load_from_h5py(path, hdf5_names, as_dict=False):
    """ Helper for loading h5-files

    :param path: str
        forward-slash separated path to h5-file
    :param hdf5_names: list of str
        names of sets that should be loaded
    :param as_dict: bool
        True: returns contained sets in dict (keys from hdf5_names)
        False: returns contained sets as list (order from hdf5_names)
    :return:
        dict or list, see as_dict
    """
    if as_dict:
        data = {}
    else:
        data = []
    try:
        f = h5py.File(path, 'r')
        for hdf5_name in hdf5_names:
            if as_dict:
                data[hdf5_name] = f[hdf5_name].value
            else:
                data.append(f[hdf5_name].value)
    except:
        raise Exception("Error at Path: %s, with labels:" % path, hdf5_names)
    f.close()
    return data


def save_to_h5py(data, path, hdf5_names=None, overwrite=False, compression=True):
    """
    Saves data to h5py File.

    Parameters
    ----------
    data: list or dict of np.arrays
        if list, hdf5_names has to be set.
    path: str
        forward-slash separated path to file
    hdf5_names: list of str
        has to be the same length as data
    overwrite : bool
        determines whether existing files are overwritten
    compression : bool
        True: compression='gzip' is used which is recommended for sparse and
        ordered data

    Returns
    -------
    nothing

    """
    if (not type(data) is dict) and hdf5_names is None:
        raise Exception("hdf5names has to be set, when data is a list")
    if os.path.isfile(path) and overwrite:
        os.remove(path)
    f = h5py.File(path, "w")
    if type(data) is dict:
        for key in data.keys():
            if compression:
                f.create_dataset(key, data=data[key], compression="gzip")
            else:
                f.create_dataset(key, data=data[key])
    else:
        if len(hdf5_names) != len(data):
            f.close()
            raise Exception("Not enough or to much hdf5-names given!")
        for nb_data in range(len(data)):
            if compression:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data],
                                 compression="gzip")
            else:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data])
    f.close()


def save_to_pickle(data, filename):
    """ Helper for saving pickle-file """
    f = open(filename, 'wb')
    pickle.dump(data, f, -1)
    f.close()


def load_from_pickle(filename):
    """ Helper for loading pickle-file """
    return pickle.load(open(filename))


def _find_and_delete_cubes_process(args):
    """ Function which is called by an multiprocessing call
        from delete_all_overlaycubes"""
    if args[1]:
        _print(args[0])
    all_files = our_glob(args[0])
    for f in all_files:
        os.remove(f)


class KnossosDataset(object):
    """ Class that contains information and operations for a Knossos-Dataset
    """
    def __init__(self):
        moduleInit()
        global module_wide
        self.module_wide = module_wide
        self._knossos_path = None
        self._conf_path = None
        self._http_url = None
        self._http_user = None
        self._http_passwd = None
        self._experiment_name = None
        self._mag = []
        self._name_mag_folder = None
        self._boundary = np.zeros(3, dtype=np.int)
        self._scale = np.ones(3, dtype=np.float)
        self._number_of_cubes = np.zeros(3)
        self._cube_shape = np.full(3, 128, dtype=np.int)
        self._initialized = False

    @property
    def mag(self):
        return self._mag

    @property
    def name_mag_folder(self):
        return self._name_mag_folder

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def boundary(self):
        return self._boundary

    @property
    def scale(self):
        return self._scale

    @property
    def knossos_path(self):
        if self.in_http_mode:
            return self.http_url
        elif self._knossos_path:
            return self._knossos_path
        else:
            raise Exception("No knossos path available")

    @property
    def conf_path(self):
        return self._conf_path

    @property
    def number_of_cubes(self):
        return self._number_of_cubes

    @property
    def cube_shape(self):
        return self._cube_shape

    @property
    def initialized(self):
        return self._initialized

    @property
    def http_url(self):
        return self._http_url

    @property
    def http_user(self):
        return self._http_user

    @property
    def http_passwd(self):
        return self._http_passwd

    @property
    def in_http_mode(self):
        if self.http_url and self.http_user and self.http_passwd:
            return True
        else:
            return False

    @property
    def http_auth(self):
        if self.in_http_mode:
            return self.http_user, self.http_passwd
        else:
            return None

    def get_first_blocks(self, offset):
        return offset // self.cube_shape

    def get_last_blocks(self, offset, size):
        return ((offset+size-1) // self.cube_shape) + 1

    def _initialize_cache(self, cache_size):
        """ Initializes the internal RAM cache for repeated look-ups.
        max_size: Maximum number of cubes to hold before replacing existing cubes.

        :param max_size: int
            path to knossos.conf

        :return:
            nothing
        """

        self._cache_mutex = Lock()

        self._cube_cache = collections.OrderedDict()
        self._cube_cache_size = cache_size

    def _add_to_cube_cache(self, c, mode, values):
        if not self._cube_cache_size:
            return

        self._cache_mutex.acquire()
        if len(self._cube_cache) >= self._cube_cache_size:
            # remove the oldest (i.e. first inserted) cache element
            self._cube_cache.popitem(last=False)

        self._cube_cache[str(c) + str(mode)] = values
        self._cache_mutex.release()

        return

    def _test_all_cache_satisfied(self, coordinates, mode):
        """
        Tests whether all supplied cube coordinates can be
        provided from the cache.

        :param coordinates: iterable
            cube coordinate iterable
        :return: bool
            Whether all cubes are currently in the cache
        """
        return all([self._cube_cache.has_key(str(c) + str(mode)) for c in coordinates])

    def _cube_from_cache(self, c, mode):

        self._cache_mutex.acquire()

        try:
            values = self._cube_cache[str(c) + str(mode)]
            if np.sum(values) == 0:
                raise KeyError
        except KeyError:
            values = None

        self._cache_mutex.release()
        return values


    def parse_knossos_conf(self, path_to_knossos_conf, verbose=False):
        """ Parse a knossos.conf

        :param path_to_knossos_conf: str
            path to knossos.conf
        :param verbose: bool
            several information is printed when set to True
        :return:
            nothing
        """
        try:
            f = open(path_to_knossos_conf)
            lines = f.readlines()
            f.close()
        except:
            raise NotImplementedError("Could not find/read *mag1/knossos.conf")

        self._conf_path = path_to_knossos_conf

        parsed_dict = {}
        for line in lines:
            if line.startswith("ftp_mode"):
                line_s = line.split(" ")
                self._http_url = "http://" + line_s[1] + line_s[2] + "/"
                self._http_user = line_s[3]
                self._http_passwd = line_s[4]
            else:
                try:
                    match = re.search(r'(?P<key>[A-Za-z _]+)'
                                      r'((?P<numeric_value>[0-9\.]+)'
                                      r'|"(?P<string_value>[A-Za-z0-9._/-]+)");',
                                      line).groupdict()

                    if match['string_value']:
                        val = match['string_value']
                    elif '.' in match['numeric_value']:
                        val = float(match['numeric_value'])
                    elif match['numeric_value']:
                        val = int(match['numeric_value'])
                    else:
                        raise Exception('Malformed knossos.conf')

                    parsed_dict[match["key"]] = val
                except:
                    if verbose:
                        _print("Unreadable line in knossos.conf - ignored.")

        self._boundary[0] = parsed_dict['boundary x ']
        self._boundary[1] = parsed_dict['boundary y ']
        self._boundary[2] = parsed_dict['boundary z ']
        self._scale[0] = parsed_dict['scale x ']
        self._scale[1] = parsed_dict['scale y ']
        self._scale[2] = parsed_dict['scale z ']
        self._experiment_name = parsed_dict['experiment name ']
        if self._experiment_name.endswith("mag1"):
            self._experiment_name = self._experiment_name[:-5]

        self._number_of_cubes = \
            np.array(np.ceil(self.boundary.astype(np.float) /
                             self.cube_shape), dtype=np.int)

    def initialize_from_knossos_path(self, path, fixed_mag=None, http_max_tries=10,
                                     use_abs_path=False, verbose=False, cache_size=0):
        """ Initializes the dataset by parsing the knossos.conf in path + "mag1"

        :param path: str
            forward-slash separated path
        :param fixed_mag: int
            fixes available mag to one specific value
        :param verbose: bool
            several information is printed when set to True
        :param use_abs_path: bool
            the absolut path to the knossos dataset will be used
        :return:
            nothing
        """
        while path.endswith("/"):
            path = path[:-1]

        if not os.path.exists(path):
            raise Exception("No directory or file found")

        if os.path.isfile(path):
            self.parse_knossos_conf(path, verbose=verbose)
            if self.in_http_mode:
                self._name_mag_folder = "mag"

                if fixed_mag:
                    if isinstance(fixed_mag, int):
                        self._mag.append(fixed_mag)
                    else:
                        raise Exception("Fixed mag must be integer.")
                else:
                    for mag_test_nb in range(10):
                        mag_folder = self.http_url + \
                                     self.name_mag_folder + str(2**mag_test_nb)

                        tries = 0
                        while tries < http_max_tries:
                            try:
                                request = requests.get(mag_folder,
                                                       auth=self.http_auth,
                                                       timeout=10)

                                if request.status_code == 200:
                                    self._mag.append(2 ** mag_test_nb)
                                    break
                                request.raise_for_status()
                            except:
                                tries += 1
                                if tries >= http_max_tries:
                                    break
                                else:
                                    continue

                        if tries >= http_max_tries:
                            break
            else:
                folder = os.path.basename(os.path.dirname(path))
                match = re.search(r'(?<=mag)[\d]+$', folder)
                if match:
                    self._knossos_path = \
                        os.path.dirname(os.path.dirname(path)) + "/"
                else:
                    raise Exception("Corrupt folder structure - knossos.conf"
                                    "is not inside a mag folder")
        else:
            match = re.search(r'(?<=mag)[\d]+$', path)
            if match:
                self._knossos_path = os.path.dirname(path) + "/"
            else:
                self._knossos_path = path + "/"

        if not self.in_http_mode:
            all_mag_folders = our_glob(self._knossos_path+"/*mag*")

            if fixed_mag:
                if isinstance(fixed_mag, int):
                    self._mag.append(fixed_mag)
                else:
                    raise Exception("Fixed mag must be integer.")
            else:
                for mag_test_nb in range(10):
                    for mag_folder in all_mag_folders:
                        if "mag"+str(2**mag_test_nb) in mag_folder:
                            self._mag.append(2**mag_test_nb)
                            break

            if len(all_mag_folders) == 0:
                raise Exception("No valid mag folders found")

            mag_folder = all_mag_folders[0].split("/")
            if len(mag_folder[-1]) > 1:
                mag_folder = mag_folder[-1]
            else:
                mag_folder = mag_folder[-2]

            self._name_mag_folder = \
                mag_folder[:-len(re.findall("[\d]+", mag_folder)[-1])]

            self.parse_knossos_conf(self.knossos_path +
                                    self.name_mag_folder +
                                    "%d/knossos.conf" % self.mag[0],
                                    verbose=verbose)

        if use_abs_path:
            self._knossos_path = os.path.abspath(self.knossos_path)

        self._initialize_cache(cache_size)

        if verbose:
            _print("Initialization finished successfully")
        self._initialized = True

    def initialize_without_conf(self, path, boundary, scale, experiment_name,
                                mags=None, make_mag_folders=True,
                                create_knossos_conf=True, verbose=False, cache_size=0):
        """ Initializes the dataset without a knossos.conf

            This function creates mag folders and knossos.conf's if requested.
            Hence it can be used to create a new dataset from scratch.

        :param path: str
            forward-slash separated path to the datasetfolder - not .../mag !
        :param boundary: 3 sequence of ints
            boundaries of the knossos dataset
        :param scale: 3 sequence of floats
            scaling between original data and knossos data
        :param experiment_name: str
            name of the experiment
        :param mags: sequence of ints
            available magnifications of the knossos dataset
        :param make_mag_folders: bool
            True: makes not-existing mag directories if not
        :param create_knossos_conf: bool
            True: creates not-existing knoosos.conf files
        :param verbose:
            True: prints several information
        :return:
            nothing
        """

        self._knossos_path = path
        all_mag_folders = our_glob(path+"*mag*")

        if not mags is None:
            self._mag = mags
            if make_mag_folders:
                for mag in mags:
                    exists = False
                    for mag_folder in all_mag_folders:
                        if "mag"+str(mag) in mag_folder:
                            exists = True
                            break
                    if not exists:
                        if len(all_mag_folders) > 0:
                            os.makedirs(path+"/"+ re.findall('[a-zA-Z0-9,_ -]+',
                                        all_mag_folders[0][:-1])[-1] + str(mag))
                        else:
                            os.makedirs(path+"/mag"+str(mag))
        else:
            for mag_test_nb in range(32):
                for mag_folder in all_mag_folders:
                    if "mag"+str(2**mag_test_nb) in mag_folder:
                        self._mag.append(2**mag_test_nb)
                        break

        mag_folder = our_glob(path+"*mag*")[0].split("/")
        if len(mag_folder[-1]) > 1:
            mag_folder = mag_folder[-1]
        else:
            mag_folder = mag_folder[-2]

        self._name_mag_folder = \
            mag_folder[:-len(re.findall("[\d]+", mag_folder)[-1])]

        self._scale = scale
        self._boundary = boundary
        self._experiment_name = experiment_name

        self._number_of_cubes = np.array(np.ceil(
            self.boundary.astype(np.float) / self.cube_shape), dtype=np.int)

        if create_knossos_conf:
            all_mag_folders = our_glob(path+"*mag*")
            for mag_folder in all_mag_folders:
                this_mag = re.findall("[\d]+", mag_folder)[-1]
                with open(mag_folder+"/knossos.conf", "w") as f:
                    f.write('experiment name "%s_mag%s";\n' %(experiment_name,
                                                              this_mag))
                    f.write('boundary x %d;\n' % boundary[0])
                    f.write('boundary y %d;\n' % boundary[1])
                    f.write('boundary z %d;\n' % boundary[2])
                    f.write('scale x %.2f;\n' % scale[0])
                    f.write('scale y %.2f;\n' % scale[1])
                    f.write('scale z %.2f;\n' % scale[2])
                    f.write('magnification %s;' % this_mag)
        if verbose:
            _print("Initialization finished successfully")

        self._initialize_cache(cache_size)

        self._initialized = True

    def initialize_from_matrix(self, path, scale, experiment_name,
                               offset=None, boundary=None, fast_downsampling=True,
                               data=None, data_path=None, hdf5_names=None,
                               mags=None, verbose=False, cache_size=0):
        """ Initializes the dataset with matrix
            Only for use with "small" matrices (~10^3 edgelength)

            This function creates mag folders and knossos.conf's.

        :param path: str
            forward-slash separated path to the datasetfolder - not .../mag !
        :param scale: 3 sequence of floats
            scaling between original data and knossos data
        :param experiment_name: str
            name of the experiment
        :param offset: 3 sequence of ints or None
            offset of the given data
            if None offset is set to [0, 0, 0]
        :param boundary: 3 sequence of ints or None
            boundary of the knossos dataset
            if None boundary is calculated from offset and data
        :param fast_downsampling: bool
            True: uses order 1 downsampling(striding)
            False: uses order 3 downsampling
        :param data: 3D numpy array or list of 3D numpy arrays of ints
            exported data
            if list: data is combined to a single array by np.maximum()
        :param data_path: str
            path for loading data (hdf5 and pickle files are supported)
        :param hdf5_names: str or list of str
            hdf5 setnames in data_path
        :param mags: sequence of ints
            available magnifications of the knossos dataset
        :param verbose:
            True: prints several information
        :return:
            nothing
        """

        if (data is None) and (data_path is None or hdf5_names is None):
            raise Exception("No data given")

        if data is None:
            data = load_from_h5py(data_path, hdf5_names, False)[0]

        if offset is None:
            offset = np.array([0, 0, 0], dtype=np.int)
        else:
            offset = np.array(offset, dtype=np.int)

        if boundary is None:
            boundary = np.array(data.shape) + offset
        else:
            if np.any(boundary < np.array(data.shape) + offset):
                raise Exception("Given size is too small for data")

        if mags is None:
            mags = [1]

        self._initialize_cache(cache_size)

        self.initialize_without_conf(path, boundary, scale, experiment_name,
                                     mags=mags, make_mag_folders=True,
                                     create_knossos_conf=True, verbose=verbose)

        self.from_matrix_to_cubes(offset, mags=mags, data=data,
                                  datatype=np.uint8,
                                  fast_downsampling=fast_downsampling,
                                  as_raw=True)

    def copy_dataset(self, path, data_range=None, do_raw=True, mags=None,
                     stride=256, return_errors=False, nb_threads=20,
                     verbose=True, apply_func=None):
        """ Copies a dataset to another dataset - especially useful for
            downloading remote datasets

        :param path: str
            path to new knossosdataset (will be created)
        :param data_range: list of list
            specifies subvolume: [[x, y, z], [x, y, z]]
            None: whole dataset will be copied
        :param do_raw: boolean
            True: raw data will be copied
            False: overlaycubes will be copied
            do not do both at once in different processes!
        :param mags: list of int or int
            mags from which data should be copied (automatically 1 for
            overlaycubes). Default: all available mags
        :param stride: int
            stride for copying
        :param nb_threads: int
            number of threads to be used (recommended: 2 * number of cpus)
        :param apply_func: function
            function which will be applied to raw data before writing to new
            dataset folder
        """
        if apply_func is not None:
            assert callable(apply_func)

        def _copy_block_thread(args):
            mag, size, offset, do_raw = args
            if do_raw:
                raw = self.from_raw_cubes_to_matrix(size, offset, mag=mag,
                                                    http_verbose=True,
                                                    nb_threads=1,
                                                    show_progress=False,
                                                    verbose=verbose)

                if isinstance(raw, tuple):
                    err = raw[1]
                    raw = raw[0]
                else:
                    err = None
                if apply_func is not None:
                    raw = apply_func(raw)
                new_kd.from_matrix_to_cubes(offset=offset, mags=mag,
                                            data=raw, datatype=np.uint8,
                                            as_raw=True, nb_threads=1,
                                            verbose=verbose)

                return err
            else:
                overlay = self.from_overlaycubes_to_matrix(size, offset,
                                                           mag=mag,
                                                           http_verbose=True,
                                                           nb_threads=1,
                                                           show_progress=False)

                if isinstance(overlay, tuple):
                    err = overlay[1]
                    overlay = overlay[0]
                else:
                    err = None
                if apply_func is not None:
                    overlay = apply_func(overlay)
                new_kd.from_matrix_to_cubes(offset=offset, mags=mag,
                                            data=overlay, datatype=np.uint64,
                                            nb_threads=1)
                return err

        if data_range:
            assert isinstance(data_range, list)
            assert len(data_range[0]) == 3
            assert len(data_range[1]) == 3
        else:
            data_range = [[0, 0, 0], self.boundary]

        if mags is None:
            mags = self.mag

        if isinstance(mags, int):
            mags = [mags]

        new_kd = KnossosDataset()
        new_kd.initialize_without_conf(path=path, boundary=self.boundary,
                                       scale=self.scale,
                                       experiment_name=self.experiment_name,
                                       mags=self.mag)

        multi_params = []
        if do_raw:
            for mag in mags:
                for x in range(data_range[0][0],
                               data_range[1][0] / mag, stride):
                    for y in range(data_range[0][1],
                                   data_range[1][1] / mag, stride):
                        for z in range(data_range[0][2],
                                       data_range[1][2] / mag, stride):
                            multi_params.append([mag, [stride]*3, [x, y, z],
                                                 True])
        else:
            for x in range(data_range[0][0],
                           data_range[1][0], stride):
                for y in range(data_range[0][1],
                               data_range[1][1], stride):
                    for z in range(data_range[0][2],
                                   data_range[1][2], stride):
                        multi_params.append([1, [stride]*3, [x, y, z],
                                             False])

        if nb_threads > 1:
            pool = ThreadPool(nb_threads)
            results = pool.map(_copy_block_thread, multi_params)
            pool.close()
            pool.join()
        else:
            results = map(_copy_block_thread, multi_params)

        errors = {}
        for result in results:
            if result:
                for errno in result:
                    if errno in errors:
                        errors[errno] += result[errno]
                    else:
                        errors[errno] = result[errno]
        if errors:
            _print("Errors appeared! Keep in mind that Error 404 might be "
                   "totally fine. Overview:")
            for errno in errors:
                _print("%d: %dx" % (errno, errors[errno]))
        if return_errors:
            return errors

    def from_cubes_to_list(self, vx_list, raw=True, datatype=np.uint32):
        """ Read voxel values vectorized
        WARNING: voxels have to be clustered, otherwise: RAM & runtime -> inf

        :param vx_list:  list or array of 3 sequence of int
            list of voxels which values should be returned
        :param raw: bool
            True: read from raw cubes
            False: read from overlaycubes
        :param datatype: np.dtype
            defines np.dtype, only relevant for overlaycubes (raw=False)
        :return: array of int
            array of voxel values corresponding to vx_list
        """
        vx_list = np.array(vx_list, dtype=np.int)
        boundary_box = [np.min(vx_list, axis=0),
                        np.max(vx_list, axis=0)]
        size = boundary_box[1] - boundary_box[0] + np.array([1, 1, 1])

        if raw:
            block = self.from_raw_cubes_to_matrix(size, boundary_box[0],
                                                  show_progress=False,
                                                  mirror_oob=True)
        else:
            block = self.from_overlaycubes_to_matrix(size, boundary_box[0],
                                                     datatype=datatype,
                                                     show_progress=False,
                                                     mirror_oob=True)

        vx_list -= boundary_box[0]

        return block[vx_list[:, 0], vx_list[:, 1], vx_list[:, 2]]

    def from_raw_cubes_to_list(self, vx_list):
        """ Read voxel values vectorized
        WARNING: voxels have to be clustered, otherwise: RAM & runtime -> inf

        :param vx_list:  list or array of 3 sequence of int
            list of voxels which values should be returned
        :return: array of int
            array of voxel values corresponding to vx_list
        """

        return self.from_cubes_to_list(vx_list, raw=True, datatype=np.uint8)

    def from_overlaycubes_to_list(self, vx_list, datatype=np.uint32):
        """ Read voxel values vectorized
        WARNING: voxels have to be clustered, otherwise: RAM & runtime -> inf

        :param vx_list:  list or array of 3 sequence of int
            list of voxels which values should be returned
        :param datatype: np.dtype
            defines np.dtype
        :return: array of int
            array of voxel values corresponding to vx_list
        """

        return self.from_cubes_to_list(vx_list, raw=False, datatype=datatype)

    def from_cubes_to_matrix(self, size, offset, mode, mag=1, datatype=np.uint8,
                             mirror_oob=True, hdf5_path=None,
                             hdf5_name="raw", pickle_path=None,
                             invert_data=False, zyx_mode=False,
                             nb_threads=40, verbose=True, show_progress=True,
                             http_max_tries=2000, http_verbose=False):
        """ Extracts a 3D matrix from the KNOSSOS-dataset
            NOTE: You should use one of the two wrappers below

        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param mode: str
            either 'raw' or 'overlay'
        :param mag: int
            magnification of the requested data block
        :param datatype: numpy datatype
            typically:  'raw' = np.uint8
                        'overlay' = np.uint64
        :param mirror_oob: bool
            pads the raw data with mirrored data if given box is out of bounce
        :param hdf5_path: str
            if given the output is written as hdf5 file
        :param hdf5_name: str
            name of hdf5-set
        :param pickle_path: str
            if given the output is written as (c)Pickle file
        :param invert_data: bool
            True: inverts the output
        :param zyx_mode: bool
            activates zyx-order, size and offset have to in zyx if activated
        :param nb_threads: int
            number of threads - twice the number of cores is recommended
        :param verbose: bool
            True: prints several information
        :param show_progress: bool
            True: progress is printed to the terminal
        :return: 3D numpy array or nothing
            if a path is given no data is returned
        """

        def _read_cube(c):
            pos = np.subtract([c[0], c[1], c[2]], start) * self.cube_shape
            valid_values = False

            # check cache first
            values = self._cube_from_cache(c, mode)

            if values is not None:
                #print('Cache hit')
                if zyx_mode:
                    output[pos[2]: pos[2] + self.cube_shape[2],
                           pos[1]: pos[1] + self.cube_shape[1],
                           pos[0]: pos[0] + self.cube_shape[0]] = \
                        values
                else:
                    output[pos[0]: pos[0] + self.cube_shape[0],
                           pos[1]: pos[1] + self.cube_shape[1],
                           pos[2]: pos[2] + self.cube_shape[2]] = \
                        values
            else:
                if from_raw:
                    path = self.knossos_path + \
                           self.name_mag_folder + \
                           "%d/x%04d/y%04d/z%04d/" % (mag, c[0], c[1], c[2]) + \
                           self.experiment_name + \
                           "_mag%d_x%04d_y%04d_z%04d.raw" % (mag, c[0], c[1], c[2])

                    if self.in_http_mode:
                        tries = 0
                        while True:
                            try:
                                request = requests.get(path,
                                                       auth=self.http_auth,
                                                       timeout=60)
                                request.raise_for_status()
                                values = np.fromstring(request.content,
                                                       dtype=datatype)
                                if values.sum() == 0:
                                    pass
                                    # if http_verbose:
                                    #     _print("Zero value array encountered("
                                    #           "%d/%d) [%s]\n" % (1+tries, http_max_tries, path))
                                else:
                                    try:
                                        values.reshape(self.cube_shape)
                                        valid_values = True
                                        break
                                    except ValueError:
                                        # if verbose:
                                        #     _print("Reshape error("
                                        #       "%d/%d) [%s]\n" % (1+tries, http_max_tries, path))
                                        pass
                            except requests.exceptions.Timeout as e:
                                if http_verbose:
                                    _print(e)
                            except requests.exceptions.TooManyRedirects as e:
                                if http_verbose:
                                    _print(e)
                            except requests.exceptions.RequestException as e:
                                if http_verbose:
                                    _print(e)
                            except requests.exceptions.ConnectionError as e:
                                if http_verbose:
                                    _print(e)
                            except requests.exceptions.HTTPError as e:
                                if http_verbose:
                                    _print(e)
                            tries += 1
                            if tries >= http_max_tries:
                                _print("Max. #tries reached.")
                                return "Max-try error"
                            _print("[%s] Error occured (%d/%d)\n" %
                                   (path, tries, http_max_tries))
                            time.sleep(1)
                    else:
                        try:
                            flat_shape = int(np.prod(self.cube_shape))
                            values = np.fromfile(path, dtype=np.uint8,
                                                 count=flat_shape)
                            valid_values = True
                        except:
                            if verbose:
                                _print("Cube does not exist, cube with zeros "
                                       "only assigned")
                else:
                    path = self.knossos_path + \
                           self.name_mag_folder + \
                           "%d/x%04d/y%04d/z%04d/" % (mag, c[0], c[1], c[2]) + \
                           self.experiment_name + \
                           "_mag%d_x%04d_y%04d_z%04d.seg.sz" % \
                           (mag, c[0], c[1], c[2])

                    if self.in_http_mode:
                        tries = 0
                        while tries < http_max_tries:
                            try:
                                request = requests.get(path + ".zip",
                                                       auth=self.http_auth,
                                                       timeout=60)
                                request.raise_for_status()
                                with zipfile.ZipFile(BytesIO(
                                        request.content), "r") \
                                        as zf:
                                    values = np.fromstring(
                                        self.module_wide["snappy"].decompress(
                                            zf.read(os.path.basename(path))),
                                        dtype=datatype)
                                # check if requested values match shape
                                try:
                                    values.reshape(self.cube_shape)
                                except ValueError:
                                    if verbose:
                                        _print("\nReshape error encountered for"
                                               " %d time. (%s)\n" %
                                               (1 + tries, path))
                                    tries += 1
                                    time.sleep(0.1)
                                    if tries == http_max_tries:
                                        if verbose:
                                            _print("Reshape error.")
                                        return "Reshape error"
                                    else:
                                        continue
                                valid_values = True
                            except requests.exceptions.Timeout as e:
                                return e
                            except requests.exceptions.TooManyRedirects as e:
                                return e
                            except requests.exceptions.RequestException as e:
                                return e
                            except requests.exceptions.ConnectionError as e:
                                tries += 1
                                time.sleep(1)
                                if tries == http_max_tries:
                                    if verbose:
                                        _print("Max. #tries reached.")
                                else:
                                    continue
                            except requests.exceptions.HTTPError as e:
                                return e
                            break
                    else:
                        try:
                            with zipfile.ZipFile(path + ".zip", "r") as zf:
                                values = np.fromstring(
                                    self.module_wide["snappy"].decompress(
                                        zf.read(os.path.basename(path))),
                                    dtype=datatype)
                                valid_values = True
                        except:
                            if verbose:
                                _print("Cube does not exist, cube with zeros "
                                       "only assigned")

                if valid_values:
                    if zyx_mode:
                        values = values.reshape(self.cube_shape)
                        self._add_to_cube_cache(c, mode, values)
                        output[pos[2]: pos[2]+self.cube_shape[2],
                               pos[1]: pos[1]+self.cube_shape[1],
                               pos[0]: pos[0]+self.cube_shape[0]] = \
                            values

                    else:
                        try:
                            values = values.reshape(self.cube_shape).T
                        except Exception as e:
                            # _print('Exception in reshape: values.shape {0}'.
                            #        format(values.shape))
                            # _print('Exception in reshape:self.cube_shape {0}'.
                            #        format(self.cube_shape))
                            # if verbose:
                            _print("Cube is invalid, cube with zeros "
                                   "only assigned")
                            _print(c)
                            _print(e)
                            values = np.zeros(self.cube_shape)

                        self._add_to_cube_cache(c, mode, values)
                        output[pos[0]: pos[0]+self.cube_shape[0],
                               pos[1]: pos[1]+self.cube_shape[1],
                               pos[2]: pos[2]+self.cube_shape[2]] = \
                            values

        t0 = time.time()

        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if mag not in self._mag:
            raise Exception("Magnification not supported")

        if 0 in size:
            raise Exception("The first parameter is size! - "
                            "at least one dimension was set to 0 ...")

        if verbose and show_progress:
            show_progress = False
            _print("when choosing verbose, show_progress is automatically "
                   "disabled")

        if mode == 'raw':
            from_raw = True
        elif mode == 'overlay':
            from_raw = False
        else:
            raise NotImplementedError("mode has to be 'raw' or 'overlay'")

        size = np.array(size, dtype=np.int)
        offset = np.array(offset, dtype=np.int)

        if zyx_mode:
            size = size[::-1]
            offset = offset[::-1]

        mirror_overlap = [[0, 0], [0, 0], [0, 0]]

        for dim in range(3):
            if offset[dim] < 0:
                size[dim] += offset[dim]
                mirror_overlap[dim][0] = - offset[dim]
                offset[dim] = 0

            if offset[dim]+size[dim] > self.boundary[dim]:
                mirror_overlap[dim][1] = offset[dim] + size[dim] - \
                                         self.boundary[dim]
                size[dim] -= offset[dim] + size[dim] - self.boundary[dim]

            if size[dim] < 0:
                raise Exception("Given block is totally out ouf bounds with "
                                "offset: [%d, %d, %d]!" %
                                (offset[0], offset[1], offset[2]))

        start = self.get_first_blocks(offset)
        end = self.get_last_blocks(offset, size)

        uncut_matrix_size = (end - start) * self.cube_shape
        if zyx_mode:
            uncut_matrix_size = uncut_matrix_size[::-1]

        output = np.zeros(uncut_matrix_size, dtype=datatype)

        offset_start = offset % self.cube_shape
        offset_end = (self.cube_shape - (offset + size)
                      % self.cube_shape) % self.cube_shape

        cnt = 0
        nb_cubes_to_process = int(np.prod(end - start))

        cube_coordinates = []

        for z in range(start[2], end[2]):
            for y in range(start[1], end[1]):
                for x in range(start[0], end[0]):
                    cube_coordinates.append([x, y, z])

        if nb_threads > 1:
            if not self._test_all_cache_satisfied(cube_coordinates, mode)\
                    and len(cube_coordinates) > 1:
                pool = ThreadPool(nb_threads)
                results = pool.map(_read_cube, cube_coordinates)
                pool.close()
                pool.join()
            else:
                results = []
                for c in cube_coordinates:
                    results.append(_read_cube(c))
        else:
            results = []
            for c in cube_coordinates:
                results.append(_read_cube(c))

        if (verbose or http_verbose) and self.in_http_mode:
            errors = {}
            for result in results:
                if result:
                    if result.response:
                        errno = result.response.status_code
                        if errno in errors:
                            errors[errno] += 1
                        else:
                            errors[errno] = 1
            if verbose and len(errors) > 0:
                _print("%d errors appeared! Keep in mind that Error 404 might be "
                       "totally fine. Overview:" %len(errors))
                for errno in errors:
                    _print("%d: %dx" % (errno, errors[errno]))

        if zyx_mode:
            output = cut_matrix(output, offset_start[::-1], offset_end[::-1],
                                self.cube_shape[::-1], start[::-1], end[::-1])
        else:
            output = cut_matrix(output, offset_start, offset_end,
                                self.cube_shape, start, end)

        if show_progress:
            progress = 100.0 * cnt / nb_cubes_to_process
            _stdout('\rProgress: %.2f%%' % progress)
            _stdout('\rProgress: finished\n')
            dt = time.time()-t0
            speed = np.product(output.shape) * 1.0/1000000/dt
            if mode == "raw":
                _stdout('\rSpeed: %.3f MB or MPix /s, time %s\n' % (speed, dt))
            else:
                _stdout('\rSpeed: %.3f MPix /s, time %s\n' % (speed, dt))

        ref_size = size[::-1] if zyx_mode else size
        if not np.all(output.shape == ref_size):
            raise Exception("Incorrect shape! Should be", ref_size, "; got:",
                            output.shape)
        else:
            if verbose:
                _print("Shape was verified")

        if np.any(mirror_overlap != 0):
            if not zyx_mode:
                if mirror_oob:
                    output = np.lib.pad(output, mirror_overlap, 'symmetric')
                else:
                    output = np.lib.pad(output, mirror_overlap, 'constant')
            else:
                if mirror_oob:
                    output = np.lib.pad(output, mirror_overlap[::-1], 'symmetric')
                else:
                    output = np.lib.pad(output, mirror_overlap[::-1], 'constant')

        if output.dtype != datatype:
            raise Exception("Wrong datatype! - for unknown reasons...")

        if invert_data:
            output = np.invert(output)

        if hdf5_path and hdf5_name:
            save_to_h5py(output, hdf5_path, hdf5_names=[hdf5_name])

        if pickle_path:
            save_to_pickle(output, pickle_path)

        if http_verbose and self.in_http_mode:
            return output, errors
        else:
            return output

    def from_raw_cubes_to_matrix(self, size, offset, mag=1,
                                 datatype=np.uint8, mirror_oob=False,
                                 hdf5_path=None, hdf5_name="raw",
                                 pickle_path=None, invert_data=False,
                                 zyx_mode=False, nb_threads=40,
                                 verbose=False, http_verbose=False,
                                 http_max_tries=2000, show_progress=True):
        """ Extracts a 3D matrix from the KNOSSOS-dataset raw cubes

        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param mag: int
            magnification of the requested data block
        :param datatype: numpy datatype
            typically np.uint8
        :param mirror_oob: bool
            pads the raw data with mirrored data if given box is out of bounce
        :param hdf5_path: str
            if given the output is written as hdf5 file
        :param hdf5_name: str
            name of hdf5-set
        :param pickle_path: str
            if given the output is written as (c)Pickle file
        :param invert_data: bool
            True: inverts the output
        :param zyx_mode: bool
            activates zyx-order, size and offset have to in zyx if activated
        :param nb_threads: int
            number of threads - twice the number of cores is recommended
        :param verbose: bool
            True: prints several information
        :param show_progress: bool
            True: progress is printed to the terminal
        :return: 3D numpy array or nothing
            if a path is given no data is returned
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        return self.from_cubes_to_matrix(size, offset,
                                         mode='raw',
                                         mag=mag,
                                         datatype=datatype,
                                         mirror_oob=mirror_oob,
                                         hdf5_path=hdf5_path,
                                         hdf5_name=hdf5_name,
                                         pickle_path=pickle_path,
                                         invert_data=invert_data,
                                         zyx_mode=zyx_mode,
                                         nb_threads=nb_threads,
                                         verbose=verbose,
                                         http_max_tries=http_max_tries,
                                         http_verbose=http_verbose,
                                         show_progress=show_progress)

    def from_overlaycubes_to_matrix(self, size, offset, mag=1,
                                    datatype=np.uint64, mirror_oob=False,
                                    hdf5_path=None, hdf5_name="raw",
                                    pickle_path=None, invert_data=False,
                                    zyx_mode=False, nb_threads=40,
                                    verbose=False, http_verbose=False,
                                    show_progress=True):
        """ Extracts a 3D matrix from the KNOSSOS-dataset overlay cubes

        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param mag: int
            magnification of the requested data block
        :param datatype: numpy datatype
            typically np.uint64
        :param mirror_oob: bool
            pads the raw data with mirrored data if given box is out of bounce
        :param hdf5_path: str
            if given the output is written as hdf5 file
        :param hdf5_name: str
            name of hdf5-set
        :param pickle_path: str
            if given the output is written as (c)Pickle file
        :param invert_data: bool
            True: inverts the output
        :param zyx_mode: bool
            activates zyx-order, size and offset have to in zyx if activated
        :param nb_threads: int
            number of threads - twice the number of cores is recommended
        :param verbose: bool
            True: prints several information
        :param show_progress: bool
            True: progress is printed to the terminal
        :return: 3D numpy array or nothing
            if a path is given no data is returned
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        return self.from_cubes_to_matrix(size, offset,
                                         mode='overlay',
                                         mag=mag,
                                         datatype=datatype,
                                         mirror_oob=mirror_oob,
                                         hdf5_path=hdf5_path,
                                         hdf5_name=hdf5_name,
                                         pickle_path=pickle_path,
                                         invert_data=invert_data,
                                         zyx_mode=zyx_mode,
                                         nb_threads=nb_threads,
                                         verbose=verbose,
                                         http_verbose=http_verbose,
                                         show_progress=show_progress)

    def from_kzip_to_matrix(self, path, size, offset, mag=8, empty_cube_label=0,
                            datatype=np.uint64, verbose=False):
        """ Extracts a 3D matrix from a kzip file

        :param path: str
            forward-slash separated path to kzip file
        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param empty_cube_label: int
            label for empty cubes
        :param datatype: numpy datatype
            typically np.uint8
        :param verbose: bool
            True: prints several information
        :return: 3D numpy array
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if not self.module_wide["snappy"]:
            raise Exception("Snappy is not available - you cannot read "
                            "overlaycubes or kzips.")
        archive = zipfile.ZipFile(path, 'r')

        size = np.array(size, dtype=np.int)
        offset = np.array(offset, dtype=np.int)

        start = np.array([get_first_block(dim, offset, self._cube_shape)
                          for dim in range(3)])
        end = np.array([get_last_block(dim, size, offset, self._cube_shape) + 1
                        for dim in range(3)])

        matrix_size = (end - start)*self.cube_shape
        output = np.zeros(matrix_size, dtype=datatype)

        offset_start = offset % self.cube_shape
        offset_end = (self.cube_shape - (offset + size) % self.cube_shape) % \
                     self.cube_shape

        current = np.array([start[dim] for dim in range(3)])
        cnt = 1
        nb_cubes_to_process = \
            (end[2]-start[2]) * (end[1] - start[1]) * (end[0] - start[0])

        while current[2] < end[2]:
            current[1] = start[1]
            while current[1] < end[1]:
                current[0] = start[0]
                while current[0] < end[0]:
                    if not verbose:
                        progress = 100*cnt/float(nb_cubes_to_process)
                        _stdout('\rProgress: %.2f%%' % progress)

                    this_path = self._experiment_name +\
                                '_mag1_mag%dx%dy%dz%d.seg.sz' % \
                                (mag, current[0], current[1], current[2])
                    print(this_path)

                    if self._experiment_name == \
                                "20130410.membrane.striatum.10x10x30nm":
                        this_path = self._experiment_name +\
                                    '_mag1x%dy%dz%d.segmentation.snappy' % \
                                    (current[0], current[1], current[2])

                    try:
                        values = np.fromstring(
                            module_wide["snappy"].decompress(
                                archive.read(this_path)), dtype=datatype)
                    except Exception:
                        if verbose:
                            _print("Cube does not exist, cube with %d only " \
                                  "assigned" % empty_cube_label)
                        values = np.full(self.cube_shape, empty_cube_label,
                                         dtype=datatype)

                    pos = (current-start)*self.cube_shape

                    values = np.swapaxes(values.reshape(self.cube_shape), 0, 2)
                    output[pos[0] : pos[0] + self.cube_shape[0],
                           pos[1] : pos[1] + self.cube_shape[1],
                           pos[2] : pos[2] + self.cube_shape[2]] = values
                    cnt += 1
                    current[0] += 1
                current[1] += 1
            current[2] += 1

        output = cut_matrix(output, offset_start, offset_end, self.cube_shape,
                            start, end)
        if verbose:
            _print("applying mergelist now")
        mergelist_tools.apply_mergelist(output, archive.read("mergelist.txt"))

        if False in [output.shape[dim] == size[dim] for dim in range(3)]:
            raise Exception("Incorrect shape! Should be", size, "; got:",
                            output.shape)
        else:
            if verbose:
                _print("Correct shape")

        return output

    def from_raw_cubes_to_image_stack(self, size, offset, output_path,
                                      name="img", output_format='png', mag=1,
                                      swap_xy=False, overwrite=False,
                                      delete_dir_first=False, verbose=False):
        """ Exports 2D images (x/y) from raw cubes to one folder

        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param output_path: str
            output folder
        :param name: str
            prefix of image name
        :param output_format: str
            only formats supported by scipy.misc.imsave can be used
        :param mag: int
            magnification of the requested data
        :param swap_xy: bool
            swaps x and y axis
        :param overwrite: bool
            False: raises Exception if directory already exists
        :param delete_dir_first: bool
            True: deletes directory and creates new one before processing
        :param verbose: bool
            True: prints several information
        :return:
            nothing
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        elif not overwrite:
            raise Exception("Directory already exists and overwriting is not "
                            "allowed.")
        elif delete_dir_first:
            if verbose:
                _print("Deleting directory")
            shutil.rmtree(output_path)
            os.makedirs(output_path)

        data = self.from_raw_cubes_to_matrix(size, offset, mag=mag,
                                             verbose=verbose)
        if swap_xy:
            data = np.swapaxes(data, 0, 1)

        if verbose:
            _print("Writing Images")
        for z in range(data.shape[2]):
            scipy.misc.imsave(output_path + "/" + name + "_%d." + output_format,
                              data[:, :, z])

    def export_to_image_stack(self,
                              mode='raw',
                              out_dtype=np.uint8,
                              out_path='',
                              xy_zoom=1.,
                              out_format='tif',
                              mag=1):
        """
        Simple exporter, NOT RAM friendly. Always loads entire cube layers ATM.
        Make sure to have enough RAM available. Supports raw data and
        overlay export (only raw file).
        Please be aware that overlay tif export can be problematic, regarding
        the datatype. Usage of the raw format is advised.

        :param mode: string
        :param out_dtype: numpy dtype
        :param out_format: string
        :param out_path: string
        :return:
        """

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        z_coord_cnt = 0

        stop = False

        scaled_cube_layer_size = (self.boundary[0]//mag,
                                  self.boundary[1]//mag,
                                  self._cube_shape[2])

        for curr_z_cube in range(0, 1 + int(np.ceil(
                self._number_of_cubes[2]) / float(mag))):
            if stop:
                break
            if mode == 'raw':
                layer = self.from_raw_cubes_to_matrix(
                    size=scaled_cube_layer_size,
                    offset=np.array([0, 0, curr_z_cube * self._cube_shape[2]]),
                    mag=mag)
            elif mode == 'overlay':
                layer = self.from_overlaycubes_to_matrix(
                    size=scaled_cube_layer_size,
                    offset=np.array([0, 0, curr_z_cube * self._cube_shape[2]]),
                    mag=mag)

            for curr_z_coord in range(0, self._cube_shape[2]):

                file_path = "{0}{1}_{2:06d}.{3}".format(out_path,
                                                         mode,
                                                         z_coord_cnt,
                                                         out_format)

                # the swap is necessary to have the same visual
                # appearence in knossos and the resulting image stack
                # => needs further investigation?
                try:
                    swapped = np.swapaxes(layer[:, :, curr_z_coord], 0, 1).astype(out_dtype)
                except IndexError:
                    stop = True
                    break

                if xy_zoom != 1.:
                    if mode == 'overlay':
                        swapped = scipy.ndimage.zoom(swapped, xy_zoom, order=0)
                    elif mode == 'raw':
                        swapped = scipy.ndimage.zoom(swapped, xy_zoom, order=1)

                if out_format != 'raw':
                    img = Image.fromarray(swapped)
                    with open(file_path, 'w') as fp:
                        img.save(fp)
                else:
                    swapped.tofile(file_path)

                _print("Writing layer {0} of {1} in total.".format(
                    z_coord_cnt+1, self.boundary[2]//mag))

                z_coord_cnt += 1
            del layer
        return

    def export_partially_to_image_stack(self,
                              mode='raw',
                              out_dtype=np.uint8,
                              out_path='',
                              xy_zoom=1., bounding_box=None,
                              out_format='tif',
                              mag=1):
        """
        Simple exporter, NOT RAM friendly. Always loads entire cube layers ATM.
        Make sure to have enough RAM available. Supports raw data and
        overlay export (only raw file).
        Please be aware that overlay tif export can be problematic, regarding
        the datatype. Usage of the raw format is advised.

        :param mode: string
        :param out_dtype: numpy dtype
        :param out_format: string
        :param out_path: string
        :return:
        """
        if not bounding_box:
            self.export_partially_to_image_stack(mode=mode, out_dtype=out_dtype, out_path=out_path,
            xy_zoom=xy_zoom, out_format=out_format, mag=mag)
        starting_offset = bounding_box[0]
        size = bounding_box[1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        z_coord_cnt = 0

        stop = False

        scaled_cube_layer_size = (size[0]//mag,
                                  size[1]//mag,
                                  self._cube_shape[2])

        end_z = 1 + int(np.ceil((starting_offset[2] + size[2]) // self._cube_shape[2]))
        for curr_z_cube in range(starting_offset[2] // self.cube_shape[2], end_z):
            if stop:
                break
            if mode == 'raw':
                layer = self.from_raw_cubes_to_matrix(
                    size=scaled_cube_layer_size,
                    offset=np.array([starting_offset[0], starting_offset[1], curr_z_cube * self._cube_shape[2]]),
                    mag=mag)
            elif mode == 'overlay':
                layer = self.from_overlaycubes_to_matrix(
                    size=scaled_cube_layer_size,
                    offset=np.array([starting_offset[0], starting_offset[1], curr_z_cube * self._cube_shape[2]]),
                    mag=mag)

            layer = layer.astype(out_dtype)

            for curr_z_coord in range(0, self._cube_shape[2]):

                file_path = "{0}{1}_{2:06d}.{3}".format(out_path,
                                                         mode,
                                                         z_coord_cnt,
                                                         out_format)

                # the swap is necessary to have the same visual
                # appearence in knossos and the resulting image stack
                # => needs further investigation?
                try:
                    swapped = np.swapaxes(layer[:, :, curr_z_coord], 0, 1)
                except IndexError:
                    stop = True
                    break

                if xy_zoom != 1.:
                    if mode == 'overlay':
                        swapped = scipy.ndimage.zoom(swapped, xy_zoom, order=0)
                    elif mode == 'raw':
                        swapped = scipy.ndimage.zoom(swapped, xy_zoom, order=1)

                if out_format != 'raw':
                    img = Image.fromarray(swapped)
                    with open(file_path, 'w') as fp:
                        img.save(fp)
                else:
                    swapped.tofile(file_path)

                _print("Writing layer {0} of {1} in total.".format(
                    z_coord_cnt+1, self.boundary[2]//mag))

                z_coord_cnt += 1

        return

    def from_matrix_to_cubes(self, offset, mags=1, data=None, data_mag=1,
                             data_path=None, hdf5_names=None,
                             datatype=np.uint64, fast_downsampling=True,
                             force_unique_labels=False, verbose=True,
                             overwrite=True, kzip_path=None,
                             overwrite_kzip=False, annotation_str=None,
                             as_raw=False, nb_threads=20):
        """ Cubes data for viewing and editing in KNOSSOS
            one can choose from
                a) (Over-)writing overlay cubes in the dataset
                b) Writing a kzip which can be loaded in KNOSSOS
                c) (Over-)writing raw cubes

        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param mags: sequence of ints
            exported magnifications
        :param data: 3D numpy array or list of 3D numpy arrays of ints
            exported data
            if list: data is combined to a single array by np.maximum()
        :param data_path: str
            path for loading data (hdf5 and pickle files are supported)
        :param hdf5_names: str or list of str
            hdf5 setnames in data_path
        :param datatype: numpy dtype
            typically:  raw = np.uint8
                        overlays = np.uint64
        :param fast_downsampling: bool
            True: uses order 1 downsampling (striding)
            False: uses order 3 downsampling
        :param force_unique_labels: bool
            True and len(data) > 0: Assures unique data before combining all
            list entries
        :param verbose: bool
            True: prints several information
        :param overwrite: bool
            True: whole KNOSSOS cube is overwritten with new data
            False: cube entries where new data == 0 are contained
            eg.: Two different function calls write to two non-overlapping parts
                 of one KNOSSOS cube. When overwrite is set to False, the second
                 call won't overwrite the output of the first one. When they
                 overlap however, the second call will overwrite the data
                 from the first call at all voxels where the second data block
                 has non-zero entries. If overwrite is set to True for the
                 second call the full block gets replaced with the new data
                 regardless of its values. In the current implementation this
                 effects all data within all knossos cubes that are accessed.
        :param kzip_path: str
            is not None: overlay data is written as kzip to this path
        :param annotation_str: str
            is not None: if writing to k.zip, include this as annotation.xml
        :param as_raw: bool
            True: outputs data as normal KNOSSOS raw cubes
        :param nb_threads: int
            if < 2: no multithreading
        :return:
            nothing
        """

        def _write_cubes(args):
            """ Helper function for multithreading """
            folder_path = args[0]
            path = args[1]
            cube_offset = args[2]
            cube_limit = args[3]
            start = args[4]
            end = args[5]

            cube = np.zeros(self.cube_shape, dtype=datatype)

            cube[cube_offset[0]: cube_limit[0],
                 cube_offset[1]: cube_limit[1],
                 cube_offset[2]: cube_limit[2]]\
                = data_inter[start[0]: start[0] + end[0],
                             start[1]: start[1] + end[1],
                             start[2]: start[2] + end[2]]

            cube = np.swapaxes(cube, 0, 2)
            cube = cube.reshape(np.prod(self.cube_shape))

            if kzip_path is None:

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                while True:
                    try:
                        os.makedirs(folder_path+"block")    # Semaphore --------
                        break
                    except:
                        if time.time() - \
                                os.stat(folder_path+"block").st_mtime > 5:
                            os.rmdir(folder_path+"block")
                            os.makedirs(folder_path+"block")
                            break
                        time.sleep(1)

                if (not overwrite) and os.path.isfile(path) and as_raw:
                    existing_cube = np.fromfile(path, dtype=datatype)
                    indices = np.where(cube == 0)

                    cube[indices] = existing_cube[indices]

                elif (not overwrite) and os.path.isfile(path+".zip") and \
                        not as_raw:
                    with zipfile.ZipFile(path+".zip", "r") as zf:
                        existing_cube = \
                            np.fromstring(
                                self.module_wide["snappy"].decompress(
                                    zf.read(os.path.basename(path))),
                                dtype=np.uint64)
                    indices = np.where(cube == 0)
                    cube[indices] = existing_cube[indices]
                if as_raw:
                    f = open(path, "wb")
                    f.write(cube)
                    f.close()
                else:

                    arc_path = os.path.basename(path)
                    with zipfile.ZipFile(path + ".zip", "w") as zf:
                        zf.writestr(arc_path,
                                    self.module_wide["snappy"].compress(cube),
                                    compress_type=zipfile.ZIP_DEFLATED)

                os.rmdir(folder_path+"block")   # ------------------------------

            else:
                f = open(path, "wb")
                f.write(self.module_wide["snappy"].compress(cube))
                f.close()

        # Main Function
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if not (as_raw or self.module_wide["snappy"]):
            raise Exception("Snappy is not available - you cannot write "
                            "overlaycubes or kzips.")

        if not isinstance(mags, list):
            mags = [mags]

        if (data is None) and (data_path is None or hdf5_names is None):
            raise Exception("No data given")

        if not kzip_path is None:
            if as_raw:
                raise Exception("You have to choose between kzip and raw cubes")
            try:
                if ".k.zip" == kzip_path[-6:]:
                    kzip_path = kzip_path[:-6]
            except:
                pass

            if not os.path.exists(kzip_path):
                os.makedirs(kzip_path)
            if verbose:
                _print("kzip path created, notice that kzips can only be "
                       "created in mag1")

            #mags = [1]
            # if not 1 in self.mag:
            #     raise Exception("kzips have to be in mag1 but dataset does not"
            #                     "support mag1")

        if not data_path is None:
            if '.h5' in data_path:
                if not hdf5_names is None:
                    if not isinstance(hdf5_names, list):
                        hdf5_names = [hdf5_names]
                else:
                    raise Exception("No hdf5 names given to read hdf5 file.")

                data = load_from_h5py(data_path, hdf5_names)

            elif '.pkl' in data_path:
                data = load_from_pickle(data_path)
            else:
                raise Exception("File has to be of type hdf5 or pickle.")

        elif len(data) > 0:
            pass
        else:
            raise Exception("No data or path given!")

        if (not isinstance(data, list)) and \
                (not data is None):
            data = np.array(data, copy=False)
        else:
            for ii in range(len(data)):
                data[ii] = np.array(data[ii])

            if force_unique_labels:
                max_label_so_far = np.max(data[0])
                for ii in range(1, len(data)):
                    data[ii][data[ii] > 0] += max_label_so_far
                    max_label_so_far = np.max(data[ii])

            data = np.max(np.array(data), axis=0)

        for mag in mags:
            mag_ratio = float(mag) / data_mag
            if mag_ratio > 1:
                mag_ratio = int(mag_ratio)
                if fast_downsampling:
                    data_inter = np.array(data[::mag_ratio, ::mag_ratio, ::mag_ratio],
                                          dtype=datatype)
                else:
                    data_inter = \
                        scipy.ndimage.zoom(data, 1.0/mag_ratio, order=3).\
                            astype(datatype, copy=False)
            elif mag_ratio < 1:
                inv_mag_ratio = int(1./mag_ratio)
                if fast_downsampling:
                    data_inter = np.zeros(
                        np.array(data.shape) * inv_mag_ratio,
                        dtype=data.dtype)

                    for i_step in range(inv_mag_ratio):
                        data_inter[i_step:: inv_mag_ratio,
                                   i_step:: inv_mag_ratio,
                                   i_step:: inv_mag_ratio] = data

                    data_inter = data_inter.astype(dtype=datatype, copy=False)
                else:
                    data_inter = \
                        scipy.ndimage.zoom(data, inv_mag_ratio, order=3).\
                            astype(datatype, copy=False)
            else:
                # copy=False means in this context that a copy is only made
                # when necessary (e.g. type change)
                data_inter = data.astype(datatype, copy=False)

            offset_mag = np.array(offset, dtype=np.int) // mag_ratio
            size_mag = np.array(data_inter.shape, dtype=np.int)

            if verbose:
                _print("box_offset: {0}".format(offset_mag))
                _print("box_size: {0}".format(size_mag))

            start = np.array([get_first_block(dim, offset_mag, self._cube_shape)
                              for dim in range(3)])
            end = np.array([get_last_block(dim, size_mag, offset_mag,
                                           self._cube_shape) + 1
                            for dim in range(3)])

            if verbose:
                _print("start_cube: {0}".format(start))
                _print("end_cube: {0}".format(end))

            current = np.array([start[dim] for dim in range(3)])
            multithreading_params = []

            while current[2] < end[2]:
                current[1] = start[1]
                while current[1] < end[1]:
                    current[0] = start[0]
                    while current[0] < end[0]:
                        this_cube_info = []
                        path = self.knossos_path + self.name_mag_folder + \
                               str(mag) + "/" + "x%04d/y%04d/z%04d/" \
                                        % (current[0], current[1], current[2])

                        this_cube_info.append(path)

                        if kzip_path is None:
                            if as_raw:
                                path += self.experiment_name \
                                        + "_mag"+str(mag)+\
                                        "_x%04d_y%04d_z%04d.raw" \
                                        % (current[0], current[1], current[2])
                            else:
                                path += self.experiment_name \
                                        + "_mag"+str(mag) + \
                                        "_x%04d_y%04d_z%04d.seg.sz" \
                                        % (current[0], current[1], current[2])
                        else:
                            path = kzip_path+"/"+self._experiment_name + \
                                   '_mag1_mag%dx%dy%dz%d.seg.sz' % \
                                   (mag, current[0], current[1], current[2])
                        this_cube_info.append(path)

                        cube_coords = current*self.cube_shape
                        cube_offset = np.zeros(3)
                        cube_limit = np.ones(3)*self.cube_shape

                        for dim in range(3):
                            if cube_coords[dim] < offset_mag[dim]:
                                cube_offset[dim] += offset_mag[dim] \
                                                    - cube_coords[dim]
                            if cube_coords[dim] + self.cube_shape[dim] > \
                                            offset_mag[dim] + size_mag[dim]:
                                cube_limit[dim] -= \
                                    self.cube_shape[dim] + cube_coords[dim]\
                                        - (offset_mag[dim] + size_mag[dim])

                        start_coord = cube_coords-offset_mag+cube_offset
                        end_coord = cube_limit-cube_offset

                        this_cube_info.append(cube_offset.astype(np.int))
                        this_cube_info.append(cube_limit.astype(np.int))
                        this_cube_info.append(start_coord.astype(np.int))
                        this_cube_info.append(end_coord.astype(np.int))

                        multithreading_params.append(this_cube_info)
                        current[0] += 1
                    current[1] += 1
                current[2] += 1
            if nb_threads > 1:
                pool = ThreadPool(nb_threads)
                pool.map(_write_cubes, multithreading_params)
                pool.close()
                pool.join()
            else:
                for params in multithreading_params:
                    _write_cubes(params)

        if kzip_path is not None:
            with zipfile.ZipFile(kzip_path+".k.zip", "w",
                                 zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(kzip_path):
                    for file in files:
                        zf.write(os.path.join(root, file), file)
                zf.writestr("mergelist.txt",
                            mergelist_tools.gen_mergelist_from_segmentation(
                                data.astype(datatype, copy=False),
                                offsets=np.array(offset, dtype=np.uint64)))
                if annotation_str is not None:
                    zf.writestr("annotation.xml", annotation_str)
            shutil.rmtree(kzip_path)

    def from_overlaycubes_to_kzip(self, size, offset, output_path,
                                  src_mag=1, trg_mags=[1,2,4,8],
                                  nb_threads=5):
        """ Copies chunk from overlay cubes and saves them as kzip

        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param output_path: str
            path to .k.zip file without extension
        :param src_mag: int
            source mag from knossos dataset
        :param trg_mags: iterable of ints
            target mags to write to kzip
        :param nb_threads: int
            number of worker threads
        :return:
            nothing
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        overlay = self.from_overlaycubes_to_matrix(size,
                                                   offset,
                                                   mag=src_mag,
                                                   nb_threads=nb_threads)

        self.from_matrix_to_cubes(offset, data=overlay,
                                  kzip_path=output_path,
                                  nb_threads=nb_threads,
                                  mags=trg_mags)

    def delete_all_overlaycubes(self, nb_processes=4, verbose=False):
        """  Deletes all overlaycubes

        :param nb_processes: int
            if < 2: no multiprocessing
        :param verbose: bool
            True: prints several information
        :return:
            nothing
        """
        self.delete_all_cubes(raw=False, nb_processes=nb_processes,
                              verbose=verbose)

    def delete_all_rawcubes(self, nb_processes=4, verbose=False):
        """  Deletes all overlaycubes

        :param nb_processes: int
            if < 2: no multiprocessing
        :param verbose: bool
            True: prints several information
        :return:
            nothing
        """
        self.delete_all_cubes(raw=True, nb_processes=nb_processes,
                              verbose=verbose)

    def delete_all_cubes(self, raw, nb_processes=4, verbose=False):
        """  Deletes all overlaycubes

        :param raw: bool
            wether to delete raw or overlay cubes
        :param nb_processes: int
            if < 2: no multiprocessing
        :param verbose: bool
            True: prints several information
        :return:
            nothing
        """
        multi_params = []
        for mag in range(32):
            if os.path.exists(self._knossos_path+self._name_mag_folder +
                              str(2**mag)):
                for x_cube in range(self._number_of_cubes[0] // 2**mag+1):
                    if raw:
                        glob_input = self._knossos_path + \
                                     self._name_mag_folder + \
                                     str(2**mag) + "/x%04d/y*/z*/" % x_cube + \
                                     self._experiment_name + "*.raw"
                    else:
                        glob_input = self._knossos_path + \
                                     self._name_mag_folder + \
                                     str(2**mag) + "/x%04d/y*/z*/" % x_cube + \
                                     self._experiment_name + "*seg*"

                    multi_params.append([glob_input, verbose])

        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if nb_processes > 1:
            pool = Pool(nb_processes)
            pool.map(_find_and_delete_cubes_process, multi_params)
            pool.close()
            pool.join()
        else:
            for params in multi_params:
                _find_and_delete_cubes_process(params)
