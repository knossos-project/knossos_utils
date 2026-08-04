################################################################################
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


"""This file provides a class representation of a KNOSSOS-dataset for
reading and writing raw and overlay data."""


from __future__ import annotations
import collections
import copy
import dataclasses
from dataclasses import dataclass
import glob
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import time
import tomli
from typing import List, Optional, Union
import urllib
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from enum import Enum
from io import BytesIO
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Lock
from typing import Optional, Sequence
from xml.etree import ElementTree as ET
import warnings

import imageio
import h5py
import numpy as np
import requests
import scipy.misc
import scipy.ndimage
from PIL import Image
import tensorstore as ts

try:
    from . import mergelist_tools
except ImportError:
    print('mergelist_tools not available, using slow python fallback. '
          'Try to build the cython version of it.')
    from . import mergelist_tools_fallback as mergelist_tools

module_wide = {"init": False, "noprint": False, "snappy": None, "fadvise": None}

# Limits used by KnossosDataset._calculate_optimal_shard_size to bound the number of
# neuroglancer_precomputed shards and the on-disk size of a single shard.
# Note: tensorstore needs roughly 2-3x MAX_SHARD_SIZE as temporary memory while writing a shard.
MAX_NUMBER_OF_SHARDS = 200
MAX_SHARD_SIZE = 10**10  # 10 GB


def our_glob(s):
    l = []
    for g in glob.glob(s):
        l.append(g.replace(os.path.sep, "/"))
    return l


def _print(*args, **kwargs):
    global module_wide
    if not module_wide["noprint"]:
        print(*args, **kwargs)
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
        array = np.fromiter(x, dtype=int, count=dim)
    except TypeError:
        array = np.full(dim, x, dtype=int)
    return array


def _normalize_dtype(dtype):
    if dtype is not None:
        _dtype = np.dtype(dtype)
        if _dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint64)):
            raise ValueError(f"dtype must be np.uint8 or np.uint16 or np.uint64, got {_dtype}.")
        return _dtype
    return dtype


def _file_url_to_path(url: str):
    # Do not pass Windows drive paths through urlparse; it treats "C:" as a URL scheme.
    url_path = url[7:]
    if re.match(r"^/[a-zA-Z]:[\\/]", url_path):
        url_path = url_path[1:]
    return os.path.abspath(urllib.parse.unquote(url_path))


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
        print("snappy is not available - you won't be able to write/read "
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


def _precomputed_kvstore_config(url: str, cdn_token: Optional[dict] = None):
    dataset_url = copy.deepcopy(url)
    dataset_url = dataset_url[:-5] if dataset_url.endswith("/info") else dataset_url
    if dataset_url.startswith("http"):
        split_url = urllib.parse.urlsplit(dataset_url)
        query_params = ""
        if cdn_token is not None:
            bcdn_token = cdn_token.get('token', '')
            expires = cdn_token.get('expires', '')
            token_path = cdn_token.get('token_path', '')
            query_params = f"token={bcdn_token}&expires={expires}&token_path={token_path}"
        base_url = urllib.parse.urlunsplit(
            (split_url.scheme, split_url.netloc, "", query_params, "")
        )
        return {
            "driver": "http",
            "base_url": base_url,
            "path": split_url.path,
        }
    return {
        "driver": "file",
        "base_url": None,
        "path": _file_url_to_path(dataset_url) if dataset_url.startswith("file://") else dataset_url,
    }


def cut_matrix(data, offset_start, offset_end, cube_shape, start, end):
    """ Helper for cutting matrices extracted from cubes to a required size """
    cube_shape = _as_shapearray(cube_shape)

    cut_start = np.array(offset_start, dtype=int)
    number_cubes = np.array(end) - np.array(start)
    cut_end = np.array(number_cubes * cube_shape - offset_end, dtype=int)

    return data[cut_start[2]: cut_end[2],
                cut_start[1]: cut_end[1],
                cut_start[0]: cut_end[0]]


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
    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __init__(self, path: str=None, show_progress: bool=True, reentrant: bool=True):
        '''
        Args:
            path: Path to KnossosDataset streaming configuration file
            show_progress: Output speed and progress when loading cubes
            reentrant: If True, multiple parallel calls for cube writing are safe.
                Should not be turned off if calling save_raw/save_seg/save_to_kzip for the same destination in parallel.
        '''
        moduleInit()
        global module_wide
        self.module_wide = module_wide
        self._knossos_path = None
        self._conf_path = None
        self.url = None
        self._http_user = None
        self._http_passwd = None
        self._cdn_token = None
        self.server_format = None
        self._experiment_name = None
        self.description = None
        self.reentrant = reentrant
        self.layers = []
        self._name_mag_folder = 'mag'
        self._ordinal_mags = False
        self._boundary = np.zeros(3, dtype=int)
        self._scale = np.ones(3, dtype=float)
        self.scales = []
        self._number_of_cubes = np.zeros(3)
        self._cube_shape = np.full(3, 128, dtype=int)
        self._initialized = False
        self._mags = None
        self.verbose = False
        self.show_progress = show_progress
        self.background_label = 0
        self.http_max_tries = 5
        self.description = ''
        self.color = None
        self.visible = None # unspecified
        self.write_empty_cubes = False
        self._tensorstore_datasets = None
        self._rgb_channel = None
        self._dtype = None
        self._shard_size = None

        if path is not None:
            if str(path).endswith(".k.zip"):
                self.initialize_from_kzip(path_to_kzip=path)
            else:
                self.initialize_from_conf(path)

    @property
    def is_embedded(self):
        if ".k.zip" in str(self._conf_path):
            return True
        else:
            return False

    def _embedded_kzip_path(self):
        embedded_path = str(self._knossos_path or self._conf_path)
        kzip_end = embedded_path.rfind(".k.zip")
        assert kzip_end != -1, "Embedded dataset path does not contain a .k.zip archive."
        return embedded_path[:kzip_end + len(".k.zip")]

    @property
    def shard_size(self):
        if self._shard_size is None and self._tensorstore_datasets:
            dataset = self._tensorstore_datasets[min(self._tensorstore_datasets)]
            self._shard_size = np.asarray(
                dataset.chunk_layout.write_chunk.shape[:3], dtype=int
            )
        return self._shard_size

    @property
    def mag(self):
        print('mag is DEPRECATED\nPlease use available_mags')
        return self.available_mags

    @property
    def available_mags(self):
        if self._mags is None:
            self._mags = []
            if self.server_format == "precomputed" and self.scales:
                if self._ordinal_mags:
                    self._mags = list(range(1, len(self.scales) + 1))
                else:
                    self._mags = [2 ** i for i in range(len(self.scales))]
            elif self.in_http_mode:
                for mag_test_nb in range(10):
                    mag_num = mag_test_nb+1 if self._ordinal_mags else 2 ** mag_test_nb
                    url = copy.deepcopy(self.url)
                    url = url.replace("/info", "")
                    mag_folder = "{}/{}{}".format(url, self.name_mag_folder, mag_num)
                    for tries in range(10):
                        try:
                            request = requests.get(mag_folder,
                                                   auth=self.http_auth,
                                                   params=self._cdn_token,
                                                   timeout=10)
                            request.raise_for_status()
                            self._mags.append(mag_num)
                            break
                        except requests.exceptions.HTTPError:
                            if request.status_code < requests.codes.server_error:
                                break # no use retrying if client error (e.g. 404)
                            continue
            elif self.is_embedded:
                regex = re.compile("mag([1-9][0-9]*)")
                kzip_path = self._embedded_kzip_path()
                with zipfile.ZipFile(kzip_path, "r") as archive:
                    for file in archive.namelist():
                        if (file.startswith("embedded/") and
                            self.experiment_name in file and
                            not file.endswith('/') and  # Not a directory
                            any(ext in file for ext in ['.png', '.jpg', '.jpeg', '.raw'])):  # Actual data files
                            match = regex.search(file)
                            if match is not None:
                                self._mags.append(int(match.group(1))) # mag number
                self._mags = list(np.unique(self._mags))
            else:
                regex = re.compile("mag[1-9][0-9]*$")
                for mag_folder in glob.glob(os.path.join(self.knossos_path, "*mag*")):
                    match = regex.search(mag_folder)
                    if match is not None:
                        self._mags.append(int(mag_folder[match.start() + 3:])) # mag number
        return self._mags

    @property
    def existing_mags(self):
        """Magnifications that exist on disk / in the precomputed info (may contain data).

        For precomputed datasets this is the set of scales present in the info file
        (open tensorstore handles). For classic KNOSSOS datasets it matches
        ``available_mags`` (mag folders that were discovered).
        """
        if self.server_format == "precomputed":
            if not self._tensorstore_datasets:
                return []
            return sorted(self._tensorstore_datasets.keys())
        return list(self.available_mags)

    @property
    def name_mag_folder(self):
        return self._name_mag_folder

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def boundary(self):
        return np.array(self._boundary)

    @property
    def scale(self):
        return self._scale

    @property
    def knossos_path(self):
        if self.in_http_mode:
            return self.url
        elif self.url or self._knossos_path:
            if self.url:
                if self.url.startswith("file://"):
                    path = _file_url_to_path(self.url)
                else:
                    path = urllib.parse.urlparse(self.url).path
                return path
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
    def http_user(self):
        return self._http_user

    @property
    def http_passwd(self):
        return self._http_passwd

    @property
    def in_http_mode(self):
        return self.url and self.url.startswith('http')

    @property
    def http_auth(self):# when auth is contained in URL we can return None here
        if self.http_user and self.http_passwd:
            return (self.http_user, self.http_passwd)
        else:
            return None

    @property
    def highest_mag(self):
        return len(self.scales) + 1\
               if self._ordinal_mags else\
               max(np.ceil(np.array(self._boundary) / np.array(self._cube_shape)))

    def mag_scale(self, mag): # get scale in specific mag
        index = mag - 1 if self._ordinal_mags else int(np.log2(mag))
        return self.scales[index]

    def scale_ratio(self, mag, base_mag) -> np.ndarray: # ratio between scale in mag and scale in base_mag
        return (self.mag_scale(mag) / self.mag_scale(base_mag)) if self._ordinal_mags else np.array(3 * [float(mag) / base_mag])

    def iter(self, offset=(0, 0, 0), end=None, step=(512, 512, 512)):
        end = self.boundary if end is None else np.minimum(end, self.boundary)
        step = np.minimum(step, end - offset)
        if step[2] == 0:
            return ((x, y, 0) for x in range(offset[0], end[0], step[0])
                              for y in range(offset[1], end[1], step[1]))
        else:
            return ((x, y, z) for x in range(offset[0], end[0], step[0])
                            for y in range(offset[1], end[1], step[1])
                            for z in range(offset[2], end[2], step[2]))

    def get_first_blocks(self, offset):
        return offset // self.cube_shape

    def get_last_blocks(self, offset, size):
        return ((offset + size - 1) // self.cube_shape) + 1

    def get_cube_coordinates(self, cube_name):
        x_pos = cube_name.rfind("x")
        y_pos = cube_name.find("y", x_pos, len(cube_name))
        z_pos = cube_name.find("z", y_pos, len(cube_name))
        dot_pos = cube_name.find(".", z_pos, len(cube_name))
        x = int(cube_name[x_pos + 1:y_pos])
        y = int(cube_name[y_pos + 1:z_pos])
        z = int(cube_name[z_pos + 1:dot_pos])
        return [x, y, z]

    def get_intervals(self, offset, size, cube_coord):
        global_end = offset + size
        out_start = np.maximum(0, cube_coord * self.cube_shape - offset)
        out_end = (cube_coord + 1) * self.cube_shape - global_end
        out_end = size * (out_end >= 0) + out_end * (out_end < 0) # cube contains this output edge
        incube_start = np.maximum(0, offset - cube_coord * self.cube_shape)
        incube_end = global_end - (cube_coord + 1) * self.cube_shape
        incube_end = self.cube_shape * (incube_end >= 0) + incube_end * (incube_end < 0) # output contains this cube edge
        return out_start, out_end, incube_start, incube_end

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
        return all([str(c) + str(mode) in self._cube_cache.keys() for c in coordinates])

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

    def generate_scales(self, mag1_scale, ds_factor=(2,2,2)):
        if ds_factor[0] < 2 or ds_factor[1] < 2:
            raise ValueError('In xy only downsampling factors ≥ 2 are allowed.')
        x, y, z = np.ceil(np.array(self.boundary) / self.cube_shape)
        scales = []
        scale = list(mag1_scale)
        while True:
            scales.append(np.array([scale[0], scale[1], scale[2]]))
            if x < ds_factor[0] and y < ds_factor[1] and (z < ds_factor[2] or ds_factor[2] == 1):
                break
            x = np.ceil(x / ds_factor[0])
            y = np.ceil(y / ds_factor[1])
            scale[0] *= ds_factor[0]
            scale[1] *= ds_factor[1]
            if scale[2] < scale[0] and not ds_factor[2] == 1:
                scale[2] *= ds_factor[2]
                z = np.ceil(z / ds_factor[1])
        return scales

    def initialize_from_conf(self, path_to_conf: str):

        if path_to_conf.startswith("http") and path_to_conf.endswith(".k.toml"):
            try:
                response = requests.get(path_to_conf)
                response.raise_for_status()
                self._initialize_from_dict(tomli.loads(response.text))
            except Exception as e:
                raise NotImplementedError(f"Could not read .conf from url {path_to_conf}: {e}")
        else:
            path_to_conf = Path(path_to_conf)
            if path_to_conf.name.endswith('.k.toml'):
                self.initialize_from_toml(path_to_conf)
            elif not path_to_conf.exists():
                try:
                    for suffix in ('.k.conf', '.pyk.conf', '.pyknossos.conf', '.conf'):
                        if path_to_conf.name.endswith(suffix):
                            break
                    name = path_to_conf.name[:-len(suffix)]
                    new_path_to_conf = path_to_conf.with_name(f'{name}.k.toml')
                    self.initialize_from_toml(new_path_to_conf)
                    print(f'{path_to_conf} does not exist. Loaded {new_path_to_conf} instead.')
                except Exception as e:
                    print(f'{path_to_conf} does not exist. Also failed to load {new_path_to_conf} instead: {e}')
            elif path_to_conf.name.endswith("ariadne.conf") or path_to_conf.name.endswith(".pyknossos.conf") or path_to_conf.name.endswith(".pyk.conf"):
                self.initialize_from_pyknossos_path(path_to_conf)
            else:
                self.initialize_from_knossos_path(str(path_to_conf))
                self.layers = [self]

    @staticmethod
    def from_toml_string(toml_str: str) -> KnossosDataset:
        ds = KnossosDataset()
        return ds._initialize_from_dict(tomli.loads(toml_str))

    def initialize_from_toml(self, path_to_toml: Union[str, Path]):
        try:
            with open(path_to_toml, 'rb') as conf_file:
                conf = tomli.load(conf_file)
        except FileNotFoundError as e:
            raise NotImplementedError("Could not read .conf: {}".format(e))
        self._initialize_from_dict(conf, path_to_toml)

    def initialize_from_kzip(self, path_to_kzip: Union[str, Path]):
        path_to_kzip = str(path_to_kzip)
        dataset_path = None
        conf = None

        with zipfile.ZipFile(path_to_kzip, "r") as zf:
            # Try to read the dataset path from annotation.xml
            try:
                xml_str = zf.read("annotation.xml").decode()
                annotation_xml = ET.fromstring(xml_str)
                dataset = annotation_xml.find("parameters/dataset")
                if dataset is not None:
                    dataset_path = dataset.attrib.get("path")
            except KeyError:
                pass

            if dataset_path is not None:
                embedded_dataset_path = None
                if dataset_path.startswith(f"{path_to_kzip}/"):
                    embedded_dataset_path = dataset_path[len(path_to_kzip) + 1:]
                elif dataset_path.startswith("embedded/"):
                    embedded_dataset_path = dataset_path

                if embedded_dataset_path in zf.namelist():
                    conf = zf.read(embedded_dataset_path).decode()
                    dataset_path = f"{path_to_kzip}/{embedded_dataset_path}"
                else:
                    if dataset_path.startswith("file://"):
                        dataset_path = _file_url_to_path(dataset_path)
                    with open(dataset_path, "r") as conf_file:
                        conf = conf_file.read()
            else:
                # Check if there is an "embedded" folder with a dataset config file
                for file_info in zf.infolist():
                    if (file_info.filename.startswith("embedded/") and
                        "/" not in file_info.filename[len("embedded/"):] and
                        (file_info.filename.endswith(".conf") or
                        file_info.filename.endswith(".toml"))):
                            dataset_name = file_info.filename
                            dataset_path = f"{path_to_kzip}/{dataset_name}"
                            conf = zf.read(dataset_name).decode()
                            break
        assert dataset_path is not None, "No dataset path has been found in the provided kzip."
        assert conf is not None, f"Could not read dataset configuration from {dataset_path}."

        if dataset_path.endswith(".k.toml"):
            toml_conf = tomli.loads(conf)
            self._initialize_from_dict(toml_conf, dataset_path)
        elif (dataset_path.endswith("ariadne.conf") or dataset_path.endswith(".pyknossos.conf") or
              dataset_path.endswith(".pyk.conf")):
            lines = conf.split("\n")
            lines = [line + "\n" for line in lines] # mimic readlines behaviour
            self._initialize_from_pyknossos_conf(dataset_path, lines)
        else:
            raise NotImplementedError("Only toml and pyknossos confs are implemented for embedded kzips")

    def _copy_configuration(self, source_layer: KnossosDataset) -> KnossosDataset:
        layer = KnossosDataset(show_progress=self.show_progress)
        layer._conf_path = copy.deepcopy(source_layer._conf_path)
        layer._knossos_path = copy.deepcopy(source_layer._knossos_path)
        layer._initialized = copy.deepcopy(source_layer._initialized)
        layer._initialize_cache(0)
        layer._ordinal_mags = copy.deepcopy(source_layer._ordinal_mags)
        layer._cube_shape = copy.deepcopy(source_layer._cube_shape)
        layer._boundary = copy.deepcopy(source_layer._boundary)
        layer.scales = copy.deepcopy(source_layer.scales)
        layer._scale = copy.deepcopy(source_layer._scale)
        layer.layers = [layer]
        layer._experiment_name = copy.deepcopy(source_layer._experiment_name)
        layer.file_extensions = copy.deepcopy(source_layer.file_extensions)
        layer.server_format = copy.deepcopy(source_layer.server_format)
        layer.url = copy.deepcopy(source_layer.url)
        layer._http_user = copy.deepcopy(source_layer._http_user)
        layer._http_passwd = copy.deepcopy(source_layer._http_passwd)
        layer._cdn_token = copy.deepcopy(source_layer._cdn_token)
        layer.description = copy.deepcopy(source_layer.description)
        layer.color = copy.deepcopy(source_layer.color)
        layer.visible = copy.deepcopy(source_layer.visible)
        layer._tensorstore_datasets = source_layer._tensorstore_datasets
        layer._rgb_channel = copy.deepcopy(source_layer._rgb_channel)
        layer._dtype = copy.deepcopy(source_layer._dtype)
        return layer

    def _initialize_from_dict(self, conf: dict, conf_path: Optional[str] = None):
        fail_fast_cdn = False
        layers = []
        for layer_conf in conf['Layer']:
            layer = KnossosDataset(show_progress=self.show_progress)
            layer._conf_path = conf_path
            layer._knossos_path = os.path.dirname(conf_path) + "/" if conf_path is not None else None
            layer._initialized = True
            layer._initialize_cache(0)
            layer._ordinal_mags = True
            layer._cube_shape = [128, 128, 128]
            layer.layers = [layer]
            layers.append(layer)
            layer._experiment_name = layer_conf['Name']
            layer.file_extensions = layer_conf['FileExtension']
            layer._dtype = _normalize_dtype(layer_conf.get('DataType', None))
            if layer._dtype is None:
                layer._dtype = np.uint64 if ".seg.sz.zip" in layer.file_extensions else np.uint8
            num_channels = layer_conf.get('NumChannels', 1)
            if num_channels == 3:
                layer._rgb_channel = True
            elif num_channels not in (1, None):
                raise ValueError(
                    f"NumChannels must be 1 or 3, got {num_channels} for layer {layer_conf['Name']}"
                )
            layer.server_format = layer_conf.get('ServerFormat', layer.server_format)
            layer.url = f'file://{layer._knossos_path}' if layer._knossos_path is not None else None
            if 'URL' in layer_conf:
                layer.url = layer_conf['URL']
                if layer.server_format == None and layer.url.endswith("info"):
                    layer.server_format = "precomputed"
                split_url = urllib.parse.urlsplit(layer.url)
                layer._http_user = split_url.username
                layer._http_passwd = split_url.password
                if layer._http_user is not None and layer._http_passwd is not None and not fail_fast_cdn:
                    try:
                        auth_path = split_url.path
                        if layer.server_format == "precomputed":
                            auth_path = auth_path[:-5] if auth_path.endswith("/info") else auth_path
                        response = requests.get(
                            f'{split_url.scheme}://{split_url.hostname}/auth', 
                            auth=(layer._http_user, layer._http_passwd), 
                            params={"path": auth_path}
                        )
                        if response.status_code == 200:
                            token_data = response.json()
                            token_str = token_data['token_string']
                            token_dict = {}
                            for param in token_str.split('&'):
                                key, value = param.split('=')
                                token_dict[key] = value
                            token_dict["token_path"] = token_dict["token_path"].replace("%2F", "/")
                            layer._cdn_token = token_dict
                        else:
                            print(f'Unable to get CDN token: {response.status_code} for {layer.url}')
                            fail_fast_cdn = True
                    except Exception as e:
                        print(f'Failed to get CDN token: {e} for {layer.url}')
                        fail_fast_cdn = True
                if layer.server_format is None:
                    if layer.url.startswith("file://"):
                        local_path = _file_url_to_path(layer.url)
                        if os.path.exists(local_path):
                            layer.server_format = "precomputed"
                    else:
                        auth=None
                        cdn_token = None
                        if layer._http_user and layer._http_passwd:
                            auth = (layer._http_user, layer._http_passwd)
                            cdn_token = copy.deepcopy(layer._cdn_token)
                        response = requests.get(layer.url + "/info", auth=auth, params=cdn_token)
                        if response.status_code == 200:
                            layer.server_format = "precomputed"
            elif layer._knossos_path is not None:
                info_file_path = os.path.join(layer._knossos_path, "info")
                if os.path.exists(info_file_path):
                    layer.server_format = "precomputed"

            if layer.server_format == "precomputed":
                import json
                info_json = None
                if layer.url is not None and not layer.url.endswith("info"):
                    layer.url = layer.url + "/info"
                if layer.url and layer.url.startswith("file://"):
                    local_path = _file_url_to_path(layer.url)
                    try:
                        with open(local_path, 'r') as f:
                            info_json = json.load(f)
                    except Exception as e:
                        print(f"Failed to load info json from local file '{local_path}': {e}")
                        if os.path.exists(local_path):
                            raise Exception(f"Failed to load info json from local file '{local_path}': {e}")
                elif layer.url:
                    try:
                        auth=None
                        cdn_token = None
                        if layer._http_user and layer._http_passwd:
                            auth = (layer._http_user, layer._http_passwd)
                            cdn_token = copy.deepcopy(layer._cdn_token)
                        response = requests.get(layer.url, auth=auth, params=cdn_token)
                        response.raise_for_status()
                        info_json = response.json()
                    except Exception as e:
                        print(f"Failed to load info json from url '{layer.url}': {e}")
                        raise Exception(f"Failed to load info json from url '{layer.url}': {e}")

                extent_px = layer_conf.get('Extent_px', None)
                cube_shape_px = layer_conf.get('CubeShape_px', None)
                # voxel_sizes = layer_conf.get('VoxelSize_nm', None)
                voxel_sizes = None
                if 'VoxelSize_nm' in layer_conf:
                    voxel_sizes = [np.array(scale) for scale in layer_conf['VoxelSize_nm']]

                if info_json is not None:
                    assert info_json["data_type"] == "uint8" or info_json["data_type"] == "uint16" or info_json["data_type"] == "uint64", f"Expected data_type to be uint8 or uint16 or uint64, got {info_json['data_type']}"
                    assert info_json["num_channels"] == 1 or info_json["num_channels"] == 3, f"Expected num_channels to be 1 or 3(rgb), got {info_json['num_channels']}"
                    file_extension = ".seg.sz.zip" if info_json["scales"][0]["encoding"] == "compressed_segmentation" else f'.{info_json["scales"][0]["encoding"]}'
                    if layer.file_extensions != [file_extension]:
                        warnings.warn(f"Expected file extensions from .toml file to be {layer.file_extensions}, got {info_json['scales'][0]['encoding']} from info file. Using file extension from info file...")
                        layer.file_extensions = [file_extension]
                    info_dtype = _normalize_dtype(info_json["data_type"])
                    if layer._dtype is not None and layer._dtype != info_dtype:
                        warnings.warn(
                            f"DataType in toml ({np.dtype(layer._dtype).name}) differs from "
                            f"info data_type ({info_json['data_type']}). Using info data_type."
                        )
                    layer._dtype = info_dtype

                    # Prefer TOML geometry; fall back to finest info scale (legacy).
                    finest = min(info_json["scales"], key=lambda s: float(np.prod(s["resolution"])))
                    if finest is not None:
                        mag = 0
                        if finest["key"] != "mag1":
                            warnings.warn(f"Expected finest scale key to be 'mag1', got {finest['key']}. This may cause issues with the dataset.")
                            if finest["key"].startswith("mag"):
                                mag = int(finest["key"].split("mag")[1]) - 1
                        if extent_px is None or mag == 0:
                            extent_px_json = finest["size"]
                            if extent_px_json is not None:
                                if extent_px_json[2] > 1:
                                    extent_px = [int(extent_px_json[0] * 2**mag), int(extent_px_json[1] * 2**mag), int(extent_px_json[2] * 2**mag)]
                                else:
                                    extent_px = [int(extent_px_json[0] * 2**mag), int(extent_px_json[1] * 2**mag), int(extent_px_json[2])]
                        if cube_shape_px is None or mag == 0:
                            cube_shape_px = finest["chunk_sizes"][0]
                        if voxel_sizes is None or mag == 0:
                            voxel_sizes = [np.array(finest["resolution"]) / 2**mag]

                    # assert extent_px is not None, "Extent_px is not set"
                    # assert cube_shape_px is not None, "CubeShape_px is not set"
                    # assert voxel_sizes is not None, "VoxelSize_nm is not set"

                    if info_json["num_channels"] == 3:
                        toml_channels = layer_conf.get('NumChannels', 1)
                        if toml_channels not in (None, 3) and int(toml_channels) != 3:
                            warnings.warn(
                                f"NumChannels in toml ({toml_channels}) differs from "
                                f"info num_channels (3). Using info num_channels."
                            )
                        layer._rgb_channel = True
                    elif layer._rgb_channel:
                        warnings.warn(
                            f"NumChannels in toml is 3 but info num_channels is "
                            f"{info_json['num_channels']}. Using info num_channels."
                        )
                        layer._rgb_channel = None

                    kvstore_config = _precomputed_kvstore_config(
                        layer.url, layer._cdn_token
                    )
                    layer._tensorstore_datasets = {}
                    for idx, key in enumerate([scale["key"] for scale in info_json["scales"]]):
                        mag = idx + 1
                        if "mag" in key:
                            mag = int(key.split("mag")[1])
                        layer._tensorstore_datasets[mag] = ts.open({
                                    "driver": "neuroglancer_precomputed",
                                    "kvstore": {
                                        **({"base_url": kvstore_config["base_url"]} if kvstore_config["base_url"] is not None else {}),
                                        "driver": kvstore_config["driver"],
                                        "path": kvstore_config["path"],
                                    },
                                    "scale_metadata" : {
                                        "key" : key,
                                    },
                                }).result()
                else:
                    print(f"Looking for missing information in toml file... file_extensions: {layer.file_extensions}")
                    if extent_px is None or cube_shape_px is None or voxel_sizes is None:
                        warnings.warn("MISSING INFORMATION: Could not find all missing information in toml file. Looking for missing information in other layers...")
                        if len(layers) > 1:
                            other_layer = layers[0]
                            if extent_px is None:
                                extent_px = other_layer._boundary
                            if cube_shape_px is None:
                                cube_shape_px = other_layer._cube_shape
                            if voxel_sizes is None and other_layer.scales:
                                voxel_sizes = other_layer.scales
                    layer._tensorstore_datasets = {}

                if extent_px is None or cube_shape_px is None or voxel_sizes is None:
                    raise ValueError(
                        f"No info file found at {layer.url} and could not find all "
                        f"missing information in toml file or other layers."
                    )
                
                layer._boundary = extent_px
                layer._cube_shape = cube_shape_px
                if len(voxel_sizes) == 1:
                        warnings.warn("MISSING INFORMATION: Only one scale found in toml file. Assuming isotropic scale and generating scales...")
                        voxel_sizes = layer.generate_scales(voxel_sizes[0], KnossosDataset._default_ds_factor(layer._boundary))
                layer.scales = voxel_sizes
                print("Found all information. Creating neuroglancer dataset...")

                # KnossosDataset.create_neuroglancer_layer(layer)

            if not layer.server_format == "precomputed":
                layer.scales = [np.array(mag_scale) for mag_scale in layer_conf['VoxelSize_nm']]
                layer._boundary = layer_conf['Extent_px']
                layer._cube_shape = layer_conf['CubeShape_px']
            layer._ordinal_mags = True
            layer._scale = layer.scales[0]
            layer.description = layer_conf.get('Description', layer.description)
            layer.color = layer_conf.get('Color')
            layer.visible = layer_conf.get('Visible')

            if layer._rgb_channel:
                rgb_numbers = []
                for lyr in layers:
                    val = getattr(lyr, "_rgb_channel", None)
                    if isinstance(val, str) and re.match(r'^[rgb]_(\d+)$', val):
                        num = int(val.split("_")[1])
                        rgb_numbers.append(num)
                highest_rgb_channel = max(rgb_numbers, default=0)

                layer._rgb_channel = f"r_{highest_rgb_channel + 1}"
                for channel in ["g", "b"]:
                    layer_rgb = self._copy_configuration(layer)
                    layer_rgb._rgb_channel = f"{channel}_{highest_rgb_channel + 1}"
                    layers.append(layer_rgb)

        for layer in layers:
            # set to first local layer or to first remote layer if there is no local one.
            if not self._initialized or (self.in_http_mode and not layer.in_http_mode):
                self.__dict__.update(layer.__dict__)

        self.layers = layers

        # print(self.__dict__)
        return self

    def save_toml(self, path_to_toml: Union[str, Path]):
        with open(path_to_toml, 'w') as toml_file:
            string = ''
            for layer in self.layers:
                if layer._rgb_channel and (layer._rgb_channel.startswith("g_") or layer._rgb_channel.startswith("b_")):
                    continue
                string += '[[Layer]]\n'
                string += LayerConfig(layer).to_toml_string() + '\n'
            toml_file.write(string[:-1])

    def initialize_from_pyknossos_path(self, path_to_pyknossos_conf):
        """ Parse a pyKNOSSOS conf
        :param path_to_pyknossos_conf: str
        """
        print(
            'DEPRECATION warning: The PyKNOSSOS conf format is deprecated (loaded conf: '
            f'{path_to_pyknossos_conf}). Please convert this dataset to toml using '
            'save_toml("/output/path.k.toml") or use examples/convert_conf_to_toml.py '
            f'{path_to_pyknossos_conf} /output/path.k.toml'
        )

        try:
            f = open(path_to_pyknossos_conf)
            lines = f.readlines()
            f.close()
        except FileNotFoundError as e:
            raise NotImplementedError("Could not read .conf: {}".format(e))

        self._initialize_from_pyknossos_conf(path_to_pyknossos_conf, lines)

    def _initialize_from_pyknossos_conf(self, path_to_pyknossos_conf: str, lines: list[str]):
        def initialize(layer):
            layer._knossos_path = os.path.dirname(path_to_pyknossos_conf) + "/"
            layer._initialized = True
            layer._initialize_cache(0)

        layers = []
        for line in lines:
            tokens = re.split(" = |,|\n", line.replace('"', ''))
            key = tokens[0]
            if re.match(r'\[Dataset[ \d]*]$', tokens[0]):
                layer = KnossosDataset(show_progress=self.show_progress)
                layer._conf_path = os.path.abspath(path_to_pyknossos_conf)
                layer._ordinal_mags = True # pyk.conf is ordinal by default
                layer._cube_shape = [128, 128, 128] # default cube shape
                layer.layers = [layer]
                layers.append(layer)
            if key == "_BaseName":
                layer._experiment_name = tokens[1]
            elif key == "_BaseURL":
                layer.url = tokens[1]
            elif key == "_UserName":
                layer._http_user = tokens[1]
            elif key == "_Password":
                layer._http_passwd = tokens[1]
            elif key == "_ServerFormat":
                layer._ordinal_mags = tokens[1] != "knossos";
            elif key == "_DataScale":
                layer.scales = []
                for x, y, z in zip(tokens[1::3], tokens[2::3], tokens[3::3]):
                    layer.scales.append(np.array([float(x), float(y), float(z)]))
                layer._scale = layer.scales[0]
            elif key == "_FileType":
                type_map = {'0': '.raw', '2': '.png', '3': '.jpg'}
                assert tokens[1] in type_map, f'unsupported _FileType ({tokens[1]})'
                layer.file_extensions = [type_map[tokens[1]]]
            elif key == "_NumberofCubes":
                layer._number_of_cubes[0] = int(tokens[1])
                layer._number_of_cubes[1] = int(tokens[2])
                layer._number_of_cubes[2] = int(tokens[3])
            elif key == "_Extent":
                layer._boundary[0] = float(tokens[1])
                layer._boundary[1] = float(tokens[2])
                layer._boundary[2] = float(tokens[3])
            elif key == '_Description':
                layer.description = tokens[1]
            elif key == '_CubeSize':
                layer._cube_shape = [int(tokens[1]), int(tokens[2]), int(tokens[3])]
            elif key == "_BaseExt":
                layer.file_extensions = ['.' * (not tokens[1].startswith('.')) + tokens[1]]
            elif key == '_Color':
                layer.color = tokens[1]
            elif key == '_Visible':
                try:
                    layer.visible = bool(int(tokens[1]))
                except ValueError:
                    layer.visible = bool(tokens[1])

        for layer in layers:
            initialize(layer)
            if layer.url is None:
                layer.url = f'file://{layer._knossos_path}'
            # set to first local layer or to first remote layer if there is no local one.
            if not self._initialized or (self.in_http_mode and not layer.in_http_mode):
                self.__dict__.update(layer.__dict__)

        self.layers = layers

    def write_pyknossos_conf(self, write_path):
        with open(write_path, 'w') as conf:
            for layer in self.layers:
                for ext in layer.file_extensions:
                    conf.write('[Dataset]\n')
                    if layer.url:
                        url = urllib.parse.urlparse(layer.url)
                        conf.write(f'_BaseURL = {url.scheme}://{url.netloc}{urllib.parse.quote(url.path)}\n')
                    if layer.http_auth:
                        conf.write(f'_UserName = {layer.http_user}\n')
                        conf.write(f'_Password = {layer.http_passwd}\n')
                    if not layer._ordinal_mags:
                        conf.write('_ServerFormat = knossos\n')
                    conf.write(f'_BaseName = {layer.experiment_name}\n')
                    scale_str = ''.join([f'{sx},{sy},{sz}, ' for (sx, sy, sz) in layer.scales])
                    conf.write(f'_DataScale = {scale_str}\n')
                    #conf.write(f'_NumberOfCubes = {layer.number_of_cubes}\n') var currently holds only the mag1 number of cubes
                    conf.write(f'_CubeSize = {layer.cube_shape[0]},{layer.cube_shape[1]},{layer.cube_shape[2]}\n')
                    conf.write(f'_Extent = {layer.boundary[0]},{layer.boundary[1]},{layer.boundary[2]}\n')
                    conf.write(f'_Description = {layer.description}\n')
                    conf.write(f'_BaseExt = {ext}\n')
                    if ext == '.png':
                        conf.write(f'_FileType = 2\n')
                    if layer.color is not None:
                        conf.write(f'_Color = {layer.color}\n')
                    if layer.visible is not None:
                        conf.write(f'_Visible = {int(layer.visible)}\n')
                    conf.write('\n')

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
        except FileNotFoundError:
            raise NotImplementedError("Could not find/read *mag1/knossos.conf")

        self._conf_path = os.path.abspath(path_to_knossos_conf)

        parsed_dict = {}
        for line in lines:
            if line.startswith("ftp_mode"):
                line_s = line.split(" ")
                self.url = "http://" + line_s[1] + line_s[2] + "/"
                self._http_user = line_s[3]
                self._http_passwd = line_s[4]
            else:
                match = re.search(r'(?P<key>[A-Za-z _]+)'
                                  r'((((?P<numeric_value>[0-9\.]+)'
                                  r'|"(?P<string_value>[A-Za-z0-9._/-]+)");)'
                                  r'|(?P<empty_value>;))',
                                  line)
                if match:
                    match = match.groupdict()
                    if match['empty_value']:
                        val = True
                    elif match['string_value']:
                        val = match['string_value']
                    elif '.' in match['numeric_value']:
                        val = float(match['numeric_value'])
                    elif match['numeric_value']:
                        val = int(match['numeric_value'])
                    else:
                        raise Exception('Malformed knossos.conf')

                    parsed_dict[match["key"]] = val
                elif verbose:
                        _print(f"Unreadable line in knossos.conf - ignored: {line}")

        self._boundary[0] = parsed_dict['boundary x ']
        self._boundary[1] = parsed_dict['boundary y ']
        self._boundary[2] = parsed_dict['boundary z ']
        self._scale[0] = parsed_dict['scale x ']
        self._scale[1] = parsed_dict['scale y ']
        self._scale[2] = parsed_dict['scale z ']
        self.scales = [np.multiply(2**i, self._scale) for i in range(0, int(np.ceil(np.log2(np.amax(self._boundary / self._cube_shape)))))]
        self._experiment_name = parsed_dict['experiment name ']
        if self._experiment_name.endswith("mag1"):
            self._experiment_name = self._experiment_name[:-5]

        self._number_of_cubes = \
            np.array(np.ceil(self.boundary.astype(float) /
                             self.cube_shape), dtype=int)

        if 'png' in parsed_dict:
            self.file_extensions = ['.png']
        else:
            self.file_extensions = ['.raw']

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
        print(f'DEPRECATION warning: The knossos.conf format is deprecated (loaded conf: {path}).\nPlease convert this dataset to toml using save_toml("/output/path.k.toml")')
        while path.endswith("/"):
            path = path[:-1]

        if not os.path.exists(path):
            raise Exception("Does not exist: {0}".format(path))

        if os.path.isfile(path):
            self.parse_knossos_conf(path, verbose=verbose)
            if self.in_http_mode:
                self._name_mag_folder = "mag"
            else:
                folder = os.path.basename(os.path.dirname(path))
                match = re.search(r'(?<=mag)[\d]+$', folder)
                if match:
                    self._knossos_path = \
                        os.path.dirname(os.path.dirname(path)) + "/"
                else:
                    self._knossos_path = os.path.dirname(path) + "/"
        else:
            match = re.search(r'(?<=mag)[\d]+$', path)
            if match:
                self._knossos_path = os.path.dirname(path) + "/"
            else:
                self._knossos_path = path + "/"

        if not self.in_http_mode:
            all_mag_folders = our_glob(self._knossos_path+"/*mag*")

            if len(all_mag_folders) == 0:
                self._name_mag_folder = "mag"
            else:
                mag_folder = all_mag_folders[0].split("/")
                if len(mag_folder[-1]) > 1:
                    mag_folder = mag_folder[-1]
                else:
                    mag_folder = mag_folder[-2]

                self._name_mag_folder = \
                    mag_folder[:-len(re.findall(r"[\d]+", mag_folder)[-1])]

            if not os.path.isfile(path):
                warnings.warn(
                        'You are initializing a KnossosDataset from a path to a directory. This possibility will soon be'
                        ' removed, please specify paths to configuration files instead.')
                conf_path = self.knossos_path + self.name_mag_folder + "1/knossos.conf" # legacy path
                for name in os.listdir(self.knossos_path):
                    if name == "knossos.conf" or name.endswith(".k.conf"):
                        conf_path = os.path.join(self.knossos_path, name)
                self.parse_knossos_conf(conf_path, verbose=verbose)

        if use_abs_path:
            self._knossos_path = os.path.abspath(self.knossos_path)

        self._initialize_cache(cache_size)

        if verbose:
            _print("Initialization finished successfully")
        self._initialized = True

    @staticmethod
    def _calculate_optimal_shard_size(layer: KnossosDataset) -> np.ndarray:
        """Compute an optimal 3D shard size for a neuroglancer_precomputed layer.

        Starts at the layer's cube_shape and iteratively doubles it until either
        the total number of shards fits within MAX_NUMBER_OF_SHARDS or doubling
        would push a single shard above MAX_SHARD_SIZE. When the current shard
        already covers the full Z extent of the dataset, only X and Y are doubled.

        The returned shard size is later used as the tensorstore write_chunk size
        (i.e. the on-disk shard); the read_chunk stays at layer.cube_shape.
        """
        shard_size = np.asarray(layer.cube_shape, dtype=float)
        dataset_size = np.asarray(layer.boundary, dtype=float)
        number_of_shards = np.maximum(1, np.ceil(dataset_size / shard_size).astype(int))
        total_number_of_shards = np.prod(number_of_shards)
        while total_number_of_shards > MAX_NUMBER_OF_SHARDS:
            if shard_size[2] >= dataset_size[2]:
                shard_size = np.array(
                    [shard_size[0] * 2, shard_size[1] * 2, shard_size[2]],
                    dtype=float,
                )
            else:
                shard_size = shard_size * 2

            number_of_shards = np.maximum(
                1, np.ceil(dataset_size / shard_size).astype(int)
            )
            total_number_of_shards = np.prod(number_of_shards)

            # Heuristic bytes-per-voxel-times-channel factor used to bound shard size:
            # 4 for flat (single-Z-slab) shards, 8 otherwise.
            factor = 4 if shard_size[2] >= dataset_size[2] else 8
            if np.prod(shard_size) * factor > MAX_SHARD_SIZE:
                break
        return shard_size.astype(int)

    @staticmethod
    def _default_ds_factor(boundary) -> list:
        ds_factor = [2, 2, 2]
        if int(boundary[2]) == 1:
            ds_factor[2] = 1
        return ds_factor

    @staticmethod
    def _sort_precomputed_info_scales(layer: KnossosDataset):
        """Sort info.json scales by ascending resolution (Neuroglancer requirement)."""
        import json
        if layer._knossos_path is None:
            return
        info_path = Path(layer._knossos_path) / "info"
        if not info_path.is_file():
            return
        with open(info_path, "r") as f:
            info = json.load(f)
        scales = info.get("scales", [])
        if len(scales) <= 1:
            return
        sorted_scales = sorted(
            scales,
            key=lambda s: (float(s["resolution"][0]), float(s["resolution"][1]), float(s["resolution"][2])),
        )
        if sorted_scales == scales:
            return
        info["scales"] = sorted_scales
        with open(info_path, "w") as f:
            json.dump(info, f)

    def _ensure_precomputed_mag(self, mag: int, create: bool = True):
        """Return the tensorstore handle for mag; optionally create the scale on demand."""
        if self._tensorstore_datasets is None:
            self._tensorstore_datasets = {}
        if mag in self._tensorstore_datasets:
            return self._tensorstore_datasets[mag]
        if not create:
            raise Exception(
                f"No precomputed data for mag {mag}. Available scales in info: "
                f"{sorted(self._tensorstore_datasets.keys())}."
            )
        if mag < 1 or mag > len(self.scales):
            raise Exception(
                f"Requested mag {mag} not available, only mags {self.available_mags} are available."
            )
        KnossosDataset.create_neuroglancer_layer(
            self,
            as_rgb=bool(self._rgb_channel),
            shard_size=self._shard_size,
            dtype=self._dtype,
            mags=[mag],
        )
        return self._tensorstore_datasets[mag]

    @staticmethod
    def create_neuroglancer_layer(
        layer: KnossosDataset,
        as_rgb: bool = False,
        shard_size: Optional[Sequence[int]] = None,
        dtype=np.uint8,
        mags: Optional[Sequence[int]] = None,
    ):
        """Create neuroglancer_precomputed tensorstore dataset(s) for selected magnifications.

        Supports image layers (.raw / .png/.jpg/.jpeg, uint8/uint16(only for precomputed)) and
        segmentation layers (.seg.sz.zip, uint64 + compressed_segmentation).

        Only the magnifications listed in `mags` are created (or all pyramid levels if
        `mags` is None). Existing entries in `layer._tensorstore_datasets` are kept.
        After creation, on-disk info scales are sorted by ascending resolution.

        Sharding is configured implicitly via tensorstore's chunk_layout: the
        write_chunk drives the on-disk shard size, the read_chunk stays at
        layer.cube_shape so existing readers keep working. If `shard_size` is
        None, an optimal value is derived from the dataset boundary and cube
        shape via _calculate_optimal_shard_size; otherwise the provided value
        is validated and used as-is.
        """
        supported_extensions = ('.seg.sz.zip', '.raw', '.png', '.jpg', '.jpeg')
        ext = next((ext for ext in layer.file_extensions if ext in supported_extensions), None)
        if ext is None:
            return
        
        if len(layer.file_extensions) != 1:
            print(f"Warning: {layer.experiment_name} has multiple file extensions: {layer.file_extensions}. Will only create a layer for the first extension: {ext}.")

        dtype = _normalize_dtype(dtype) if dtype is not None else np.dtype(np.uint8)
        layer._dtype = dtype

        if ext == '.seg.sz.zip':
            assert not as_rgb, "Cannot create neuroglancer layer for segmentation data with as_rgb=True"
            dtype = "uint64"
            encoding = "compressed_segmentation"
            encoding_level_key = None
            encoding_level_value = None
        elif ext == '.raw':
            dtype = dtype.name
            encoding = "raw"
            encoding_level_key = None
            encoding_level_value = None
        elif ext == '.png':
            dtype = dtype.name
            encoding = "png"
            encoding_level_key = "png_level"
            encoding_level_value = 6
        elif ext in ('.jpg', '.jpeg'):
            dtype = dtype.name
            encoding = "jpeg"
            encoding_level_key = "jpeg_quality"
            encoding_level_value = 75
        else:
            return  # Unsupported extension; leave layer untouched.

        if shard_size is None:
            shard_size = KnossosDataset._calculate_optimal_shard_size(layer)
        else:
            if int(shard_size[0]) != int(shard_size[1]):
                raise ValueError(
                    f"shard_size must be equal in x and y (got {tuple(shard_size)})"
                )
            if int(shard_size[2]) > int(shard_size[0]):
                raise ValueError(
                    f"shard_size z must be <= x/y (got {tuple(shard_size)})"
                )
            shard_size = np.asarray(shard_size, dtype=int)

        layer._shard_size = np.asarray(shard_size, dtype=int)
        layer.server_format = "precomputed"
        if layer._tensorstore_datasets is None:
            layer._tensorstore_datasets = {}

        num_channels = 3 if as_rgb else 1
        cube_shape = [int(c) for c in layer.cube_shape]
        shard_size_int = [int(s) for s in shard_size]
        base_scale = np.asarray(layer.scales[0], dtype=float)

        if mags is None:
            mags = list(range(1, len(layer.scales) + 1))
        else:
            mags = list(mags)

        created_any = False
        for mag in mags:
            if mag in layer._tensorstore_datasets:
                continue
            mag_idx = mag - 1
            if mag_idx < 0 or mag_idx >= len(layer.scales):
                raise ValueError(
                    f"Cannot create mag {mag}: only {len(layer.scales)} scales in pyramid."
                )
            curr_scale = np.asarray(layer.scales[mag_idx], dtype=float)
            # factors is base/curr (<= 1 for higher mags); dataset_size shrinks accordingly.
            factors = base_scale / curr_scale
            dataset_size = [
                int(np.ceil(float(layer.boundary[i]) * float(factors[i])))
                for i in range(3)
            ]
            resolution = [float(s) for s in curr_scale]

            spec_seed = {
                "driver": "neuroglancer_precomputed",
                "schema": {
                    "rank": 4,
                    "dtype": dtype,
                    "chunk_layout": {
                        "write_chunk": {
                            "shape_soft_constraint": [
                                shard_size_int[0],
                                shard_size_int[1],
                                shard_size_int[2],
                                num_channels,
                            ]
                        },
                        "read_chunk": {
                            "shape": [
                                cube_shape[0],
                                cube_shape[1],
                                cube_shape[2],
                                num_channels,
                            ]
                        },
                    },
                    "codec": {
                        "driver": "neuroglancer_precomputed",
                        "encoding": encoding,
                        "shard_data_encoding": "raw",
                    },
                    "domain": {
                        "shape": [
                            dataset_size[0],
                            dataset_size[1],
                            dataset_size[2],
                            num_channels,
                        ],
                    },
                    "dimension_units": [
                        [resolution[0], "nm"],
                        [resolution[1], "nm"],
                        [resolution[2], "nm"],
                        None,
                    ],
                },
                "kvstore": {"driver": "memory"},
                "create": True,
            }

            # Open on a memory kvstore first so tensorstore computes the sharding
            # parameters from the chunk layout; then patch the resulting JSON spec
            # and open the real on-disk dataset.
            tmp_dataset = ts.open(spec_seed).result()
            json_spec = tmp_dataset.spec().to_json()
            sharding = json_spec.get("scale_metadata", {}).get("sharding")
            if sharding is not None:
                sharding["minishard_index_encoding"] = "raw"
            json_spec["scale_metadata"]["key"] = f"mag{mag}"
            if encoding_level_key is not None:
                json_spec["scale_metadata"][encoding_level_key] = encoding_level_value
            if layer._knossos_path is not None:
                json_spec["kvstore"] = {
                    "driver": "file",
                    "path": str(layer._knossos_path),
                }
            json_spec.pop("scale_index", None)

            layer._tensorstore_datasets[mag] = ts.open(
                ts.Spec(json_spec), open=True, create=True, delete_existing=False
            ).result()
            created_any = True

        if created_any:
            KnossosDataset._sort_precomputed_info_scales(layer)

    @staticmethod
    def initialize(path, experiment_name, boundary, cube_shape, scale, ds_factor=None, file_extensions=['.png'], description = '', channel='', parent_dataset=None, server_format="precomputed", as_rgb: bool = False, shard_size: Optional[Sequence[int]] = None, dtype = None):
        assert server_format == "precomputed" or not as_rgb, "as_rgb=True is only supported for server_format=precomputed"
        assert server_format == "precomputed" or shard_size is None, "shard_size is only supported for server_format=precomputed"
        conf_path = Path(path) / channel / f'{experiment_name}.k.toml'
        if parent_dataset is None and conf_path.exists():
            raise ValueError(f"Cannot initialize dataset at {conf_path}. File already exists.")
        layers = []
        layer = KnossosDataset()
        layer._conf_path = str(conf_path)
        layer._knossos_path = str(conf_path.parent)
        if channel != '':
            layer.url = f'file://{layer._knossos_path}/'
        layer._experiment_name = experiment_name
        layer.server_format = server_format
        layer._boundary = boundary
        layer._scale = scale
        layer._cube_shape = cube_shape
        layer.scales = layer.generate_scales(scale, ds_factor if ds_factor is not None else KnossosDataset._default_ds_factor(boundary))
        layer._ordinal_mags = True
        layer.description = description
        layer.file_extensions = []
        for ext in file_extensions:
            if not ext.startswith('.'):
                ext = f'.{ext}'
            if ext.lower() not in {'.raw', '.png', '.jpg', '.jpeg', '.seg.sz.zip'}:
                raise ValueError(f'Invalid extension {ext}. Supported extensions: .raw, .png, .jpg, .jpeg, .seg.sz.zip')
            layer.file_extensions.append(ext)
        layer._dtype = _normalize_dtype(dtype) if dtype is not None else (np.uint64 if ".seg.sz.zip" in file_extensions else np.uint8)
        layer.layers = [layer]
        layer._initialize_cache(0)
        layer._initialized = True
        layers = [layer]

        if server_format == "precomputed":
            # Defer info/tensorstore creation until the first write (on-demand).
            layer._tensorstore_datasets = {}
            if shard_size is not None:
                assert int(shard_size[0]) == int(shard_size[1]), "shard_size must be equal in x and y"
                assert int(shard_size[2]) <= int(shard_size[0]), "shard_size z must be <= x/y"
                layer._shard_size = np.asarray(shard_size, dtype=int)
            if as_rgb:
                highest_rgb_channel = 0
                if parent_dataset:
                    rgb_numbers = []
                    for lyr in parent_dataset.layers:
                        val = getattr(lyr, "_rgb_channel", None)
                        if isinstance(val, str) and re.match(r'^[rgb]_(\d+)$', val):
                            num = int(val.split("_")[1])
                            rgb_numbers.append(num)
                    highest_rgb_channel = max(rgb_numbers, default=0)
                layer._rgb_channel = f"r_{highest_rgb_channel + 1}"
                for channel in ["g", "b"]:
                    layer_rgb = layer._copy_configuration(layer)
                    layer_rgb._rgb_channel = f"{channel}_{highest_rgb_channel + 1}"
                    layers.append(layer_rgb)

        if parent_dataset:
            d = parent_dataset
            d.layers.extend(layers)
        else:
            d = KnossosDataset()
            d.__dict__.update(layer.__dict__)
            d._conf_path = str(Path(path) / f'{experiment_name}.k.toml')
            d._knossos_path = str(Path(d._conf_path).parent)
            d.layers = layers
        Path(d._conf_path).parent.mkdir(exist_ok=True, parents=True)
        d.save_toml(d._conf_path)
        return d

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
        print('DEPRECATION warning: initialize_without_conf is deprecated. Please use initialize. This will not generate neuroglancer datasets.')
        self._knossos_path = path
        all_mag_folders = our_glob(path+"/*mag*")

        if not mags is None and make_mag_folders:
            for mag in mags:
                exists = False
                for mag_folder in all_mag_folders:
                    if mag_folder.endswith(f'mag{mag}'):
                        exists = True
                        break
                if not exists:
                    if len(all_mag_folders) > 0:
                        assert(not re.match(r'.*mag\d+$', all_mag_folders[0]) is None)
                        os.makedirs(re.sub(r'mag\d+$', f'mag{mag}', all_mag_folders[0]))
                    else:
                        os.makedirs(path+"/mag"+str(mag))
        else:
            assert(len(all_mag_folders) > 0)

        mag_folder = our_glob(path+"/*mag*")[0].split("/")
        if len(mag_folder[-1]) > 1:
            mag_folder = mag_folder[-1]
        else:
            mag_folder = mag_folder[-2]

        self._name_mag_folder = \
            mag_folder[:-len(re.findall(r"[\d]+", mag_folder)[-1])]

        self._scale = scale
        self._boundary = boundary
        self._experiment_name = experiment_name

        self._number_of_cubes = np.array(np.ceil(
            np.array(self.boundary).astype(float) / self.cube_shape), dtype=int)

        if create_knossos_conf:
            for mag_folder in our_glob(path + '/*mag*'): # need (empty) knossos.conf files for mag discovery when streaming
                open(mag_folder + '/knossos.conf', 'a').close()
            # create base conf in dataset root
            self._conf_path = self.knossos_path + f'/{experiment_name}.k.conf'
            with open(self.conf_path, 'w') as f:
                f.write(f'experiment name {experiment_name};\n')
                f.write('boundary x %d;\n' % boundary[0])
                f.write('boundary y %d;\n' % boundary[1])
                f.write('boundary z %d;\n' % boundary[2])
                f.write('scale x %.2f;\n' % scale[0])
                f.write('scale y %.2f;\n' % scale[1])
                f.write('scale z %.2f;\n' % scale[2])
                f.write('magnification 1;\n')

        if verbose:
            _print("Initialization finished successfully")

        self._initialize_cache(cache_size)

        self._initialized = True

    @staticmethod
    def initialize_from_array(data: np.ndarray, experiment_name: str, cube_shape: Sequence[int], scale: Sequence[int], ds_factor: Sequence[int], file_extensions: Sequence[str] = ['.png'], channels: Optional[Sequence[str]] = ('',), write_path: Optional[str] = None, parent_dataset: Optional[KnossosDataset] = None, server_format="precomputed", as_rgb: bool = False, shard_size: Optional[Sequence[int]] = None, dtype=None):
        if write_path and parent_dataset:
            raise ValueError(f"Specify either `write_path` (to create a new dataset) or `parent_dataset` (to add a layer to an existing dataset).")
        if parent_dataset and not parent_dataset.initialized:
            raise ValueError("Parent dataset must be initialized, see `KnossosDataset.initialize`.")

        write_path = os.path.abspath(write_path) if write_path else str(Path(parent_dataset._conf_path).parent)
        conf_path = f'{write_path}/{experiment_name}.k.toml'
        if not parent_dataset and Path(conf_path).exists():
            raise ValueError(f"Cannot initialize dataset at {conf_path}. File already exists.")

        dtype = _normalize_dtype(dtype or data.dtype)

        if as_rgb:
            if server_format != "precomputed":
                raise ValueError("as_rgb=True is only supported for server_format='precomputed'.")
            if channels not in (None, '', ('',), ['']):
                raise ValueError("as_rgb=True stores one RGB layer; do not pass separate channels.")
            if data.ndim != len(cube_shape) + 1 or data.shape[-1] != 3:
                raise ValueError(f'Cube shape: {cube_shape}, as_rgb=True. Expected data.shape == {(*cube_shape, 3)}, found actual shape {data.shape}.')

            boundary = data.shape[:-1][::-1]
            number_existing_layers = len(parent_dataset.layers) if parent_dataset else 0
            parent = KnossosDataset.initialize(
                write_path,
                experiment_name,
                boundary,
                cube_shape,
                scale,
                ds_factor,
                file_extensions,
                channel='',
                parent_dataset=parent_dataset,
                server_format=server_format,
                as_rgb=True,
                shard_size=shard_size,
                dtype=dtype,
            )
            layers = parent.layers[number_existing_layers:]
            for idx, layer in enumerate(layers):
                Path(layer._conf_path).parent.mkdir(exist_ok=True)
                layer.save_raw(data[..., idx], offset=(0, 0, 0), data_mag=1)
            return parent

        if len(channels) > 1 and (data.ndim < len(cube_shape) + 1 or data.shape[-1] != len(channels)):
            raise ValueError(f'Cube shape: {cube_shape}, channels: {channels}.  Expected data.shape == {(*cube_shape, len(channels))}, found actual shape {data.shape}.')

        if len(channels) == 1 and data.ndim == len(cube_shape):
            data = data[...,None]

        boundary = data.shape[:-1][::-1]
        parent = parent_dataset or None
        layers = []
        number_existing_layers = 0
        if parent_dataset:
            number_existing_layers = len(parent_dataset.layers)
        for channel in channels:
            ds = KnossosDataset.initialize(write_path, experiment_name, boundary, cube_shape, scale, ds_factor, file_extensions, channel=channel, parent_dataset=parent, server_format=server_format, shard_size=shard_size, dtype=dtype)
            if parent is None:
                parent = ds
        layers.extend(parent.layers[number_existing_layers:])
        for idx, layer in enumerate(layers):
            save_func = layer.save_seg if '.seg.sz.zip' in file_extensions else layer.save_raw
            Path(layer._conf_path).parent.mkdir(exist_ok=True)
            print(f"Saving layer {idx} with dtype {data[...,idx].dtype} to {layer._conf_path}")
            save_func(data[...,idx], offset=(0, 0, 0), data_mag=1)
        return parent


    def initialize_from_matrix(self, path, scale, experiment_name,
                               offset=None, boundary=None, fast_downsampling=True,
                               data=None, data_path=None, hdf5_names=None,
                               mags=None, verbose=False, cache_size=0):
        """
            Initializes the dataset with matrix
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
        print('DEPRECATION warning: initialize_from_matrix is deprecated. Please use initialize_from_array. This will not generate neuroglancer datasets.')

        if (data is None) and (data_path is None or hdf5_names is None):
            raise Exception("No data given")

        if data is None:
            data = load_from_h5py(data_path, hdf5_names, False)[0]

        if offset is None:
            offset = np.array([0, 0, 0], dtype=int)
        else:
            offset = np.array(offset, dtype=int)

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
                assert apply_func is None
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

                new_kd.from_matrix_to_cubes(offset=offset, mags=mag,
                                            data=overlay, datatype=np.uint32,
                                            nb_threads=1)
                return err

        if data_range:
            assert isinstance(data_range, list)
            assert len(data_range[0]) == 3
            assert len(data_range[1]) == 3
        else:
            data_range = [[0, 0, 0], self.boundary]

        if mags is None:
            mags = self.available_mags

        if isinstance(mags, int):
            mags = [mags]

        new_kd = KnossosDataset()
        new_kd.initialize_without_conf(path=path, boundary=self.boundary,
                                       scale=self.scale,
                                       experiment_name=self.experiment_name,
                                       mags=mags)

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

        vx_list = np.array(vx_list, dtype=int)
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


    def _load(self, offset, size, from_overlay, mag, ext, expand_area_to_mag=False, padding=0, datatype=None):
        """ Extracts a 3D matrix from the KNOSSOS-dataset NOTE: You should use one of the two wrappers below

        :param offset: 3 sequence of ints
            mag 1 coordinate of the corner closest to (0, 0, 0)
        :param size: 3 sequence of ints
            mag 1 size of requested data block
        :param from_overlay: bool
            loads overlay instead of raw cubes
        :param mag: int
            magnification of the requested data block
            Enlarges area to true voxels of mag in case offset and size don’t exist in that mag.
        :param ext: str
            File extension to load
        :param expand_area_to_mag: bool, int
            Enlarges area to true voxels of specified mag in case offset and size don’t exist in that mag.
            False: no expansion, True: expansion to ``mag``, int: expansion to ``expand_area_to_mag``
        :param padding: str or int
            Pad mode for matrix parts outside the dataset. See https://www.pydoc.io/pypi/numpy-1.9.3/autoapi/numpy/lib/arraypad/index.html?highlight=pad#numpy.lib.arraypad.pad
            When passing an it, will pad with that int in 'constant' mode
        :param datatype: numpy datatype
            typically: for mode 'raw' this is np.uint8, and for 'overlay' np.uint64
        :return: 3D numpy array or nothing
        """
        def _read_cube(cube_coord):
            out_start, out_end, incube_start, incube_end = self.get_intervals(offset, size, cube_coord)

            valid_values = False

            # check cache first
            values = self._cube_from_cache(cube_coord, from_overlay)
            from_cache = values is not None

            if not from_cache:
                filename = f'{self.experiment_name}_{self.name_mag_folder}{mag}_x{cube_coord[0]:04d}_y{cube_coord[1]:04d}_z{cube_coord[2]:04d}{ext}'
                path = f'{self.knossos_path}/{self.name_mag_folder}{mag}/x{cube_coord[0]:04d}/y{cube_coord[1]:04d}/z{cube_coord[2]:04d}/{filename}'

                if self.in_http_mode:
                    for tries in range(1, self.http_max_tries + 1):
                        try:
                            request = requests.get(path, auth=self.http_auth, params=self._cdn_token, timeout=60)
                            request.raise_for_status()
                            if not from_overlay:
                                if ext == '.raw':
                                    values = np.frombuffer(request.content, dtype=np.uint8).astype(datatype)
                                else:
                                    values = imageio.imread(request.content)
                            else:
                                with zipfile.ZipFile(BytesIO(request.content), 'r') as zf:
                                    snappy_cube = zf.read(zf.namelist()[0]) # seg.sz (without .zip)
                                    raw_cube = self.module_wide['snappy'].decompress(snappy_cube)
                                    values = np.frombuffer(raw_cube, dtype=np.uint64).astype(datatype)
                            try:# check if requested values match shape
                                values.reshape(self.cube_shape[::-1])
                                valid_values = True
                                break
                            except ValueError:
                                self._print(f'Reshape error encountered for {1 + tries} time. ({path}). Content length: {len(request.content)}')
                                time.sleep(random.uniform(0.1, 1.0))
                                if tries == self.http_max_tries:
                                    raise Exception(f'Reshape errors exceed http_max_tries ({self.http_max_tries}).')
                        except requests.exceptions.RequestException as e:
                            if isinstance(e, requests.exceptions.ConnectionError) and tries < self.http_max_tries:
                                time.sleep(random.uniform(0.1, 1.0))
                                continue
                            return e
                        self._print(f'[{path}] Error occured ({tries}/{self.http_max_tries})')
                    if not valid_values:
                        raise Exception(f'Max. #tries reached. ({self.http_max_tries})')
                else:
                    if os.path.exists(path):
                        try:
                            if from_overlay:
                                with zipfile.ZipFile(path, 'r') as zf:
                                    snappy_cube = zf.read(zf.namelist()[0]) # seg.sz (without .zip)
                                raw_cube = self.module_wide['snappy'].decompress(snappy_cube)
                                values = np.frombuffer(raw_cube, dtype=np.uint64).astype(datatype)
                            elif ext == '.raw':
                                flat_shape = int(np.prod(self.cube_shape))
                                values = np.fromfile(path, dtype=np.uint8, count=flat_shape).astype(datatype)
                            else: # compressed
                                values = imageio.imread(path)
                            valid_values = True
                        except Exception as e:
                            print(f'Reading cube failed: {path}')
                            raise e
                    elif self.is_embedded:
                        kzip_path = self._embedded_kzip_path()
                        embedded_path = f'embedded/{self.name_mag_folder}{mag}/x{cube_coord[0]:04d}/y{cube_coord[1]:04d}/z{cube_coord[2]:04d}/{filename}'
                        try:
                            if from_overlay:
                                with zipfile.ZipFile(kzip_path, "r") as archive:
                                    with archive.open(embedded_path, "r") as inner_zip:
                                        zf = zipfile.ZipFile(inner_zip)
                                        snappy_cube = zf.read(zf.namelist()[0]) # seg.sz (without .zip)
                                raw_cube = self.module_wide['snappy'].decompress(snappy_cube)
                                values = np.frombuffer(raw_cube, dtype=np.uint64).astype(datatype)
                            elif ext == '.raw':
                                flat_shape = int(np.prod(self.cube_shape))
                                with zipfile.ZipFile(kzip_path, "r") as archive:
                                    with archive.open(embedded_path) as file:
                                        values = np.frombuffer(file.read(), dtype=np.uint8, count=flat_shape).astype(datatype)
                            else: # compressed
                                with zipfile.ZipFile(kzip_path, "r") as archive:
                                    with archive.open(embedded_path) as file:
                                        values = imageio.imread(file)
                            valid_values = True
                        except KeyError:
                            self._print(f"Cube »{path}« does not exist, cube with zeros only assigned")
                        except Exception as e:
                            print(f'Reading cube failed: {path}')
                            raise e
                    else:
                        self._print(f'Cube »{path}« does not exist, cube with zeros only assigned')

            if valid_values:
                values = values.reshape(self.cube_shape[::-1])
                if not from_cache:
                    self._add_to_cube_cache(cube_coord, from_overlay, values)
                output[out_start[2]:out_end[2], out_start[1]:out_end[1], out_start[0]:out_end[0]] \
                    = values[incube_start[2]:incube_end[2], incube_start[1]:incube_end[1], incube_start[0]:incube_end[0]]

        t0 = time.time()

        assert self.initialized, 'Dataset is not initialized'

        if len(self.available_mags) == 0:
            warnings.warn(f'Dataset has no available mags or mags could not be determined')
        elif mag not in self.available_mags:
            raise Exception(f'Requested mag {mag} not available, only mags {self.available_mags} are available.')

        if 0 in size:
            raise Exception(f'The second parameter is size! - at least one dimension was set to 0 ({size})')

        ratio = self.scale_ratio(mag, 1)
        if expand_area_to_mag:
            if expand_area_to_mag is True:
                expand_area_to_mag = mag
            expand_ratio = self.scale_ratio(expand_area_to_mag, 1)
            # mag1 coords rounded such that when converting back from target mag to mag1 the specified offset and size can be extracted.
            # i.e. for higher mags the matrix will be larger rather than smaller
            boundary = np.ceil(np.array(self.boundary, dtype=int) / expand_ratio).astype(int)
            end = np.ceil(np.add(offset, size) / expand_ratio) * expand_ratio
            offset = np.floor(np.array(offset, dtype=int) / expand_ratio) * expand_ratio
            # offset and size in target mag
            size = ((end - offset) // ratio).astype(int)
            offset = (offset // ratio).astype(int)
        else:
            size = (np.array(size, dtype=int) // ratio).astype(int)
            offset = (np.array(offset, dtype=int) // ratio).astype(int)
            boundary = (np.array(self.boundary, dtype=int) // ratio).astype(int)
        orig_size = np.copy(size)

        mirror_overlap = [[0, 0], [0, 0], [0, 0]]

        for dim in range(3):
            if offset[dim] < 0:
                size[dim] += offset[dim]
                mirror_overlap[dim][0] = -offset[dim]
                offset[dim] = 0

            if offset[dim] + size[dim] > boundary[dim]:
                mirror_overlap[dim][1] = offset[dim] + size[dim] - boundary[dim]
                size[dim] = boundary[dim] - offset[dim]

            if size[dim] < 0:
                raise Exception("Given block is totally out of bounds with "
                                "offset: [%d, %d, %d]!" %
                                (offset[0], offset[1], offset[2]))

        output = np.zeros(size[::-1], dtype=datatype)

        if self.server_format == "precomputed" and not self.is_embedded:
            channel = 0
            if self._rgb_channel:
                if self._rgb_channel.startswith("r_"):
                    channel = 0
                elif self._rgb_channel.startswith("g_"):
                    channel = 1
                elif self._rgb_channel.startswith("b_"):
                    channel = 2
            dataset = self._ensure_precomputed_mag(mag, create=False)
            data = np.array(dataset[offset[0]:offset[0]+size[0], offset[1]:offset[1]+size[1], offset[2]:offset[2]+size[2], channel])
            output = data.swapaxes(0, 2)
        else:
            start = self.get_first_blocks(offset).astype(int)
            end = self.get_last_blocks(offset, size).astype(int)

            nb_cubes_to_process = int(np.prod(end - start))
            if nb_cubes_to_process == 0:
                return np.zeros(orig_size[::-1], dtype=datatype)

            cube_coordinates = []

            for z in range(start[2], end[2]):
                for y in range(start[1], end[1]):
                    for x in range(start[0], end[0]):
                        cube_coordinates.append(np.array([x, y, z]))

            with ThreadPoolExecutor() as pool:
                results = list(pool.map(_read_cube, cube_coordinates)) # convert generator to list so we can count

            if results.count(None) < len(results):
                errors = defaultdict(int)
                for result in results: # None results are no error
                    if result is not None and result.response is not None: # errors with server response
                        errors[result.response.status_code] += 1
                    elif result is not None: # errors without server response
                        errors[result.__class__.__name__] += 1
                self._print(f'{len(errors)} non-ok http responses: {list(errors.items())}')

        if self.show_progress:
            dt = time.time() - t0
            speed = np.prod(output.shape) * 1.0/1000000/dt
            print(f'\rSpeed: {speed:.2f} Mvx/s, time {dt}')

        if not np.all(output.shape == size[::-1]):
            raise Exception(f'Incorrect shape! Should be {size[::-1]}; got {output.shape}')

        if np.any(mirror_overlap):
            if isinstance(padding, int):
                output = np.pad(output, mirror_overlap[::-1], 'constant', constant_values=padding)
            else:
                output = np.pad(output, mirror_overlap[::-1], mode=padding)

        return output

    def preferred_raw_layer(self):
        # legacy
        preferred_raw_layer = None; ext = None
        for layer in self.layers:
            ext = layer.preferred_raw_extension()
            if ext == '.raw' or ext == '.png':
                preferred_raw_layer = layer
                break
            if ext != '.seg.sz.zip':
                preferred_raw_layer = layer
        return preferred_raw_layer, ext

    def preferred_raw_extension(self):
        # preference raw → png → jpg
        preferred_raw_extension = None
        for ext in self.file_extensions:
            if ext == '.raw' or ext == '.png':
                preferred_raw_extension = ext
                break
            preferred_raw_extension = ext
        return preferred_raw_extension

    def load_raw(self, **kwargs):
        """
        :param offset: 3 sequence of ints
            mag 1 coordinate of the corner closest to (0, 0, 0)
        :param size: 3 sequence of ints
            mag 1 size of requested data block
        :param mag: int
            magnification of the requested data block
            Enlarges area to true voxels of mag in case offset and size don’t exist in that mag.
        :param expand_area_to_mag: bool
        :param padding: str or int
            Pad mode for matrix parts outside the dataset. See https://www.pydoc.io/pypi/numpy-1.9.3/autoapi/numpy/lib/arraypad/index.html?highlight=pad#numpy.lib.arraypad.pad
            When passing an it, will pad with that int in 'constant' mode
        :param datatype: numpy datatype
            default is np.uint8
        :return: 3D numpy array or nothing
        """
        assert 'from_overlay' not in kwargs, 'Don’t pass from_overlay, from_overlay is automatically set to False here.'
        kwargs.update({'from_overlay': False})
        if 'datatype' not in kwargs:
            kwargs.update({'datatype': np.uint8})

        preferred_raw_layer, ext = self.preferred_raw_layer()

        assert preferred_raw_layer is not None, 'Tried to load raw data, but the loaded dataset configuration contains no raw layer.'
        kwargs['ext'] = ext
        return preferred_raw_layer._load(**kwargs)

    def load_seg(self, **kwargs):
        """
        :param offset: 3 sequence of ints
            mag 1 coordinate of the corner closest to (0, 0, 0)
        :param size: 3 sequence of ints
            mag 1 size of requested data block
        :param mag: int
            magnification of the requested data block
            Enlarges area to true voxels of mag in case offset and size don’t exist in that mag.
        :param expand_area_to_mag: bool
        :param padding: str or int
            Pad mode for matrix parts outside the dataset. See https://www.pydoc.io/pypi/numpy-1.9.3/autoapi/numpy/lib/arraypad/index.html?highlight=pad#numpy.lib.arraypad.pad
            When passing an it, will pad with that int in 'constant' mode
        :param datatype: numpy datatype
            default is np.uint64
        :return: 3D numpy array or nothing
        """
        assert 'from_overlay' not in kwargs, 'Don’t pass from_overlay, from_overlay is automatically set to True here.'
        kwargs.update({'from_overlay': True})
        kwargs['ext'] = '.seg.sz.zip'
        if 'datatype' not in kwargs:
            kwargs.update({'datatype': np.uint64})

        for layer in self.layers: # prefer local seg
            if not layer.in_http_mode and '.seg.sz.zip' in layer.file_extensions:
                return layer._load(**kwargs)
        for layer in self.layers:
            if '.seg.sz.zip' in layer.file_extensions:
                return layer._load(**kwargs)
        raise Exception("Tried to load segmentation but the loaded dataset configuration contains no segmentation layer.")

    def from_cubes_to_matrix(self, size, offset, mode, mag=1, datatype=np.uint8,
                             mirror_oob=True, hdf5_path=None,
                             hdf5_name="raw", pickle_path=None,
                             invert_data=False, zyx_mode=False,
                             nb_threads=40, verbose=False, show_progress=True,
                             http_max_tries=2000, http_verbose=False):
        print('from_*cubes_to_matrix is DEPRECATED.\n Please use load_raw or load_seg.')
        self.verbose = verbose or http_verbose
        self.show_progress = show_progress
        self.http_max_tries = http_max_tries

        if zyx_mode:
            offset = offset[::-1]
            size = size[::-1]
        ratio = self.scale_ratio(mag, 1)
        size = (np.array(size) * ratio).astype(int)
        offset = (np.array(offset) * ratio).astype(int)

        from_overlay = mode == 'overlay'
        padding = 'symmetric' if mirror_oob else 0

        data = self._load(offset=offset, size=size, from_overlay=from_overlay, mag=mag, padding=padding, datatype=datatype)

        if invert_data:
            data = np.invert(data)

        if not zyx_mode:
            data = data.swapaxes(0, 2)

        if hdf5_path and hdf5_name:
            save_to_h5py(data, hdf5_path, hdf5_names=[hdf5_name])

        if pickle_path:
            save_to_pickle(data, pickle_path)

        return data

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

    def read_movement_area(self, kzip_path):
        try:
            with zipfile.ZipFile(kzip_path, "r") as zf:
                xml_str = zf.read('annotation.xml').decode()
            annotation_xml = ET.fromstring(xml_str)
            area_elem = annotation_xml.find("parameters/MovementArea")
            area_min = [0, 0, 0]
            area_size = np.copy(self.boundary)
            area_max = np.copy(self.boundary)
            size_exists = False
            for key, value in area_elem.items():
                if key == 'min.x':
                    area_min[0] = int(value)
                elif key == 'min.y':
                    area_min[1] = int(value)
                elif key == 'min.z':
                    area_min[2] = int(value)
                elif key == 'size.x':
                    size_exists = True
                    area_size[0] = int(value)
                elif key == 'size.y':
                    size_exists = True
                    area_size[1] = int(value)
                elif key == 'size.z':
                    size_exists = True
                    area_size[2] = int(value)
                elif key == 'max.x':
                    area_max[0] = int(value)
                elif key == 'max.y':
                    area_max[1] = int(value)
                elif key == 'max.z':
                    area_max[2] = int(value)
            if not size_exists:
                area_size = area_max - area_min
        except (KeyError, AttributeError):
            # KeyError: annotation.xml does not exist, AttributeError: xml elem does not exist
            return np.array([0, 0, 0]), self.boundary
        return (np.array(area_min), np.array(area_size))

    def get_movement_area(self, kzip_path):
        print('get_movement_area is DEPRECATED.\nPlease use read_movement_area. Instead of movement area min and max, it will return min and size.')
        area_min, area_size = self.read_movement_area(kzip_path)
        return area_min, area_min + area_size

    def load_kzip_seg(self, path, mag, return_area=False):
        area_min, area_size = self.read_movement_area(path)
        matrix = self._load_kzip_seg(path=path, offset=area_min, size=area_size, mag=mag)
        return (matrix, area_min, area_size) if return_area else matrix

    def from_kzip_to_matrix(self, path, size, offset, mag=8, empty_cube_label=0,
                            datatype=np.uint64,
                            verbose=False,
                            show_progress=True,
                            apply_mergelist=True,
                            binarize_overlay=False,
                            return_dataset_cube_if_nonexistent=False,
                            expand_area_to_mag=False):
        print('from_kzip_to_matrix is DEPRECATED.\n Please use load_kzip_seg.')
        self.verbose = verbose
        self.show_progress = show_progress
        self.background_label = empty_cube_label

        ratio = self.scale_ratio(mag, 1)
        size = (np.array(size) * ratio).astype(int)
        offset = (np.array(offset) * ratio).astype(int)

        data = self._load_kzip_seg(path, offset, size, mag, datatype, apply_mergelist, return_dataset_cube_if_nonexistent, expand_area_to_mag)

        if binarize_overlay:
            data[data > 1] = 1

        return data.swapaxes(0, 2)

    def _load_kzip_seg(self, path, offset, size, mag, datatype=np.uint64, padding=0, apply_mergelist=True, return_dataset_cube_if_nonexistent=False, expand_area_to_mag=False, kzip_experiment_name=None):
        """ Extracts a 3D matrix from a kzip file

        :param path: str
            forward-slash separated path to kzip file
        :param offset: 3 sequence of ints
            mag 1 coordinate of the corner closest to (0, 0, 0)
        :param size: 3 sequence of ints
            size of requested data block
        :param datatype: numpy datatype
            typically np.uint8
        :param apply_mergelist: bool
            True: Merges IDs based on the kzip mergelist
        :param expand_area_to_mag: bool, int
            Enlarges area to true voxels of specified mag in case offset and size don’t exist in that mag.
            False: no expansion, True: expansion to ``mag``, int: expansion to ``expand_area_to_mag``
        :param return_empty_cube_if_nonexistent: bool
            True: if kzip doesn't contain specified cube,
            an empty cube (cube filled with empty_cube_label) is returned.
            False: returns None instead.
        :return: 3D numpy array
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if not self.module_wide["snappy"]:
            raise Exception("Snappy is not available - you cannot read "
                            "overlaycubes or kzips.")
        ratio = self.scale_ratio(mag, 1)
        if expand_area_to_mag:
            if expand_area_to_mag is True:
                expand_area_to_mag = mag
            expand_ratio = self.scale_ratio(expand_area_to_mag, 1)
            end = np.ceil(np.add(offset, size) / expand_ratio) * expand_ratio
            offset = np.floor(np.array(offset, dtype=int) / expand_ratio) * expand_ratio
            size = (end - offset) // ratio
            offset = offset // ratio
        else:
            size = np.array(size, dtype=int) // ratio
            offset = np.array(offset, dtype=int) // ratio
        offset = offset.astype(np.int64)
        size = size.astype(np.int64)

        start = np.array([get_first_block(dim, offset, self._cube_shape)
                          for dim in range(3)])
        end = np.array([get_last_block(dim, size, offset, self._cube_shape) + 1
                        for dim in range(3)])

        output = np.zeros(size[::-1], dtype=datatype)

        offset_start = offset % self.cube_shape
        offset_end = (self.cube_shape - (offset + size) % self.cube_shape) % self.cube_shape

        current = np.array([start[dim] for dim in range(3)])
        cnt = 1
        nb_cubes_to_process = (end - start).prod()
        experiment_name = kzip_experiment_name or self.experiment_name
        with zipfile.ZipFile(path, 'r') as archive:
            archive_names = archive.namelist()
            for file in archive_names:
                if file.endswith('.seg.sz'):
                    match = re.search(r'_mag\d+x\d+y\d+z\d+.seg.sz', file)
                    if match is None:
                        warnings.warn(f'{path}: found seg cube with invalid name: {file}')
                    else:
                        experiment_name = file[0:match.span()[0]]
                        break
            for z in range(start[2], end[2]):
                for y in range(start[1], end[1]):
                    for x in range(start[0], end[0]):
                        current = np.array([x, y, z])
                        if self.show_progress:
                            progress = 100*cnt/float(nb_cubes_to_process)
                            _stdout(f'\rProgress: {progress:.2f}% ')

                        this_path = f'{experiment_name}_mag{mag}x{x}y{y}z{z}.seg.sz'
                        try:
                            self._print(f'{current}: loading from .k.zip')
                            scube = archive.read(this_path)
                            values = np.frombuffer(module_wide["snappy"].decompress(scube), dtype=np.uint64)
                        except KeyError:
                            self._print(f'{current}: {"dataset" if return_dataset_cube_if_nonexistent else self.background_label} cube assigned')
                            if return_dataset_cube_if_nonexistent:
                                values = self.load_seg(offset=current * ratio * self.cube_shape, size=ratio * self.cube_shape, mag=mag,
                                                       datatype=datatype, padding=padding, expand_area_to_mag=expand_area_to_mag)
                            else:
                                values = np.full(self.cube_shape[::-1], self.background_label, dtype=datatype)

                        out_start, out_end, incube_start, incube_end = self.get_intervals(offset, size, current)
                        output[out_start[2]:out_end[2], out_start[1]:out_end[1], out_start[0]:out_end[0]] \
                            = values.reshape(self.cube_shape[::-1]).astype(datatype, copy=False) \
                                [incube_start[2]:incube_end[2], incube_start[1]:incube_end[1], incube_start[0]:incube_end[0]]

                        cnt += 1

            if self.show_progress and not self.verbose:
                print() # newline after sys.stdout.writes inside loop

            if apply_mergelist:
                if "mergelist.txt" not in archive.namelist():
                    self._print("no mergelist to apply")
                else:
                    self._print("applying mergelist now")
                    mergelist_tools.apply_mergelist(output, archive.read("mergelist.txt").decode())

        return output

    def set_experiment_name_for_kzip(self, kzip_path):
        with tempfile.TemporaryDirectory() as tempdir_path:
            with zipfile.ZipFile(kzip_path, 'r') as original_kzip:
                original_kzip.extractall(tempdir_path)
            tempdir_path = Path(tempdir_path)
            with zipfile.ZipFile(kzip_path, 'w', zipfile.ZIP_DEFLATED) as new_kzip:
                for member in tempdir_path.iterdir():
                    if member.name == 'annotation.xml':
                        tree = ET.parse(member)
                        experiment = tree.find('parameters/experiment')
                        experiment.attrib['name'] = self.experiment_name
                        tree.write(member)
                    hit = re.search('_mag[0-9]+x[0-9]+y[0-9]+z[0-9]+.seg.sz', member.name)
                    new_path = member
                    if hit:
                        new_path = member.parent / (self.experiment_name + member.name[hit.span()[0]:])
                        member.rename(new_path)
                    new_kzip.write(new_path, new_path.name)

    def downsample_upsample_kzip_cubes(self, kzip_path, source_mag, out_mags=None, upsample=True, downsample=True, dest_path=None, chunk_size=None):
        from knossos_utils import skeleton as k_skel
        if dest_path is None:
            dest_path = kzip_path
        if out_mags is None:
            out_mags = []
        area_min, area_size = self.read_movement_area(str(kzip_path))
        if chunk_size is None:
            mat = self._load_kzip_seg(str(kzip_path), offset=area_min, size=area_size, mag=source_mag, apply_mergelist=False)
        else:
            for offset in self.iter(area_min, area_min + area_size, chunk_size):
                mat = self._load_kzip_seg(path=str(kzip_path), offset=offset, size=chunk_size, mag=source_mag, apply_mergelist=False)
                self.save_to_kzip(offset=offset, data=mat, data_mag=source_mag, kzip_path=dest_path, gen_mergelist=True,
                                  mags=out_mags, downsample=downsample, upsample=upsample, compress_kzip=False)
            area_min = offset
        skel = k_skel.Skeleton()
        mag_limit = 1
        if len(out_mags) > 0:
            mag_limit = np.log2(max(out_mags)) if not self._ordinal_mags else max(out_mags)
        elif downsample:
            mag_limit = self.highest_mag
        skel.movement_area_min = np.array(area_min) + (mag_limit - np.array(area_min) % mag_limit)
        area_max = area_min + area_size
        area_max = np.maximum(area_max - np.array(area_max) % mag_limit, skel.movement_area_min + 1)
        skel.movement_area_size = area_max - skel.movement_area_min
        skel.set_scaling(self.scales[0])
        skel.experiment_name = self.experiment_name
        annotation_str = skel.to_xml_string()
        self.save_to_kzip(offset=area_min, data=mat, data_mag=source_mag, kzip_path=dest_path, mags=out_mags, gen_mergelist=True,
                          downsample=downsample, upsample=upsample, annotation_str=annotation_str)

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

        for curr_z_cube in range(0, int(np.ceil(self._number_of_cubes[2]) / float(mag))):
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

            layer = layer.astype(out_dtype)

            for curr_z_coord in range(0, self._cube_shape[2]):
                if (z_coord_cnt >= self.boundary[2]):
                    break;

                file_path = os.path.join(out_path, "{0}_{1:06d}.{2}".format(mode, z_coord_cnt, out_format))

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
                    with open(file_path, 'wb') as fp:
                        if out_format == 'tif' or out_format == 'tiff':
                            img.save(fp, compression='tiff_lzw')
                        else:
                            img.save(fp)
                else:
                    swapped.tofile(file_path)

                _print("Writing layer {0} of {1} in total.".format(
                    z_coord_cnt+1, self.boundary[2]//mag))

                z_coord_cnt += 1

        return

    def save_cube(self, cube_path, data, overwrite_offset=None, overwrite_limit=None):
        """
        Helper function for from_matrix_to_cubes. Can also be used independently to overwrite individual cubes.
        Expects data, offset and limit in xyz and data.shape == self.cube_shape.
        :param cube_path: absolute path to destination cube (*.seg.sz.zip, *.seg.sz, *.raw, *.[ending known by imageio.imread])
        :param data: data to be written to the cube
        :param overwrite_offset: overwrite area offset. Defaults to (0, 0, 0) if overwrite_limit is set.
        :param overwrite_limit: overwrite area offset. Defaults to self.cube_shape if overwrite_offset is set.
        """
        assert np.array_equal(data.shape, self.cube_shape[::-1]), 'Can only save cubes of shape self.cube_shape ({}). found shape {}'.format(self.cube_shape[::-1], data.shape)
        dest_cube = data
        if os.path.isfile(cube_path):
            # read
            try:
                if cube_path.endswith('.seg.sz.zip'):
                    with zipfile.ZipFile(cube_path, "r") as zf:
                        in_zip_name = os.path.basename(cube_path)[:-4]
                        dest_cube = np.frombuffer(self.module_wide["snappy"].decompress(zf.read(in_zip_name)), dtype=np.uint64)
                elif cube_path.endswith('.seg.sz'):
                    with open(cube_path, "rb") as existing_file:
                        dest_cube = np.frombuffer(self.module_wide["snappy"].decompress(existing_file.read()), dtype=np.uint64)
                elif cube_path.endswith('.raw'):
                    dest_cube = np.fromfile(cube_path, dtype=np.uint8)
                else: # png or jpg
                    dest_cube = imageio.imread(cube_path)
            except Exception as e:
                print(f'Cube is broken and will be overwritten: {cube_path}')
            dest_cube = dest_cube.reshape(self.cube_shape[::-1])
            dest_cube = dest_cube.astype(data.dtype)
            if overwrite_offset is not None or overwrite_limit is not None:
                overwrite_offset = overwrite_offset if overwrite_offset is not None else (0, 0, 0)
                overwrite_limit = overwrite_limit if overwrite_offset is not None else self.cube_shape
                dest_cube[overwrite_offset[2]: overwrite_limit[2],
                          overwrite_offset[1]: overwrite_limit[1],
                          overwrite_offset[0]: overwrite_limit[0]] = data[overwrite_offset[2]: overwrite_limit[2],
                                                                          overwrite_offset[1]: overwrite_limit[1],
                                                                          overwrite_offset[0]: overwrite_limit[0]]
            else:
                indices = np.where(data != 0)
                dest_cube[indices] = data[indices]
        # write
        if self.write_empty_cubes or np.any(dest_cube):
            dest_cube = dest_cube.reshape(np.prod(dest_cube.shape))
            if cube_path.endswith('.seg.sz.zip'):
                in_zip_name = os.path.basename(cube_path)[:-4]
                with zipfile.ZipFile(cube_path, "w") as zf:
                    zf.writestr(in_zip_name, self.module_wide["snappy"].compress(dest_cube.astype(np.uint64)), compress_type=zipfile.ZIP_DEFLATED)
            elif cube_path.endswith('.seg.sz'):
                with open(cube_path, "wb") as dest_file:
                    dest_file.write(self.module_wide["snappy"].compress(dest_cube.astype(np.uint64)))
            elif cube_path.endswith('.raw'):
                with open(cube_path, "wb") as dest_file:
                    dest_file.write(dest_cube.astype(np.uint8))
            else:  # png or jpg
                imageio.imwrite(cube_path, dest_cube.reshape(self._cube_shape[2] * self._cube_shape[1], self._cube_shape[0]))
        elif (overwrite_offset is not None or overwrite_limit is not None) and os.path.exists(cube_path):
            os.remove(cube_path)

    def from_matrix_to_cubes(self, offset, mags=[], data=None, data_mag=1,
                             data_path=None, hdf5_names=None,
                             datatype=np.uint64, fast_downsampling=True,
                             force_unique_labels=False, verbose=True,
                             overwrite='area', kzip_path=None, compress_kzip=True,
                             annotation_str=None, as_raw=False, nb_threads=20,
                             upsample=True, downsample=True, gen_mergelist=True):
        """ Cubes data for viewing and editing in KNOSSOS
            one can choose from
                a) (Over-)writing overlay cubes in the dataset
                b) Writing a kzip which can be loaded in KNOSSOS
                c) (Over-)writing raw cubes
        :param compress_kzip: bool
            If kzip_path selected, indicates if tmp output folder should be
            compressed to the kzip. For multiple calls to this function with
            same kzip target, it makes sense to only compress in the last call.
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
            unsupported
        :param verbose: bool
            True: prints several information
        :param overwrite: True (overwrites all values within offset and offset+data.shape)
                         | False (preserves original cube values at 0-locations of new data)
        :param kzip_path: str
            is not None: overlay data is written as kzip to this path
        :param annotation_str: str
            is not None: if writing to k.zip, include this as annotation.xml
        :param as_raw: bool
            True: outputs data as normal KNOSSOS raw cubes
        :param gen_mergelist: bool
            True: generates a mergelist when writing into a kzip
        :param nb_threads: int
            if < 2: no multithreading
        :return:
            nothing
        """
        print('from_matrix_to_cubes is DEPRECATED.\n Please use save_raw or save_seg instead.')
        if data_path is not None:
            if '.h5' in data_path:
                assert hdf5_names is not None, 'No hdf5 names given to read hdf5 file.'
                data = load_from_h5py(data_path, list(hdf5_names))
            elif '.pkl' in data_path:
                data = load_from_pickle(data_path)
            else:
                raise Exception("File has to be .h5 pr .pkl")

        assert data is not None
        if len(data) == 0:
            raise Exception("No data or path given!")

        data = np.array(data)
        data = np.swapaxes(data, 0, 2)
        assert not force_unique_labels, 'force_unique_labels unsupported'

        if kzip_path:
            if compress_kzip:
                self.save_to_kzip(data, data_mag, kzip_path, offset, mags, gen_mergelist, annotation_str)
            else:
                self.save_to_kzip_path_only(data, data_mag, kzip_path, offset, mags, gen_mergelist, annotation_str)
        else:
            self._save(data, data_mag, offset, mags, as_raw, None, upsample, downsample, fast_downsampling)

    def _save(self, data, data_mag, offset, mags, as_raw, kzip_path, upsample, downsample, fast_resampling, datatype=None):
        if datatype is not None:
            datatype = np.dtype(datatype)
        else:
            datatype = data.dtype

        if (as_raw and datatype not in (np.dtype(np.uint8), np.dtype(np.uint16))) or (not as_raw and datatype != np.dtype(np.uint64)):
            raise ValueError('Currently, saving only accepts destination datatypes np.uint8 or np.uint16 (raw) or np.uint64 (segmentation).')
        if as_raw and datatype == np.dtype(np.uint16) and self.server_format != "precomputed":
            raise ValueError('uint16 raw data is only supported for precomputed Tensorstore datasets; classic KNOSSOS raw cubes only support uint8.')

        if as_raw and datatype != self._dtype:
            if datatype == np.dtype(np.uint16) and self._dtype == np.dtype(np.uint8):
                raise ValueError(f'Can not save uint16 raw data to initialized with uint8 raw data.')
            else:
                warnings.warn(f'Data type mismatch: will convert data(dtype={datatype}) to {self._dtype}.')
                datatype = self._dtype
        if not as_raw and datatype != np.dtype(np.uint64):
            raise ValueError(f'Segmentation dataset expects np.uint64 data, got requested datatype {datatype}.')
        overwrite=True

        def _write_cubes(args):
            """ Helper function for multithreading """
            folder_path, path, file_extensions, cube_offset, cube_limit, start, end = args

            cube = np.zeros(self.cube_shape[::-1], dtype=datatype)
            cube[cube_offset[2]: cube_limit[2],
                 cube_offset[1]: cube_limit[1],
                 cube_offset[0]: cube_limit[0]] = data_inter[start[2]: start[2] + end[2],
                                                             start[1]: start[1] + end[1],
                                                             start[0]: start[0] + end[0]]


            nothing_to_write = not self.write_empty_cubes and not np.any(cube)
            if nothing_to_write:
               self._print(path, 'no data to write, cube will be removed if present')

            if not kzip_path and not nothing_to_write:
                while True:
                    try:
                        os.makedirs(folder_path, exist_ok=True)
                        break
                    except PermissionError: # sometimes happens via sshfs with multiple workers
                        print('Permission error while creating cube folder. Sleeping on', folder_path)
                        time.sleep(random.uniform(0.1, 1.0))
                        pass

            block_path = f'{path}-block'
            while self.reentrant:
                try:
                    os.makedirs(block_path)    # file lock -------------
                    break
                except (FileExistsError, PermissionError):
                    try:
                        tdelta = time.time() - filesystem_process_time_diff - os.stat(block_path).st_mtime
                        if tdelta <= 30:
                            time.sleep(random.uniform(0.1, 1.0)) # wait for other workers to finish
                        else:
                            print(f'had to remove block folder {block_path} that wasn’t accessed recently {tdelta}')
                            os.rmdir(block_path)
                    except FileNotFoundError:
                        pass # folder was removed by another worker in the meantime
            for ext in file_extensions:
                self.save_cube(cube_path=f'{path}{ext}' if as_raw or kzip_path else f'{path}{ext}.zip', data=cube,
                                overwrite_offset=cube_offset if overwrite else None,
                                overwrite_limit=cube_limit if overwrite else None)

            if self.reentrant:
                try:
                    os.rmdir(block_path)   # ------------------------------
                except FileNotFoundError:
                    print(f'another worker removed our semaphore {block_path}')
                    pass

        # Main Function
        assert self.initialized, 'Dataset is not initialized'
        assert as_raw or self.module_wide["snappy"], 'Snappy is not available - you cannot write overlaycubes or kzips.'
        mags = list(mags)

        if not mags:
            start_mag = 1 if upsample else data_mag
            end_mag = self.highest_mag if downsample else data_mag
            if self._ordinal_mags:
                mags = np.arange(start_mag, end_mag, dtype=int)
            else: # power of 2 mags (KNOSSOS style)
                mags = np.power(2, np.arange(np.log2(start_mag), np.log2(end_mag), dtype=int))
        self._print(f'mags to write: {mags}')

        if kzip_path is not None:
            kzip_path = str(kzip_path)
            assert not as_raw, 'You have to choose between kzip and raw cubes'
            if kzip_path.endswith(".k.zip"):
                kzip_path = kzip_path[:-6]
            os.makedirs(kzip_path, exist_ok=True)

        if self.reentrant:
            # obtain clock difference between write destination and process system for correct block file age determination
            time_file_dir = kzip_path if kzip_path else None
            if time_file_dir is None and self._conf_path is not None:
                time_file_dir = os.path.dirname(self._conf_path)
            if time_file_dir is not None:
                with tempfile.NamedTemporaryFile(dir=time_file_dir) as time_file:
                    filesystem_process_time_diff = time.time() - os.stat(time_file.name).st_mtime
            else:
                filesystem_process_time_diff = 0

        for mag in mags:
            ratio = self.scale_ratio(mag, data_mag)[::-1]
            inv_mag_ratio = 1.0/np.array(ratio)
            fast = fast_resampling or (not as_raw and mag > data_mag)
            if fast and all(mag_ratio.is_integer() for mag_ratio in ratio):
                data_inter = np.array(data[::int(ratio[0]), ::int(ratio[1]), ::int(ratio[2])])
            elif all(mag_ratio == 1 for mag_ratio in ratio):
                data_inter = data
            elif fast:
                data_inter = scipy.ndimage.zoom(data, inv_mag_ratio, order=0).astype(datatype, copy=False)
            elif as_raw:
                quality = 3 if mag > data_mag else 1
                data_inter = scipy.ndimage.zoom(data, inv_mag_ratio, order=quality).astype(datatype, copy=False)
            else: # fancy seg upsampling
                data_inter = np.zeros(shape=(inv_mag_ratio * np.array(data.shape)).astype(int), dtype=datatype)
                for value in np.unique(data):
                    if value == 0: continue # no 0 upsampling
                    up_chunk_channel = scipy.ndimage.zoom((data == value).astype(np.uint8), inv_mag_ratio, order=1)
                    data_inter += (up_chunk_channel * value).astype(datatype, copy=False)

            offset_mag = np.array(offset, dtype=int) // self.scale_ratio(mag, 1)
            size_mag = np.array(data_inter.shape[::-1], dtype=int)

            self._print(f'mag: {mag}')
            self._print(f'box_offset: {offset_mag}')
            self._print(f'box_size: {size_mag}')

            if self.server_format == "precomputed" and kzip_path is None:
                channel = 0
                if self._rgb_channel:
                    if self._rgb_channel.startswith("r_"):
                        channel = 0
                    elif self._rgb_channel.startswith("g_"):
                        channel = 1
                    elif self._rgb_channel.startswith("b_"):
                        channel = 2
                dataset = self._ensure_precomputed_mag(mag, create=True)
                dataset[int(offset_mag[0]):int(offset_mag[0]+size_mag[0]), int(offset_mag[1]):int(offset_mag[1]+size_mag[1]), int(offset_mag[2]):int(offset_mag[2]+size_mag[2]), channel] = data_inter.swapaxes(0,2).astype(datatype)
            else:
                start = np.array([get_first_block(dim, offset_mag, self._cube_shape) for dim in range(3)])
                end = np.array([get_last_block(dim, size_mag, offset_mag, self._cube_shape) + 1 for dim in range(3)])

                self._print(f'start_cube: {start}')
                self._print(f'end_cube: {end}')

                multithreading_params = []

                conf_folder = os.path.dirname(self._conf_path)
                conf_folder_name = "/" + Path(conf_folder).name
                index = self.knossos_path.rfind(conf_folder_name)
                if index != -1:
                    conf_folder = Path(conf_folder) / self.knossos_path[index + len(conf_folder_name):]

                for z in range(start[2], end[2]):
                    for y in range(start[1], end[1]):
                        for x in range(start[0], end[0]):
                            current = np.array([x, y, z])

                            this_cube_info = []
                            path = f'{conf_folder}/{self.name_mag_folder}{mag}/x{current[0]:04d}/y{current[1]:04d}/z{current[2]:04d}/'
                            this_cube_info.append(path)

                            extensions = ['.seg.sz']
                            if kzip_path is None:
                                if as_raw:
                                    save_layer, _ = self.preferred_raw_layer()
                                    extensions = save_layer.file_extensions
                                else:
                                    save_layer = self
                                path += f'{save_layer.experiment_name}_{save_layer.name_mag_folder}{mag}_x{current[0]:04d}_y{current[1]:04d}_z{current[2]:04d}'
                            else:
                                path = f'{kzip_path}/{self._experiment_name}_{self.name_mag_folder}{mag}x{current[0]}y{current[1]}z{current[2]}'
                            this_cube_info.extend([path, extensions])
                            cube_coords = current * self.cube_shape
                            cube_offset = np.zeros(3)
                            cube_limit = np.ones(3) * self.cube_shape

                            for dim in range(3):
                                if cube_coords[dim] < offset_mag[dim]:
                                    cube_offset[dim] = offset_mag[dim] - cube_coords[dim]
                                if cube_coords[dim] + cube_limit[dim] > offset_mag[dim] + size_mag[dim]:
                                    cube_limit[dim] = offset_mag[dim] + size_mag[dim] - cube_coords[dim]

                            start_coord = cube_coords - offset_mag + cube_offset
                            end_coord = cube_limit - cube_offset

                            this_cube_info.append(cube_offset.astype(int))
                            this_cube_info.append(cube_limit.astype(int))
                            this_cube_info.append(start_coord.astype(int))
                            this_cube_info.append(end_coord.astype(int))

                            multithreading_params.append(this_cube_info)

                with ThreadPoolExecutor() as pool:
                    list(pool.map(_write_cubes, multithreading_params)) # convert generator to list to unsilence errors

    def save_raw(self, data, data_mag, offset, mags=[], upsample=True, downsample=True, fast_resampling=True, datatype=None):
        self._save(data=data, data_mag=data_mag, offset=offset, mags=mags, as_raw=True, kzip_path=None, upsample=upsample, downsample=downsample, fast_resampling=fast_resampling, datatype=datatype)

    def save_seg(self, data, data_mag, offset, mags=[], upsample=True, downsample=True, fast_resampling=True, datatype=None):
        self._save(data=data, data_mag=data_mag, offset=offset, mags=mags, as_raw=False, kzip_path=None, upsample=upsample, downsample=downsample, fast_resampling=fast_resampling, datatype=datatype)

    def save_to_kzip(self, data, data_mag, kzip_path, offset, mags=[], gen_mergelist=True, annotation_str=None, upsample=True, downsample=True, fast_resampling=True):
        kzip_path = str(kzip_path)
        kzip_dir_path = kzip_path[:-6] if kzip_path.endswith('.k.zip') else kzip_path
        assert not Path(kzip_dir_path).exists(), f'the folder used for kzip compression already exists: {kzip_dir_path}'
        self.save_to_kzip_path_only(data=data, data_mag=data_mag, kzip_path=kzip_path, offset=offset, mags=mags, gen_mergelist=gen_mergelist, annotation_str=annotation_str, upsample=upsample, downsample=downsample, fast_resampling=fast_resampling)
        self.compress_kzip(kzip_path=kzip_path)

    def save_to_kzip_path_only(self, data, data_mag, kzip_path, offset, mags=[], gen_mergelist=True, annotation_str=None, upsample=True, downsample=True, fast_resampling=True):
        kzip_path = str(kzip_path)
        if kzip_path.endswith('.k.zip'):
            kzip_path = kzip_path[:-6]
        self._save(data=data, data_mag=data_mag, offset=offset, mags=mags, as_raw=False, kzip_path=kzip_path, upsample=upsample, downsample=downsample, fast_resampling=fast_resampling)
        if gen_mergelist:
            with open(os.path.join(kzip_path, 'mergelist.txt'), 'w') as mergelist:
                start = time.time();
                mergelist.write(mergelist_tools.gen_mergelist_from_segmentation(data, offsets=np.array(offset, dtype=np.uint64), scale=self.scale_ratio(data_mag,1)))
                print('gen mergelist', time.time() - start)
        if annotation_str is not None:
            with open(os.path.join(kzip_path, 'annotation.xml'), 'w') as annotation:
                annotation.write(annotation_str)

    def compress_kzip(self, kzip_path):
        kzip_path = str(kzip_path)
        while kzip_path.endswith('/'):
            kzip_path = kzip_path[:-1]
        if kzip_path.endswith('.k.zip'):
            kzip_path = kzip_path[:-6]
        assert os.path.isdir(kzip_path), f"Could not find folder for compression to kzip: {kzip_path}"
        with zipfile.ZipFile(kzip_path + '.k.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(kzip_path):
                for file in files:
                    zf.write(os.path.join(root, file), file)
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

    def add_mergelist_to_kzip(self, kzip_path, subobj_map={}):
        ids = defaultdict(lambda: [0, 0, 0])
        ids_count = defaultdict(int)
        obj_map = defaultdict(set)
        for x, y, z in self.iter((0, 0, 0), self.boundary.tolist(), (128, 128, 128)):
            cube = self.from_kzip_to_matrix(kzip_path, size=(128, 128, 128), offset=(x, y, z), mag=1,
                                            return_dataset_cube_if_nonexistent=True, apply_mergelist=False,
                                            show_progress=False, verbose=False)
            if not np.any(cube): continue
            labels = np.unique(cube)[1:]  # no 0
            for sv_id in labels:
                obj_id = subobj_map.get(sv_id, sv_id)
                obj_map[obj_id].add(sv_id)
                indices = np.where(cube == sv_id)
                ids[obj_id][0] += np.sum(indices[0] + x)
                ids[obj_id][1] += np.sum(indices[1] + y)
                ids[obj_id][2] += np.sum(indices[2] + z)
                ids_count[obj_id] += len(indices[0])

        obj_dict = {}
        for obj_id, indices in ids.items():
            center = np.divide(indices, ids_count[obj_id])
            obj_dict[obj_id] = (obj_map[obj_id], center)

        with zipfile.ZipFile(kzip_path, "a") as zf:
            mergelist = mergelist_tools.gen_mergelist_from_objects(obj_dict)
            zf.writestr("mergelist.txt", mergelist)

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
            for ext in self.file_extensions:
                if os.path.exists(self.knossos_path+self._name_mag_folder +
                                str(2**mag)):
                    for x_cube in range(int(self._number_of_cubes[0] // 2**mag+1)):
                        if raw:
                            glob_input = self.knossos_path + \
                                        self._name_mag_folder + \
                                        str(2**mag) + "/x%04d/y*/z*/" % x_cube + \
                                        self._experiment_name + "*" + ext
                        else:
                            glob_input = self.knossos_path + \
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


@dataclass
class LayerConfig:
    URL: Optional[str]
    Name: str
    ServerFormat: Optional[str]
    FileExtension: List[str]
    Extent_px: List[int]
    VoxelSize_nm: List[List[float]]
    CubeShape_px: List[int]
    DataType: Optional[str]
    NumChannels: Optional[int]
    Description: Optional[str]
    Color: Optional[str]
    Visible: Optional[bool]

    def __init__(self, layer: KnossosDataset):
        self.URL = layer.url
        if layer.http_auth is not None:
            parsed_url = urllib.parse.urlparse(layer.url)
            if parsed_url.username is None:
                self.URL = layer.url.replace(f'{parsed_url.scheme}://', f'{parsed_url.scheme}://{layer.http_user}:{layer.http_passwd}@')
        self.Name = layer.experiment_name
        self.ServerFormat = layer.server_format
        self.FileExtension = layer.file_extensions
        self.Extent_px = list(layer.boundary)
        self.VoxelSize_nm = [scale.tolist() for scale in layer.scales]
        self.CubeShape_px = list(layer.cube_shape)
        self.DataType = None if layer._dtype is None else np.dtype(layer._dtype).name
        self.NumChannels = 3 if layer._rgb_channel else 1
        self.Description = layer.description
        self.Color = layer.color
        self.Visible = layer.visible

    def to_toml_string(self):
        string = ''
        for key, value in dataclasses.asdict(self).items():
            if value is not None:
                string += f'{key} = {self.elem_to_toml_string(value)}\n'
        return string

    def elem_to_toml_string(self, elem):
        if isinstance(elem, list):
            return '[' + ', '.join([self.elem_to_toml_string(sub_elem) for sub_elem in elem]) + ']'
        elif isinstance(elem, str):
            return f"'{elem}'"
        elif isinstance(elem, bool):
            return str(elem).lower()
        else:
            return str(elem)
