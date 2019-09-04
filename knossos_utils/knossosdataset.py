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


import collections
import glob
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import time
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
from xml.etree import ElementTree as ET

import imageio
import h5py
import numpy as np
import requests
import scipy.misc
import scipy.ndimage
from PIL import Image

try:
    from . import mergelist_tools
except ImportError:
    print('mergelist_tools not available, using slow python fallback. '
          'Try to build the cython version of it.')
    from . import mergelist_tools_fallback as mergelist_tools

module_wide = {"init": False, "noprint": False, "snappy": None, "fadvise": None}


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


def cut_matrix(data, offset_start, offset_end, cube_shape, start, end):
    """ Helper for cutting matrices extracted from cubes to a required size """
    cube_shape = _as_shapearray(cube_shape)

    cut_start = np.array(offset_start, dtype=np.int)
    number_cubes = np.array(end) - np.array(start)
    cut_end = np.array(number_cubes * cube_shape - offset_end, dtype=np.int)

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

    class CubeType(Enum):
        RAW = 0,
        COMPRESSED = 1

    def __init__(self, path=None, show_progress=True):
        moduleInit()
        global module_wide
        self.module_wide = module_wide
        self._knossos_path = None
        self._conf_path = None
        self._http_url = None
        self._http_user = None
        self._http_passwd = None
        self._experiment_name = None
        self._name_mag_folder = None
        self._ordinal_mags = False
        self._boundary = np.zeros(3, dtype=np.int)
        self._scale = np.ones(3, dtype=np.float)
        self.scales = []
        self._number_of_cubes = np.zeros(3)
        self._cube_shape = np.full(3, 128, dtype=np.int)
        self._cube_type = KnossosDataset.CubeType.RAW
        self._raw_ext = 'raw'
        self._initialized = False
        self._mags = None
        self.verbose = False
        self.show_progress = show_progress
        self.background_label = 0
        self.http_max_tries = 5

        if path is not None:
            self.initialize_from_conf(path)

    @property
    def mag(self):
        print('mag is DEPRECATED\nPlease use available_mags')
        return self.available_mags

    @property
    def available_mags(self):
        if self._mags is None:
            self._mags = []
            if self.in_http_mode:
                for mag_test_nb in range(10):
                    mag_num = mag_test_nb+1 if self._ordinal_mags else 2 ** mag_test_nb
                    mag_folder = "{}/{}{}".format(self.http_url, self.name_mag_folder, mag_num)
                    for tries in range(10):
                        try:
                            request = requests.get(mag_folder,
                                                   auth=self.http_auth,
                                                   timeout=10)
                            request.raise_for_status()
                            self._mags.append(mag_num)
                            break
                        except requests.exceptions.HTTPError:
                            if request.status_code < requests.codes.server_error:
                                break # no use retrying if client error (e.g. 404)
                            continue
            else:
                regex = re.compile("mag[1-9][0-9]*$")
                for mag_folder in glob.glob(os.path.join(self._knossos_path, "*mag*")):
                    match = regex.search(mag_folder)
                    if match is not None:
                        self._mags.append(int(mag_folder[match.start() + 3:])) # mag number
        return self._mags

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
        return bool(self.http_url)

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
               np.ceil(np.log2(max(np.ceil(np.array(self._boundary) / np.array(self._cube_shape)))))

    def mag_scale(self, mag): # get scale in specific mag
        index = mag - 1 if self._ordinal_mags else int(np.log2(mag))
        return self.scales[index]

    def scale_ratio(self, mag, base_mag): # ratio between scale in mag and scale in base_mag
        return (self.mag_scale(mag) / self.mag_scale(base_mag)) if self._ordinal_mags else np.array(3 * [float(mag) / base_mag])

    def iter(self, offset=(0, 0, 0), end=None, step=(512, 512, 512)):
        end = np.minimum(end or self.boundary, self.boundary)
        step = np.minimum(step, end - offset)
        return ((x, y, z) for x in range(offset[0], end[0], step[0])
                          for y in range(offset[1], end[1], step[1])
                          for z in range(offset[2], end[2], step[2]))

    def set_channel(self, channel):
        if channel == 'implicit': return
        cube_types = {
            'raw': KnossosDataset.CubeType.RAW,
            'png': KnossosDataset.CubeType.COMPRESSED,
            'jpg': KnossosDataset.CubeType.COMPRESSED
        }
        if channel in cube_types:
            self._cube_type = cube_types[channel]
            self._raw_ext = channel
        else:
            raise ValueError(f'channel must be one of {cube_types.keys()}')

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

    def initialize_from_conf(self, path_to_conf):
        if path_to_conf.endswith("ariadne.conf") or path_to_conf.endswith(".pyknossos.conf") or path_to_conf.endswith(".pyk.conf"):
            self.initialize_from_pyknossos_path(path_to_conf)
        else:
            self.initialize_from_knossos_path(path_to_conf)

    def parse_pyknossos_conf(self, path_to_pyknossos_conf):
        """ Parse a pyknossos conf

        :param path_to_pyknossos_conf: str
        :param verbose: bool
        :return:
            nothing
        """
        try:
            f = open(path_to_pyknossos_conf)
            lines = f.readlines()
            f.close()
        except FileNotFoundError as e:
            raise NotImplementedError("Could not read .conf: {}".format(e))

        self._conf_path = path_to_pyknossos_conf
        self._ordinal_mags = True # pyk.conf is ordinal by default

        for line in lines:
            tokens = re.split(" = |,|\n", line)
            key = tokens[0]
            if key == "_BaseName":
                self._experiment_name = tokens[1]
            elif key == "_BaseURL":
                self._http_url = tokens[1]
            elif key == "_UserName":
                self._http_user = tokens[1]
            elif key == "_Password":
                self._http_passwd = tokens[1]
            elif key == "_ServerFormat":
                self._ordinal_mags = tokens[1] != "knossos";
            elif key == "_DataScale":
                self.scales = []
                for x, y, z in zip(tokens[1::3], tokens[2::3], tokens[3::3]):
                    self.scales.append(np.array([float(x), float(y), float(z)]))
                self._scale = self.scales[0]
            elif key == "_FileType":
                type_token = int(tokens[1])
                self._cube_type = KnossosDataset.CubeType.RAW\
                                  if type_token == 0\
                                  else KnossosDataset.CubeType.COMPRESSED
                # set extension automatically if not specified
                self._raw_ext = "raw" if type_token == 0\
                                else "jpg" if type_token == 3 \
                                else "png" # type_token == 2 and default
            elif key == "_NumberofCubes":
                self._number_of_cubes[0] = int(tokens[1])
                self._number_of_cubes[1] = int(tokens[2])
                self._number_of_cubes[2] = int(tokens[3])
            elif key == "_Extent":
                self._boundary[0] = float(tokens[1])
                self._boundary[1] = float(tokens[2])
                self._boundary[2] = float(tokens[3])
            elif key == "_BaseExt":
                self._raw_ext = tokens[1].replace('.', '', 1)
                self._cube_type = KnossosDataset.CubeType.RAW if self._raw_ext == "raw" else KnossosDataset.CubeType.COMPRESSED
        self._cube_shape = [128, 128, 128]  # hardcoded cube shape, others are not supported

    def initialize_from_pyknossos_path(self, path):
        self.parse_pyknossos_conf(path)
        self._knossos_path = os.path.dirname(path) + "/"
        self._name_mag_folder = "mag"
        self._initialized = True
        self._initialize_cache(0)

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

        self._conf_path = path_to_knossos_conf

        parsed_dict = {}
        for line in lines:
            if line.startswith("ftp_mode"):
                line_s = line.split(" ")
                self._http_url = "http://" + line_s[1] + line_s[2] + "/"
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
            np.array(np.ceil(self.boundary.astype(np.float) /
                             self.cube_shape), dtype=np.int)

        if 'png' in parsed_dict:
            self._cube_type = KnossosDataset.CubeType.COMPRESSED
            self._raw_ext = 'png'
        else:
            self._cube_type = KnossosDataset.CubeType.RAW
            self._raw_ext = 'raw'

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
                    mag_folder[:-len(re.findall("[\d]+", mag_folder)[-1])]

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


    def _load(self, offset, size, from_overlay, mag, expand_area_to_mag=False, padding=0, datatype=None):
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
        :param expand_area_to_mag: bool
        :param padding: str or int
            Pad mode for matrix parts outside the dataset. See https://www.pydoc.io/pypi/numpy-1.9.3/autoapi/numpy/lib/arraypad/index.html?highlight=pad#numpy.lib.arraypad.pad
            When passing an it, will pad with that int in 'constant' mode
        :param datatype: numpy datatype
            typically: for mode 'raw' this is np.uint8, and for 'overlay' np.uint64
        :return: 3D numpy array or nothing
            if a path is given no data is returned
        """
        def _read_cube(c):
            local_offset = np.subtract([c[0], c[1], c[2]], start) * self.cube_shape
            valid_values = False

            # check cache first
            values = self._cube_from_cache(c, from_overlay)
            from_cache = values is not None

            if not from_cache:
                filename = f'{self.experiment_name}_mag{mag}_x{c[0]:04d}_y{c[1]:04d}_z{c[2]:04d}.{"seg.sz.zip" if from_overlay else self._raw_ext}'
                path = f'{self.knossos_path}/{self.name_mag_folder}{mag}/x{c[0]:04d}/y{c[1]:04d}/z{c[2]:04d}/{filename}'

                if self.in_http_mode:
                    for tries in range(1, self.http_max_tries + 1):
                        try:
                            request = requests.get(path, auth=self.http_auth, timeout=60)
                            request.raise_for_status()
                            if not from_overlay:
                                if self._raw_ext == 'raw':
                                    values = np.fromstring(request.content, dtype=np.uint8).astype(datatype)
                                else:
                                    values = imageio.imread(request.content)
                            else:
                                with zipfile.ZipFile(BytesIO(request.content), 'r') as zf:
                                    snappy_cube = zf.read(os.path.basename(path[:-4])) # seg.sz (without .zip)
                                    raw_cube = self.module_wide['snappy'].decompress(snappy_cube)
                                    values = np.fromstring(raw_cube, dtype=np.uint64).astype(datatype)
                            try:# check if requested values match shape
                                values.reshape(self.cube_shape)
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
                        if from_overlay:
                            with zipfile.ZipFile(path, 'r') as zf:
                                snappy_cube = zf.read(os.path.basename(path[:-4])) # seg.sz (without .zip)
                            raw_cube = self.module_wide['snappy'].decompress(snappy_cube)
                            values = np.fromstring(raw_cube, dtype=np.uint64).astype(datatype)
                        elif self._cube_type == KnossosDataset.CubeType.RAW:
                            flat_shape = int(np.prod(self.cube_shape))
                            values = np.fromfile(path, dtype=np.uint8, count=flat_shape).astype(datatype)
                        else: # compressed
                            values = imageio.imread(path)
                        valid_values = True
                    self. _print(f'Cube »{path}« does not exist, cube with zeros only assigned')

            if valid_values:
                values = values.reshape(self.cube_shape)
                if not from_cache:
                    self._add_to_cube_cache(c, from_overlay, values)
                local_end = local_offset + self.cube_shape
                output[local_offset[2]:local_end[2], local_offset[1]:local_end[1], local_offset[0]:local_end[0]] = values

        t0 = time.time()

        if not self.initialized:
            raise Exception('Dataset is not initialized')

        if mag not in self.available_mags:
            raise Exception(f'Requested mag {mag} not available, only mags {self.available_mags} are available.')

        if 0 in size:
            raise Exception(f'The second parameter is size! - at least one dimension was set to 0 ({size})')

        ratio = self.scale_ratio(mag, 1)
        if expand_area_to_mag:
            # mag1 coords rounded such that when converting back from target mag to mag1 the specified offset and size can be extracted.
            # i.e. for higher mags the matrix will be larger rather than smaller
            boundary = np.ceil(np.array(self.boundary, dtype=np.int) / ratio).astype(int)
            end = np.ceil(np.add(offset, size) / ratio) * ratio
            offset = np.floor(np.array(offset, dtype=np.int) / ratio) * ratio
            # offset and size in target mag
            size = ((end - offset) // ratio).astype(int)
            offset = (offset // ratio).astype(int)
        else:
            size = (np.array(size, dtype=np.int) // ratio).astype(int)
            offset = (np.array(offset, dtype=np.int) // ratio).astype(int)
            boundary = (np.array(self.boundary, dtype=np.int) // ratio).astype(int)
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
                raise Exception("Given block is totally out ouf bounds with "
                                "offset: [%d, %d, %d]!" %
                                (offset[0], offset[1], offset[2]))

        start = self.get_first_blocks(offset).astype(int)
        end = self.get_last_blocks(offset, size).astype(int)
        uncut_matrix_size = (end - start) * self.cube_shape

        output = np.zeros(uncut_matrix_size[::-1], dtype=datatype)

        offset_start = offset % self.cube_shape
        offset_end = (self.cube_shape - (offset + size)
                      % self.cube_shape) % self.cube_shape

        nb_cubes_to_process = int(np.prod(end - start))
        if nb_cubes_to_process == 0:
            return np.zeros(orig_size[::-1], dtype=datatype)

        cube_coordinates = []

        for z in range(start[2], end[2]):
            for y in range(start[1], end[1]):
                for x in range(start[0], end[0]):
                    cube_coordinates.append([x, y, z])

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

        output = cut_matrix(output, offset_start, offset_end, self.cube_shape, start, end)

        if (uncut_matrix_size / output.shape).prod() > 1.5: # shrink allocation
            output = output.astype(datatype, copy=True)

        if self.show_progress:
            dt = time.time()-t0
            speed = np.product(output.shape) * 1.0/1000000/dt
            if not from_overlay:
                _stdout('\rSpeed: %.3f MB or Mpx/s, time %s\n' % (speed, dt))
            else:
                _stdout('\rSpeed: %.3f Mpx/s, time %s\n' % (speed, dt))

        if not np.all(output.shape == size[::-1]):
            raise Exception("Incorrect shape! Should be", size[::-1], "; got:", output.shape)

        if np.any(mirror_overlap):
            if isinstance(padding, int):
                output = np.pad(output, mirror_overlap[::-1], 'constant', constant_values=padding)
            else:
                output = np.pad(output, mirror_overlap[::-1], mode=padding)

        return output

    def load_raw(self, **kwargs):
        """ from_cubes_to_matrix helper func with mode=raw.
        datatype default is np.uint8, but can be overriden.
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")
        assert 'from_overlay' not in kwargs, 'Don’t pass from_overlay, from_overlay is automatically set to False here.'
        kwargs.update({'from_overlay': False})
        if 'datatype' not in kwargs:
            kwargs.update({'datatype': np.uint8})
        return self._load(**kwargs)

    def load_seg(self, **kwargs):
        """ from_cubes_to_matrix helper func with mode=overlay.
        datatype default is np.uint64, but can be overriden.
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")
        assert 'from_overlay' not in kwargs, 'Don’t pass from_overlay, from_overlay is automatically set to True here.'
        kwargs.update({'from_overlay': True})
        if 'datatype' not in kwargs:
            kwargs.update({'datatype': np.uint64})
        return self._load(**kwargs)

    def from_cubes_to_matrix(self, size, offset, mode, mag=1, datatype=np.uint8,
                             mirror_oob=True, hdf5_path=None,
                             hdf5_name="raw", pickle_path=None,
                             invert_data=False, zyx_mode=False,
                             nb_threads=40, verbose=False, show_progress=True,
                             http_max_tries=2000, http_verbose=False):
        self.verbose = verbose or http_verbose
        self.show_progress = show_progress
        self.http_max_tries = http_max_tries

        if zyx_mode:
            offset = offset[::-1]
            size = size[::-1]
        ratio = self.scale_ratio(mag, 1)
        size = (np.array(size) * ratio).astype(np.int)
        offset = (np.array(offset) * ratio).astype(np.int)

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
        print('from_raw_cubes_to_matrix is DEPRECATED.\nPlease use load_raw, this function will not be updated with from_cubes_to_matrix anymore.')
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
        print('from_overlaycubes_to_matrix is DEPRECATED.\nPlease use load_seg, this function will not be updated with from_cubes_to_matrix anymore.')
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

    @staticmethod
    def get_movement_area(kzip_path):
        with zipfile.ZipFile(kzip_path, "r") as zf:
            xml_str = zf.read('annotation.xml').decode()
        annotation_xml = ET.fromstring(xml_str)
        area_elem = annotation_xml.find("parameters").find("MovementArea")
        area_min = (int(area_elem.get("min.x")),
                  int(area_elem.get("min.y")),
                  int(area_elem.get("min.z")))
        area_max = (int(area_elem.get("max.x")),
                int(area_elem.get("max.y")),
                int(area_elem.get("max.z")))
        return (np.array(area_min), np.array(area_max))

    def from_kzip_movement_area_to_matrix(self, path, mag=8, apply_mergelist=True, return_area=False):
        area_min, area_max = self.get_movement_area(path)
        size = area_max - area_min
        matrix = self.from_kzip_to_matrix(path, size=size, offset=area_min, mag=mag, apply_mergelist=apply_mergelist)
        return (matrix, area_min, size) if return_area else matrix

    def from_kzip_to_matrix(self, path, size, offset, mag=8, empty_cube_label=0,
                            datatype=np.uint64,
                            verbose=False,
                            show_progress=True,
                            apply_mergelist=True,
                            binarize_overlay=False,
                            return_dataset_cube_if_nonexistent=False,
                            expand_area_to_mag=False):
        self.verbose = verbose
        self.show_progress = show_progress
        self.background_label = empty_cube_label

        ratio = self.scale_ratio(mag, 1)
        size = (np.array(size) * ratio).astype(np.int)
        offset = (np.array(offset) * ratio).astype(np.int)

        data = self.load_kzip_seg(path, offset, size, mag, datatype, apply_mergelist, return_dataset_cube_if_nonexistent, expand_area_to_mag)

        if binarize_overlay:
            data[data > 1] = 1

        return data.swapaxes(0, 2)

    def load_kzip_seg(self, path, offset, size, mag, datatype=np.uint64, padding=0, apply_mergelist=True, return_dataset_cube_if_nonexistent=False, expand_area_to_mag=False):
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
        :param expand_area_to_mag: bool
            Enlarges area to true voxels of mag in case offset and size don’t exist in that mag.
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
        archive = zipfile.ZipFile(path, 'r')

        ratio = self.scale_ratio(mag, 1)
        if expand_area_to_mag:
            end = np.ceil(np.add(offset, size) / ratio) * ratio
            offset = np.floor(np.array(offset, dtype=np.int) / ratio) * ratio
            size = (end - offset) // ratio
            offset = offset // ratio
        else:
            size = np.array(size, dtype=np.int)//ratio
            offset = np.array(offset, dtype=np.int)//ratio


        start = np.array([get_first_block(dim, offset, self._cube_shape)
                          for dim in range(3)])
        end = np.array([get_last_block(dim, size, offset, self._cube_shape) + 1
                        for dim in range(3)])

        matrix_size = (end - start) * self.cube_shape
        output = np.zeros(matrix_size[::-1], dtype=datatype)

        offset_start = offset % self.cube_shape
        offset_end = (self.cube_shape - (offset + size) % self.cube_shape) % self.cube_shape

        current = np.array([start[dim] for dim in range(3)])
        cnt = 1
        nb_cubes_to_process = (end - start).prod()

        for z in range(start[2], end[2]):
            for y in range(start[1], end[1]):
                for x in range(start[0], end[0]):
                    current = [x, y, z]
                    if self.show_progress:
                        progress = 100*cnt/float(nb_cubes_to_process)
                        _stdout(f'\rProgress: {progress:.2f}% ') # 
                    this_path = f'{self._experiment_name}_mag{mag}x{x}y{y}z{z}.seg.sz'
                    try:
                        scube = archive.read(this_path)
                        values = np.fromstring(module_wide["snappy"].decompress(scube), dtype=np.uint64)
                        self._print(f'{current}: loaded from .k.zip')
                    except KeyError:
                        self._print(f'{current}: {"dataset" if return_dataset_cube_if_nonexistent else self.background_label} cube assigned')
                        if return_dataset_cube_if_nonexistent:
                            values = self.load_seg(offset=current * ratio * self.cube_shape, size=ratio * self.cube_shape, mag=mag, 
                                                   datatype=datatype, padding=padding, expand_area_to_mag=expand_area_to_mag)
                        else:
                            values = np.full(self.cube_shape, self.background_label, dtype=datatype)

                    local_offset = (current - start) * self.cube_shape
                    local_end = local_offset + self.cube_shape
                    output[local_offset[2]:local_end[2], 
                           local_offset[1]:local_end[1], 
                           local_offset[0]:local_end[0]] = values.reshape(self.cube_shape).astype(datatype, copy=False)
                    cnt += 1

        if self.show_progress and not self.verbose:
            print() # newline after sys.stdout.writes inside loop
        output = cut_matrix(output, offset_start, offset_end, self.cube_shape, start, end)
        if apply_mergelist:
            if "mergelist.txt" not in archive.namelist():
                self._print("no mergelist to apply")
            else:
                self._print("applying mergelist now")
                mergelist_tools.apply_mergelist(output, archive.read("mergelist.txt").decode())

        assert np.array_equal(output.shape, size[::-1]), f'Incorrect shape! Should be {size[::-1]}; got {output.shape}'

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
        if chunk_size is None:
            mat, area_min, size = self.from_kzip_movement_area_to_matrix(str(kzip_path), mag=source_mag, apply_mergelist=False, return_area=True)
            area_max = np.array(area_min) + np.array(size) - 1
        else:
            area_min, area_max = self.get_movement_area(str(kzip_path))
            for offset in self.iter(area_min, area_max, chunk_size):
                mat = self.from_kzip_to_matrix(path=str(kzip_path), offset=offset, size=chunk_size, mag=source_mag, apply_mergelist=False)
                self.from_matrix_to_cubes(offset=offset, data=mat, data_mag=source_mag, kzip_path=dest_path,
                                          mags=out_mags, downsample=downsample, upsample=upsample, compress_kzip=False)
            area_min = offset
        skel = k_skel.Skeleton()
        mag_limit = 1
        if len(out_mags) > 0:
            mag_limit = np.log2(max(out_mags)) if self._ordinal_mags else max(out_mags)
        elif downsample:
            mag_limit = self.highest_mag
        skel.movement_area_min = np.array(area_min) + (mag_limit - np.array(area_min) % mag_limit)
        skel.movement_area_max = np.maximum(area_max - np.array(area_max) % mag_limit, skel.movement_area_min + 1)
        skel.set_scaling(self.scales[0])
        skel.experiment_name = self.experiment_name
        annotation_str = skel.to_xml_string()
        self.from_matrix_to_cubes(offset=area_min, data=mat, data_mag=source_mag, kzip_path=dest_path, mags=out_mags,
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
        assert np.array_equal(data.shape, self.cube_shape), 'Can only save cubes of shape self.cube_shape ({}). found shape {}'.format(self.cube_shape, data.shape)
        data = np.swapaxes(data, 0, 2)
        data = data.reshape(np.prod(self.cube_shape))
        dest_cube = data
        if os.path.isfile(cube_path):
            # read
            if cube_path.endswith('.seg.sz.zip'):
                try:
                    with zipfile.ZipFile(cube_path, "r") as zf:
                        in_zip_name = os.path.basename(cube_path)[:-4]
                        dest_cube = np.fromstring(self.module_wide["snappy"].decompress(zf.read(in_zip_name)), dtype=np.uint64)
                except zipfile.BadZipFile:
                    print(cube_path, "is broken and will be overwritten")
            elif cube_path.endswith('.seg.sz'):
                with open(cube_path, "rb") as existing_file:
                    dest_cube = np.fromstring(self.module_wide["snappy"].decompress(existing_file.read()), dtype=np.uint64)
            elif cube_path.endswith('.raw'):
                dest_cube = np.fromfile(cube_path, dtype=np.uint8)
            else: # png or jpg
                try:
                    dest_cube = imageio.imread(cube_path)
                except ValueError:
                    print(cube_path, "is broken and will be overwritten")
            dest_cube = dest_cube.reshape(self.cube_shape)
            data = data.reshape(self.cube_shape)

            if overwrite_offset is not None or overwrite_limit is not None:
                overwrite_offset = (overwrite_offset if overwrite_offset is not None else (0, 0, 0))[::-1]
                overwrite_limit = (overwrite_limit if overwrite_offset is not None else self.cube_shape)[::-1]
                dest_cube[overwrite_offset[0]: overwrite_limit[0],
                          overwrite_offset[1]: overwrite_limit[1],
                          overwrite_offset[2]: overwrite_limit[2]] = data[overwrite_offset[0]: overwrite_limit[0],
                                                                          overwrite_offset[1]: overwrite_limit[1],
                                                                          overwrite_offset[2]: overwrite_limit[2]]
            else:
                indices = np.where(data != 0)
                dest_cube[indices] = data[indices]
        # write
        if np.any(dest_cube):
            dest_cube = dest_cube.reshape(np.prod(dest_cube.shape))
            if cube_path.endswith('.seg.sz.zip'):
                in_zip_name = os.path.basename(cube_path)[:-4]
                with zipfile.ZipFile(cube_path, "w") as zf:
                    zf.writestr(in_zip_name, self.module_wide["snappy"].compress(dest_cube), compress_type=zipfile.ZIP_DEFLATED)
            elif cube_path.endswith('.seg.sz'):
                with open(cube_path, "wb") as dest_file:
                    dest_file.write(self.module_wide["snappy"].compress(dest_cube))
            elif cube_path.endswith('.raw'):
                with open(cube_path, "wb") as dest_file:
                    dest_file.write(dest_cube)
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
            True and len(data) > 0: Assures unique data before combining all
            list entries
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
        def _write_cubes(args):
            """ Helper function for multithreading """
            folder_path, path, cube_offset, cube_limit, start, end = args

            cube = np.zeros(self.cube_shape, dtype=datatype)
            cube[cube_offset[0]: cube_limit[0],
                 cube_offset[1]: cube_limit[1],
                 cube_offset[2]: cube_limit[2]] = data_inter[start[0]: start[0] + end[0],
                                                             start[1]: start[1] + end[1],
                                                             start[2]: start[2] + end[2]]
            if verbose and not np.any(cube):
               _print(path, 'no data to write, cube will be removed if present')

            if kzip_path is None:
                while True:
                    try:
                        os.makedirs(folder_path, exist_ok=True)
                        break
                    except PermissionError: # sometimes happens via sshfs with multiple workers
                        print('Permission error while creating cube folder. Sleeping on', folder_path)
                        time.sleep(random.uniform(0.1, 1.0))
                        pass

                while True:
                    try:
                        os.makedirs(folder_path+"block")    # Semaphore --------
                        break
                    except (FileExistsError, PermissionError):
                        try:
                            if time.time() - os.stat(folder_path+"block").st_mtime <= 30:
                                time.sleep(random.uniform(0.1, 1.0)) # wait for other workers to finish
                            else:
                                print(f'had to remove block folder {folder_path+"block"} that wasn’t accessed recently {os.stat(folder_path+"block").st_mtime}')
                                os.rmdir(folder_path+"block")
                        except FileNotFoundError:
                            pass # folder was removed by another worker in the meantime

                self.save_cube(cube_path=path if as_raw else path + '.zip', data=cube,
                                overwrite_offset=cube_offset if overwrite else None,
                                overwrite_limit=cube_limit if overwrite else None)
                try:
                    os.rmdir(folder_path+"block")   # ------------------------------
                except FileNotFoundError:
                    print(f'another worker removed our semaphore {folder_path+"block"}')
                    pass
            else:
                self.save_cube(cube_path=path, data=cube,
                                overwrite_offset=cube_offset if overwrite else None,
                                overwrite_limit=cube_limit if overwrite else None)

        # Main Function
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if not (as_raw or self.module_wide["snappy"]):
            raise Exception("Snappy is not available - you cannot write "
                            "overlaycubes or kzips.")

        if not isinstance(mags, list):
            mags = [mags]

        if not mags:
            start_mag = 1 if upsample else data_mag
            end_mag = self.highest_mag if downsample else data_mag
            if self._ordinal_mags:
                mags = np.arange(start_mag, end_mag, dtype=np.int)
            else: # power of 2 mags (KNOSSOS style)
                mags = np.power(2, np.arange(np.log2(start_mag), np.log2(end_mag), dtype=np.int))
        _print("mags to write: {}".format(mags))
        if (data is None) and (data_path is None or hdf5_names is None):
            raise Exception("No data given")

        if kzip_path is not None:
            if as_raw:
                raise Exception("You have to choose between kzip and raw cubes")

            if kzip_path.endswith(".k.zip"):
                kzip_path = kzip_path[:-6]

            if not os.path.exists(kzip_path):
                os.makedirs(kzip_path)
                #if verbose:
                #    _print("kzip path created, notice that kzips can only be "
                #           "created in mag1")

        if data_path is not None:
            if '.h5' in data_path:
                if hdf5_names is not None:
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
            ratio = self.scale_ratio(mag, data_mag)
            inv_mag_ratio = 1.0/np.array(ratio)
            fast = fast_downsampling or (not as_raw and mag > data_mag)
            if fast and all(mag_ratio.is_integer() for mag_ratio in ratio):
                data_inter = np.array(data[::int(ratio[0]), ::int(ratio[1]), ::int(ratio[2])], dtype=datatype)
            elif all(mag_ratio == 1 for mag_ratio in ratio):
                # copy=False means in this context that a copy is only made
                # when necessary (e.g. type change)
                data_inter = data.astype(datatype, copy=False)
            elif fast:
                data_inter = scipy.ndimage.zoom(data, inv_mag_ratio, order=0).astype(datatype, copy=False)
            elif as_raw:
                quality = 3 if mag > data_mag else 1
                data_inter = scipy.ndimage.zoom(data, inv_mag_ratio, order=quality).astype(datatype, copy=False)
            else: # fancy seg upsampling
                data_inter = np.zeros(shape=(inv_mag_ratio * np.array(data.shape)).astype(np.int), dtype=datatype)
                for value in np.unique(data):
                    if value == 0: continue # no 0 upsampling
                    up_chunk_channel = scipy.ndimage.zoom((data == value).astype(np.uint8), inv_mag_ratio, order=1)
                    data_inter += (up_chunk_channel * value).astype(datatype, copy=False)

            offset_mag = np.array(offset, dtype=np.int) // self.scale_ratio(mag, 1)
            size_mag = np.array(data_inter.shape, dtype=np.int)

            if verbose:
                _print('mag: {}'.format(mag))
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
                                        "_x%04d_y%04d_z%04d.%s" \
                                        % (current[0], current[1], current[2], self._raw_ext)
                            else:
                                path += self.experiment_name \
                                        + "_mag"+str(mag) + \
                                        "_x%04d_y%04d_z%04d.seg.sz" \
                                        % (current[0], current[1], current[2])
                        else:
                            path = kzip_path+"/"+self._experiment_name + \
                                   '_mag%dx%dy%dz%d.seg.sz' % \
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
            if compress_kzip:
                with zipfile.ZipFile(kzip_path+".k.zip", "w",
                                     zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(kzip_path):
                        for file in files:
                            zf.write(os.path.join(root, file), file)
                    if gen_mergelist:
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
            if os.path.exists(self._knossos_path+self._name_mag_folder +
                              str(2**mag)):
                for x_cube in range(int(self._number_of_cubes[0] // 2**mag+1)):
                    if raw:
                        glob_input = self._knossos_path + \
                                     self._name_mag_folder + \
                                     str(2**mag) + "/x%04d/y*/z*/" % x_cube + \
                                     self._experiment_name + "*." + self._raw_ext
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
