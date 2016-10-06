# coding=utf-8
################################################################################
#  This file provides a class representation of a KNOSSOS-dataset for reading
#  and writing raw and overlay data.
#
#  (C) Copyright 2015
#  Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V.
#
#  DatasetUtils.py is free software: you can redistribute it and/or modify
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


import cPickle as pickle
import glob
import h5py
import io
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
try:
    import mergelist_tools
except ImportError:
    print('mergelist_tools not available, using slow python fallback. '
          'Try to build the cython version of it.')
    import mergelist_tools_fallback as mergelist_tools
import numpy as np
import re
import scipy.misc
import scipy.ndimage
import shutil
import sys
import time
import os
import zipfile

module_wide = {"init":False,"noprint":False,"snappy":None,"fadvise":None}

def our_glob(s):
    l = []
    for g in glob.glob(s):
        l.append(g.replace(os.path.sep,"/"))
    return l

def _print(s):
    global module_wide
    if not module_wide["noprint"]:
        print s
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

def moduleInit():
    global module_wide
    if module_wide["init"]:
        return
    module_wide["init"] = True
    try:
        import snappy
        module_wide["snappy"] = snappy
    except:
        _print("snappy is not available - you won't be able to write/read " \
              "overlaycubes and k.zips. Reference for snappy: " \
              "https://pypi.python.org/pypi/python-snappy/")
    try:
        import fadvise
        module_wide["fadvise"] = fadvise
    except:
        pass
    return

def get_first_block(dim, offset, edgelength):
    """ Helper for iterating over cubes """
    try:
        return int(np.floor(offset[dim]/edgelength[dim]))
    except:
        return int(np.floor(offset[dim]/edgelength))


def get_last_block(dim, size, offset, edgelength):
    """ Helper for iterating over cubes """
    try:
        return int(np.floor((offset[dim]+size[dim]-1)/edgelength[dim]))
    except:
        return int(np.floor((offset[dim]+size[dim]-1)/edgelength))


def cut_matrix(data, offset_start, offset_end, edgelength, start, end):
    """ Helper for cutting matrices extracted from cubes to a required size """
    try:
        len(edgelength)
    except:
        edgelength = np.array([edgelength, ]*3)

    cut_start = np.array(offset_start, dtype=np.int)
    number_cubes = np.array(end) - np.array(start)
    cut_end = np.array(number_cubes*edgelength-offset_end, dtype=np.int)

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


def save_to_h5py(data, path, hdf5_names=None, compression=False,
                 overwrite=True):
    """ Helper for saving h5-files

    :param data: list or dict of arrays
        if list, hdf5_names has to be set.
    :param path: str
        forward-slash separated path to file
    :param hdf5_names: list of str
        same order as data
    :param compression: bool
        True: compression='gzip' is used which is recommended for sparse and ordered data
    :param overwrite: bool
        determines whether an existing file is overwritten
    :return:
        nothing
    """
    if (not type(data) is dict) and hdf5_names is None:
        raise Exception("hdf5names has to be given, when data is a list")
    if os.path.isfile(path) and overwrite:
        os.remove(path)
    else:
        raise Exception("File already exists and overwriting it is not "
                        "allowed.")
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
        self._experiment_name = None
        self._mag = []
        self._name_mag_folder = None
        self._boundary = np.zeros(3, dtype=np.int)
        self._scale = np.ones(3, dtype=np.float)
        self._number_of_cubes = np.zeros(3)
        self._edgelength = 128
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
        return self._knossos_path

    @property
    def number_of_cubes(self):
        return self._number_of_cubes

    @property
    def edgelength(self):
        return self._edgelength

    @property
    def initialized(self):
        return self._initialized

    def initialize_from_knossos_path(self, path, fixed_mag=None, verbose=False):
        """ Initializes the dataset by parsing the knossos.conf in path + "mag1"

        :param path: str
            forward-slash separated path to the datasetfolder - not .../mag !
        :param fixed_mag: int
            fixes available mag to one specific value
        :param verbose: bool
            several information is printed when set to True
        :return:
            nothing
        """

        self._knossos_path = path
        all_mag_folders = our_glob(path+"/*mag*")

        if fixed_mag > 0:
            self._mag.append(fixed_mag)
        else:
            for mag_test_nb in range(32):
                for mag_folder in all_mag_folders:
                    if "mag"+str(2**mag_test_nb) in mag_folder:
                        self._mag.append(2**mag_test_nb)
                        break

        mag_folder = our_glob(path+"/*mag*")[0].split("/")
        if len(mag_folder[-1]) > 1:
            mag_folder = mag_folder[-1]
        else:
            mag_folder = mag_folder[-2]

        self._name_mag_folder = \
            mag_folder[:-len(re.findall("[\d]+", mag_folder)[-1])]

        try:
            f = open(our_glob(path+"/*mag1")[0]+"/knossos.conf")
            lines = f.readlines()
            f.close()
        except:
            raise NotImplementedError("Could not find/read *mag1/knossos.conf")

        parsed_dict = {}
        for line in lines:
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

        self._number_of_cubes = np.array(np.ceil(self.boundary /
                                                 float(self.edgelength)),
                                         dtype=np.int)

        if verbose:
            _print("Initialization finished successfully")
        self._initialized = True

    def initialize_without_conf(self, path, boundary, scale, experiment_name,
                                mags=None, make_mag_folders=True,
                                create_knossos_conf=True, verbose=False):
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

        self._number_of_cubes = np.array(np.ceil(self.boundary /
                                                 float(self.edgelength)),
                                         dtype=np.int)

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
        self._initialized = True

    def initialize_from_matrix(self, path, scale, experiment_name,
                               offset=None, boundary=None, fast_downsampling=True,
                               data=None, data_path=None, hdf5_names=None,
                               mags=None, verbose=False):
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

        self.initialize_without_conf(path, boundary, scale, experiment_name,
                                     mags=mags, make_mag_folders=True,
                                     create_knossos_conf=True, verbose=verbose)

        self.from_matrix_to_cubes(offset, mags=mags, data=data, datatype=np.uint8,
                                  fast_downsampling=fast_downsampling, as_raw=True)

    def from_raw_cubes_to_list(self, vx_list):
        """ Read voxel values vectorized
        WARNING: voxels have to be clustered, otherwise: runtime -> inf

        :param vx_list:  list or array of 3 sequence of int
            list of voxels which values should be returned
        :return: array of int
            array of voxel values corresponding to vx_list
        """
        vx_list = np.array(vx_list, dtype=np.int)
        boundary_box = [np.min(vx_list, axis=0),
                        np.max(vx_list, axis=0)]
        size = boundary_box[1] - boundary_box[0] + np.array([1,1,1])

        block = self.from_raw_cubes_to_matrix(size, boundary_box[0], show_progress=False)

        vx_list -= boundary_box[0]

        return block[vx_list[:, 0], vx_list[:, 1], vx_list[:, 2]]
        
    def from_cubes_to_matrix(self, size, offset, type, mag=1, datatype=np.uint8,
                             mirror_oob=True, hdf5_path=None,
                             hdf5_name="raw", pickle_path=None,
                             invert_data=False, verbose=False,
                             show_progress=True):
        """ Extracts a 3D matrix from the KNOSSOS-dataset
            NOTE: You should use one of the two wrappers below

        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param type: str
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
        :param verbose: bool
            True: prints several information
        :param show_progress: bool
            True: progress is printed to the terminal
        :return: 3D numpy array or nothing
            if a path is given no data is returned
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if mag not in self._mag:
            raise Exception("Magnification not supported")

        if verbose and show_progress:
            show_progress = False
            _print("when choosing verbose, show_progress is automatically set " \
                  "to False")

        if type == 'raw':
            from_raw = True
        elif type == 'overlay':
            from_raw = False
        else:
            raise NotImplementedError("type has to be 'raw' or 'overlay'")

        size = np.array(size, dtype=np.int)
        offset = np.array(offset, dtype=np.int)
        mirror_overlap = [[0, 0], [0, 0], [0, 0]]

        for dim in range(3):
            if offset[dim] < 0:
                size[dim] += offset[dim]
                mirror_overlap[dim][0] = offset[dim]*(-1)
                offset[dim] = 0
            if offset[dim]+size[dim] > self.boundary[dim]:
                mirror_overlap[dim][1] = offset[dim]+size[dim]\
                                         -self.boundary[dim]
                size[dim] -= offset[dim]+size[dim]-self.boundary[dim]
            if size[dim] < 0:
                raise Exception("Given block is totally out ouf bounce!")

        start = np.array([get_first_block(dim, offset, self._edgelength)
                          for dim in xrange(3)])
        end = np.array([get_last_block(dim, size, offset, self._edgelength)+1
                        for dim in xrange(3)])

        #TODO: Describe method
        uncut_matrix_size = (end - start)*self.edgelength
        output = np.zeros(uncut_matrix_size, dtype=datatype)

        offset_start = np.array([offset[dim] % self.edgelength
                                 for dim in range(3)])
        offset_end = np.array([(self.edgelength-(offset[dim]+size[dim])
                               % self.edgelength) % self.edgelength
                              for dim in range(3)])

        current = np.array([start[dim] for dim in range(3)])
        cnt = 1
        nb_cubes_to_process = \
            (end[2]-start[2]) * (end[1] - start[1]) * (end[0] - start[0])

        while current[2] < end[2]:
            current[1] = start[1]
            while current[1] < end[1]:
                current[0] = start[0]
                while current[0] < end[0]:
                    if show_progress:
                        progress = 100*cnt/float(nb_cubes_to_process)
                        if progress < 100:
                            _stdout('\rProgress: %.2f%%' % progress)
                        else:
                            _stdout('\rProgress: finished\n')

                    if from_raw:
                        path = self._knossos_path+self._name_mag_folder+\
                               str(mag) + "/x%04d/y%04d/z%04d/" \
                               % (current[0], current[1], current[2]) + \
                               self._experiment_name + '_mag' + str(mag) + \
                               "_x%04d_y%04d_z%04d.raw" \
                               % (current[0], current[1], current[2])

                        try:
                            if self.module_wide["fadvise"]:
                                self.module_wide["fadvise"].willneed(path)

                            l = []
                            buffersize = 32768
                            fd = io.open(path, 'rb',
                                         buffering=buffersize)
                            for i in range(0,(self._edgelength**3/buffersize)+1):
                                l.append(fd.read(buffersize))
                            content = "".join(l)
                            fd.close()

                            values = np.fromstring(content, dtype=np.uint8)

                        except:
                            values = np.zeros([self.edgelength,]*3)
                            if verbose:
                                _print("Cube does not exist, cube with zeros " \
                                      "only assigned")

                    else:
                        path = self._knossos_path+self._name_mag_folder+\
                               str(mag)+"/x%04d/y%04d/z%04d/" \
                               % (current[0], current[1], current[2]) + \
                               self._experiment_name + '_mag' + str(mag) + \
                               "_x%04d_y%04d_z%04d.seg.sz" \
                               % (current[0], current[1], current[2])
                        try:
                            with zipfile.ZipFile(path+".zip", "r") as zf:
                                values = np.fromstring(self.module_wide["snappy"].decompress(
                                    zf.read(os.path.basename(path))),
                                                       dtype=datatype)

                        except:
                            values = np.zeros([self.edgelength,]*3)
                            if verbose:
                                _print("Cube does not exist, cube with zeros " \
                                      "only assigned")

                    pos = (current-start)*self.edgelength

                    values = np.swapaxes(values.reshape([self.edgelength, ]*3),
                                         0, 2)
                    output[pos[0]: pos[0]+self.edgelength,
                           pos[1]: pos[1]+self.edgelength,
                           pos[2]: pos[2]+self.edgelength] = values

                    cnt += 1
                    current[0] += 1
                current[1] += 1
            current[2] += 1

        output = cut_matrix(output, offset_start, offset_end, self.edgelength,
                            start, end)

        if False in [output.shape[dim] == size[dim] for dim in xrange(3)]:
            raise Exception("Incorrect shape! Should be", size, "; got:",
                            output.shape)
        else:
            if verbose:
                _print("Correct shape")

        if mirror_oob and np.any(mirror_overlap!=0):
            output = np.lib.pad(output, mirror_overlap, 'symmetric')

        if output.dtype != datatype:
            raise Exception("Wrong datatype! - for unknown reasons...")

        if invert_data:
            output = np.invert(output)

        if hdf5_path and hdf5_name:
            save_to_h5py(output, hdf5_path, hdf5_names=[hdf5_name])

        if pickle_path:
            save_to_pickle(output, pickle_path)

        return output

    def from_raw_cubes_to_matrix(self, size, offset, mag=1,
                                 datatype=np.uint8, mirror_oob=True,
                                 hdf5_path=None, hdf5_name="raw",
                                 pickle_path=None, invert_data=False,
                                 verbose=False, show_progress=True):
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
        :param verbose: bool
            True: prints several information
        :param show_progress: bool
            True: progress is printed to the terminal
        :return: 3D numpy array or nothing
            if a path is given no data is returned
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if (not hdf5_name is None and not hdf5_path is None) or \
                not pickle_path is None:
            self.from_cubes_to_matrix(size, offset, 'raw', mag, datatype,
                                      mirror_oob, hdf5_path, hdf5_name,
                                      pickle_path, invert_data, verbose,
                                      show_progress)
        else:
            return self.from_cubes_to_matrix(size, offset, 'raw', mag, datatype,
                                             mirror_oob, hdf5_path,
                                             hdf5_name, pickle_path,
                                             invert_data, verbose,
                                             show_progress)

    def from_overlaycubes_to_matrix(self, size, offset, mag=1,
                                    datatype=np.uint64, mirror_oob=True,
                                    hdf5_path=None, hdf5_name="raw",
                                    pickle_path=None, invert_data=False,
                                    verbose=False, show_progress=True):
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
        :param verbose: bool
            True: prints several information
        :param show_progress: bool
            True: progress is printed to the terminal
        :return: 3D numpy array or nothing
            if a path is given no data is returned
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        if (not hdf5_name is None and not hdf5_path is None) or \
                not pickle_path is None:
            self.from_cubes_to_matrix(size, offset, 'overlay', mag, datatype,
                                      mirror_oob, hdf5_path, hdf5_name,
                                      pickle_path, invert_data, verbose,
                                      show_progress)
        else:
            return self.from_cubes_to_matrix(size, offset, 'overlay', mag,
                                             datatype, mirror_oob,
                                             hdf5_path, hdf5_name, pickle_path,
                                             invert_data, verbose,
                                             show_progress)

    def from_kzip_to_matrix(self, path, size, offset, empty_cube_label=0,
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

        start = np.array([get_first_block(dim, offset, self._edgelength)
                          for dim in xrange(3)])
        end = np.array([get_last_block(dim, size, offset, self._edgelength)+1
                        for dim in xrange(3)])

        matrix_size = (end - start)*self.edgelength
        output = np.zeros(matrix_size, dtype=datatype)

        offset_start = np.array([offset[dim] % self.edgelength
                                 for dim in range(3)])
        offset_end = np.array([(self.edgelength-(offset[dim]+size[dim])
                               % self.edgelength) % self.edgelength
                               for dim in range(3)])

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
                                '_mag1_mag1x%dy%dz%d.seg.sz' % \
                                (current[0], current[1], current[2])

                    if self._experiment_name == \
                                "20130410.membrane.striatum.10x10x30nm":
                        this_path = self._experiment_name +\
                                    '_mag1x%dy%dz%d.segmentation.snappy' % \
                                    (current[0], current[1], current[2])

                    try:
                        values = np.fromstring(module_wide["snappy"].decompress(
                            archive.read(this_path)), dtype=datatype)
                    except:
                        if verbose:
                            _print("Cube does not exist, cube with %d only " \
                                  "assigned" % empty_cube_label)
                        values = np.ones([self.edgelength,]*3)*empty_cube_label

                    pos = (current-start)*self.edgelength

                    values = np.swapaxes(values.reshape([self.edgelength, ]*3),
                                         0, 2)
                    output[pos[0]: pos[0]+self.edgelength,
                           pos[1]: pos[1]+self.edgelength,
                           pos[2]: pos[2]+self.edgelength] = values
                    cnt += 1
                    current[0] += 1
                current[1] += 1
            current[2] += 1

        output = cut_matrix(output, offset_start, offset_end, self.edgelength,
                            start, end)
        if verbose:
            _print("applying mergelist now")
        mergelist_tools.apply_mergelist(output, archive.read("mergelist.txt"))

        if False in [output.shape[dim] == size[dim] for dim in xrange(3)]:
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
            scipy.misc.imsave(output_path+"/"+name+"_%d."+output_format,
                              data[:, :, z])


    def export_to_image_stack(self, out_format='png', out_path='', mag=1):
        """
        Simple exporter, NOT RAM friendly. Always loads entire cube layers ATM.
        Make sure to have enough RAM available. There is still a bug at the
        final layers, tifs containing no image data are written out at the end.

        :param out_format: string
        :param out_path: string
        :return:
        """

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        z_coord_cnt = 0

        scaled_cube_layer_size = (self.boundary[0]/mag,
                                  self.boundary[1]/mag,
                                  self._edgelength)

        for curr_z_cube in range(0, 1+int(np.ceil(self._number_of_cubes[
            2])/float(mag))):

            layer = self.from_raw_cubes_to_matrix(size=scaled_cube_layer_size,
                                                  offset=[0,0,
                                                          curr_z_cube *
                                                          self._edgelength],
                                                  mag=mag)

            for curr_z_coord in range(0, self._edgelength):

                file_path = "{0}_{1}_{2:06d}.{3}".format(out_path,
                                                         self.experiment_name,
                                                         z_coord_cnt,
                                                         out_format)

                # the swap is necessary to have the same visual
                # appearence in knossos and the resulting image stack
                swapped = np.swapaxes(layer[:,:,curr_z_coord], 0,0)
                if out_format == 'png':
                    scipy.misc.imsave(file_path, swapped)
                    # this_img = Image.fromarray(swapped)
                    # this_img.save(file_path)
                elif out_format == 'raw':
                    swapped.tofile(file_path)

                _print("Writing layer {0} of {1} in total.".format(z_coord_cnt, self.boundary[2]/mag))

                z_coord_cnt += 1



        return

    def from_matrix_to_cubes(self, offset, mags=1, data=None, data_path=None,
                             hdf5_names=None, datatype=np.uint64, fast_downsampling=True,
                             force_unique_labels=False, verbose=False,
                             overwrite=True, kzip_path=None, annotation_str=None, as_raw=False,
                             nb_threads=10):
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
            True: whole KNOSSOS cube is overwritten
            False: cube entries where data == 0 are contained
            eg.: Two different function calls write different parts of one
                 KNOSSOS cube. When overwrite is set to False, the second call
                 won't overwrite the output of the first one.
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

            cube = np.zeros([self.edgelength,]*3, dtype=datatype)

            cube[cube_offset[0]: cube_limit[0],
                 cube_offset[1]: cube_limit[1],
                 cube_offset[2]: cube_limit[2]]\
                = data_inter[start[0]: start[0]+end[0],
                             start[1]: start[1]+end[1],
                             start[2]: start[2]+end[2]]

            cube = np.swapaxes(cube, 0, 2)
            cube = cube.reshape(self.edgelength**3)

            if kzip_path is None:
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                while True:
                    try:
                        os.makedirs(folder_path+"block")# Semaphore --------------------
                        break
                    except:
                        if time.time()-os.stat(folder_path+"block").st_mtime > 5:
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
                        existing_cube = np.fromstring(self.module_wide["snappy"].decompress(
                            zf.read(os.path.basename(path))), dtype=np.uint64)
                    indices = np.where(cube == 0)
                    cube[indices] = existing_cube[indices]

                if as_raw:
                    f = open(path, "wb")
                    f.write(cube)
                    f.close()
                else:
                    arc_path = os.path.basename(path)
                    with zipfile.ZipFile(path + ".zip", "w") as zf:
                        zf.writestr(arc_path, self.module_wide["snappy"].compress(cube),
                                    compress_type=zipfile.ZIP_DEFLATED)

                os.rmdir(folder_path+"block")#---------------------------–-

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
                _print("kzip path created, notice that kzips can only be " \
                      "created in mag1")

            mags = [1]
            if not 1 in self.mag\
                    :
                raise Exception("kzips have to be in mag1 but dataset does not"
                                "support mag1")

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
            data = np.array(data)
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
            if mag > 1:
                if fast_downsampling:
                    data_inter = np.array(data[::mag, ::mag, ::mag], dtype=datatype)
                else:
                    data_inter = np.array(scipy.ndimage.zoom(data, 1./mag, order=3), dtype=datatype)
            else:
                data_inter = np.array(np.copy(data), dtype=datatype)

            offset_mag = np.array(offset, dtype=np.int) / mag
            size_mag = np.array(data_inter.shape, dtype=np.int)

            if verbose:
                _print("box_offset: {0}".format(offset_mag))
                _print("box_size: {0}".format(size_mag))

            start = np.array([get_first_block(dim, offset_mag, self._edgelength)
                              for dim in xrange(3)])
            end = np.array([get_last_block(dim, size_mag, offset_mag,
                                           self._edgelength)+1
                            for dim in xrange(3)])

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
                        path = self._knossos_path+self._name_mag_folder+\
                               str(mag)+"/"+"x%04d/y%04d/z%04d/" \
                               % (current[0], current[1], current[2])

                        this_cube_info.append(path)

                        if kzip_path is None:
                            if as_raw:
                                path += self._experiment_name \
                                        + "_mag"+str(mag)+\
                                        "_x%04d_y%04d_z%04d.raw" \
                                        % (current[0], current[1], current[2])
                            else:
                                path += self._experiment_name \
                                        + "_mag"+str(mag)+\
                                        "_x%04d_y%04d_z%04d.seg.sz" \
                                        % (current[0], current[1], current[2])
                        else:
                            path = kzip_path+"/"+self.experiment_name+ \
                                   "_mag"+str(mag)+"_mag"+str(mag)+ \
                                   "x%dy%dz%d.seg.sz" \
                                   % (current[0], current[1], current[2])

                        this_cube_info.append(path)

                        cube_coords = current*self.edgelength
                        cube_offset = np.zeros(3)
                        cube_limit = np.ones(3)*self.edgelength

                        for dim in range(3):
                            if cube_coords[dim] < offset_mag[dim]:
                                cube_offset[dim] += offset_mag[dim] \
                                                    - cube_coords[dim]
                            if cube_coords[dim] + self.edgelength > \
                                            offset_mag[dim] + size_mag[dim]:
                                cube_limit[dim] -= \
                                    self.edgelength + cube_coords[dim]\
                                        - (offset_mag[dim] + size_mag[dim])

                        start_coord = cube_coords-offset_mag+cube_offset
                        end_coord = cube_limit-cube_offset

                        this_cube_info.append(cube_offset)
                        this_cube_info.append(cube_limit)
                        this_cube_info.append(start_coord)
                        this_cube_info.append(end_coord)

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
                map(_write_cubes, multithreading_params)

        if kzip_path is not None:
            with zipfile.ZipFile(kzip_path+".k.zip", "w",
                                 zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(kzip_path):
                    for file in files:
                        zf.write(os.path.join(root, file), file)
                zf.writestr("mergelist.txt", mergelist_tools.gen_mergelist_from_segmentation(data, offsets=np.array(offset, dtype=np.uint64)))
                if annotation_str is not None:
                    zf.writestr("annotation.xml", annotation_str)
            shutil.rmtree(kzip_path)

    def from_overlaycubes_to_kzip(self, size, offset, output_path, mag=1):
        """ Copies chunk from overlay cubes and saves them as kzip

        :param size: 3 sequence of ints
            size of requested data block
        :param offset: 3 sequence of ints
            coordinate of the corner closest to (0, 0, 0)
        :param output_path: str
            path to .k.zip file without extension
        :param mag: int
            desired magnification
        :return:
            nothing
        """
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        overlay = self.from_overlaycubes_to_matrix(size, offset, mag=mag)

        self.from_matrix_to_cubes(offset, data=overlay,
                                  kzip_path=output_path)

    def delete_all_overlaycubes(self, nb_processes=4, verbose=False):
        """  Deletes all overlaycubes

        :param nb_processes: int
            if < 2: no multiprocessing
        :param verbose: bool
            True: prints several information
        :return:
            nothing
        """
        multi_params = []
        for mag in range(32):
            if os.path.exists(self._knossos_path+self._name_mag_folder+
                              str(2**mag)):
                for x_cube in range(self._number_of_cubes[0]/2**mag+1):
                    glob_input = self._knossos_path+self._name_mag_folder+\
                                 str(2**mag)+"/x%04d/y*/z*/" % x_cube + \
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
            map(_find_and_delete_cubes_process, multi_params)