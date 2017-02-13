# coding=utf-8
################################################################################
#  This file provides a class representation of a KNOSSOS-dataset for reading
#  and writing raw and overlay data.
#
#  (C) Copyright 2015 - now
#  Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V.
#
#  remote_datasets.py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 2 of
#  the License as published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
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

################################################################################
#
# This examples covers the handling of knossos datasets. Using remote datasets
# as the one hosted under <http://connectomes.org/knossos.conf> works exactly
# the same.
#
################################################################################

from knossos_utils import knossosdataset


# Initialization ---------------------------------------------------------------

# Initialize your knossosdataset from a the knossos.conf (the dataset folder
# also works). In case you are using a remote dataset you have to download the
# knossos.conf as a textfile first.
kd = knossosdataset.KnossosDataset()
kd.initialize_from_knossos_path("path_to_knossos.conf")


# Reading ----------------------------------------------------------------------

# Now you are able to pull data from this dataset. For a 1000 x 1000 x 500 vx
# block at the origin that would look like
raw = kd.from_raw_cubes_to_matrix(size=[1000, 1000, 500], offset=[0, 0, 0])


# In case you want to load the whole dataset into an array, you can use the
# boundary property. This is safe for small enough datasets, but can easily
# exceed your memory for larger ones.
raw = kd.from_raw_cubes_to_matrix(size=kd.boundary, offset=[0, 0, 0])


# Especially helpful for large remote datasets is the ability to copy datasets.
# You can use this to duplicate datasets on disk, but also to download remote
# ones.
kd.copy_dataset("path_to_new_knossos_dataset_folder", do_raw=True)

# Alternatively to raw data, KnossosDatasets support interacting with
# overlaycubes and k.zips. The usage is the same as for raw data
overlay = kd.from_overlaycubes_to_matrix(size=kd.boundary, offset=[0, 0, 0])
overlay_from_kzip = kd.from_kzip_to_matrix(path="path_to_k.zip",
                                           size=kd.boundary, offset=[0, 0, 0])


# Writing ----------------------------------------------------------------------

# You can also use KnossosDataset to write to your local knossos dataset.
# Please have a look at the specific docstring for handling writing from hdf5,
# dealing with existing data and more.
# raw data
import numpy as np
kd.from_matrix_to_cubes(offset=[0, 0, 0], data=raw, as_raw=True,
                        datatype=np.uint8)

# overlays
kd.from_matrix_to_cubes(offset=[0, 0, 0], data=overlay, as_raw=False)

# kzips
kd.from_matrix_to_cubes(offset=[0, 0, 0], data=overlay_from_kzip, as_raw=False,
                        kzip_path="path_to_k.zip")

# In case you want to delete all your overlay data:
kd.delete_all_overlaycubes()


# New datasets -----------------------------------------------------------------

# KnossosDataset can also be used to create new datasets. You can then use
# from_matrix_to_cubes to populate it with data - KnossosDataset is essentially
# also a cuber.
kd.initialize_without_conf("path_to_new_knossos_dataset_folder",
                           boundary=size,
                           scale=resolution,
                           experiment_name="fancy_name")

# In case you already have your data for the new dataset in an array, you can
# also use
kd.initialize_from_matrix("path_to_new_knossos_dataset_folder",
                          scale=resolution,
                          experiment_name="fancy_name",
                          data=raw)

