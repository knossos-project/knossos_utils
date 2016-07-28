from __future__ import absolute_import, division, print_function
# builtins is either provided by Python 3 or by the "future" module for Python 2 (http://python-future.org/)
from builtins import range, map, zip, filter, round, next, input, bytes, hex, oct, chr, int  # TODO: Import all other necessary builtins after testing
from functools import reduce

from knossos_utils import knossosdataset
from knossos_utils import chunky
kd = knossosdataset.KnossosDataset()
kd.initialize_from_knossos_path("/path/to/knossosdir/")

cd = chunky.ChunkDataset()

# Example: Initialize chunkdataset to span the whole knossosdataset with
# chunk-edgelength 512; box_size refers to the box the chunkdataset is
# operating on, this can also be a subset of the total volume. Use box_coords to
# define the offset of your box.
cd.initialize(kd, kd.boundary, [512, 512, 512], "/path/to/cd_home/",
              box_coords=[0, 0, 0], fit_box_size=True)

# After initializing once the cd can be loaded via
cd = chunky.load_dataset("/path/to/cd_home/")

# All chunks are accessible via the chunk_dict. Say one wants number 10
chunk = cd.chunk_dict[10]

# Raw data should never be saved in the cd. One can load with
raw = cd.chunk_dict[0].raw_data(show_progress=True)

# Saving data into chunks
# Data from the same project should have the same name. One can write multiple
# representations in one project by using different setnames.
chunk.save_chunk(your_data, name="my_project", setname="rep1")

# Loading data from chunks
# Either directly from one chunk:
data = chunk.load_chunk(name="my_project", setname="rep1")

# Or in analogy to from_raw_cubes_to_matrix (knossosdataset):
data = cd.from_chunky_to_matrix(size, offset, name="my_project",
                                setnames=["rep1"], show_progress=True)

# Note: from_chunky_to_matrix returns a dictionary containing the different
# representations defined in setnames.
# Note: One can "overload" chunks with arrays that exceed the size of the chunk.
# When loading singles chunks, the whole array is returned, but when using
# from_chunky_to_matrix the array is centered and cut to be stitched with the
# data from neighbouring chunks.

