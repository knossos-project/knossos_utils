# knossos_utils
A Python 3.x library for interacting with **KNOSSOS** datasets and annotation files.


# KnossosDataset

The KnossosDataset class can read data chunk-wise from datasets and .k.zips into NumPy arrays, or vice versa.

## Reading

A chunk is described by its offset into the dataset and its size, both specified in x,y,z order, and the desired magnification. The reading functions return numpy arrays in C-order, i.e. z,y,x. Per default, grayscale images are loaded as np.uint8 and segmentation as np.uint64.

```python
from knossos_utils import KnossosDataset

inp_dataset = KnossosDataset('/path/to/input_dataset_conf')

# loading a grayscale dataset chunk
raw_chunk = inp_dataset.load_raw(offset=(0, 0, 0), size=(1024, 512, 256), mag=1)
print(raw_chunk.shape) # output: (256, 512, 1024)

# loading the entire segmentation dataset
seg_chunk = inp_dataset.load_seg(offset=(0, 0, 0), size=inp_dataset.boundary, mag=1)

# loading segmentation from .k.zip annotation file. the region is specified by the movement_area inside the .k.zip
kzip_chunk = inp_dataset.load_kzip_seg(path='/path/to/segmentation.k.zip', mag=1)

# load a custom region from .k.zip:
kzip_chunk = inp_dataset._load_kzip_seg(path='/path/to/segmentation.k.zip', mag=1, offset=(0, 0, 0), size=(256,256,256))
```

## Writing

Writing a data chunk requires the z,y,x ordered numpy array to be written, the offset at which it should be saved and the chunk’s magnification. Per default KnossosDataset will automatically produce all other magnifications from it.

```python
out_dataset = KnossosDataset('/path/to/destination_dataset_conf')

out_dataset.save_raw(data=raw_chunk, data_mag=1, offset=(0, 0, 0))
out_dataset.save_seg(data=seg_chunk, data_mag=1, offset=(0, 0, 0))
out_dataset.save_to_kzip(data=kzip_chunk, data_mag=1, kzip_path='/write/destination.k.zip', offset=(0,0,0))
```

## Neuroglancer Precomputed Datasets

`KnossosDataset` can also create and read [Neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) datasets. This is the default for newly initialized datasets created with `KnossosDataset.initialize()` or `KnossosDataset.initialize_from_array()`.

Precomputed datasets are backed by `tensorstore` and store one dataset `info` file plus one scale per generated magnification. The public read/write API stays the same as for KNOSSOS cubes: offsets and sizes are passed in x,y,z order, while NumPy arrays are z,y,x.

```python
import numpy as np
from knossos_utils import KnossosDataset

data = np.zeros((64, 256, 256), dtype=np.uint8)  # z,y,x

dataset = KnossosDataset.initialize_from_array(
    data=data,
    experiment_name='example',
    cube_shape=(128, 128, 64),     # x,y,z
    scale=(8, 8, 40),              # nm per voxel at mag1
    ds_factor=(2, 2, 1),
    file_extensions=['.raw'],
    write_path='/path/to/output',
    server_format='precomputed',
)

chunk = dataset.load_raw(offset=(0, 0, 0), size=(256, 256, 64), mag=1)
dataset.save_raw(data=chunk, data_mag=1, offset=(0, 0, 0))
```

Supported precomputed encodings are selected through `file_extensions`:

- `'.raw'`: uint8 raw image data
- `'.png'`: uint8 PNG image data
- `'.jpg'` / `'.jpeg'`: uint8 JPEG image data
- `'.seg.sz.zip'`: uint64 segmentation data using Neuroglancer `compressed_segmentation`

RGB image data can be stored as a single 3-channel precomputed dataset by passing `as_rgb=True`. In that mode, input data must have shape z,y,x,3 and no separate `channels` argument should be passed. The returned dataset exposes three logical layers (`r_*`, `g_*`, `b_*`) that read and write the corresponding precomputed channel.

```python
rgb = np.zeros((64, 256, 256, 3), dtype=np.uint8)  # z,y,x,channel

dataset = KnossosDataset.initialize_from_array(
    data=rgb,
    experiment_name='rgb_example',
    cube_shape=(128, 128, 64),
    scale=(8, 8, 40),
    ds_factor=(2, 2, 1),
    file_extensions=['.raw'],
    write_path='/path/to/output_rgb',
    server_format='precomputed',
    as_rgb=True,
)

red = dataset.layers[0].load_raw(offset=(0, 0, 0), size=(256, 256, 64), mag=1)
```

Existing local precomputed datasets can be opened by pointing `KnossosDataset` at their `.k.toml` configuration. If a layer directory contains an `info` file, it is detected as `server_format='precomputed'`. Remote precomputed datasets can be referenced from the layer config with a `URL` ending in `/info`; HTTP(S) URLs may include credentials and ports, and CDN token parameters are forwarded when available.

For explicit creation without data, use `KnossosDataset.initialize(..., server_format='precomputed')`. A custom `shard_size=(x, y, z)` can be supplied; otherwise a shard size is chosen automatically from the dataset boundary and cube shape.

# Skeleton

A KNOSSOS skeleton is a graph structure with nodes and edges that are grouped into trees. This class can read skeletons from .k.zip or the legacy .nml format, but also import/export [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html).

## Basic Usage

```python
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode

skel = Skeleton()
# loading from .k.zip or .nml
skel.fromNml('/path/to/input.k.zip')

# importing SWC
skel.fromSWC('/path/to/input.swc')

# iterating over nodes per tree
for tree: SkeletonAnnotation in skel.getAnnotations():
    for node: SkeletonNode in tree.getNodes():
        ...

# iterating over all nodes
for node: skeletonNode in skel.getNodes():
    ...

# saving
skel.to_kzip('/path/to/output.k.zip')

# exporting to SWC. Each tree will be saved as one SWC with the specified basename, e.g. /output/folder/prefix0.swc
skel.toSWC(basename='prefix', dest_folder='/output/folder')
```
