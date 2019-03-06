import requests
import numpy as np
import imageio
import snappy
from zipfile import ZipFile
from io import BytesIO
import itertools as it
import os
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from typing import Tuple, Dict


class DatasetInfo(object):
    def __init__(self, exp_name, bounds, mag_scales):
        self.bounds = bounds
        self.mag_scales = mag_scales
        self.exp_name = exp_name
        self.cube_edge = 128


def _cube_bound_space(lo, hi, cube_edge):
    """
    Return a list of tuples describing how the linear space between lo and hi is split up into subregions
    by data cubes of size cube_edge. E.g., for parameters (50, 300, 128), return:
        ((50, 128), (128, 256), (256, 300))
    """

    lo_cube = int(np.floor(lo / cube_edge) + 1) * cube_edge
    hi_cube = int(np.ceil(hi / cube_edge) - 1) * cube_edge

    out_vec = [lo] + list(range(lo_cube, hi_cube+1, cube_edge)) + [hi]
    out_vec = list(zip(out_vec, out_vec[1:]))

    return out_vec


def _iter_cubes(size_px, offset_px, cube_edge):
    """
    Generator that iterates the cubes and corresponding cube slice bounds making up a 3-dimensional
    volume.

    Given offset and size in pixels (defining a 3-dimensional region of a dataset), yields the following:
        ((xcube, ycube, zcube),
         (x_local_bounds, y_local_bounds, z_local_bounds),
         (x_output_bounds, y_output_bounds, z_output_bounds))

    ... where the first element is a tuple of int, the second is a tuple of tuple of 2 ints, the third is
    a tuple of tuple of 2 ints.

    The first element (xcube, ...) specifies a cube, the second specifies a slice into that cube, and the third
    specifies a slice into an output array.
    """

    lo_px = np.array(offset_px)
    size_px = np.array(size_px)

    hi_px = offset_px + size_px

    cube_bounds = [_cube_bound_space(xx, yy, cube_edge)
                   for xx, yy in zip(lo_px, hi_px)]

    for xbounds, ybounds, zbounds in it.product(*cube_bounds):
        # xbounds, ybounds, zbounds are the boundaries for the current cube slice
        # in dataset-global coordinates

        x_cube, y_cube, z_cube = tuple(
                int(np.floor(xx[0] / cube_edge)) for xx in (xbounds, ybounds, zbounds))
        x_cube_px, y_cube_px, z_cube_px = tuple(
                xx * cube_edge for xx in (x_cube, y_cube, z_cube))
        # these are these same boundaries, in cube-local coordinates
        x_local_bounds = np.array(xbounds) - x_cube_px
        y_local_bounds = np.array(ybounds) - y_cube_px
        z_local_bounds = np.array(zbounds) - z_cube_px

        # these are the same boundaries, in output-array coordinates
        x_output_bounds = np.array(xbounds) - lo_px[0]
        y_output_bounds = np.array(ybounds) - lo_px[1]
        z_output_bounds = np.array(zbounds) - lo_px[2]

        yield ((x_cube, y_cube, z_cube),
               (x_local_bounds, y_local_bounds, z_local_bounds),
               (x_output_bounds, y_output_bounds, z_output_bounds))


def _get_extension(channel):
    return channel


def _maybe_decompress_cube_data(data_str, channel):
    if channel == 'raw':
        return data_str
    elif channel in ['png', 'jpg']:
        return imageio.imread(data_str).flatten()
    elif channel == 'seg.sz':
        return snappy.decompress(data_str)
    elif channel == 'seg.sz.zip':
        with ZipFile(BytesIO(data_str), 'r') as zf:
            in_archive = zf.namelist()[0]
            data_str = zf.read(in_archive)
        return snappy.decompress(data_str)


def _get_pth_to_cube(src, cube, exp_name, mag_scale, channel):
    """
    Generate a cube path: on filesystem, remote (http) or inside a zip file.

    If src is str, generate a filesystem or http path. Otherwise, assume that we want a path within a zip file.
    """

    xc, yc, zc = cube
    extension = _get_extension(channel)
    if isinstance(src, str):
        cur_pth = f'{src}/mag{mag_scale}/x{xc:04d}/y{yc:04d}/z{zc:04d}/{exp_name}_mag{mag_scale}_x{xc:04d}_y{yc:04d}_z{zc:04d}.{extension}'
    else:
        cur_pth = f'{exp_name}_mag{mag_scale}x{xc}y{yc}z{zc}.{extension}'

    print(cur_pth)

    return cur_pth


def _get_raw_filesystem_cube(src, pth, **kw):
    if os.path.exists(pth):
        with open(pth, 'rb') as fp:
            data_str = fp.read()
    else:
        data_str = None

    return data_str


def _get_http_cube(src, pth, **kw):
    hopeless_codes = [404, ]

    for _ in range(0, kw['http_retries']):
        r = requests.get(pth, auth=kw['auth'], timeout=60)
        if r.status_code == 200:
            return r.content
        elif r.status_code in hopeless_codes:
            break

    return None


def _get_kzip_cube(src, pth, **kw):
    try:
        return src.read(pth)
    except KeyError:
        return None


def _get_cube(src, cube, exp_name, mag_scale, channel, cube_edge, **kw):
    """
    Helper function to get contents of a cube.

    src : str or ZipFile
        If str, a filesystem or URL prefix. If ZipFile, a previously opened zip archive (.k.zip).

    cube : 3-tuple of int
    exp_name : str
    mag_scale : dict int -> 3-tuple of float
    channel : str
    cube_edge : int
    """

    pth = _get_pth_to_cube(src, cube, exp_name, mag_scale, channel)

    if isinstance(src, ZipFile):
        get_cube_fn = _get_kzip_cube
    elif src.lower().startswith('http'):
        get_cube_fn = _get_http_cube
    else:
        get_cube_fn = _get_raw_filesystem_cube

    data_str = get_cube_fn(src, pth, **kw)

    if data_str is None:
        return None
    else:
        data_str = _maybe_decompress_cube_data(data_str, channel)

    vx_size = len(data_str) / (cube_edge**3)
    if vx_size == 1:
        dtype = np.uint8
    elif vx_size == 2:
        dtype = np.uint16
    elif vx_size == 4:
        dtype = np.uint32
    elif vx_size == 8:
        dtype = np.uint64
    else:
        raise Exception('Unknown word size.')

    cube_data = np.frombuffer(data_str, dtype=dtype).reshape((cube_edge, cube_edge, cube_edge))

    return cube_data


def np_matrix_from_knossos(
        pth, ds_info, size, offset, channel='raw', mag_scale=1, to_dtype='default', thread_count=40, auth=None, http_retries=10):
    """
    Read knossos data into a numpy matrix from various sources.

    Important note: The numpy array returned is indexed in opposite order compared to the Knossos coordinates. That is,
    a voxel that Knossos finds at (x, y, z) will be at O[z, y, x] if O is the output array. This avoids any expensive
    array axes swaps in this code. Note also that this allows you to write slices out into images conveniently,
    e.g. by passing O[z, :, :] to PIL's Image.fromarray().

    However, all coordinates passed to this function are exactly like in Knossos.

    pth : str
        If the string starts with "http", try to read cubes from a http source. If the string ends with "zip", try to
        read cubes from inside a k.zip. Otherwise, try to read cubes directly from the local filesystem.

    ds_info : DatasetInfo

    size : 3-tuple of int
        Size of volume to read. Given in mag1 coordinates.

    offset : 3-tuple of int
        Offset to volume to read. Given in mag1 coordinates.

    channel : str
        One of ['jpg', 'png', 'raw', 'seg.sz']

        Note: We would like to replace this with a more general channel handling mechanism in the future, where there
        might be multiple channels for the same data format.

    mag_scale : int
        Index into the mag_scales dict of ds_info and component of a mag directory name.

        Note: In old-style Knossos, we only had mags [1, 2, 4, 8, ...], with factor 2 downscaling along each axis
        with every step. PyKnossos is more flexible about this, with the mags being numbered without gaps and
        arbitrary, potentially anisotropic downscaling. The mag_scales attribute on ds_info reflects this, allowing to
        arbitrarily map "mag numbers" to physical voxel scaling. The mag numbers are still used in the knossos dataset
        directory names and cube file names in any case (but they do not have to be powers of two).

    to_dtype : numpy dtype or str
        Datatype of the returned numpy array.

        If str ('default'), use whatever datatype comes out of the first cube that is read, or np.uint8 if there is
        no valid data. Otherwise, if numpy dtype, enforce that datatype.

    thread_count : int
        Number of threads used to load cubes

    auth : None or 2-tuple of str
        If http auth is required, specify (username, password).
    """

    # This is a 1-element list because we are loading cubes in parallel below, but only allocating the output array
    # once we have successfully read the first cube (so that we can probe for the data type if to_dtype=='default').
    # This requires a mutable variable visible to all worker threads. The first (and only) element will be None until
    # the first cube is read, at which point it will be replaced by a np array.
    output = [None, ]

    if pth.lower().endswith('zip'):
        pth = ZipFile(pth, 'r')

    down_factor = tuple(xx / yy for xx, yy in zip(ds_info.mag_scales[mag_scale], ds_info.mag_scales[1]))
    size = tuple(round(xx / yy) for xx, yy in zip(size, down_factor))
    size_out = (size[2], size[1], size[0])
    offset = tuple(round(xx / yy) for xx, yy in zip(offset, down_factor))

    output_lock = Lock()
    def _worker(*args):
        cube, lb, ob = args[0]
        cube_data = _get_cube(pth, cube, ds_info.exp_name, mag_scale, channel, ds_info.cube_edge, auth=auth, http_retries=http_retries)

        if cube_data is None:
            print('No data.')
            return

        output_lock.acquire()
        if output[0] is None:
            # Doing this here so that we can use a cube that has been read to probe for the data type
            if to_dtype == 'default':
                output[0] = np.zeros(size_out, dtype=cube_data.dtype)
            else:
                output[0] = np.zeros(size_out, dtype=to_dtype)
        output_lock.release()

        output[0][ob[2][0]:ob[2][1],
                  ob[1][0]:ob[1][1],
                  ob[0][0]:ob[0][1]] = cube_data[
                       lb[2][0]:lb[2][1],
                       lb[1][0]:lb[1][1],
                       lb[0][0]:lb[0][1]]

    cube_slices_to_get = list(_iter_cubes(size, offset, ds_info.cube_edge))
    print(cube_slices_to_get)
    with ThreadPoolExecutor(max_workers=thread_count) as p:
        r = p.map(_worker, cube_slices_to_get)
        r = list(r) # wait for results

    if output[0] is None:
        # We have never read from a valid cube
        print('No data found in requested range, returning zeros.')
        if to_dtype == 'default':
            output[0] = np.zeros(size_out, dtype=np.uint8)
        else:
            output[0] = np.zeros(size_out, dtype=to_dtype)

    if isinstance(pth, ZipFile):
        pth.close()

    return output[0]


def oldstyle_knossos_mag_range(mag_1 : Tuple[float, float, float], mag_count : int) -> Dict[int, Tuple[float, float, float]]:
    """
    Convenience function to create the mag_scales dictionary required for DatasetInfo when the mags are downsampled
    in the "old style" Knossos way, i.e. the mags are named 1, 2, 4, 8, ... and the downsampling is by a factor of
    2 in each dimension for every step.

    mag_1 : Resolution of mag 1
    mag_count : How many mags there are in total
    """

    mag_dict = {}
    for cur_mag in (2**x for x in range(0, mag_count)):
        mag_dict[cur_mag] = tuple(xx * cur_mag for xx in mag_1)

    return mag_dict


def main():
    exp_name = 'Dataset'
    bounds = (1024, 1024, 1024)
    mag_scales = {1: (100., 100., 100.)}
    pth = '/path/to/data/'
    size = (256, 256, 256)
    of = (1024, 1024, 1024)


if __name__ == '__main__':
    main()