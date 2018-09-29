import numpy as np
import imageio
import snappy
from zipfile import ZipFile
from io import BytesIO
import itertools as it
import os


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
    Generator that iterates the cubes and corresponding cube slice bounds required to build up a 3-dimensional
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


def _interpret_read_cube_data(data_str, channel):
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
    Generate a cube path, filesystem, remote or inside zipfile.

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


def _get_raw_filesystem_cube(src, pth):
    if os.path.exists(pth):
        with open(pth, 'rb') as fp:
            data_str = fp.read()
    else:
        data_str = None

    return data_str


def _get_http_cube(src, pth):
    raise Exception('Implement _get_cube_http() please')


def _get_kzip_cube(src, pth):
    try:
        return src.read(pth)
    except KeyError:
        return None


def _get_cube(src, cube, exp_name, mag_scale, channel, cube_edge):
    """
    Helper function to get contents of a cube. The idea is to make this fetch from http and load from zip in the
    future, too (by having src being a ZipFile object or 'http://'-prefixed str.
    """

    pth = _get_pth_to_cube(src, cube, exp_name, mag_scale, channel)

    if isinstance(src, ZipFile):
        get_cube_fn = _get_kzip_cube
    elif src.lower().startswith('http'):
        get_cube_fn = _get_http_cube
    else:
        get_cube_fn = _get_raw_filesystem_cube

    data_str = get_cube_fn(src, pth)

    if data_str is None:
        return None
    else:
        data_str = _interpret_read_cube_data(data_str, channel)

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

    cube_data = np.frombuffer(data_str, dtype=dtype).reshape((cube_edge, cube_edge, cube_edge)).T

    return cube_data


def np_matrix_from_knossos(pth, ds_info, size, offset, channel='raw', mag_scale=1, to_dtype='default'):
    """
    Read knossos data into a numpy matrix from various sources.

    pth : str
        If the string starts with "http", try to read data from a http source. If the string ends with "zip", try to
        read data from a k.zip. Otherwise, try to read from the local filesystem.

    ds_info : DatasetInfo

    size : 3-tuple of int
        Size of volume to read

    offset : 3-tuple of int
        Offset to volume to read

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
    """

    output = None

    if pth.lower().endswith('zip'):
        pth = ZipFile(pth, 'r')

    down_factor = tuple(xx / yy for xx, yy in zip(ds_info.mag_scales[mag_scale], ds_info.mag_scales[1]))
    size = tuple(round(xx / yy) for xx, yy in zip(size, down_factor))
    offset = tuple(round(xx / yy) for xx, yy in zip(offset, down_factor))

    for cube, lb, ob in _iter_cubes(size, offset, ds_info.cube_edge):
        cube_data = _get_cube(pth, cube, ds_info.exp_name, mag_scale, channel, ds_info.cube_edge)

        if cube_data is None:
            continue

        if output is None:
            # Doing this here so that we can use a read cube to probe for the output datatype.
            if to_dtype == 'default':
                output = np.zeros(size, dtype=cube_data.dtype)
            else:
                output = np.zeros(size, dtype=to_dtype)

        output[ob[0][0]:ob[0][1],
               ob[1][0]:ob[1][1],
               ob[2][0]:ob[2][1]] = cube_data[
                       lb[0][0]:lb[0][1],
                       lb[1][0]:lb[1][1],
                       lb[2][0]:lb[2][1]]

    if output is None:
        # We have never read from a valid cube
        print('No data found in requested range, returning zeros.')
        if to_dtype == 'default':
            output = np.zeros(size, dtype=np.uint8)
        else:
            output = np.zeros(size, dtype=to_dtype)

    if isinstance(pth, ZipFile):
        pth.close()

    return output


def main():
    exp_name = 'Platy1607'
    bounds = (2750, 2592, 2854)
    mag_scales = {1: (100., 100., 100.)}
    pth = '/mnt/storage02/projects/darendt-1-as/membrane/membrane/prob_map_cubes_100nm_2/'
    size = (256, 256, 256)
    of = (1279, 1279, 511)

    ds_info = DatasetInfo(exp_name, bounds, mag_scales)

    O = np_matrix_from_knossos(pth, ds_info, size, of, channel='png')


if __name__ == '__main__':
    main()
