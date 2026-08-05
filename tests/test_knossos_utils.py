import json
import os
import zipfile

import numpy as np
import pytest

from knossos_utils import KnossosDataset
from knossos_utils.knossosdataset import (
    _file_url_to_path,
    _precomputed_kvstore_config,
)


def _minimal_toml_config(experiment_name="TestDataset"):
    return f"""[[Layer]]
Name = "{experiment_name}"
FileExtension = [".raw"]
Extent_px = [4, 3, 2]
VoxelSize_nm = [[8, 8, 8]]
CubeShape_px = [2, 2, 2]
Description = "test"

"""


@pytest.mark.parametrize("boundary", [np.array([7, 9, 10]), (7, 9, 10), [7, 9, 10]])
def test_KnossosDataset_initialize_without_conf__boundary(tmp_path, boundary):
    kd = KnossosDataset()
    kd.initialize_without_conf(
        str(tmp_path),
        boundary=boundary,
        scale=(1, 1, 1),
        experiment_name="test",
        mags=[1],
        verbose=True,
    )


def test_KnossosDataset_initialize_without_conf__mags(tmp_path):
    kd = KnossosDataset()
    with pytest.raises(AssertionError):
        kd.initialize_without_conf(
            str(tmp_path),
            boundary=(7, 9, 10),
            scale=(1, 1, 1),
            experiment_name="test",
            mags=None,
            verbose=True,
        )


def test_KnossosDataset_initalize_without_conf__mags_1_make_mag_folder_False(tmp_path):
    kd = KnossosDataset()
    with pytest.raises(AssertionError):
        kd.initialize_without_conf(
            str(tmp_path),
            boundary=(7, 9, 19),
            scale=(1, 1, 1),
            experiment_name="test",
            mags=[1],
            make_mag_folders=False,
            verbose=True,
        )


def test_KnossosDataset_initialize_without_conf__conf_exist(tmp_path):
    from pathlib import Path

    kd = KnossosDataset()
    kd.initialize_without_conf(
        str(tmp_path),
        boundary=(7, 9, 10),
        scale=(1, 1, 1),
        experiment_name="test",
        mags=[1],
        verbose=True,
    )
    assert Path(kd.conf_path).is_file()


@pytest.mark.parametrize(
    "existing_mag,expected_mag",
    [
        ("test_mag16", "test_mag1"),
        ("test_mag1", "test_mag1"),
        ("mag16", "mag1"),
        ("mag1", "mag1"),
    ],
)
def test_Knossosdataset__initalize_without_conf__robust_magfolder_detection(
    tmp_path, existing_mag, expected_mag
):
    (tmp_path / existing_mag).mkdir()
    kd = KnossosDataset()
    kd.initialize_without_conf(
        str(tmp_path),
        boundary=(7, 9, 10),
        scale=(1, 1, 1),
        experiment_name="test",
        mags=[1],
        verbose=True,
    )
    assert (tmp_path / expected_mag).is_dir()


def test_KnossosDataset_from_toml_string_without_conf_path():
    kd = KnossosDataset.from_toml_string("""[[Layer]]
Name = "TestDataset"
FileExtension = [".jpg", ".png"]
Extent_px = [128,128, 128]
VoxelSize_nm = [[8,8,8]]
CubeShape_px = [128, 128, 128]
Description = "test"

""")

    assert isinstance(kd, KnossosDataset)
    assert kd.experiment_name == "TestDataset"
    assert len(kd.layers) == 1
    assert kd.layers[0].experiment_name == kd.experiment_name
    assert kd.conf_path is None
    assert kd.url is None
    assert np.array_equal(kd.boundary, [128, 128, 128])
    assert np.array_equal(kd.cube_shape, [128, 128, 128])
    assert np.array_equal(kd.scale, [8, 8, 8])
    assert len(kd.scales) == 1


def test_KnossosDataset_from_toml_path_without_server_format_classic_raw_roundtrip(tmp_path):
    toml_path = tmp_path / "classic.k.toml"
    toml_path.write_text(_minimal_toml_config("ClassicDataset"))
    data = np.arange(8, dtype=np.uint8).reshape((2, 2, 2))

    kd = KnossosDataset(str(toml_path))
    kd.save_raw(
        data=data,
        data_mag=1,
        offset=(0, 0, 0),
        mags=[1],
        upsample=False,
        downsample=False,
    )
    loaded = kd.load_raw(offset=(0, 0, 0), size=(2, 2, 2), mag=1)

    assert kd.server_format is None
    assert kd._tensorstore_datasets is None
    assert kd._dtype == np.uint8
    assert kd.knossos_path == os.path.abspath(str(tmp_path))
    assert np.array_equal(loaded, data)


def test_KnossosDataset_initialize_from_kzip_reads_external_toml_from_annotation(tmp_path):
    toml_path = tmp_path / "dataset.k.toml"
    toml_path.write_text(_minimal_toml_config("ExternalDataset"))
    kzip_path = tmp_path / "annotation.k.zip"
    annotation_xml = f"""<things>
    <parameters>
        <dataset path="{toml_path}" />
    </parameters>
</things>"""

    with zipfile.ZipFile(kzip_path, "w") as zf:
        zf.writestr("annotation.xml", annotation_xml)

    kd = KnossosDataset(kzip_path)

    assert kd.experiment_name == "ExternalDataset"
    assert kd.conf_path == str(toml_path)
    assert np.array_equal(kd.boundary, [4, 3, 2])
    assert np.array_equal(kd.cube_shape, [2, 2, 2])
    assert np.array_equal(kd.scale, [8, 8, 8])


def test_KnossosDataset_initialize_from_kzip_reads_file_url_toml_from_annotation(tmp_path):
    toml_path = tmp_path / "dataset.k.toml"
    toml_path.write_text(_minimal_toml_config("FileUrlDataset"))
    kzip_path = tmp_path / "annotation_file_url.k.zip"
    annotation_xml = f"""<things>
    <parameters>
        <dataset path="{toml_path.as_uri()}" />
    </parameters>
</things>"""

    with zipfile.ZipFile(kzip_path, "w") as zf:
        zf.writestr("annotation.xml", annotation_xml)

    kd = KnossosDataset(kzip_path)

    assert kd.experiment_name == "FileUrlDataset"
    assert kd.conf_path == str(toml_path)
    assert kd.knossos_path == os.path.abspath(str(tmp_path))
    assert np.array_equal(kd.boundary, [4, 3, 2])


def test_KnossosDataset_initialize_from_kzip_reads_embedded_toml(tmp_path):
    kzip_path = tmp_path / "embedded.k.zip"

    with zipfile.ZipFile(kzip_path, "w") as zf:
        zf.writestr("embedded/dataset.k.toml", _minimal_toml_config("EmbeddedDataset"))

    kd = KnossosDataset(kzip_path)

    assert kd.experiment_name == "EmbeddedDataset"
    assert kd.conf_path == f"{kzip_path}/embedded/dataset.k.toml"
    assert np.array_equal(kd.boundary, [4, 3, 2])
    assert np.array_equal(kd.cube_shape, [2, 2, 2])
    assert np.array_equal(kd.scale, [8, 8, 8])


def test_KnossosDataset_save_raw_rejects_uint16_for_classic_knossos_cubes(tmp_path):
    kd = KnossosDataset.initialize(
        tmp_path,
        experiment_name="classic",
        boundary=(4, 3, 2),
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        file_extensions=[".raw"],
        server_format="knossos",
    )
    data = np.zeros((2, 3, 4), dtype=np.uint16)

    with pytest.raises(ValueError, match="uint16 raw data is only supported for precomputed"):
        kd.save_raw(
            data=data,
            data_mag=1,
            offset=(0, 0, 0),
            mags=[1],
            upsample=False,
            downsample=False,
            datatype=np.uint16,
        )


def test_KnossosDataset_save_raw_accepts_uint16_for_precomputed_tensorstore():
    class FakeDomain:
        shape = (4, 3, 2, 1)

    class FakeTensorstoreDataset:
        def __init__(self):
            self.written = None
            self.domain = FakeDomain()

        def __setitem__(self, key, value):
            self.written = (key, value)

    dataset = FakeTensorstoreDataset()
    kd = KnossosDataset()
    kd._initialized = True
    kd.server_format = "precomputed"
    kd._tensorstore_datasets = {1: dataset}
    kd._rgb_channel = None
    kd._dtype = np.uint16
    kd.scales = [np.array([1, 1, 1])]
    kd._ordinal_mags = True
    kd._boundary = np.array([4, 3, 2])
    kd._cube_shape = np.array([2, 2, 2])
    data = np.arange(24, dtype=np.uint16).reshape((2, 3, 4))

    kd.save_raw(
        data=data,
        data_mag=1,
        offset=(0, 0, 0),
        mags=[1],
        upsample=False,
        downsample=False,
        datatype=np.uint16,
    )

    assert dataset.written is not None
    _, written = dataset.written
    assert written.dtype == np.uint16
    assert np.array_equal(written.swapaxes(0, 2), data)


def test_KnossosDataset_save_to_kzip_ignores_precomputed_server_format(monkeypatch, tmp_path):
    class FakeSnappy:
        @staticmethod
        def compress(data):
            return np.asarray(data, dtype=np.uint64).tobytes()

    kd = KnossosDataset(show_progress=False)
    monkeypatch.setitem(kd.module_wide, "snappy", FakeSnappy())
    kd._initialized = True
    kd.server_format = "precomputed"
    kd._tensorstore_datasets = None
    kd._rgb_channel = None
    kd._dtype = np.uint64
    kd.scales = [np.array([1, 1, 1])]
    kd._ordinal_mags = True
    kd._boundary = np.array([2, 2, 2])
    kd._cube_shape = np.array([2, 2, 2])
    kd._experiment_name = "precomputed"
    kd._knossos_path = str(tmp_path)
    kd._conf_path = str(tmp_path / "precomputed.k.toml")
    data = np.arange(8, dtype=np.uint64).reshape((2, 2, 2))

    kzip_path = tmp_path / "segmentation.k.zip"
    kd.save_to_kzip(
        data=data,
        data_mag=1,
        kzip_path=kzip_path,
        offset=(0, 0, 0),
        mags=[1],
        gen_mergelist=False,
        upsample=False,
        downsample=False,
    )

    with zipfile.ZipFile(kzip_path, "r") as zf:
        assert "precomputed_mag1x0y0z0.seg.sz" in zf.namelist()


def test_KnossosDataset_load_embedded_kzip_ignores_precomputed_server_format(tmp_path):
    data = np.arange(8, dtype=np.uint8).reshape((2, 2, 2))
    kzip_path = tmp_path / "embedded_precomputed.k.zip"
    cube_path = (
        "embedded/mag1/x0000/y0000/z0000/"
        "EmbeddedPrecomputed_mag1_x0000_y0000_z0000.raw"
    )

    with zipfile.ZipFile(kzip_path, "w") as zf:
        zf.writestr(cube_path, data.tobytes())

    kd = KnossosDataset(show_progress=False)
    kd._initialized = True
    kd._initialize_cache(0)
    kd.server_format = "precomputed"
    kd._tensorstore_datasets = None
    kd._rgb_channel = None
    kd._dtype = np.uint8
    kd.scales = [np.array([1, 1, 1])]
    kd._ordinal_mags = True
    kd._boundary = np.array([2, 2, 2])
    kd._cube_shape = np.array([2, 2, 2])
    kd._experiment_name = "EmbeddedPrecomputed"
    kd.file_extensions = [".raw"]
    kd._conf_path = f"{kzip_path}/embedded/dataset.k.toml"
    kd._knossos_path = f"{kzip_path}/embedded/"
    kd.layers = [kd]

    loaded = kd.load_raw(offset=(0, 0, 0), size=(2, 2, 2), mag=1)

    assert np.array_equal(loaded, data)


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter behavior")
@pytest.mark.parametrize("url", ["file://C:/data/dataset/info", "file:///C:/data/dataset/info"])
def test_file_url_to_path_preserves_windows_drive_letter(url):
    assert _file_url_to_path(url) == os.path.abspath("C:/data/dataset/info")


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter behavior")
@pytest.mark.parametrize("url", ["file://C:/data/dataset/info", "file:///C:/data/dataset/info"])
def test_precomputed_kvstore_config__file_urls_preserve_windows_drive_letter(url):
    assert _precomputed_kvstore_config(url) == {
        "driver": "file",
        "base_url": None,
        "path": os.path.abspath("C:/data/dataset"),
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter behavior")
@pytest.mark.parametrize("url", ["file://C:/data/dataset", "file:///C:/data/dataset"])
def test_KnossosDataset_knossos_path_preserves_windows_drive_letter(url):
    kd = KnossosDataset()
    kd.url = url

    assert kd.knossos_path == os.path.abspath("C:/data/dataset")


@pytest.mark.skipif(os.name != "nt", reason="Unix path behavior")
@pytest.mark.parametrize("url", ["file:///mnt/storage/dataset"])
def test_KnossosDataset_knossos_path_preserves_unix_path(url):
    kd = KnossosDataset()
    kd.url = url

    assert kd.knossos_path == os.path.abspath("/mnt/storage/dataset")


def test_KnossosDataset_from_toml_string_without_conf_path_precomputed():
    kd = KnossosDataset.from_toml_string("""[[Layer]]
Name = "TestDataset"
ServerFormat = "precomputed"
FileExtension = [".jpg", ".png"]
Extent_px = [128,128, 128]
VoxelSize_nm = [[8,8,8]]
CubeShape_px = [128, 128, 128]
Description = "test"

""")

    assert isinstance(kd, KnossosDataset)
    assert kd.experiment_name == "TestDataset"
    assert kd.server_format == "precomputed"
    assert len(kd.layers) == 1
    assert kd.layers[0].experiment_name == kd.experiment_name
    assert kd.conf_path is None
    assert kd.url is None
    assert len(kd._tensorstore_datasets) > 0
    assert np.array_equal(kd.boundary, [128, 128, 128])
    assert np.array_equal(kd.cube_shape, [128, 128, 128])
    assert np.array_equal(kd.scale, [8, 8, 8])
    assert np.array_equal(kd.scales[0], [8, 8, 8])


def test_KnossosDataset_from_toml_string_precomputed_without_conf_path_roundtrip():
    kd = KnossosDataset.from_toml_string("""[[Layer]]
Name = "TestDataset"
ServerFormat = "precomputed"
FileExtension = [".raw"]
Extent_px = [4, 3, 2]
VoxelSize_nm = [[8, 8, 8], [16, 16, 16]]
CubeShape_px = [2, 2, 2]
Description = "test"

""")
    data = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))

    kd.save_raw(
        data=data,
        data_mag=1,
        offset=(0, 0, 0),
        mags=[1],
        upsample=False,
        downsample=False,
    )
    loaded = kd.load_raw(offset=(0, 0, 0), size=(4, 3, 2), mag=1)

    assert kd.url is None
    assert tuple(kd._tensorstore_datasets[1].domain.shape) == (4, 3, 2, 1)
    assert np.array_equal(loaded, data)


def test_KnossosDataset_initialize_from_array__raw_precomputed_uint16_roundtrip(tmp_path):
    data = np.arange(24, dtype=np.uint16).reshape((2, 3, 4))
    data[0, 0, 0] = 2**12

    kd = KnossosDataset.initialize_from_array(
        data=data,
        experiment_name="raw16",
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".raw"],
        write_path=str(tmp_path),
        server_format="precomputed",
        dtype=np.uint16,
    )

    info = json.loads((tmp_path / "info").read_text())
    loaded = kd.load_raw(offset=(0, 0, 0), size=(4, 3, 2), mag=1)

    assert info["data_type"] == "uint16"
    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, data)

    reloaded = KnossosDataset(kd.conf_path)
    assert reloaded._dtype == np.uint16
    assert reloaded.load_raw(offset=(0, 0, 0), size=(4, 3, 2), mag=1).dtype == np.uint16


def test_KnossosDataset_initialize_from_array__raw_precomputed_uint8_to_uint16_roundtrip(tmp_path):
    data = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))

    with pytest.warns(UserWarning, match="Data type mismatch"):
        kd = KnossosDataset.initialize_from_array(
            data=data,
            experiment_name="raw16",
            cube_shape=(2, 2, 2),
            scale=(1, 1, 1),
            ds_factor=(2, 2, 1),
            file_extensions=[".raw"],
            write_path=str(tmp_path),
            server_format="precomputed",
            dtype=np.uint16,
        )

    loaded = kd.load_raw(offset=(0, 0, 0), size=(4, 3, 2), mag=1)

    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, data.astype(np.uint16))


def test_KnossosDataset_initialize_from_array__as_rgb_precomputed(tmp_path):
    data = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    data[..., 0] = 11
    data[..., 1] = 22
    data[..., 2] = 33

    kd = KnossosDataset.initialize_from_array(
        data=data,
        experiment_name="rgb",
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".raw"],
        write_path=str(tmp_path),
        server_format="precomputed",
        as_rgb=True,
    )

    assert len(kd.layers) == 3
    assert [layer._rgb_channel for layer in kd.layers] == ["r_1", "g_1", "b_1"]
    assert kd.layers[0]._tensorstore_datasets is kd.layers[1]._tensorstore_datasets
    assert kd.layers[0]._tensorstore_datasets is kd.layers[2]._tensorstore_datasets
    assert json.loads((tmp_path / "info").read_text())["num_channels"] == 3

    reloaded = KnossosDataset(kd.conf_path)
    assert len(reloaded.layers) == 3
    for idx, layer in enumerate(reloaded.layers):
        loaded = layer.load_raw(offset=(0, 0, 0), size=(4, 3, 2), mag=1)
        assert np.array_equal(loaded, data[..., idx])


def _assert_segmentation_precomputed_roundtrip(kd, data, tmp_path, expected_size):
    info = json.loads((tmp_path / "info").read_text())
    assert info["data_type"] == "uint64"
    assert info["scales"][0]["encoding"] == "compressed_segmentation"
    assert info["scales"][0]["size"] == expected_size
    assert np.array_equal(kd.boundary, expected_size)
    assert tuple(kd._tensorstore_datasets[1].domain.shape) == (*expected_size, 1)

    loaded = kd.load_seg(offset=(0, 0, 0), size=expected_size, mag=1)
    assert loaded.dtype == np.uint64
    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)

    reloaded = KnossosDataset(kd.conf_path)
    assert np.array_equal(reloaded.boundary, expected_size)
    assert tuple(reloaded._tensorstore_datasets[1].domain.shape) == (*expected_size, 1)

    loaded_after_reload = reloaded.load_seg(offset=(0, 0, 0), size=expected_size, mag=1)
    assert loaded_after_reload.dtype == np.uint64
    assert loaded_after_reload.shape == data.shape
    assert np.array_equal(loaded_after_reload, data)


def _segmentation_test_data():
    data = np.zeros((2, 3, 4), dtype=np.uint64)
    data[0, 0, 0] = 1
    data[0, 1, 2] = 42
    data[1, 2, 3] = 2**33
    return data


def _segmentation_2d_test_data():
    data = np.zeros((1, 3, 4), dtype=np.uint64)
    data[0, 0, 0] = 1
    data[0, 1, 2] = 42
    data[0, 2, 3] = 2**33
    return data


def test_KnossosDataset_save_seg_load_seg_precomputed_roundtrip(tmp_path):
    data = np.zeros((2, 2, 2), dtype=np.uint64)
    data[0, 0, 0] = 1
    data[0, 1, 1] = 42
    data[1, 1, 1] = 2**33
    expected = np.zeros((4, 4, 4), dtype=np.uint64)
    expected[1:3, 1:3, 1:3] = data

    kd = KnossosDataset.initialize(
        str(tmp_path),
        experiment_name="seg_direct",
        boundary=(4, 4, 4),
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".seg.sz.zip"],
        server_format="precomputed",
    )

    kd.save_seg(
        data=data,
        data_mag=1,
        offset=(1, 1, 1),
        mags=[1],
        upsample=False,
        downsample=False,
    )
    loaded = kd.load_seg(offset=(0, 0, 0), size=(4, 4, 4), mag=1)

    assert loaded.dtype == np.uint64
    assert loaded.shape == expected.shape
    assert np.array_equal(loaded, expected)


def test_KnossosDataset_save_seg_load_seg_classic_knossos_roundtrip(tmp_path):
    data = np.zeros((2, 2, 2), dtype=np.uint64)
    data[0, 0, 0] = 1
    data[0, 1, 1] = 42
    data[1, 1, 1] = 2**33
    expected = np.zeros((4, 4, 4), dtype=np.uint64)
    expected[1:3, 1:3, 1:3] = data

    kd = KnossosDataset.initialize(
        str(tmp_path),
        experiment_name="seg_direct_classic",
        boundary=(4, 4, 4),
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".seg.sz.zip"],
        server_format="knossos",
    )

    kd.save_seg(
        data=data,
        data_mag=1,
        offset=(1, 1, 1),
        mags=[1],
        upsample=False,
        downsample=False,
    )
    loaded = kd.load_seg(offset=(0, 0, 0), size=(4, 4, 4), mag=1)

    assert kd.server_format == "knossos"
    assert loaded.dtype == np.uint64
    assert loaded.shape == expected.shape
    assert np.array_equal(loaded, expected)


def test_KnossosDataset_save_raw_clips_to_dataset_boundary(tmp_path):
    kd = KnossosDataset.initialize(
        str(tmp_path),
        experiment_name="clip_boundary",
        boundary=(4, 3, 2),
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".raw"],
        server_format="knossos",
    )
    # data extends 2 voxels past the boundary in x
    data = np.arange(2 * 3 * 6, dtype=np.uint8).reshape((2, 3, 6))

    with pytest.warns(UserWarning, match="clipping write region"):
        kd.save_raw(
            data=data,
            data_mag=1,
            offset=(0, 0, 0),
            mags=[1],
            upsample=False,
            downsample=False,
        )

    loaded = kd.load_raw(offset=(0, 0, 0), size=(4, 3, 2), mag=1)
    assert np.array_equal(loaded, data[:, :, :4])


def test_KnossosDataset_save_seg_rejects_non_uint64_data(tmp_path):
    kd = KnossosDataset.initialize(
        str(tmp_path),
        experiment_name="seg_dtype",
        boundary=(2, 2, 2),
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".seg.sz.zip"],
        server_format="precomputed",
    )

    with pytest.raises(ValueError, match="np.uint64"):
        kd.save_seg(
            data=np.ones((2, 2, 2), dtype=np.uint8),
            data_mag=1,
            offset=(0, 0, 0),
            mags=[1],
            upsample=False,
            downsample=False,
        )


def test_KnossosDataset_initialize_from_array__segmentation_precomputed_2d_roundtrip(
    tmp_path,
):
    data = _segmentation_2d_test_data()

    kd = KnossosDataset.initialize_from_array(
        data=data,
        experiment_name="seg2d",
        cube_shape=(2, 2, 1),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".seg.sz.zip"],
        write_path=str(tmp_path),
        server_format="precomputed",
    )

    _assert_segmentation_precomputed_roundtrip(
        kd, data, tmp_path, expected_size=[4, 3, 1]
    )


def test_KnossosDataset_initialize_from_array__segmentation_precomputed_3d_roundtrip(
    tmp_path,
):
    data = _segmentation_test_data()

    kd = KnossosDataset.initialize_from_array(
        data=data,
        experiment_name="seg",
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".seg.sz.zip"],
        write_path=str(tmp_path),
        server_format="precomputed",
    )

    _assert_segmentation_precomputed_roundtrip(
        kd, data, tmp_path, expected_size=[4, 3, 2]
    )


def test_KnossosDataset_initialize_from_array__segmentation_precomputed_roundtrip_with_shard_size(
    tmp_path,
):
    data = _segmentation_test_data()

    kd = KnossosDataset.initialize_from_array(
        data=data,
        experiment_name="seg",
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=[".seg.sz.zip"],
        write_path=str(tmp_path),
        server_format="precomputed",
        shard_size=(4, 4, 2),
    )

    info = json.loads((tmp_path / "info").read_text())
    assert "sharding" in info["scales"][0]
    assert tuple(kd.shard_size) == (4, 4, 2)
    _assert_segmentation_precomputed_roundtrip(
        kd, data, tmp_path, expected_size=[4, 3, 2]
    )


@pytest.mark.parametrize(
    "url,cdn_token,expected",
    [
        (
            "https://example.org/dataset/info",
            None,
            {"driver": "http", "base_url": "https://example.org", "path": "/dataset"},
        ),
        (
            "https://user:pass@example.org:8443/dataset/info",
            None,
            {
                "driver": "http",
                "base_url": "https://user:pass@example.org:8443",
                "path": "/dataset",
            },
        ),
        (
            "https://example.org/dataset/info",
            {"token": "abc", "expires": "123", "token_path": "/dataset"},
            {
                "driver": "http",
                "base_url": "https://example.org?token=abc&expires=123&token_path=/dataset",
                "path": "/dataset",
            },
        ),
        (
            "https://example.org/dataset/infobox/info",
            None,
            {
                "driver": "http",
                "base_url": "https://example.org",
                "path": "/dataset/infobox",
            },
        ),
    ],
)
def test_precomputed_kvstore_config__http_urls(url, cdn_token, expected):
    assert _precomputed_kvstore_config(url, cdn_token) == expected
