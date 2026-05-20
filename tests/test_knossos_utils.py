import json
import os
import zipfile

import numpy as np
import pytest

from knossos_utils import KnossosDataset
from knossos_utils.knossosdataset import _precomputed_kvstore_config


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
    class FakeTensorstoreDataset:
        def __init__(self):
            self.written = None

        def __setitem__(self, key, value):
            self.written = (key, value)

    dataset = FakeTensorstoreDataset()
    kd = KnossosDataset()
    kd._initialized = True
    kd.server_format = "precomputed"
    kd._tensorstore_datasets = {1: dataset}
    kd._rgb_channel = None
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


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter behavior")
@pytest.mark.parametrize("url", ["file://C:/data/dataset", "file:///C:/data/dataset", "file:///mnt/storage/dataset"])
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
VoxelSize_nm = [[8, 8, 8]]
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
        shard_size=(2, 2, 2),
    )

    info = json.loads((tmp_path / "info").read_text())
    assert "sharding" in info["scales"][0]
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
