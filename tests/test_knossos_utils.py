import json

import numpy as np
import pytest

from knossos_utils import KnossosDataset
from knossos_utils.knossosdataset import _precomputed_kvstore_config


@pytest.mark.parametrize("boundary", [np.array([7,9,10]), (7,9,10), [7, 9, 10]])
def test_KnossosDataset_initialize_without_conf__boundary(tmp_path, boundary):
    kd = KnossosDataset()
    kd.initialize_without_conf(str(tmp_path), boundary=boundary, scale=(1, 1, 1), experiment_name='test', mags=[1], verbose=True)


def test_KnossosDataset_initialize_without_conf__mags(tmp_path):
    kd = KnossosDataset()
    with pytest.raises(AssertionError):
        kd.initialize_without_conf(str(tmp_path), boundary=(7, 9, 10), scale=(1, 1, 1), experiment_name='test', mags=None, verbose=True)


def test_KnossosDataset_initalize_without_conf__mags_1_make_mag_folder_False(tmp_path):
    kd = KnossosDataset()
    with pytest.raises(AssertionError):
        kd.initialize_without_conf(str(tmp_path), boundary=(7, 9, 19), scale=(1, 1, 1), experiment_name='test', mags=[1], make_mag_folders=False, verbose=True)


def test_KnossosDataset_initialize_without_conf__conf_exist(tmp_path):
    from pathlib import Path
    kd = KnossosDataset()
    kd.initialize_without_conf(str(tmp_path), boundary=(7, 9, 10), scale=(1, 1, 1), experiment_name='test', mags=[1], verbose=True)
    assert(Path(kd.conf_path).is_file())


@pytest.mark.parametrize('existing_mag,expected_mag', [
    ('test_mag16', 'test_mag1'),
    ('test_mag1', 'test_mag1'),
    ('mag16', 'mag1'),
    ('mag1', 'mag1')
])
def test_Knossosdataset__initalize_without_conf__robust_magfolder_detection(tmp_path, existing_mag, expected_mag):
    (tmp_path / existing_mag).mkdir()
    kd = KnossosDataset()
    kd.initialize_without_conf(str(tmp_path), boundary=(7, 9, 10), scale=(1, 1, 1), experiment_name='test', mags=[1], verbose=True)
    assert((tmp_path / expected_mag).is_dir())


def test_KnossosDataset_initialize_from_array__as_rgb_precomputed(tmp_path):
    data = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    data[..., 0] = 11
    data[..., 1] = 22
    data[..., 2] = 33

    kd = KnossosDataset.initialize_from_array(
        data=data,
        experiment_name='rgb',
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=['.raw'],
        write_path=str(tmp_path),
        server_format='precomputed',
        as_rgb=True,
    )

    assert len(kd.layers) == 3
    assert [layer._rgb_channel for layer in kd.layers] == ['r_1', 'g_1', 'b_1']
    assert kd.layers[0]._tensorstore_datasets is kd.layers[1]._tensorstore_datasets
    assert kd.layers[0]._tensorstore_datasets is kd.layers[2]._tensorstore_datasets
    assert json.loads((tmp_path / 'info').read_text())['num_channels'] == 3

    reloaded = KnossosDataset(kd.conf_path)
    assert len(reloaded.layers) == 3
    for idx, layer in enumerate(reloaded.layers):
        loaded = layer.load_raw(offset=(0, 0, 0), size=(4, 3, 2), mag=1)
        assert np.array_equal(loaded, data[..., idx])


def test_KnossosDataset_initialize_from_array__segmentation_precomputed_roundtrip(tmp_path):
    data = np.zeros((2, 3, 4), dtype=np.uint64)
    data[0, 0, 0] = 1
    data[0, 1, 2] = 42
    data[1, 2, 3] = 2**33

    kd = KnossosDataset.initialize_from_array(
        data=data,
        experiment_name='seg',
        cube_shape=(2, 2, 2),
        scale=(1, 1, 1),
        ds_factor=(2, 2, 1),
        file_extensions=['.seg.sz.zip'],
        write_path=str(tmp_path),
        server_format='precomputed',
    )

    info = json.loads((tmp_path / 'info').read_text())
    assert info['data_type'] == 'uint64'
    assert info['scales'][0]['encoding'] == 'compressed_segmentation'
    assert info['scales'][0]['size'] == [4, 3, 2]
    assert np.array_equal(kd.boundary, [4, 3, 2])
    assert tuple(kd._tensorstore_datasets[1].domain.shape) == (4, 3, 2, 1)

    loaded = kd.load_seg(offset=(0, 0, 0), size=(4, 3, 2), mag=1)
    assert loaded.dtype == np.uint64
    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)

    reloaded = KnossosDataset(kd.conf_path)
    assert np.array_equal(reloaded.boundary, [4, 3, 2])
    assert tuple(reloaded._tensorstore_datasets[1].domain.shape) == (4, 3, 2, 1)

    loaded_after_reload = reloaded.load_seg(offset=(0, 0, 0), size=(4, 3, 2), mag=1)
    assert loaded_after_reload.dtype == np.uint64
    assert loaded_after_reload.shape == data.shape
    assert np.array_equal(loaded_after_reload, data)


@pytest.mark.parametrize(
    'url,cdn_token,expected',
    [
        (
            'https://example.org/dataset/info',
            None,
            {'driver': 'http', 'base_url': 'https://example.org', 'path': '/dataset'},
        ),
        (
            'https://user:pass@example.org:8443/dataset/info',
            None,
            {'driver': 'http', 'base_url': 'https://user:pass@example.org:8443', 'path': '/dataset'},
        ),
        (
            'https://example.org/dataset/info',
            {'token': 'abc', 'expires': '123', 'token_path': '/dataset'},
            {'driver': 'http', 'base_url': 'https://example.org?token=abc&expires=123&token_path=/dataset', 'path': '/dataset'},
        ),
        (
            'https://example.org/dataset/infobox/info',
            None,
            {'driver': 'http', 'base_url': 'https://example.org', 'path': '/dataset/infobox'},
        ),
    ],
)
def test_precomputed_kvstore_config__http_urls(url, cdn_token, expected):
    assert _precomputed_kvstore_config(url, cdn_token) == expected
