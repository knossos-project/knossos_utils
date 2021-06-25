import numpy as np
import pytest

from knossos_utils import KnossosDataset


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
