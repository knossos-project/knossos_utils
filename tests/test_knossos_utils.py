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
