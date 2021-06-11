from knossos_utils import KnossosDataset
import pytest

def test_KnossosDataset_initialize_without_conf(tmp_path):
  from knossos_utils import KnossosDataset
  import numpy as np
  kd = KnossosDataset()
  kd.initialize_without_conf(str(tmp_path), boundary=np.array([7, 9, 10]), scale=(1, 1, 1), experiment_name='test', mags=[1], verbose=True)