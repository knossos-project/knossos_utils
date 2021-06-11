from knossos_utils import KnossosDataset
import pytest

import numpy as np

@pytest.mark.parametrize("boundary", [np.array([7,9,10]), (7,9,10), [7, 9, 10]])
def test_KnossosDataset_initialize_without_conf(tmp_path, boundary):
  from knossos_utils import KnossosDataset
  import numpy as np
  kd = KnossosDataset()
  kd.initialize_without_conf(str(tmp_path), boundary=boundary, scale=(1, 1, 1), experiment_name='test', mags=[1], verbose=True)

