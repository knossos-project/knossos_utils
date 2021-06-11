from knossos_utils import KnossosDataset
  
def test_KnossosDataset_initialize_without_conf(tmp_path):
  from knossos_utils import KnossosDataset
  kd = KnossosDataset()
  kd.initialize_without_conf(str(tmp_path), boundary=(7, 9, 10), scale=(1, 1, 1), experiment_name='test', verbose=True)
  
