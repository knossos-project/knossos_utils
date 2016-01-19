# Knossos Python-Tools
This repository serves as a place for python code that interacts with **KNOSSOS** data sets or annotation files. However, the code here is standalone and you won't find **KNOSSOS-Plugins** here (but feel free to be inspired by it when writing plugins).

# knossosdataset
knossosdataset provides easy access tp a **KNOSSOS**-dataset for reading and writing raw and overlay data via Python. It is also able to create **KNOSSOS**-datasets from scratch by writing knossos.conf's and creating the necessary directories. Moreover, it provides mergelist_tools for working with supervoxel-representations in a **KNOSSOS**-dataset.

# chunky
chunky provides a meta-representation of a knossosdataset to work on the raw data without the need to copy and save it in a different format. It is currently in an early-release stage and will be soon improved in terms of readability (comments, docstrings, examples).
