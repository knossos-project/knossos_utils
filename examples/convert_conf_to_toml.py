import argparse
from knossos_utils.knossosdataset import KnossosDataset

# Create command line argument parser and parse arguments
parser = argparse.ArgumentParser(
    description="Convert .conf files to .toml dataset configuration files."
)
parser.add_argument("input_conf_file", help="Input .conf file path")
parser.add_argument("output_toml_file", help="Output .toml file path")
args = parser.parse_args()

# Convert .conf file to .toml file
dataset = KnossosDataset(args.input_conf_file)
dataset.save_toml(args.output_toml_file)
