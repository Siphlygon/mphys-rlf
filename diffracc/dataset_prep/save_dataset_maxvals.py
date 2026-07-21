import argparse
import configparser
from pathlib import Path

import h5py
import numpy as np

from ..utils import paths
from ..utils.logger import get_logger

logger = get_logger(__name__)

def write_maxvals_of_h5_to_file(outfile: Path, infile: Path):
    """
    A function to go through an input h5 file, select the 'images' category, and sum along axes 1 and 2 to get an array
    of the maximum pixel values of each image, then save the numpy array to an output file. This saves the maximum pixel
    values neccesary for the box-cox power transform in a much more portable format, shrinking the file size by 6400x
    and allowing it to be copied for access by multiple nodes.

    Parameters
    ----------
    outfile : Path
        The output to write the numpy max vals array to
    infile : Path
        The h5 file which contains images as file['images'][:].shape = (n_images, ndim, ndim)
    """
    logger.info(f"Writing maxvals of {infile} to {outfile}")
    with h5py.File(infile, "r") as f:
        max_vals = np.max(f["images"][:], axis=(1, 2))
    np.save(outfile, max_vals)
    logger.info(f"Done writing maxvals of {infile} to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help=f"Which config to use for obtaining training dataset maxvals, as defined in "
                        f"{paths.PROGRAM_CONFIG.name}", type=str, required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(paths.PROGRAM_CONFIG)
    specific_config = config[args.config]

    # train_data_path = paths.LOFAR_DATA_PATH
    assert specific_config['train_data_path'] is not None, "(currently) train_data_path must be specified in the config"
    train_data_path = specific_config['train_data_path']

    # if train_data_path is paths.LOFAR_DATA_PATH, we also want to make sure to update the dataset subdir
    logger.info("Starting to write maxvals for training data")
    write_maxvals_of_h5_to_file(paths.NP_ARRAY_PARENT / 'dataset' / paths.MAXVALS, train_data_path)
    write_maxvals_of_h5_to_file(paths.NP_ARRAY_PARENT / specific_config['generated_subdir'] / paths.MAXVALS,
                                train_data_path)
    logger.info("Finished writing maxvals for training data")
