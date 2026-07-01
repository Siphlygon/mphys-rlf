import argparse
import configparser
from pathlib import Path

import h5py
import numpy as np

import utils.paths as pth


def write_maxvals_of_h5_to_file( outfile: Path, infile: Path ):
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
        The h5 file which contains images as file[ 'images' ][ : ].shape = (n_images, ndim, ndim)
    """
    with h5py.File( infile, "r" ) as f:
        max_vals = np.max( f[ "images" ][ : ], axis=(1, 2) )
    np.save( outfile, max_vals )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config",
                        help=f"Which config to use for obtaining training dataset maxvals, as defined in "
                        f"{pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read( pth.PROGRAM_CONFIG )
    specific_config = config[ args.config ]

    train_data_path = pth.LOFAR_DATA_PATH
    if specific_config[ 'train_data_path' ] != "None":
        train_data_path = specific_config[ 'train_data_path' ]

    # if train_data_path is pth.LOFAR_DATA_PATH, we also want to make sure to update the dataset subdir
    write_maxvals_of_h5_to_file( pth.NP_ARRAY_PARENT / 'dataset' / pth.MAXVALS, train_data_path )
    write_maxvals_of_h5_to_file( pth.NP_ARRAY_PARENT / specific_config[ 'generated_subdir' ] / pth.MAXVALS,
                                train_data_path )
