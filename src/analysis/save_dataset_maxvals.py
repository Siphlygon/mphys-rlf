from pathlib import Path
import h5py
import numpy as np
import utils.paths as pth

def write_maxvals_of_h5_to_file( outfile: Path, infile: Path ):
    """
    A function to go through an input h5 file, select the 'images' category, and sum along axes 1 and 2 to get an array of the maximum
    pixel values of each image, then save the numpy array to an output file. This saves the maximum pixel values neccesary for the box-cox
    power transform in a much more portable format, shrinking the file size by 6400x and allowing it to be copied for access by multiple nodes.

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
    write_maxvals_of_h5_to_file( pth.MAXVALS, pth.LOFAR_DATA_PATH )