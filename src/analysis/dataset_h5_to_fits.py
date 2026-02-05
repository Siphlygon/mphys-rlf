# This file was created by Ashley and Luna and defines functions neccesary to convert the LOFAR h5 dataset
# that comes with the project into FITS files needed for PyBDSF, as well as a more specific function which implements
# the first function with the files and parameters presented in utils.paths and utils.parameters

# This file also provides a main function which forces a flush of the fits images directory and remakes it according
# to the command line arguments, or utils.parameters if none are passed

import sys
from astropy.io import fits
from astropy.io.fits import ImageHDU
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from pathlib import Path, PurePath
import shutil
import utils.parameters
import utils.paths as pth
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from analysis.image_analyzer import ImageAnalyzer

def convert_LOFAR_h5_to_fits( lofar_data_h5: Path, subdir: str, cutoff: int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF, bin_sizes: list[ int ] | None = None ):
    """
    A method to convert the LOFAR h5 dataset used in this project into fits files. Outputs fits files to 
    fits_output_dir, and groups the images into directories by descending sizes.

    Parameters
    ----------
    lofar_data_h5: Path
        The directory to read the LOFAR data from
    subdir: str
        The subdirectory to output the FITS files to
    cutoff : int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF
        The number of images to convert and export. Default behaviour in utils.parameters.LOFAR_FITS_COUNT_CUTOFF
    bin_sizes : list[ int ] | None = None
        Directory bins to sort the images into for ease of use. Defaults to the value in utils.parameters

        For example: bin_sizes array of [10000, 1000, 3000] would be sorted to [10000, 3000, 1000] and files would be stored as

        00010 -> subdir/0-9999/0-2999/0-999/image10.fits

        09150 -> subdir/0-9999/9000-9999/9100-9199/image9150.fits

        12120 -> subdir/10000-19999/12000-14999/12100-12199/image12120.fits

        As can be seen with 9150, bins are funneled such that the upper bound of an inner bin is always less than or equal to the upper bound of all
        its outer bins, and the opposite with the lower bound of an inner bin.
    """
    dataset_analyzer = ImageAnalyzer( subdir )

    if bin_sizes is None:
        bin_sizes = utils.parameters.BINS_ARRAY

    bin_sizes = sorted( bin_sizes, reverse=True )
    with h5py.File( str( lofar_data_h5 ), 'r' ) as h5:
        images = h5[ 'images' ]
    
        images_len = images.shape[ 0 ]
        num_to_convert = min( cutoff, images_len ) if cutoff is not None else images_len

        # Set up the power transformer so we can scale the max fluxes
        max_vals = np.max( images[:], axis=(1, 2) )
        pt = PowerTransformer( method="box-cox" )
        pt.fit( max_vals.reshape(-1, 1) )

        for i in tqdm( range( num_to_convert ) ):
            image = images[ i ]

            # the images in the dataset *are* selected by the process in the paper but *are not* scaled 0-1
            # here we do that scaling
            im_max = np.max( image )
            im_min = np.min( image )
            if im_min < 0:
                raise ValueError( "Images not preprocessed to remove negative values" )
            image = ( image - im_min ) / ( im_max - im_min )
            
            flux_scaled = pt.transform( np.array( [ im_max ] ).reshape(-1, 1) )[ 0, 0 ]

            #Create bins based on bin_sizes
            #make sure that there are no overextending bounds (e.g. 0-9999/9000-11999)
            postfix = PurePath()
            lowest_upper_bound =  1e30
            highest_lower_bound =-1e30
            for bin_size in bin_sizes:
                lower_bound = int( math.floor( i / bin_size ) * bin_size )
                upper_bound = int( math.ceil( ( i + 1 ) / bin_size ) * bin_size ) - 1

                if lower_bound < highest_lower_bound:
                    lower_bound = highest_lower_bound #cut this lower bound to the highest before it
                else:
                    highest_lower_bound = lower_bound #new highest lower bound

                if upper_bound > lowest_upper_bound:
                    upper_bound = lowest_upper_bound  #cut this upper bound to the lowest before it
                else:
                    lowest_upper_bound = upper_bound  #new lowest upper bound

                postfix = postfix / f"{lower_bound}-{upper_bound}"

            postfix = postfix / f"image{i}.fits"

            dataset_analyzer.save_image_to_FITS( image, postfix, flux_scaled )

def validate_LOFAR_fits_images( subdir: str, clean_directory: bool = False, cutoff: int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF, bin_sizes: list[ int ] | None = None ):
    """
    Ensure FITS images from LOFAR exist in accordance with paths laid out in utils.paths

    Parameters
    ----------
    clean_directory : bool
        Whether or not to clean out the fits images directory to ensure bin compliance
    subdir : str
        The subdirectory of the dataset
    cutoff : int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF
        The optional value to cut off conversion at, in terms of number of images converted

    bin_sizes : list[ int ] = None
        Directory bins to sort the images into for ease of use. Defaults to utils.parameters.BINS_ARRAY
        Is only used in the case where the dataset folder does not already exist. Compliance with bin structure is not checked for.

        For example: bin_sizes array of [10000, 1000, 3000] would be sorted to [10000, 3000, 1000] and files would be stored as

        00010 -> fits_output_dir/0-9999/0-2999/0-999/image10.fits

        09150 -> fits_output_dir/0-9999/9000-9999/9100-9199/image9150.fits

        12120 -> fits_output_dir/10000-19999/12000-14999/12100-12199/image12120.fits

        As can be seen with 9150, bins are funneled such that the upper bound of an inner bin is always less than or equal to the upper bound of all
        its outer bins, and the opposite with the lower bound of an inner bin.
    """
    fits_dataset_folder = pth.FITS_PARENT / subdir
    if fits_dataset_folder.exists():
        if clean_directory:
            shutil.rmtree( fits_dataset_folder )
        else:
            # If we don't want to clean the directory, check if we are at the conversion cutoff
            # and if we aren't, continue on to conversion
            num_files = sum( 1 for _ in fits_dataset_folder.rglob( "*.fits" ) )
            print( f'found {num_files}, cutoff {cutoff}' )
            if num_files >= cutoff:
                return

    convert_LOFAR_h5_to_fits( pth.LOFAR_DATA_PATH, fits_dataset_folder, cutoff, bin_sizes )



if __name__ == "__main__":
    # allow for bin sizes to be specified manually as command-line arguments, or use config if nothing specified
    bin_sizes = []
    for arg in sys.argv[ 1: ]:
        bin_sizes.append( int( arg ) )
    if len( bin_sizes ) == 0:
        bin_sizes = utils.parameters.BINS_ARRAY

    validate_LOFAR_fits_images( utils.parameters.DATASET_FITS_SUBDIR, bin_sizes=bin_sizes )