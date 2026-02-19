# This file was created by Ashley and Luna and defines functions neccesary to convert the LOFAR h5 dataset
# that comes with the project into FITS files needed for PyBDSF, as well as a more specific function which implements
# the first function and checks for compliance with the arguments, and a main function which uses the config file

import sys
from astropy.io import fits
from astropy.io.fits import ImageHDU
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from pathlib import Path, PurePath
import shutil
import utils.paths as pth
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from analysis.image_analyzer import ImageAnalyzer
import configparser
import argparse

def convert_LOFAR_h5_to_fits( lofar_data_h5: Path, subdir: str, cutoff: int | None, bin_size: int | None ):
    """
    A method to convert the LOFAR h5 dataset used in this project into fits files. Outputs fits files to 
    fits_output_dir, and groups the images into directories by descending sizes.

    Parameters
    ----------
    lofar_data_h5: Path
        The directory to read the LOFAR data from
    subdir: str
        The subdirectory to output the FITS files to
    cutoff : int | None
        The number of images to convert and export.
    bin_size : int | None
        Directory bins to sort the images into for ease of use.
    """
    dataset_analyzer = ImageAnalyzer( subdir )

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

            # bin the images based on bin_sizes
            postfix = PurePath()
            lower_bound = int( math.floor( i / bin_size ) * bin_size )
            upper_bound = int( math.ceil( ( i + 1 ) / bin_size ) * bin_size ) - 1
            postfix = postfix / f"{lower_bound}-{upper_bound}"
            postfix = postfix / f"image{i}.fits"
            dataset_analyzer.save_image_to_FITS( image, postfix, flux_scaled )

def validate_LOFAR_fits_images( subdir: str, cutoff: int | None, bin_size: int | None ):
    """
    Ensure FITS images from LOFAR exist in accordance with paths laid out in utils.paths

    Parameters
    ----------
    clean_directory : bool
        Whether or not to clean out the fits images directory to ensure bin compliance
    subdir : str
        The subdirectory of the dataset
    cutoff : int | None
        The optional value to cut off conversion at, in terms of number of images converted
    bin_size : int | None
        Directory bins to sort the images into for ease of use. Compliance with the bin structure is enforced.
    """
    fits_dataset_folder = pth.FITS_PARENT / subdir
    if fits_dataset_folder.exists():
        #first check if the bin structure is the same or if it has changed
        for bin in fits_dataset_folder.glob( "*" ):
            #if cutoff is None, we can check for compliance immediately
            #if we have a directory, we need to clean the dir, otherwise we can break and assume compliance
            if cutoff is None:
                if bin.is_dir():
                    shutil.rmtree( fits_dataset_folder )
                break

            lower_bound, upper_bound = bin.name.split( '-' )
            lower_bound, upper_bound = int( lower_bound ), int( upper_bound )
            if lower_bound % bin_size == 0 and upper_bound % bin_size == bin_size - 1:
                continue
            else:
                print( f"Bin {lower_bound}-{upper_bound} ({bin.parent/bin.name}) not compliant with bin size {bin_size}, removing directory..." )
                shutil.rmtree( fits_dataset_folder )
                break

        #then check we're not at or above the cutoff if it exists
        if cutoff is not None:
            num_files = sum( 1 for _ in fits_dataset_folder.rglob( "*.fits" ) )
            print( f'found {num_files}, cutoff {cutoff}' )
            #if we have more than cutoff, we should delete the dir and regenerate to make sure we have exactly cutoff
            if num_files > cutoff:
                shutil.rmtree( fits_dataset_folder )
            elif num_files == cutoff:
                return

    convert_LOFAR_h5_to_fits( pth.LOFAR_DATA_PATH, subdir, cutoff, bin_size )



if __name__ == "__main__":
    # allow for bin sizes to be specified manually as command-line arguments, or use config if nothing specified
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", help=f"Which config to use for image generation, as defined in {pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read( pth.PROGRAM_CONFIG )
    specific_config = config[ args.config ]

    cutoff = int( specific_config[ 'VM_FITS_COUNT_CUTOFF' ] )
    bin_size = int( specific_config[ 'FOLDER_SIZE' ] )
    vm_dataset_subdir = specific_config[ 'VM_DATASET_SUBDIR' ]

    validate_LOFAR_fits_images( vm_dataset_subdir, cutoff, bin_size )