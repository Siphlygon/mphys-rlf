"""
This file was created by Ashley and Luna. It provides a complete application that can be used to sample images according
to the LOFAR model with certain parameters, and it provides a function that can be used to access that application
through other files. This application can be distributed across multiple nodes.
"""

import argparse
import configparser
import math
from pathlib import PurePath

import numpy as np
import scipy.stats
import torch

from ..analysis.image_analyzer import ImageAnalyzer, RecursiveFileAnalyzer
from ..model import sampler
from ..utils import paths
from ..utils.distributed import DistributedUtils
from ..utils.logger import LoggingLevels, get_logger
from ..utils.power_transform import PeakFluxPowerTransformer


def get_path_from_index( index: int,
                        subdir: str,
                        bin_size: int ) -> tuple[ PurePath, PurePath ]:
    """
    Given an index, a subdirectory, and a bin size, this function returns the full path to the FITS file corresponding
    to that index, as well as the postfix path (the part of the path after the subdirectory).

    Parameters
    ----------
    index : int
        The index of the image
    subdir : str
        The subdirectory containing the FITS files
    bin_size : int
        The size of each bin

    Returns
    -------
    PurePath
        The full path to the FITS file corresponding to the given index
    PurePath
        The postfix path (the part of the path after the subdirectory)
    """
    lower_bound = int( math.floor( ( index ) / bin_size ) * bin_size )
    upper_bound = int( math.ceil( ( index + 1 ) / bin_size ) * bin_size ) - 1
    postfix = PurePath( *[ f"{lower_bound}-{upper_bound}", f"image{index}.fits" ] )
    full_image_path = ( paths.FITS_PARENT / subdir ) / postfix
    return full_image_path, postfix


def sample( args : argparse.Namespace ):
    """
    A function to sample images according to the LOFAR model with certain parameters. This function can be distributed
    across multiple nodes, and it will save the generated images to disk as they are created.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments
    """
    logger = get_logger( __name__, LoggingLevels.DEBUG.value )

    #Do a sampling loop of batch_size samples and save them to the disk as they're generated, until we reach n_samples
    model_sampler = sampler.Sampler( n_samples=args.batch_size,
                                          timesteps=args.timesteps,
                                          distribute_model=not args.use_cpu )

    #SLURM distribution w/ batching
    du = DistributedUtils()
    n_samples = args.n_samples
    bin_start = du.get_bin_start( n_samples )
    bin_end = du.get_bin_end( n_samples )
    logger.debug( 'bin_end=%i, bin_start=%i, n_samples=%i', bin_end, bin_start, n_samples )

    # Figure out initial count based on number of fits files already in the directory
    logger.debug( 'Getting initial count...' )
    initial_count = 0
    generated_images_dir = paths.FITS_PARENT / args.generated_subdir
    if generated_images_dir.exists():
        analyzer = RecursiveFileAnalyzer( generated_images_dir )
        initial_count = len( analyzer.get_unwrapped_list( None, r'.*?image(\d+)\.fits$', (bin_start, bin_end) ).paths )
    n_samples_in_bin = bin_end - bin_start
    logger.debug( 'Got initial count %i, requested samples in this bin %i', initial_count, n_samples_in_bin )

    n_samples_to_generate = n_samples_in_bin - initial_count
    if n_samples_to_generate <= 0:
        logger.info( 'Skipping bin %i-%i, nothing to do', bin_start, bin_end )
        return

    # Get the power transformer for the peak fluxes and the appropriate distribution function
    pt = PeakFluxPowerTransformer( args.generated_subdir )
    if args.distribution == 'dataset':
        fpeak_model_dist = model_sampler.get_fpeak_model_dist( None, paths.NP_ARRAY_PARENT / args.generated_subdir / paths.MAXVALS )
    elif args.distribution == 'uniform':
        fpeak_model_dist = lambda n : pt.transform( scipy.stats.uniform.rvs( args.lower_bound, args.upper_bound, size=n ) )
    elif args.distribution == 'loguniform':
        fpeak_model_dist = lambda n : pt.transform( scipy.stats.loguniform.rvs( args.lower_bound, args.upper_bound, size=n ) )
    else:
        raise ValueError( f"Unknown distribution {args.distribution}" )


    # Generate/Sample the samples
    sample_generated_count = 0
    sample_index = bin_start
    image_analyzer = ImageAnalyzer( args.generated_subdir )
    while sample_generated_count < n_samples_to_generate:
         #to not double-generate at the borders
        batch_size = min( args.batch_size, n_samples_to_generate - sample_generated_count )
        context = fpeak_model_dist( batch_size )[ :, np.newaxis ]
        # if las conditioning is enabled, add it to context
        if args.las_conditioning_enabled:
            context = np.concatenate( (context, scipy.stats.uniform.rvs( 6, 120, size=batch_size )[ :, np.newaxis ]), axis=1 )

        samples = model_sampler.quick_sample( f"{args.model_name}",
                                             context=torch.from_numpy( context ),
                                             n_samples=batch_size,
                                             distribute_model=not args.use_cpu )
        sample_generated_count += batch_size

        for i in range( samples.shape[ 0 ] ):
            image = samples[ i, -1, 0, :, : ]

            # Scale the images 0-1
            # not sure if this is neccesary for generated images
            # TODO: check this
            if not args.preserve_values:
                im_max = np.max( image )
                im_min = np.min( image )
                if im_min < 0:
                    image = np.where( image > 0, image, 0 )
                image = ( image - im_min ) / ( im_max - im_min )

            fscaled = context[ i, 0 ]
            if args.las_conditioning_enabled:
                las = context[ i, 1 ]

            full_image_path, postfix = get_path_from_index( sample_index, args.generated_subdir, args.folder_size )
            # todo: apparently you can't do PurePath.exists() ?
            while full_image_path.exists():
                sample_index += 1
                full_image_path, postfix = get_path_from_index( sample_index, args.generated_subdir, args.folder_size )
            if args.las_conditioning_enabled:
                image_analyzer.save_image_to_fits( image, postfix, FXSCLD=fscaled, LASIZE=las )
            else:
                image_analyzer.save_image_to_fits( image, postfix, FXSCLD=fscaled )

            if sample_index > bin_end:
                logger.error( 'Sample index %i has gone outside allowed value %i', sample_index, bin_end )
            elif sample_index == bin_end:
                logger.info( 'Sample index %i has reached bin end %i - generated sample count %i/%i',
                            sample_index, bin_end, sample_generated_count, n_samples_to_generate )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config",
                        help=f"Which config to use for image generation, as defined in {paths.PROGRAM_CONFIG.name}",
                        type=str )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read( paths.PROGRAM_CONFIG )
    for arg in [ 'generated_subdir',
                 'batch_size', 
                 'n_samples',
                 'folder_size',
                 'timesteps',
                 'use_cpu',
                 'preserve_values',
                 'model_name',
                 'upper_bound',
                 'lower_bound',
                 'distribution',
                 'las_conditioning_enabled' ]:
        setattr( args, arg, config.get( args.config, arg ) )

    for intarg in [ 'batch_size', 'n_samples', 'folder_size', 'timesteps' ]:
        setattr( args, intarg, int( getattr( args, intarg ) ) )
    for floatarg in [ 'upper_bound', 'lower_bound' ]:
        setattr( args, floatarg, float( getattr( args, floatarg ) ) )
    for boolarg in [ 'use_cpu', 'preserve_values', 'las_conditioning_enabled' ]:
        setattr( args, boolarg, getattr( args, boolarg ) == 'True' )

    sample( args )
