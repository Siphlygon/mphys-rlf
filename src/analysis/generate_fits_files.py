# This file was created by Ashley and Luna. It provides a complete application that can be used to sample
# images according to the LOFAR model with certian parameters, and it provides a function that can be used
# to access that application through other files. This application can be distributed across multiple nodes.

import numpy as np
import math
import argparse
import model.sampler
from analysis.image_analyzer import ImageAnalyzer, RecursiveFileAnalyzer
import utils.paths
import utils.logging
import torch
import utils.parameters
import sys
from utils.distributed import DistributedUtils
import utils.paths as pth
import logging
from pathlib import PurePath
import scipy.stats
from analysis.power_transform import PeakFluxPowerTransformer


def get_path_from_index( index: int, subdir: str, bin_size: int ):
    lower_bound = int( math.floor( ( index ) / bin_size ) * bin_size )
    upper_bound = int( math.ceil( ( index + 1 ) / bin_size ) * bin_size ) - 1
    postfix = PurePath( *[ f"{lower_bound}-{upper_bound}", f"image{index}.fits" ] )
    full_image_path = ( utils.paths.FITS_PARENT / subdir ) / postfix
    return full_image_path, postfix

def sample( parameter_args ):
    logger = utils.logging.get_logger( __name__, logging.DEBUG )

    parser = argparse.ArgumentParser()
    parser.add_argument( "-s", "--subdir", help="The subdirectory to store the sampled files under", type=str )
    parser.add_argument( "-a", "--name", help="The name of the model to sample, e.g. LOFAR or FIRST", type=str, default="LOFAR" )
    parser.add_argument( "-b", "--batch-size", help="The number of batches to be sampled at a time", type=int, default=utils.parameters.FITS_SAMPLING_ARGS[ 'batch_size' ] )
    parser.add_argument( "-n", "--n-samples", help="The number of samples to generate", type=int, default=utils.parameters.FITS_SAMPLING_ARGS[ 'n_samples'] )
    parser.add_argument( "-t", "--timesteps", help="The number of timesteps in sampling", type=int, default=utils.parameters.FITS_SAMPLING_ARGS[ 'timesteps' ] )
    parser.add_argument( "-c", "--use-cpu", help="Whether or not to use CPU and RAM for sampling, as opposed to using avaliable GPUs", action='store_true' if not utils.parameters.FITS_SAMPLING_ARGS[ 'use_cpu' ] else 'store_false' )
    parser.add_argument( "--distribution", help="Distribution type: uniform, loguniform, or dataset. For uniform and loguniform specify upper and lower bounds with --upper and --lower", default='dataset' )
    parser.add_argument( "--upper", help="Distribution upper bound", type=float, default=0 )
    parser.add_argument( "--lower", help="Distribution lower bound", type=float, default=0 )
    parser.add_argument( "-p", "--preserve-values", help="Whether or not to preserve unscaled image values. By default images are scaled 0-1", action='store_true' )
    parser.add_argument( "-sz", "--bin-size", help="How large the bins the generated images are sorted into are", type=int, default=utils.parameters.FITS_SAMPLING_ARGS[ 'bin_size' ] )
    args = parser.parse_args( parameter_args ) #will automatically read from the command line if passed, else use defaults

    #Do a sampling loop of batch_size samples and save them to the disk as they're generated, until we reach n_samples
    model_sampler = model.sampler.Sampler( n_samples=args.batch_size, timesteps=args.timesteps, distribute_model=(not args.use_cpu) )

    #SLURM distribution w/ batching
    du = DistributedUtils()
    n_samples = args.n_samples
    bin_start = du.get_bin_start( n_samples )
    bin_end = du.get_bin_end( n_samples )
    logger.debug( 'bin_end=%i, bin_start=%i, n_samples=%i', bin_end, bin_start, n_samples )

    # Figure out initial count based on number of fits files already in the directory
    logger.debug( 'Getting initial count...' )
    initial_count = 0
    generated_images_dir = utils.paths.FITS_PARENT / args.subdir
    if generated_images_dir.exists():
        analyzer = RecursiveFileAnalyzer( generated_images_dir )
        initial_count = len( analyzer.get_unwrapped_list( None, r'.*?image(\d+)\.fits$', (bin_start, bin_end) ) )
    n_samples_in_bin = bin_end - bin_start
    logger.debug( 'Got initial count %i, requested samples in this bin %i', initial_count, n_samples_in_bin )

    n_samples_to_generate = n_samples_in_bin - initial_count
    if n_samples_to_generate <= 0:
        logger.info( 'Skipping bin %i-%i, nothing to do', bin_start, bin_end )
        return

    # Get the power transformer for the peak fluxes and the appropriate distribution function
    pt = PeakFluxPowerTransformer()
    if args.distribution == 'dataset':
        fpeak_model_dist = model_sampler.get_fpeak_model_dist( None, pth.MAXVALS )
    elif args.distribution == 'uniform':
        fpeak_model_dist = lambda n : pt.transform( scipy.stats.uniform.rvs( args.lower, args.upper, size=n ) )
    elif args.distribution == 'loguniform':
        fpeak_model_dist = lambda n : pt.transform( scipy.stats.loguniform.rvs( args.lower, args.upper, size=n ) )

    # Generate/Sample the samples
    sample_generated_count = 0
    sample_index = bin_start
    image_analyzer = ImageAnalyzer( args.subdir )
    while sample_generated_count < n_samples_to_generate:
        batch_size = min( args.batch_size, n_samples_to_generate - sample_generated_count ) #to not double-generate at the borders
        fpeak_model_values = fpeak_model_dist( batch_size )[ :, np.newaxis ]
        samples = model_sampler.quick_sample( f"{args.name}_model", context=torch.from_numpy( fpeak_model_values ), n_samples=batch_size, distribute_model=(not args.use_cpu) )
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

            fscaled = fpeak_model_values[ i, 0 ]

            full_image_path, postfix = get_path_from_index( sample_index, args.subdir, args.bin_size )
            while full_image_path.exists():
                sample_index += 1
                full_image_path, postfix = get_path_from_index( sample_index, args.subdir, args.bin_size )
            image_analyzer.save_image_to_FITS( image, postfix, fscaled )

            if sample_index > bin_end:
                logger.error( 'Sample index %i has gone outside allowed value %i', sample_index, bin_end )
            elif sample_index == bin_end:
                logger.info( 'Sample index %i has reached bin end %i - generated sample count %i/%i', sample_index, bin_end, sample_generated_count, n_samples_to_generate )

if __name__ == '__main__':
    sample( sys.argv[ 1: ] )