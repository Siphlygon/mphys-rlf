import matplotlib.pyplot as plt
from pybdsf_analysis.recursive_file_analyzer import HistogramErrorDrawer
from pybdsf_analysis.image_analyzer import ImageAnalyzer
import argparse
import logging
import pybdsf_analysis.pybdsf_run_analysis
import utils.paths
from utils.distributed import DistributedUtils
import logging
from pybdsf_analysis.log_analyzer import LogAnalyzer
import pybdsf_analysis.log_analyzer as la
import pybdsf_analysis.recursive_file_analyzer as rfa
import numpy as np
from completeness.img_data_arrays import ImageDataArrays
import h5py
from files.dataset import LOFAR_DATA_PATH

def plot_graphs_with_pybdsf_data( log_level: int = logging.INFO ):
    resolution = 600
    BINCOUNT = 25

    #fig = plt.figure( figsize=(int(resolution/100), int(resolution/100)) )
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec( 2, 2,
                            left=0.11, right=0.99, bottom=0.05, top=0.95,
                            wspace=0.25, hspace=0.25 )
    ax_flux = fig.add_subplot( gs[ 0, 0 ] )
    ax_mean = fig.add_subplot( gs[ 0, 1 ] )
    ax_rms = fig.add_subplot( gs[ 1, 0 ] )
    ax_pix = fig.add_subplot( gs[ 1, 1 ] )

    axes = [ ax_flux, ax_mean, ax_rms, ax_pix ]
    titles = [ "Flux", "Mean", "RMS", "Pixel Values" ]
    xlabels = [ "Integrated Flux (arbitrary units)", "Image Mean (0-1)", "Image RMS (0-1)", "Pixel Value (0-1)" ]
    ranges = [ (0, 60), (0, 0.2), (0, 0.3), (0, 1) ]
    ylabels = [ "Relative Frequency" ] * 4

    

    hist = HistogramErrorDrawer()
    for subdir in [ utils.paths.DATASET_SUBDIR, utils.paths.GENERATED_SUBDIR ]:
        fluxes_path = utils.paths.NP_ARRAY_PARENT / subdir / 'integrated_fluxes_normalized.npy'
        if fluxes_path.exists():
            normalized_model_fluxes = np.load( fluxes_path )
        else:
            log_analyzer = LogAnalyzer( subdir )
            normalized_model_fluxes = log_analyzer.for_each( la.get_model_flux, progress_bar_desc=f'{subdir} fluxes...' )
            normalized_model_fluxes = np.array( normalized_model_fluxes )
            np.save( fluxes_path, normalized_model_fluxes )

        data_path = utils.paths.NP_ARRAY_PARENT / subdir / 'histogram_data.npy'
        if data_path.exists():
            data = np.load( data_path )
        else:
            if subdir == utils.paths.GENERATED_SUBDIR:
                rf = rfa.RecursiveFileAnalyzer( utils.paths.FITS_PARENT / subdir )
                data = np.array( rf.for_each( rfa.get_fits_primaryhdu_data, progress_bar_desc=f'{subdir} data...' ) )
            else:
                with h5py.File( LOFAR_DATA_PATH, 'r' ) as h5:
                    data = h5[ 'images' ][ : ]
                    data = data / np.max( data, axis=(1,2) )
            np.save( data_path, data )
        means = np.mean( data, axis=(1,2) )
        rmsds = np.std( data, axis=(1,2) )


        axes_data = [ normalized_model_fluxes, means, rmsds, data.ravel() ]
        for ax, ax_data, range in zip( axes, axes_data, ranges ):
            hist.draw( ax_data,
                       ax=ax,
                       bins=BINCOUNT,
                       range=range,
                       label=subdir, 
                       color="b" if subdir == utils.paths.DATASET_SUBDIR else "g",
                       density=False,
                       relative=True )
                       

    for ax, title, range in zip( axes, titles, ranges ):
        ax.legend()
        ax.set_title( title )
        ax.set_yscale( 'log' )
        ax.set_xbound( lower=range )

    plt.savefig( "hist.png" )
    plt.show()


if __name__ == "__main__":
    pybdsf_analysis.pybdsf_run_analysis.analyze_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument( "-v", "--verbose", help="Print a message to the console every time a file is read or a directory is entered", action='store_true' )
    args = parser.parse_args()
    verbose = args.verbose
    log_level = logging.DEBUG if verbose else logging.INFO

    du = DistributedUtils()
    du.single_task_only_last( 'plot_graphs_with_pybdsf_data', plot_graphs_with_pybdsf_data, 0, log_level )
