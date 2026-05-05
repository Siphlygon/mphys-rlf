import matplotlib.pyplot as plt
from utils.recursive_file_analyzer import HistogramErrorDrawer
import argparse
import utils.paths
import logging
import utils.logging
from analysis.log_analyzer import LogAnalyzer
import analysis.log_analyzer as la
import utils.recursive_file_analyzer as rfa
import numpy as np
import h5py


class HistogramPlotter:

    def __init__(self, generated_subdir: str, dataset_subdir: str, config_name: str | None = None, train_data_path: str | None = None, bin_count=25):
        self.bin_count = bin_count
        self.generated_subdir = generated_subdir
        self.dataset_subdir = dataset_subdir
        self.train_data_path = train_data_path
        self.config_name = config_name
        self.logger = utils.logging.get_logger(__name__, logging.DEBUG)

        self.hist = HistogramErrorDrawer()

    def set_up_figure(self, titles, ranges, xlabels, ylabels):
        # Initialise figure
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec( 2, 2,
                                left=0.11, right=0.99, bottom=0.05, top=0.95,
                                wspace=0.25, hspace=0.30 )
        ax_flux = fig.add_subplot( gs[ 0, 0 ] )
        ax_mean = fig.add_subplot( gs[ 0, 1 ] )
        ax_rms = fig.add_subplot( gs[ 1, 0 ] )
        ax_pix = fig.add_subplot( gs[ 1, 1 ] )
        axes = [ ax_flux, ax_mean, ax_rms, ax_pix ]

        for ax, title, range, xlabel, ylabel in zip( axes, titles, ranges, xlabels, ylabels ):
            #ax.legend()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            #ax.set_xbound(range[0], range[1])

        return fig, axes

    def plot_histograms(self):
        # Plotting flux, mean, rms, and pixel values
        titles = [ "Flux", "Mean", "RMS", "Pixel Values" ]
        xlabels = [ "Integrated Flux (Jy)", "Image Mean (Jy/beam)", "Image RMS (Jy/beam)", "Pixel Value (Jy/beam)" ]
        ranges = [ (0, 30), (0, 0.25), (0, 0.5), (0, 50) ]
        ylabels = [ "Relative Frequency" ] * 4

        fig, axes = self.set_up_figure(titles, ranges, xlabels, ylabels)
        # Plot histograms for every subdir e.g., dataset, loguniform, etc
        for subdir, c in zip( [ self.generated_subdir, self.dataset_subdir ], [ 'g', 'b' ] ):

            # -- Get model fluxes; will need to get them from PyBDSF if non-existing --
            fluxes_path = utils.paths.NP_ARRAY_PARENT / subdir / 'integrated_fluxes_normalized.npy'
            if fluxes_path.exists():
                normalized_model_fluxes = np.load( fluxes_path )
            else:
                log_analyzer = LogAnalyzer( subdir )
                normalized_model_fluxes = log_analyzer.for_each( la.get_model_flux, progress_bar_desc=f'{subdir} fluxes...' )
                normalized_model_fluxes = np.array( normalized_model_fluxes )
                np.save( fluxes_path, normalized_model_fluxes )

            # -- Get the other histogram data --
            data_path = utils.paths.NP_ARRAY_PARENT / subdir / 'histogram_data.npy'
            if data_path.exists():
                data = np.load( data_path )
            else:
                if self.train_data_path is not None:
                    with h5py.File( self.train_data_path, 'r' ) as h5file:
                        data = h5file['images'][:]
                        #don't save to numpy, no need to duplicate
                else:
                    rf = rfa.RecursiveFileAnalyzer( utils.paths.FITS_PARENT / subdir )
                    data = np.array( rf.for_each( rfa.get_fits_primaryhdu_data, progress_bar_desc=f'{subdir} data...' ) )
                    np.save( data_path, data )

            means = np.mean( data, axis=(1,2) )
            rmsds = np.std( data, axis=(1,2) )

            # Plot the histograms
            axes_data = [ normalized_model_fluxes, means, rmsds, data.ravel() ]
            for ax, ax_data, range in zip( axes, axes_data, ranges ):
                ax_data_nonnan = ax_data[ ~np.isnan( ax_data ) ]
                self.logger.info( f'ax_data length: {ax_data.shape[ 0 ]}' )
                self.logger.info( f'ax_data_nonnan length: {ax_data_nonnan.shape[ 0 ]}' )
                self.hist.draw( ax_data_nonnan,
                           ax=ax,
                           bins=self.bin_count,
                           range=range,
                           #range=None,
                           label=subdir,
                           color=c,
                           density=False,
                           relative=True )
            for ax in axes:
                ax.legend()

        if self.config_name is not None:
            plt.savefig( f"hist_{self.config_name}.png" )
        else:
            plt.savefig( f"hist.png" )
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-v", "--verbose", help="Print a message to the console every time a file is read or a directory is entered", action='store_true' )
    parser.add_argument( "--config", help="Config to use to get dataset/generated directories/paths", type=str )
    args = parser.parse_args()
    verbose = args.verbose
    log_level = logging.DEBUG if verbose else logging.INFO

    config = utils.paths.config[ args.config ]
    generated_subdir = config[ 'generated_subdir' ]
    dataset_subdir = config[ 'dataset_subdir' ]
    train_data_path = None if config[ 'train_data_path' ] == 'None' else config[ 'train_data_path' ]

    hp = HistogramPlotter( generated_subdir, dataset_subdir, config_name=args.config, train_data_path=train_data_path )
    hp.plot_histograms()
