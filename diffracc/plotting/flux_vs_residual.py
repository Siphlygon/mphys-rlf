import argparse

import matplotlib.pyplot as plt
import numpy as np

from ..utils import paths
from ..utils.img_data_arrays import ImageDataArrays
from ..utils.logger import get_logger
from ..utils.power_transform import PeakFluxPowerTransformer

logger = get_logger( __name__ )

def plot_flux_vs_residuals( config_name: str ):
    """
    A function to plot the transformed peak fluxes against the summed positive residuals for the dataset and generated
    images, for comparison.

    Parameters
    ----------
    config_name : str
        The name of the config to use for the analysis, as defined in utils.paths.PROGRAM_CONFIG
    """
    config = paths.config[ config_name ]
    config_data_arrays = ImageDataArrays( config_name )

    for subdir, color, data_arrays in zip( [ config[ 'generated_subdir' ], config[ 'dataset_subdir' ] ],
                                          [ 'g', 'b' ],
                                          [ config_data_arrays.generated_data, config_data_arrays.dataset_data ] ):
        pt = PeakFluxPowerTransformer( subdir, maxvals=np.max( data_arrays.images, axis=(1,2) ) )

        #Select for peak flux >0.5 mJy
        valid = data_arrays.peak_fluxes > 0.5
        data_arrays.peak_fluxes = data_arrays.peak_fluxes[ valid ]
        data_arrays.image_scale_factors = data_arrays.image_scale_factors[ valid ]
        data_arrays.residual_images = data_arrays.residual_images[ valid, :, : ]

        transformed_peak_fluxes = pt.transform( data_arrays.peak_fluxes / 1000 )

        # Delta - summed clipped residuals, per image
        #transform to 0-1 scale
        resid_images = data_arrays.residual_images / data_arrays.image_scale_factors[ :, np.newaxis, np.newaxis ]
        rv_clipped = np.where( resid_images > 0, resid_images, 0 )
        delta = np.sum( rv_clipped, axis=tuple( [ i for i in range( 1, len( resid_images.shape ) ) ] ) )

        # Scaled flux
        scaled_flux = transformed_peak_fluxes

        plt.scatter( scaled_flux, delta, label=subdir, color=color, s=0.01 )

    plt.xlabel( 'Transformed peak flux, arbitrary units' )
    plt.ylabel( 'Summed positive residuals mJy/image' )
    plt.yscale( 'log' )
    plt.title( 'Summed positive residuals vs transformed peak flux' )
    plt.grid( True )
    plt.legend( markerscale=1 )
    plt.savefig( 'scatter.png' )
    plt.show()
    logger.info( 'Saved figure to scatter.png' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config",
                        help=f"Which config to to use for Dataset/Generated subdirs, as defined in"
                        f" {paths.PROGRAM_CONFIG.name}",
                        type=str )
    args = parser.parse_args()

    plot_flux_vs_residuals(args.config)
