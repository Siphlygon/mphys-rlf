import utils.paths
import numpy as np
import matplotlib.pyplot as plt
from utils.logging import get_logger
from utils.power_transform import PeakFluxPowerTransformer
from completeness.img_data_arrays import ImageDataArrays

logger = get_logger( __name__ )

def plot_flux_vs_residuals():
    pt = PeakFluxPowerTransformer()
    for subdir, color in zip( utils.paths.SUBDIRS, utils.paths.COLOURS ):
        data_arrays = ImageDataArrays( subdir )

        #Select for peak flux >0.5 mJy
        valid = data_arrays.peak_fluxes > 0.5
        data_arrays.peak_fluxes = data_arrays.peak_fluxes[ valid ]
        data_arrays.image_scale_factors = data_arrays.image_scale_factors[ valid ]
        data_arrays.residual_images = data_arrays.residual_images[ valid, :, : ]

        transformed_peak_fluxes = pt.transform( data_arrays.peak_fluxes / 1000 )

        # Delta - summed clipped residuals, per image
        resid_images = data_arrays.residual_images / data_arrays.image_scale_factors[ :, np.newaxis, np.newaxis ] #transform to 0-1 scale
        rv_clipped = np.where( resid_images > 0, resid_images, 0 )
        delta = np.sum( rv_clipped, axis=tuple( [ i for i in range( 1, len( resid_images.shape ) ) ] ) )

        # Scaled flux 
        scaled_flux = transformed_peak_fluxes

        plt.scatter( scaled_flux, delta, label=subdir, 
                    color=color,
                    s=0.01 )

    plt.xlabel( 'Transformed peak flux, arbitrary units' )
    plt.ylabel( 'Summed positive residuals mJy/image' )
    plt.yscale( 'log' )
    plt.title( 'Summed positive residuals vs transformed peak flux' )
    plt.grid( True )
    plt.legend( markerscale=100 )
    plt.savefig( 'scatter.png' )
    plt.show()
    logger.info( 'Saved figure to scatter.png' )


if __name__ == '__main__':
    plot_flux_vs_residuals()

