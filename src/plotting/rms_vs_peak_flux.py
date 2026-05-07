import utils.paths as pth
import numpy as np
import matplotlib.pyplot as plt
from utils.img_data_arrays import ImageDataArrays
import argparse

"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", help=f"Which config to use, as defined in {pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()

    config = pth.config[ args.config ]

    config_data_arrays = ImageDataArrays( args.config )
    for subdir, c, data_arrays in zip( [ config[ 'generated_subdir' ], config[ 'dataset_subdir' ] ], [ 'g', 'b' ], [ config_data_arrays.generated_data, config_data_arrays.dataset_data ] ):
        plt.scatter( data_arrays.peak_fluxes, np.std( data_arrays.resid_images, axis=(1,2) ), label=subdir, c=c, s=0.01 )

    plt.xscale( 'log' )
    plt.yscale( 'log' )
    plt.xlabel( 'Peak Flux (mJy/pix)' )
    plt.ylabel( 'RMS (mJy)' )
    plt.grid( True )
    plt.legend( markerscale=100 )
    plt.savefig( 'peak_vs_rms.png' )
    plt.show()