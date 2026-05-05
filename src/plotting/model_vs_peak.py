import utils.paths
import numpy as np
import matplotlib.pyplot as plt
from utils.img_data_arrays import ImageDataArrays
import argparse
import utils.paths as pth

"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", help=f"Which config to to use for Dataset/Generated subdirs, as defined in {pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()

    config = pth.config[ args.config ]
    dataset_subdir = config[ 'dataset_subdir' ]
    generated_subdir = config[ 'generated_subdir' ]

    xticks = np.logspace( -5, 5, 11 )
    yticks = np.logspace( -5, 5, 11 )

    for subdir, c in zip( [ dataset_subdir, generated_subdir ], [ 'b', 'g' ] ):
        images, resid_images, model_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays()
        plt.scatter( peak_fluxes, model_fluxes, label=subdir, c=c, s=0.01 )

    plt.plot( xticks, yticks, c='r' )
    plt.xscale( 'log' )
    plt.yscale( 'log' )
    plt.xlabel( 'Peak Flux (mJy/beam)' )
    plt.ylabel( 'Integrated Flux (mJy)' )
    plt.title( f'Peak vs Integrated Flux {args.config}' )
    plt.grid( True )
    plt.xticks( xticks )
    plt.yticks( yticks )
    plt.legend( markerscale=100 )
    plt.savefig( f'peak_vs_model_flux_{args.config}.png' )
    plt.show()
