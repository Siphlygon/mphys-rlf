import utils.paths
import numpy as np
import matplotlib.pyplot as plt
from utils.img_data_arrays import ImageDataArrays

"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
if __name__ == "__main__":
    #analysis.analysis.analyze_everything()
    for subdir, c in zip( utils.paths.SUBDIRS, utils.paths.COLOURS ):
        images, resid_images, model_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays()
        plt.scatter( peak_fluxes, np.std( resid_images, axis=(1,2) ), label=subdir, c=c, s=0.01 )

    plt.xscale( 'log' )
    plt.yscale( 'log' )
    plt.xlabel( 'Peak Flux (mJy/pix)' )
    plt.ylabel( 'RMS (mJy)' )
    plt.grid( True )
    plt.legend( markerscale=100 )
    plt.savefig( 'peak_vs_rms.png' )
    plt.show()