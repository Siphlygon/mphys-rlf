from analysis.recursive_file_analyzer import RecursiveFileAnalyzer
from analysis.log_analyzer import LogAnalyzer
import utils.paths
import analysis.log_analyzer as la
import analysis.recursive_file_analyzer as rfa
import numpy as np
import h5py
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
from completeness.img_data_arrays import ImageDataArrays

"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
if __name__ == "__main__":
    xticks = np.logspace( -5, 5, 11 )
    yticks = np.logspace( -5, 5, 11 )

    for subdir, c in zip( utils.paths.SUBDIRS, utils.paths.COLOURS ):
        images, resid_images, model_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays()
        plt.scatter( peak_fluxes, model_fluxes, label=subdir, c=c, s=0.01 )

    plt.plot( xticks, yticks, c='r' )
    plt.xscale( 'log' )
    plt.yscale( 'log' )
    plt.xlabel( 'Peak Flux (mJy/beam)' )
    plt.ylabel( 'Integrated Flux (mJy)' )
    plt.grid( True )
    plt.xticks( xticks )
    plt.yticks( yticks )
    plt.legend( markerscale=100 )
    plt.savefig( 'peak_vs_model_flux.png' )
    plt.show()