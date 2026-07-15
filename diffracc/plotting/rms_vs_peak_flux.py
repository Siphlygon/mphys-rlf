"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from ..utils import paths
from ..utils.img_data_arrays import ImageDataArrays


def plot_rms_vs_peak_flux(config_name: str):
    """
    Plot peak flux against the RMS of the residual images for the dataset and generated images, for data
    verification.

    Parameters
    ----------
    config_name : str
        The name of the config to use for the analysis, as defined in utils.paths.PROGRAM_CONFIG
    """
    config = paths.config[config_name]

    config_data_arrays = ImageDataArrays(config_name)
    # Plotting peak flux vs RMS of residuals
    for subdir, c, data_arrays in zip([config['generated_subdir'], config['dataset_subdir']],
                                      ['g', 'b'],
                                      [config_data_arrays.generated_data, config_data_arrays.dataset_data]):
        plt.scatter(data_arrays.peak_fluxes, np.std(data_arrays.residual_images, axis=(1,2)),
                    label=subdir, c=c, s=0.01)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Peak Flux (mJy/pix)')
    plt.ylabel('RMS (mJy)')
    plt.grid(True)
    plt.legend(markerscale=100)
    plt.savefig('peak_vs_rms.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help=f"Which config to use, as defined in {paths.PROGRAM_CONFIG.name}", type=str)
    args = parser.parse_args()

    plot_rms_vs_peak_flux(args.config)
