"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from ..utils import paths
from ..utils.img_data_arrays import ImageDataArrays


def plot_peak_vs_model_flux(config_name: str):
    """
    Plot peak flux against PyBDSF-measured integrated (model) flux for the dataset and generated images, for
    data verification.

    Parameters
    ----------
    config_name : str
        The name of the config to use for the analysis, as defined in utils.paths.PROGRAM_CONFIG
    """
    config = paths.config[config_name]
    dataset_subdir = config['dataset_subdir']
    generated_subdir = config['generated_subdir']

    xticks = np.logspace(-5, 5, 11)
    yticks = np.logspace(-5, 5, 11)
    config_data_arrays = ImageDataArrays(config_name)

    for subdir, c, data_arrays in zip([generated_subdir, dataset_subdir],
                                      ['g', 'b'],
                                      [config_data_arrays.generated_data, config_data_arrays.dataset_data]):
        plt.scatter(data_arrays.peak_fluxes, data_arrays.model_fluxes, label=subdir, c=c, s=0.01)

    plt.plot(xticks, yticks, c='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Peak Flux (mJy/beam)')
    plt.ylabel('Integrated Flux (mJy)')
    plt.title(f'Peak vs Integrated Flux {config_name}')
    plt.grid(True)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.legend(markerscale=100)
    plt.savefig(f'peak_vs_model_flux_{config_name}.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help=f"Which config to to use for Dataset/Generated subdirs, as defined in "
                        f" {paths.PROGRAM_CONFIG.name}",
                        type=str)
    args = parser.parse_args()

    plot_peak_vs_model_flux(args.config)
