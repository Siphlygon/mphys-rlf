import argparse

import astropy.stats
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.image_analyzer import ProcessArgs
from ..utils import paths
from ..utils.img_data_arrays import ImageDataArrays

RMS_LOFAR = 71e-6 * 1e3
BEAM_WIDTH_LOFAR = ProcessArgs().beam[:-1]
BEAM_AREA_LOFAR = BEAM_WIDTH_LOFAR[0] * BEAM_WIDTH_LOFAR[1]


def get_noise(data : np.ndarray) -> float:
    """
    Get the noise of the data using the method from the kMS package by Cyril Tasse, which is a robust estimator of the
    RMS noise in an image. This method is based on the work of Wara and is designed to handle images with outliers and
    non-Gaussian noise distributions.

    from Cyril Tasse/kMS, courtesy of Wara
    
    Parameters
    ----------
    data : np.ndarray
        The data to calculate the noise from

    Returns
    -------
    rms : float
        The estimated RMS noise of the data
    """
    maskSup = 1e-7
    m = data[np.abs(data) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)
    for _ in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold)//rmsold) < diff:
            break
        rmsold = rms
    return rms


def masking(fits_data: np.ndarray,
            threshold_level: float = 5.0) -> np.ndarray:
    """
    Masks the input fits_data array by setting all values below a certain threshold to zero. The threshold is calculated
    as a multiple of the standard deviation of the data, which is estimated using sigma-clipped statistics to reduce the
    influence of outliers.

    Parameters
    ----------
    fits_data : np.ndarray
        The input data array to be masked
    threshold_level : float, optional
        The number of standard deviations above the median to use as the threshold, by default 5.0

    Returns
    -------
    np.ndarray
        The masked data array
    """
    _, _, std_dev = astropy.stats.sigma_clipped_stats(fits_data, sigma=3.0)

    # Calculate the threshold
    threshold = threshold_level * std_dev

    # Create a mask for values less than the threshold
    mask = fits_data < threshold

    # Set values less than the threshold to zero
    fits_data_nr = np.where(mask, 0, fits_data)

    return fits_data_nr


def create_noise_lofar(shape: tuple = (80,80), rms: float = RMS_LOFAR) -> np.ndarray:
    """
    Create a 2D patch of Gaussian noise with given RMS.
    """
    return np.random.normal(loc=0.0, scale=rms, size=shape)


def get_completeness_estim(config_name: str):
    """
    A function to plot the number of samples per flux bin for the dataset and generated images, for comparison.

    Parameters
    ----------
    config_name : str
        The name of the configuration to use
    """
    config = paths.config[config_name]
    config_data_arrays = ImageDataArrays(config_name)

    plt.figure(figsize = (8, 5))
    for subdir, data_arrays in zip([config['generated_subdir']], [config_data_arrays.generated_data]):
        images = data_arrays.images
        model_fluxes = data_arrays.model_fluxes

        detectable = np.empty((images.shape[0]), dtype=bool)

        # Define flux bins and get the average samples per bin for > 10 mJy (before we start having issues)
        flux_bins = np.logspace(-2, 4, num=25)
        bin_centers = 0.5 * (flux_bins[1:] + flux_bins[:-1])
        total_counts = np.empty(len(flux_bins) - 1, dtype=float)
        for i in range(len(flux_bins) - 1):
            in_bin = (model_fluxes >= flux_bins[i]) & (model_fluxes < flux_bins[i + 1])
            total_counts[i] = np.sum(in_bin)
        #samples_per_bin_average = np.average(total_counts[bin_centers > 10])
        #print(f'Average samples per bin >10mJy: {samples_per_bin_average}')

        # Bin and count
        samples_per_bin = np.empty(len(flux_bins) - 1, dtype=float)
        detected_samples_per_bin = np.empty(len(flux_bins) - 1, dtype=float)

        for i in range(len(flux_bins) - 1):
            # Select sources in this flux bin
            total_in_bin, = np.where(np.logical_and(model_fluxes >= flux_bins[i], model_fluxes < flux_bins[i + 1]))
            #completeness[i] = detected_in_bin.shape[0] / samples_per_bin_average
            samples_per_bin[i] = total_in_bin.shape[0]

        # Handle confidence interval with poisson_conf_interval for total_counts = 0
        #conf_interval = astropy.stats.poisson_conf_interval(samples_per_bin, interval='frequentist-confidence', sigma=1.0)
        #conf_interval[:, total_counts != 0] /= total_counts[total_counts != 0]
        #yerr = conf_interval[1] - conf_interval[0]

        # Plot completeness curve

        #plt.errorbar(bin_centers, completeness, yerr, fmt='.', color='b' if subdir is utils.paths.DATASET_SUBDIR else 'g')

        plt.plot(bin_centers, samples_per_bin, marker='.', label = f'{subdir} counts', color='g')

    plt.xscale('log')
    plt.xlabel("Integrated Flux Density (mJy)")
    plt.ylabel("Samples per Bin")
    plt.title("Sources per Integrated Flux Bin")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('sources_per_bin.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help=f"Which config to use, as defined in {paths.PROGRAM_CONFIG.name}", type=str)
    args = parser.parse_args()

    get_completeness_estim(config_name=args.config)
