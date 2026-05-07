import argparse
from scipy.optimize import curve_fit
import numpy as np
from tqdm import tqdm
import astropy.stats
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path
from typing import Callable

from rms_dist import RMSDistribution
from utils.img_data_arrays import ImageDataArrays
import configparser
import utils.paths as pth
import logging
import utils.logging
from utils.functions import sigmoid


class CompletenessEstimator:
    
    def __init__(self,
                 config_str : str,
                 which_dataset : str | None = "GENERATED_SUBDIR"):
        """
        A class to estimate the completeness of the dataset by creating mock images with noise and checking if they are detectable based on a peak-flux limit. 

        Args:
            config_str (str): The specific configuration in the config file to use.
            which_dataset (str, optional): Which of the two subdir to use in the analysis. Defaults to "GENERATED_SUBDIR"
        """
        # Set up logging
        self.logger = utils.logging.get_logger("CompletenessEstimator", logging.DEBUG)

        assert which_dataset in ["GENERATED_SUBDIR", "DATASET_SUBDIR"], "which_dataset must be either 'GENERATED_SUBDIR' or 'DATASET_SUBDIR'"
        self.which_dataset = which_dataset.split("_")[0].lower()  # "generated" or "dataset"

        # Initialise the RMS distribution finder
        self.rms_dist = RMSDistribution()

        # Read parameters from the config.ini file
        _config = configparser.ConfigParser().read(pth.PROGRAM_CONFIG)
        self.config = _config[config_str]

        # Get values from config
        self.sigma_threshold = int(self.config['DETECTION_SIGMA_THRESHOLD'])
        self.num_flux_bins = int(self.config['COMPLETENESS_FLUX_BINS'])
        self.min_log_flux = float(self.config['COMPLETENESS_MIN_LOG_FLUX'])
        self.max_log_flux = float(self.config['COMPLETENESS_MAX_LOG_FLUX'])
        self.num_noise_patches = int(self.config['N_NOISE_PATCHES'])
        
        # Extract all the relevant arrays from the specified dataset
        self.logger.info(f"Extracting data arrays for dataset")
        config_data_arrays = ImageDataArrays(self.config)
        self.data = config_data_arrays.__getattribute__(self.which_dataset + "_data")

    # ---------- FITTING FUNCTION ----------
    def fit_function(self,
                     bin_centers : np.ndarray[float, np.dtype[np.float64]],
                     completeness : np.ndarray[float, np.dtype[np.float64]],
                     function: Callable = sigmoid,
                     initial_guess : list[float] | np.ndarray[float, np.dtype[np.float64]] | None = None,
                     output_file : str | Path | None = None) -> np.ndarray[float, np.dtype[np.float64]] | None:
        """
        Fit a function to the completeness curve.

        :param bin_centers: The centers of the flux bins used for calculating completeness.
        :param completeness: The completeness values calculated for each flux bin.
        :param function: The function to fit to the completeness curve. Defaults to sigmoid.
        :param initial_guess: Initial guess for the parameters of the function to be fitted. Defaults to [0.5, 7.0, 1.0, 0.0] for sigmoid.
        :param output_file: Where to save the results. Defaults to None.
        :returns: The fitted parameters for the given function.
        """
        # Use log of flux for fitting since we're on a log scale
        if initial_guess is None:
            initial_guess = [0.5, 7.0, 1.0, 0.0]

        try:
            self.logger.info(f"Fitting {function} function to completeness curve...")
            popt, _ = curve_fit(function, bin_centers, completeness, p0=initial_guess, maxfev=10000)

            # Save fitted parameters to a file for use in RLF
            if output_file:
                np.savetxt(output_file, popt)

            return popt

        except Exception as e:
            self.logger.error(f"Error: {function} fit failed: {e}")

    def plot_completeness(self,
                          bin_centers: np.ndarray[float, np.dtype[np.float64]],
                          completeness: np.ndarray[float, np.dtype[np.float64]],
                          yerr : np.ndarray[float, np.dtype[np.float64]],
                          function: Callable = sigmoid,
                          popt : list[float] | None = None,
                          save_name : str | None = None):
        """
        Plot the completeness data points and the fitted function.

        :param bin_centers: The centers of the flux bins used for calculating completeness.
        :param completeness: The completeness values calculated for each flux bin.
        :param yerr: The errors on the y-axis of the completeness points.
        :param function: The function that was fitted to the completeness curve. Defaults to sigmoid.
        :param popt: The fitted parameters to the function.
        :param save_name: The name of the file to save the plot to. Defaults to None.
        """
        assert popt, "You need a fitted completeness function to plot."
        if save_name is None:
            save_name = f"completeness_curve.png"
    
        # Start plotting the measured completeness first
        plt.figure()
        plt.errorbar(bin_centers, completeness, yerr, fmt='.', color='g', label=f'data')
        plt.plot(bin_centers, completeness, marker='.', linestyle='None', color='g')

        # Generate smooth curve for plotting on log scale
        log_flux_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 200)
        completeness_fit = sigmoid(log_flux_smooth, *popt)

        # Convert back to linear scale for plotting
        flux_smooth = 10 ** log_flux_smooth
        plt.plot(flux_smooth, completeness_fit, 'r--', linewidth=2, label=f'{function.__name__} fit', alpha=0.7)

        # Now add fit curves in a consistent way
        x_fit = np.logspace(self.min_log_flux, self.max_log_flux, 100)
        y_fit_sig = sigmoid(np.log10(x_fit), *popt)  # function was fit in log x space, so evaluate at log10(x_fit)
        plt.plot(x_fit, y_fit_sig, label=f'{function.__name__} fit', color='r')

        plt.xscale('log')
        plt.xlabel("Integrated Flux Density (mJy/beam)")
        plt.ylabel("Completeness")
        plt.legend()
        plt.savefig(dpi=1000, fname=save_name)
        plt.show()

    # ---------- COMPLETENESS CALCULATION ----------
    def create_noise_LOFAR(self,
                           filter_kernel: np.ndarray,
                           rms : float | np.ndarray = 95e-3,
                           shape: tuple[int, int, int] = (5, 80, 80),
                           ) -> np.ndarray:
        """
        Create a 2D patch of Gaussian noise with given RMS.

        :param filter_kernel: A 2D kernel to convolve the noise with, simulating the beam-correlated noise in LOFAR images.
        :param rms: The RMS of the noise to be generated. Can be a single float or an array of floats for multiple patches.
        :param shape: The shape of the noise array to be generated. Default is (5, 80, 80) for 5 noise patches of size 80x80 pixels.
        :return: A numpy array of shape `shape` containing the generated noise patches.
        """
        # Add beam-correlated noise

        # Source - https://stackoverflow.com/a/63868276
        # Posted by Igor
        # Retrieved 2026-02-12, License - CC BY-SA 4.0
        noise = np.random.normal(loc=0.0, scale=rms, size=shape)
        noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        return noise

    def detect_mock_sources(self,
                            images: np.ndarray,
                            model_fluxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        For a given set of input images and the model fluxes of those images from PyBDSF, creates mock images by adding
        noise patches to the input images, and checks if the mock sources are detectable based on a peak-flux threshold.

        :param images: The input images to which we will add noise patches
        :param model_fluxes: The fluxes of the sources in the input images
        :return: mock_fluxes, detectable - arrays of the fluxes of the mock sources and whether they are detectable
        """
        # Precompute correlation / blur parameters used to create beam-correlated noise.
        # correlation_scale chosen to match previous behaviour: (6 arcsec / beam) / (1.5 arcsec / pix)
        correlation_scale = 6 / 1.5
        x = np.arange(-correlation_scale, correlation_scale)
        y = np.arange(-correlation_scale, correlation_scale)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X * X + Y * Y)
        dist = dist[np.newaxis, :, :]
        _filter_kernel_2d = np.exp(-dist ** 2 / (2 * correlation_scale)) * (1 / (2 * np.pi * correlation_scale ** 2))  # Normalise the kernel

        # Initialise empty arrays to store the mock fluxes (real images w/ noise) and whether they are detectable
        mock_fluxes = np.empty((images.shape[0] * self.num_noise_patches), dtype=float)
        detectable = np.empty((images.shape[0] * self.num_noise_patches), dtype=bool)

        for i in tqdm(range(images.shape[0]), desc='Creating mock images and running detection logic'):
            # Use start/end indices so each image occupies a contiguous block of the arrays.
            start = i * self.num_noise_patches
            end = start + self.num_noise_patches

            # Randomly draw a RMS from the distribution of values present in the Hardcastle catalogue sources
            rms = self.rms_dist.sample()

            # Create and apply noise patches for every input image
            mock_fluxes[start:end] = np.full((self.num_noise_patches,), model_fluxes[i], dtype=float)
            noise_patches = self.create_noise_LOFAR(_filter_kernel_2d, rms=rms, shape=(self.num_noise_patches, 80, 80))
            sim_data = noise_patches + images[i][np.newaxis, :, :]

            # Determine if the mock sources are detectable based on a peak flux threshold (e.g., 5 sigma)
            peak_fluxes = np.max(sim_data, axis=(1, 2))
            threshold = self.sigma_threshold * rms
            detectable[start:end] = peak_fluxes >= threshold

        return mock_fluxes, detectable

    def compute_completeness_per_bin(self,
                             int_flux_bins : np.ndarray,
                             mock_sources : pd.DataFrame):
        """
        Computes the completeness by calculating the fraction of detectable sources in every integrated flux bin.
        Also produces y-axis errors for these values based on the Poisson 1-sigma confidence itnervals.

        Args:
            int_flux_bins (np.ndarray): The integrated flux bins.
            mock_sources (pd.DataFrame): A dataframe containing the mock fluxes and whether they were detected.

        Returns:
            tuple(np.ndarray, nd.ndarray) : The completeness values and calculated y errors for each integrated flux bin.
        """
        # Count detected sources in each bin and calculate completeness
        n_bins = len(int_flux_bins) - 1
        completeness = np.zeros(n_bins, dtype=float)  # to store completeness per bin
        total_counts = np.zeros(n_bins, dtype=int)  # optional: for diagnostics

        # For all bins
        for i in tqdm(range(n_bins), desc='Calculating completeness per flux bin'):
            # Select sources in this flux bin
            in_bin = (mock_sources['mock_flux'] >= int_flux_bins[i]) & (mock_sources['mock_flux'] < int_flux_bins[i + 1])

            # Calculate the fraction of detectable sources in this bin
            n_detect = mock_sources[
                (mock_sources['mock_flux'] >= int_flux_bins[i]) & (mock_sources['mock_flux'] < int_flux_bins[i + 1])]
            if np.sum(in_bin) > 0:
                frac_recovered = np.sum(n_detect['detectable']) / np.sum(in_bin)
            else:
                self.logger.warning(
                    f"No sources in flux bin {int_flux_bins[i]}-{int_flux_bins[i + 1]} mJy, setting completeness to 0")
                frac_recovered = 0

            completeness[i] = frac_recovered
            total_counts[i] = np.sum(in_bin)

        # Handle confidence interval with poisson_conf_interval for total_counts = 0
        total_counts = np.array(total_counts)
        zero_counts = total_counts == 0
        total_counts = np.where(zero_counts, 1e-10, total_counts)

        # Handle confidence interval which is the error on our completeness
        self.logger.info(f"Calculating confidence intervals for completeness estimates...")
        conf_interval = astropy.stats.poisson_conf_interval(np.array(completeness) * total_counts,
                                                            interval='frequentist-confidence', sigma=1.0)
        conf_interval /= total_counts
        conf_interval[:, zero_counts] = 0
        yerr = np.array(conf_interval[1] - conf_interval[0])
        
        return completeness, yerr

    def estimate_completeness(self,
                               comp_output_file : str | Path | None = None,
                               func_output_file : str | Path | None = None,
                               plot_completeness : bool = True):
        """
        Estimate a completeness curve for a specified dataset.

        It does this by creating mock images, which are the original sources with added noise patches that are convolved
        with a kernel to simulate the beam-correlated noise in LOFAR images. These mock sources are then checked for
        detectability based on a peak flux threshold (e.g., 5 sigma). The completeness is calculated as the fraction of
        detectable sources in bins of flux, and confidence intervals are calculated using Poisson statistics.
        
        :param comp_output_file: The file in which to save completeness results to. Defaults to None.
        :param func_output_file: The file in which to save fitted parameters to. Defaults to None.
        :param plot_completeness: Whether the fitted completeness should be plotted. Defaults to True.
        :return: The bin centers in log space, estimated completeness for those bins, y error on the completeness,
        and the fitted parameters for the fitting function.
        """
        # Get the mock fluxes and whether they are detectable for all the images in the dataset
        self.logger.info(f"Creating mock images and running detection logic...")
        mock_fluxes, detectable = self.detect_mock_sources(self.data.m_images, self.data.model_fluxes)

        # Combine these into a dataframe for easier analysis
        mock_sources = pd.DataFrame()
        mock_sources['mock_flux'] = mock_fluxes
        mock_sources['detectable'] = detectable

        # Compute completeness for each bin
        int_flux_bins = np.logspace(self.min_log_flux, self.max_log_flux, num=self.num_flux_bins)  # in mJy
        bin_centers = 0.5 * (int_flux_bins[1:] + int_flux_bins[:-1])
        completeness, yerr = self.compute_completeness_per_bin(int_flux_bins, mock_sources)

        # Store in a file for later use
        if comp_output_file is not None:
            self.logger.info(f"Saving binned completeness estimates to file...")
            with open(comp_output_file, "w") as f:
                f.write("Flux_bin_center(mJy/beam)\tCompleteness\tError\n")
                for center, comp, err in zip(bin_centers, completeness, yerr):
                    f.write(f"{center}\t{comp}\t{err}\n")

        # Fit a function to the completeness curve and plot
        log_bin_centers = np.log10(bin_centers)
        fitted_params = self.fit_function(log_bin_centers, completeness, output_file=func_output_file)

        if plot_completeness:
            self.plot_completeness(log_bin_centers, completeness, yerr, sigmoid, fitted_params)
        
        return log_bin_centers, completeness, yerr, fitted_params


class SizeBinnedCompleteness(CompletenessEstimator):
    def __init__(self,
                 config_str : str,
                 which_dataset : str | None = "GENERATED_SUBDIR"):
        """
        A class to generate completeness curves binned by angular size, to investigate how completeness varies with source size.

        Args:
            config_str (str): The specific configuration in the config file to use.
            which_dataset (str, optional): Which of the two subdir to use in the analysis. Defaults to "GENERATED_SUBDIR"
        """
        super().__init__(config_str, which_dataset)
    
    def estimate_size_binned_completeness(self):
        max_size = 120 # in arcseconds, adjust as needed
        max_bins = 12
        size_bins = np.linspace(0, max_size, num=max_bins)

        for i in range(len(size_bins) - 1):
            size_min = size_bins[i]
            size_max = size_bins[i + 1]
            self.logger.info(f"Estimating completeness for sources with sizes between {size_min} and {size_max} arcseconds")
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", help=f"Which config to to use for Dataset/Generated subdirs, as defined in {pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()
    
    completeness_estim = CompletenessEstimator(args.config)
    completeness_estim.estimate_completeness()
