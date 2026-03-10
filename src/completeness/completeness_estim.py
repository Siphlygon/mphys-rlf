from scipy.optimize import curve_fit
import numpy as np
import scipy.stats
from tqdm import tqdm
import astropy.stats
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

import rms_dist
from utils.img_data_arrays import ImageDataArrays
import configparser
import utils.paths as pth
import logging
import utils.logging
from utils.fitfunctions import sigmoid, polynomial_deg4


class CompletenessEstimator:
    """
    A class to estimate the completeness of the dataset by creating mock images with noise and checking if they are
    detectable based on a peak-flux limit.
    """

    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("completeness estimator", logging.DEBUG)

        # Initialise the RMS distribution finder
        self.rms_dist = rms_dist.RMSDistribution()

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(pth.PROGRAM_CONFIG)

        # we are using sources generated in a loguniform way
        de_config = config['DEFAULT']

        # Get values from config
        self.sigma_threshold = int(de_config['DETECTION_SIGMA_THRESHOLD'])
        self.num_flux_bins = int(de_config['COMPLETENESS_FLUX_BINS'])
        self.num_noise_patches = int(de_config['N_NOISE_PATCHES'])

        # Parse dataset names properly
        datasets_input = de_config['COMPLETENESS_DATASET_NAMES']
        if "," in datasets_input:
            self.datasets = [dataset.strip() for dataset in datasets_input.split(",")]
        else:
            self.datasets = [datasets_input.strip()]

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

        for i in tqdm(range(images.shape[0]), desc='Creating mock images and running detection...'):
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

    def fit_function(self,
                     bin_centers : np.ndarray,
                     completeness : np.ndarray,
                     yerr : np.ndarray,
                     dataset : str):
        """
        Fit a function to the completeness curve and plot the results.

        :param bin_centers: The centers of the flux bins used for calculating completeness.
        :param completeness: The completeness values calculated for each flux bin.
        :param yerr:  The error on the completeness values, typically calculated from confidence intervals.
        :param dataset: The name of the dataset being analyzed, used for labeling the plot and fit results.
        """
        # Use log of flux for fitting since we're on a log scale
        log_bin_centers = np.log10(bin_centers)

        # Initial parameter guesses: x0 (midpoint), k (steepness), a (amplitude), b (offset)
        initial_guess = [0.5, 2.0, 1.0, 0.0]

        popt_sigmoid = None
        pcov_sigmoid = None
        popt_poly = None
        pcov_poly = None

        # Start plotting the measured completeness first
        plt.figure()
        plt.errorbar(bin_centers, completeness, yerr, fmt='.', color='g', label=f'{dataset} data')
        plt.plot(bin_centers, completeness, marker='.', linestyle='None', color='g')

        try:
            self.logger.info(f"Fitting sigmoid function to completeness curve for dataset {dataset}")
            # Fit the sigmoid
            popt_sigmoid, pcov_sigmoid = curve_fit(sigmoid, log_bin_centers, completeness, p0=initial_guess, maxfev=10000)

            # Generate smooth curve for plotting on log scale
            log_flux_smooth = np.linspace(log_bin_centers.min(), log_bin_centers.max(), 200)
            completeness_fit = sigmoid(log_flux_smooth, *popt_sigmoid)

            # Convert back to linear scale for plotting
            flux_smooth = 10 ** log_flux_smooth
            plt.plot(flux_smooth, completeness_fit, 'r--', linewidth=2, label=f'Sigmoid fit', alpha=0.7)

            # Show parameters and errors
            print("Sigmoid fit parameters:")
            print(f"  x0 (log midpoint): {popt_sigmoid[0]:.3f} (flux: {10 ** popt_sigmoid[0]:.3f} mJy)")
            print(f"  k (steepness): {popt_sigmoid[1]:.3f}")
            print(f"  a (amplitude): {popt_sigmoid[2]:.3f}")
            print(f"  b (offset): {popt_sigmoid[3]:.3f}")
            np.save( pth.NP_ARRAY_PARENT / 'completeness_args_sigmoid.npy', popt_sigmoid )
            if pcov_sigmoid is not None:
                print(f"  covariance (diag): {np.sqrt(np.diag(pcov_sigmoid))}")
        except Exception as e:
            print(f"Warning: Sigmoid fit failed: {e}")

        try:
            self.logger.info(f"Fitting polynomial function to completeness curve for dataset {dataset}")
            # Fit the polynomial (on the log flux)
            popt_poly, pcov_poly = curve_fit(polynomial_deg4, log_bin_centers, completeness, p0=[1, 1, 1, 1, 0], maxfev=10000)

            # Generate smooth curve for plotting
            log_flux_smooth = np.linspace(log_bin_centers.min(), log_bin_centers.max(), 200)
            completeness_fit_poly = polynomial_deg4(log_flux_smooth, *popt_poly)

            # Convert back to linear scale for plotting
            flux_smooth = 10 ** log_flux_smooth
            plt.plot(flux_smooth, completeness_fit_poly, 'b--', linewidth=2, label=f'Polynomial fit', alpha=0.7)

            # Show parameters and errors
            print("Polynomial fit parameters:")
            print(f"  a (x^4): {popt_poly[0]:.3e}")
            print(f"  b (x^3): {popt_poly[1]:.3e}")
            print(f"  c (x^2): {popt_poly[2]:.3e}")
            print(f"  d (x): {popt_poly[3]:.3e}")
            print(f"  e (constant): {popt_poly[4]:.3e}")
            np.save( pth.NP_ARRAY_PARENT / 'completeness_args_poly4.npy', popt_poly )
            if pcov_poly is not None:
                print(f"  covariance (diag): {np.sqrt(np.diag(pcov_poly))}")
        except Exception as e:
            print(f"Warning: Polynomial fit failed: {e}")

        # Now add fit curves in a consistent way if they exist
        x_fit = np.logspace(-2, 2, 100)
        if popt_sigmoid is not None:
            # sigmoid was fit in log space, so evaluate at log10(x_fit)
            y_fit_sig = sigmoid(np.log10(x_fit), *popt_sigmoid)
            plt.plot(x_fit, y_fit_sig, label=f'{dataset} sigmoid fit', color='r')

        if popt_poly is not None:
            y_fit_poly = polynomial(np.log10(x_fit), *popt_poly)
            plt.plot(x_fit, y_fit_poly, label=f'{dataset} polynomial fit', color='b')

        plt.xscale('log')
        plt.xlabel("Integrated Flux Density (mJy/beam)")
        plt.ylabel("Completeness")
        plt.legend()
        plt.savefig(dpi=1000, fname=f"completeness_curve_{dataset}.png")
        plt.show()

    def get_completeness_estim(self):
        """
        Estimate a completeness curve for datasets specified in the config file.

        It does this by creating mock images, which are the original sources with added noise patches that are convolved
        with a kernel to simulate the beam-correlated noise in LOFAR images. These mock sources are then checked for
        detectability based on a peak flux threshold (e.g., 5 sigma). The completeness is calculated as the fraction of
        detectable sources in bins of flux, and confidence intervals are calculated using Poisson statistics.
        """
        # completeness for now is designed to use loguniform, as that creates a more even distribution of samples
        # per log flux bin, which is important for an informational completeness curve.

        for dataset in self.datasets:
            self.logger.info("Estimating completeness for dataset {}".format(dataset))
            # Extract all the relevant arrays from the generated dataset
            self.logger.info("Extracting data arrays for dataset {}".format(dataset))
            _, _, m_images, model_fluxes, _, _, _ = ImageDataArrays(dataset).get_all_arrays()

            # Get the mock fluxes and whether they are detectable for all the images in the dataset
            self.logger.info("Creating mock images and running detection logic for dataset {}".format(dataset))
            mock_fluxes, detectable = self.detect_mock_sources(m_images, model_fluxes)

            # Combine these into a dataframe for easier analysis
            mock_sources = pd.DataFrame()
            mock_sources['mock_flux'] = mock_fluxes
            mock_sources['detectable'] = detectable

            # Define integrated flux bins
            int_flux_bins = np.logspace(-2, 2, num=self.num_flux_bins)  # in mJy, adjust as needed
            bin_centers = 0.5 * (int_flux_bins[1:] + int_flux_bins[:-1])

            # Count detected sources in each bin and calculate completeness
            n_bins = len(int_flux_bins) - 1
            completeness = np.zeros(n_bins, dtype=float)  # to store completeness per bin
            total_counts = np.zeros(n_bins, dtype=int)  # optional: for diagnostics

            # For all bins
            self.logger.info("Calculating completeness per flux bin for dataset {}".format(dataset))
            for i in tqdm(range(n_bins), desc='Calculating completeness per flux bin'):
                # Select sources in this flux bin
                in_bin = (mock_fluxes >= int_flux_bins[i]) & (mock_fluxes < int_flux_bins[i + 1])

                # Calculate the fraction of detectable sources in this bin
                n_detect = mock_sources[
                    (mock_sources['mock_flux'] >= int_flux_bins[i]) & (mock_sources['mock_flux'] < int_flux_bins[i + 1])]
                if np.sum(in_bin) > 0:
                    frac_recovered = np.sum(n_detect['detectable']) / np.sum(in_bin)
                else:
                    self.logger.warning(
                        "No sources in flux bin {}-{} mJy for dataset {}, setting completeness to 0".format(
                            int_flux_bins[i], int_flux_bins[i + 1], dataset))
                    frac_recovered = 0

                completeness[i] = frac_recovered
                total_counts[i] = np.sum(in_bin)

            # Handle confidence interval with poisson_conf_interval for total_counts = 0
            total_counts = np.array(total_counts)
            zero_counts = total_counts == 0
            total_counts = np.where(zero_counts, 1e-10, total_counts)

            # Handle confidence interval which is the error on our completeness
            self.logger.info(
                "Calculating confidence intervals for completeness estimates for dataset {}".format(dataset))
            conf_interval = astropy.stats.poisson_conf_interval(np.array(completeness) * total_counts,
                                                                interval='frequentist-confidence', sigma=1.0)
            conf_interval /= total_counts
            conf_interval[:, zero_counts] = 0
            yerr = conf_interval[1] - conf_interval[0]

            # Store in a file for later use
            self.logger.info("Saving binned completeness estimates to file for dataset {}".format(dataset))
            with open(f"completeness_{dataset}.txt", "w") as f:
                f.write("Flux_bin_center(mJy/beam)\tCompleteness\tError\n")
                for center, comp, err in zip(bin_centers, completeness, yerr):
                    f.write(f"{center}\t{comp}\t{err}\n")

            # Fit a sigmoid function to the completeness curve and plot
            completeness = np.array(completeness)
            yerr = np.array(yerr)

            # Fit a sigmoid function to the completeness curve and plot
            self.fit_function(bin_centers, completeness, yerr, dataset)


if __name__ == "__main__":
    completeness_estim = CompletenessEstimator()
    completeness_estim.get_completeness_estim()
