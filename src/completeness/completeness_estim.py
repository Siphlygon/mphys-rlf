from scipy.optimize import curve_fit
import numpy as np
import scipy.stats
from tqdm import tqdm
import astropy.stats
import pandas as pd
import matplotlib.pyplot as plt
from utils.img_data_arrays import ImageDataArrays
import scipy.signal
import configparser
import utils.paths as pth
import logging
import utils.logging


class CompletenessEstimator:
    """
    A class to estimate the completeness of the dataset by creating mock images with noise and checking if they are
    detectable based on a peak-flux limit.
    """

    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("completeness estimator", logging.DEBUG)

        # Average LOFAR beam rms in mJy/beam
        self.rms_LOFAR = 95e-3

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


    def create_noise_LOFAR(self, shape=(80, 80)):
        """
        Create a 2D patch of Gaussian noise with given RMS.
        """
        # Add beam-correlated noise

        # Source - https://stackoverflow.com/a/63868276
        # Posted by Igor
        # Retrieved 2026-02-12, License - CC BY-SA 4.0

        # Compute filter kernel with radius correlation_scale
        correlation_scale = 6 / 1.5  # ( 6 arcsec / beam ) / ( 1.5 arcsec / pix )
        x = np.arange(-correlation_scale, correlation_scale)
        y = np.arange(-correlation_scale, correlation_scale)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X * X + Y * Y)
        if len( shape ) == 3:
            dist = dist[ np.newaxis, :, : ]
        filter_kernel = np.exp(-dist ** 2 / (2 * correlation_scale))

        noise = np.random.normal(loc=0.0, scale=self.rms_LOFAR, size=shape)
        noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')

        return noise


    def detect_sources(self, images, model_fluxes):
        """
        For a given set of input images and the model fluxes of those images from PyBDSF, creates mock images by adding
        noise patches to the input images, and checks if the mock sources are detectable based on a peak-flux threshold.

        :param images: The input images to which we will add noise patches
        :param model_fluxes: The fluxes of the sources in the input images
        :return: mock_fluxes, detectable - arrays of the fluxes of the mock sources and whether they are detectable
        """
        # Initialise empty arrays to store the mock fluxes (real images w/ noise) and whether they are detectable
        mock_fluxes = np.empty((images.shape[0] * self.num_noise_patches), dtype=float)
        detectable = np.empty((images.shape[0] * self.num_noise_patches), dtype=bool)

        for i in tqdm(range(images.shape[0]), desc='Creating mock images and running detection...'):
            # rms = image_rmss_actual[ random_image ]
            # noise_patch = resid_images[ random_image ]
            # Using rms=image_rmss_actual[ random_image ] is technically correct yet utterly useless because the
            # majority of the noise is from the artificial 1% noise added for pybdsf

            # TODO: Use raw LOFAR data so we can get rms locally based on strength of source, potential code commented above
            rms = self.rms_LOFAR

            # Create and apply noise patches for every input image
            mock_fluxes[i:(i + self.num_noise_patches)] = model_fluxes[i][np.newaxis]
            noise_patches = self.create_noise_LOFAR(shape=(self.num_noise_patches, 80, 80))
            sim_data = noise_patches + images[i][np.newaxis, :, :]

            # Determine if the mock sources are detectable based on a peak flux threshold (e.g., 5 sigma)
            peak_fluxes = np.max(sim_data, axis=(1, 2))
            threshold = self.sigma_threshold * rms
            detectable[i:(i + self.num_noise_patches)] = peak_fluxes >= threshold

        return mock_fluxes, detectable


    def get_completeness_estim(self):
        """
        Estimate a completeness curve for the generated dataset
        """
        # completeness for now is designed to use loguniform, as that creates a more even distribution of samples
        # per log flux bin, which is important for an informational completeness curve.

        for dataset in self.datasets:
            self.logger.info("Estimating completeness for dataset {}".format(dataset))
            # Extract all the relevant arrays from the generated dataset
            self.logger.info("Extracting data arrays for dataset {}".format(dataset))
            images, resid_images, m_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays(
                dataset).get_all_arrays()

            # Get the mock fluxes and whether they are detectable for all the images in the dataset
            self.logger.info("Creating mock images and running detection logic for dataset {}".format(dataset))
            mock_fluxes, detectable = self.detect_sources(images, model_fluxes)

            # Combine these into a dataframe for easier analysis
            mock_sources = pd.DataFrame()
            mock_sources['mock_flux'] = mock_fluxes
            mock_sources['detectable'] = detectable

            # Define flux bins
            flux_bins = np.logspace(-2, 2, num=self.num_flux_bins)  # in mJy, adjust as needed
            bin_centers = 0.5 * (flux_bins[1:] + flux_bins[:-1])

            # Bin and count
            completeness = []  # to store completeness per bin
            total_counts = []  # optional: for diagnostics

            # For all bins
            self.logger.info("Calculating completeness per flux bin for dataset {}".format(dataset))
            for i in tqdm(range(len(flux_bins) - 1), desc='Calculating completeness per flux bin'):
                # Select sources in this flux bin
                in_bin = (mock_fluxes >= flux_bins[i]) & (mock_fluxes < flux_bins[i + 1])

                # Calculuate the fraction of detectable sources in this bin
                n_detect = mock_sources[(mock_sources['mock_flux'] >= flux_bins[i]) & (mock_sources['mock_flux'] < flux_bins[i + 1])]
                if np.sum(in_bin) > 0:
                    frac_recovered = np.sum(n_detect['detectable']) / np.sum(in_bin)
                else:
                    self.logger.warning("No sources in flux bin {}-{} mJy for dataset {}, setting completeness to 0".format(flux_bins[i], flux_bins[i + 1], dataset))
                    frac_recovered = 0

                completeness.append(frac_recovered)
                total_counts.append(np.sum(in_bin))

            # Handle confidence interval with poisson_conf_interval for total_counts = 0
            total_counts = np.array(total_counts)
            zero_counts = total_counts == 0
            total_counts = np.where(zero_counts, 1e-10, total_counts)

            # Handle confidence interval which is the error on our completeness
            self.logger.info("Calculating confidence intervals for completeness estimates for dataset {}".format(dataset))
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

            # Fit sigmoid to completeness curve
            def sigmoid(x, x0, k, a, b):
                """Sigmoid function: a / (1 + exp(-k*(x-x0))) + b"""
                return a / (1 + np.exp(-k * (x - x0))) + b

            # Try just some polynomial
            def polynomial(x, a, b, c, d, e):
                """Quadratic polynomial: ax^2 + bx + c"""
                return a * x**4 + b * x**3 + c * x**2 + d * x + e

            # Use log of flux for fitting since we're on a log scale
            log_bin_centers = np.log10(bin_centers)

            # Initial parameter guesses: x0 (midpoint), k (steepness), a (amplitude), b (offset)
            initial_guess = [0.5, 2.0, 1.0, 0.0]

            try:
                # Fit the sigmoid
                popt, pcov = curve_fit(sigmoid, log_bin_centers, completeness, p0=initial_guess, maxfev=10000)

                # Generate smooth curve for plotting
                log_flux_smooth = np.linspace(log_bin_centers.min(), log_bin_centers.max(), 200)
                completeness_fit = sigmoid(log_flux_smooth, *popt)

                # Convert back to linear scale for plotting
                flux_smooth = 10 ** log_flux_smooth
                plt.plot(flux_smooth, completeness_fit, 'r--', linewidth=2, label=f'Sigmoid fit', alpha=0.7)

                # Show parameters and errors
                print(f"Sigmoid fit parameters:")
                print(f"  x0 (log midpoint): {popt[0]:.3f +- {pcov}} (flux: {10 ** popt[0]:.3f} mJy)")
                print(f"  k (steepness): {popt[1]:.3f}")
                print(f"  a (amplitude): {popt[2]:.3f}")
                print(f"  b (offset): {popt[3]:.3f}")
            except Exception as e:
                print(f"Warning: Sigmoid fit failed: {e}")

            try:
                # Fit the polynomial
                popt, pcov = curve_fit(polynomial, log_bin_centers, completeness, p0=[1, 1, 1, 1, 0], maxfev=10000)

                # Generate smooth curve for plotting
                log_flux_smooth = np.linspace(log_bin_centers.min(), log_bin_centers.max(), 200)
                completeness_fit = polynomial(log_flux_smooth, *popt)

                # Convert back to linear scale for plotting
                flux_smooth = 10 ** log_flux_smooth
                plt.plot(flux_smooth, completeness_fit, 'b--', linewidth=2, label=f'Polynomial fit', alpha=0.7)

                # Show parameters and errors
                print(f"Polynomial fit parameters:")
                print(f"  a (x^4): {popt[0]:.3e}")
                print(f"  b (x^3): {popt[1]:.3e}")
                print(f"  c (x^2): {popt[2]:.3e}")
                print(f"  d (x): {popt[3]:.3e}")
                print(f"  e (constant): {popt[4]:.3e}")
            except Exception as e:
                print(f"Warning: Sigmoid fit failed: {e}")


            # Plot completeness curve
            plt.figure()
            plt.errorbar(bin_centers, completeness, yerr, fmt='.', color='g')
            plt.plot(bin_centers, completeness, marker='.', label=f'{dataset} completeness', color='g')

            # Plot the fitted sigmoid curve
            x_fit = np.logspace(-2, 2, 100)
            y_fit = sigmoid(x_fit, *popt)
            plt.plot(x_fit, y_fit, label=f'{dataset} sigmoid fit', color='b')

            # And the polynomial fit
            y_fit_poly = polynomial(np.log10(x_fit), *popt)
            plt.plot(x_fit, y_fit_poly, label=f'{dataset} polynomial fit', color='r')

            plt.xscale('log')
            plt.xlabel("Integrated Flux Density (mJy/beam)")
            plt.ylabel("Completeness")
            plt.show()
            plt.savefig(dpi=1000, fname=f"completeness_curve_{dataset}.png")

if __name__ == "__main__":
    completeness_estim = CompletenessEstimator()
    completeness_estim.get_completeness_estim()

