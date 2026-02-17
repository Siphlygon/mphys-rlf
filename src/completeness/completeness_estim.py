from scipy.optimize import curve_fit
import numpy as np
import scipy.stats
from tqdm import tqdm
import astropy.stats
import pandas as pd
import matplotlib.pyplot as plt
from utils.img_data_arrays import ImageDataArrays
import scipy.signal


class CompletenessEstimator:
    """
    A class to estimate the completeness of the dataset by creating mock images with noise and checking if they are
    detectable based on a peak-flux limit.
    """

    def __init__(self):
        # Average LOFAR beam rms
        self.rms_LOFAR = 95e-3

        # Number of noise patches to create per image for the completeness estimation
        self.num_noise_patches = 5


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
            threshold = 5 * rms
            detectable[i:(i + self.num_noise_patches)] = peak_fluxes >= threshold

        return mock_fluxes, detectable


    def get_completeness_estim(self):
        """
        Estimate a completeness curve for the LOFAR

        :return:
        """
        # completeness for now is designed to use loguniform, as that creates a more even distribution of samples
        # per log flux bin, which is important for an informational completeness curve.

        # Store the completeness curve data for later use in a file
        completeness_curve = []

        for subdir in ["generated_loguniform"]:
            # Extract all the relevant arrays from the generated dataset
            images, resid_images, m_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays(
                subdir).get_all_arrays()

            # Get the mock fluxes and whether they are detectable for all the images in the dataset
            mock_fluxes, detectable = self.detect_sources(images, model_fluxes)

            # Combine these into a dataframe for easier analysis
            mock_sources = pd.DataFrame()
            mock_sources['mock_flux'] = mock_fluxes
            mock_sources['detectable'] = detectable

            # Define flux bins
            flux_bins = np.logspace(-2, 2, num=25)  # in Jy, adjust as needed
            bin_centers = 0.5 * (flux_bins[1:] + flux_bins[:-1])

            # Bin and count
            completeness = []  # to store completeness per bin
            total_counts = []  # optional: for diagnostics

            # For all bins
            for i in range(len(flux_bins) - 1):
                # Select sources in this flux bin
                in_bin = (mock_fluxes >= flux_bins[i]) & (mock_fluxes < flux_bins[i + 1])

                # Calculuate the fraction of detectable sources in this bin
                n_detect = mock_sources[(mock_sources['mock_flux'] >= flux_bins[i]) & (mock_sources['mock_flux'] < flux_bins[i + 1])]
                if np.sum(in_bin) > 0:
                    frac_recovered = np.sum(n_detect['detectable']) / np.sum(in_bin)
                else:
                    frac_recovered = 0

                completeness.append(frac_recovered)
                total_counts.append(np.sum(in_bin))

            # Handle confidence interval with poisson_conf_interval for total_counts = 0
            total_counts = np.array(total_counts)
            zero_counts = total_counts == 0
            total_counts = np.where(zero_counts, 1e-10, total_counts)

            # Handle confidence interval which is the error on our completeness
            conf_interval = astropy.stats.poisson_conf_interval(np.array(completeness) * total_counts,
                                                                interval='frequentist-confidence', sigma=1.0)
            conf_interval /= total_counts
            conf_interval[:, zero_counts] = 0
            yerr = conf_interval[1] - conf_interval[0]

if __name__ == "__main__":
    completeness_estim = CompletenessEstimator()
    completeness_estim.get_completeness_estim()

