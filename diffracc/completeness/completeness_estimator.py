import argparse
import configparser
import inspect
from pathlib import Path
from typing import Callable

import astropy.stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from scipy.optimize import curve_fit
from tqdm import tqdm

from ..utils import paths
from ..utils.catalogue_dist import RMSDistribution
from ..utils.functions import sigmoid
from ..utils.img_data_arrays import ImageDataArrays, SubdirData
from ..utils.logger import LoggingLevels, get_logger


class CompletenessEstimator:
    """
    A class to estimate the completeness of the dataset by creating mock images with noise and checking if they are
    detectable based on a peak-flux limit.
    """
    def __init__(self,
                 config_str : str,
                 which_dataset : str | None = "GENERATED_SUBDIR",
                 override_data : bool = False):
        """
        A class to estimate the completeness of the dataset by creating mock images with noise and checking if they are
        detectable based on a peak-flux limit. 

        Parameters
        ----------
        config_str : str
            The specific configuration in the config file to use.
        which_dataset : str | None, optional
            Which of the two subdir to use in the analysis. Defaults to "GENERATED_SUBDIR".
        override_data : bool, optional
            Whether to not use ImageDataArrays as the source of data. Defaults to False.
        """
        self.logger = get_logger("CompletenessEstimator", LoggingLevels.DEBUG.value)

        assert which_dataset in ["GENERATED_SUBDIR", "DATASET_SUBDIR"],(
            "which_dataset must be either 'GENERATED_SUBDIR' or 'DATASET_SUBDIR'")
        self.which_dataset = which_dataset.split("_")[0].lower()  # "generated" or "dataset"

        self.rms_dist = RMSDistribution()

        # Read parameters from the config.ini file
        _config = configparser.ConfigParser()
        _config.read(paths.PROGRAM_CONFIG)
        self.config = dict(_config[config_str])
        self.sigma_threshold = int(self.config['DETECTION_SIGMA_THRESHOLD'])
        self.num_flux_bins = int(self.config['COMPLETENESS_FLUX_BINS'])
        self.min_log_flux = float(self.config['COMPLETENESS_MIN_LOG_FLUX'])
        self.max_log_flux = float(self.config['COMPLETENESS_MAX_LOG_FLUX'])
        self.num_noise_patches = int(self.config['N_NOISE_PATCHES'])

        if not override_data:
            # Extract all the relevant arrays from the specified dataset
            self.logger.info("Extracting data arrays for dataset")
            config_data_arrays = ImageDataArrays(config_str)
            self.data = config_data_arrays.__getattribute__(self.which_dataset + "_data")
        else:
            self.data = SubdirData()



    # ---------- FITTING FUNCTION ----------
    def _fit_function(self,
                     bin_centers : np.ndarray,
                     completeness : np.ndarray,
                     yerr : np.ndarray,
                     function: Callable = sigmoid,
                     initial_guess : list[float] | np.ndarray | None = None,
                     output_file : str | Path | None = None,
                     show_progress : bool = True,
                     **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a function to a completeness curve and return the fitted parameters and covariance matrix.

        Parameters
        ----------
        bin_centers : np.ndarray
            The centers of the flux bins used for calculating completeness. Already expected to be in log10(flux) space.
        completeness : np.ndarray
            The completeness values calculated for each flux bin.
        yerr : np.ndarray
            The errors on the y-axis of the completeness points.
        function : Callable, optional
            The function to fit to the completeness curve. Defaults to sigmoid.
        initial_guess : list[float] | np.ndarray | None, optional
            Initial guess for the parameters of the function. If None, a default guess will be generated based on the
            function signature. Defaults to None.
        output_file : str | Path | None, optional
            The name of the file to save the fitted parameters and covariance matrix. If None, the parameters will not
            be saved. Defaults to None.
        show_progress : bool, optional
            Whether to show progress bars for the different stages of the completeness estimation. Defaults to True.
        **kwargs : dict
            Additional keyword arguments to pass to the curve_fit function.

        Returns
        -------
        popt : np.ndarray
            The fitted parameters for the given function.
        pcov : np.ndarray
            The covariance matrix of the fitted parameters.
        """
        bin_centers = np.asarray(bin_centers, dtype=float)
        completeness = np.asarray(completeness, dtype=float)
        yerr = np.asarray(yerr, dtype=float)

        # Provide a sensible default initial guess based on the function signature.
        if initial_guess is None:
            # Determine the number of parameters in the function signature, excluding the first parameter (the data).
            try:
                sig = inspect.signature(function)
                params = list(sig.parameters.values())[1:]  # drop x
                param_names = [p.name for p in params]
                n_params = max(len(params), 0)
            except Exception:
                self.logger.warning(
                    f"Could not determine function signature for {function.__name__}, using default initial guess")
                param_names = []
                n_params = 0

            # Guess the 50% point from the data if possible.
            if completeness.size > 0:
                x0_guess = float(bin_centers[int(np.argmin(np.abs(completeness - 0.5)))])
            else:
                x0_guess = float(np.median(bin_centers))

            # Guess the slope/width based on the range of the bin centers.
            span = float(np.ptp(bin_centers)) if bin_centers.size > 1 else 1.0
            span = span if span > 0 else 1.0
            k_guess = 5.0 / span
            width_guess = span / 5.0

            # Guess the initial parameters based on the number of parameters.
            if n_params == 4:
                initial_guess = [x0_guess, k_guess, 1.0, 0.0]
            elif n_params == 3:
                initial_guess = [x0_guess, k_guess, 1.0]
            elif n_params == 2:
                # If the second parameter is a width/scale, guess in x-units.
                if len(param_names) >= 2 and param_names[1].lower() in {"sigma", "width", "w", "scale", "s"}:
                    initial_guess = [x0_guess, width_guess]
                else:
                    initial_guess = [x0_guess, k_guess]
            else:
                initial_guess = None

        try:
            if not callable(function):
                raise TypeError(f"`function` must be callable; got {type(function)}")

            if bin_centers.shape != completeness.shape:
                raise ValueError(
                    f"bin_centers and completeness must have the same shape; got {bin_centers.shape}"
                    f" vs {completeness.shape}"
                )
            if yerr.shape != completeness.shape:
                raise ValueError(
                    f"yerr must have same shape as completeness; got {yerr.shape} vs {completeness.shape}"
                )

            # Drop non-finite points and bins with non-positive uncertainty.
            finite_mask = np.isfinite(bin_centers) & np.isfinite(completeness) & np.isfinite(yerr)
            positive_sigma_mask = yerr > 0
            mask = finite_mask & positive_sigma_mask
            if not np.all(mask):
                dropped = int(np.size(mask) - np.count_nonzero(mask))
                if show_progress and dropped > 0:
                    self.logger.info(f"Dropping {dropped} points with non-finite/zero sigma before fitting")
                bin_centers = bin_centers[mask]
                completeness = completeness[mask]
                yerr = yerr[mask]

            if show_progress:
                self.logger.info(f"Fitting {function.__name__} function to completeness curve...")

            if initial_guess is None:
                popt, pcov = curve_fit(function, bin_centers, completeness, sigma=yerr, maxfev=100000, **kwargs)
            else:
                popt, pcov = curve_fit(function, bin_centers, completeness,
                                       p0=initial_guess, sigma=yerr, maxfev=100000, **kwargs)

            # Save fitted parameters to a file for use in RLF
            if output_file:
                out_path = Path(output_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("# Fitted parameters (popt)\n")
                    np.savetxt(f, np.asarray(popt)[None, :])
                    f.write("\n# Covariance matrix (pcov)\n")
                    np.savetxt(f, pcov)

            return popt, pcov

        except Exception as e:
            self.logger.error(f"Error: {function.__name__} fit failed: {e}")
            return np.array([]), np.array([])


    def plot_completeness(self,
                          bin_centers: np.ndarray,
                          completeness: np.ndarray,
                          yerr : np.ndarray,
                          function: Callable = sigmoid,
                          popt : list[float] | np.ndarray | None = None,
                          save_name : str | None = None):
        """
        Plot the completeness data points and the fitted function.

        Parameters
        ----------
        bin_centers : np.ndarray
            The centers of the flux bins used for calculating completeness. Already expected to be in log10(flux) space.
        completeness : np.ndarray
            The completeness values calculated for each flux bin.
        yerr : np.ndarray
            The errors on the y-axis of the completeness points.
        function : Callable, optional
            The function to fit to the completeness curve. Defaults to sigmoid.
        popt : list[float] | np.ndarray | None, optional
            The fitted parameters for the given function. If None, the function will not be plotted. Defaults to None.
        save_name : str | None, optional
            The name of the file to save the plot. If None, the plot will not be saved. Defaults to None.
        """
        assert popt is not None, "You need a fitted completeness function to plot."

        # Plot in linear flux on a log-scaled x-axis for readability.
        flux_centers = 10 ** bin_centers

        plt.figure()
        plt.errorbar(flux_centers, completeness, yerr, fmt='.', color='g', label='data')
        plt.plot(flux_centers, completeness, marker='.', linestyle='None', color='g')

        # Generate smooth curve for plotting.
        smooth_flux = np.logspace(bin_centers.min(), bin_centers.max(), 200)
        smooth_log_flux = np.log10(smooth_flux)
        completeness_fit = function(smooth_log_flux, *popt)
        plt.plot(smooth_flux, completeness_fit, color='c', label=f'{function.__name__} fit')

        plt.xscale('log')
        plt.xlabel("Integrated Flux Density (mJy/beam)")
        plt.ylabel("Completeness")
        plt.legend()
        if save_name is not None:
            plt.savefig(dpi=1000, fname=save_name)
        plt.show()



    # ---------- COMPLETENESS CALCULATION ----------
    def _create_beam_corr_noise(self,
                               filter_kernel: np.ndarray,
                               rms : float | np.ndarray = 95e-3,
                               shape: tuple[int, int, int] = (5, 80, 80),
                               ) -> np.ndarray:
        """
        Create a 2D patch of Gaussian noise with given RMS.

        Parameters
        ----------
        filter_kernel : np.ndarray
            The 2D filter kernel used to create beam-correlated noise.
        rms : float | np.ndarray, optional
            The RMS value(s) for the Gaussian noise. Can be a single float or an array of floats for multiple patches.
            Defaults to 95e-3.
        shape : tuple[int, int, int], optional
            The shape of the output noise array. Defaults to (5, 80, 80), where 5 is the number of patches and 80x80 is
            the size of each patch.
        
        Returns
        -------
        np.ndarray
            A 3D array of shape `shape` containing the generated noise patches.
        """
        # Source - https://stackoverflow.com/a/63868276
        # Posted by Igor
        # Retrieved 2026-02-12, License - CC BY-SA 4.0
        noise = np.random.normal(loc=0.0, scale=rms, size=shape)
        noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        return noise


    def _detect_mock_sources(self,
                            images: np.ndarray,
                            model_fluxes: np.ndarray,
                            show_progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        For a given set of input images and the model fluxes of those images from PyBDSF, creates mock images by adding
        noise patches to the input images, and checks if the mock sources are detectable based on a peak-flux threshold.

        Parameters
        ----------
        images : np.ndarray
            The input images to which noise patches will be added.
        model_fluxes : np.ndarray
            The model fluxes of the input images, used to determine the detectability of the mock sources.
        show_progress : bool, optional
            Whether to show progress bars for the different stages of the completeness estimation. Defaults to True.
        
        Returns
        -------
        mock_fluxes : np.ndarray
            The fluxes of the mock sources created by adding noise patches to the input images.
        detectable : np.ndarray
            A boolean array indicating whether each mock source is detectable based on the peak-flux threshold.
        """
        # Precompute correlation / blur parameters used to create beam-correlated noise.
        # correlation_scale chosen to match previous behaviour: (6 arcsec / beam) / (1.5 arcsec / pix)
        correlation_scale = 6 / 1.5
        x = np.arange(-correlation_scale, correlation_scale)
        y = np.arange(-correlation_scale, correlation_scale)
        x, y = np.meshgrid(x, y)

        # Compute the distance from the center of the kernel for each pixel
        dist = np.sqrt(x * x + y * y)
        dist = dist[np.newaxis, :, :]

        # Normalise the kernel
        _filter_kernel_2d = np.exp(-dist**2 / (2*correlation_scale)) * (1 / (2*np.pi*correlation_scale**2))

        # Initialise empty arrays to store the mock fluxes (real images w/ noise) and whether they are detectable
        mock_fluxes = np.empty((images.shape[0] * self.num_noise_patches), dtype=float)
        detectable = np.empty((images.shape[0] * self.num_noise_patches), dtype=bool)

        for i in tqdm(range(images.shape[0]),
                      desc='Creating mock images and running detection logic', disable=not show_progress):
            # Use start/end indices so each image occupies a contiguous block of the arrays.
            start = i * self.num_noise_patches
            end = start + self.num_noise_patches

            # Randomly draw a RMS from the distribution of values present in the Hardcastle catalogue sources
            rms = self.rms_dist.sample()

            # Create and apply noise patches for every input image
            mock_fluxes[start:end] = np.full((self.num_noise_patches,), model_fluxes[i], dtype=float)
            noise_patches = self._create_beam_corr_noise(_filter_kernel_2d,
                                                        rms=rms, shape=(self.num_noise_patches, 80, 80))

            # Ensure the image slice is 2D so it can broadcast against (n_patches, 80, 80).
            # Some FITS readers return shapes like (1, 80, 80) for a single image.
            image_2d = images[i]
            while (getattr(image_2d, "ndim", 0) > 2) and (image_2d.shape[0] == 1):
                image_2d = image_2d[0]
            if image_2d.ndim != 2:
                raise ValueError(f"Expected image[{i}] to be 2D after squeezing; got shape {image_2d.shape}")

            sim_data = noise_patches + image_2d[np.newaxis, :, :]

            # Determine if the mock sources are detectable based on a peak flux threshold (e.g., 5 sigma)
            peak_fluxes = np.max(sim_data, axis=(1, 2))
            threshold = self.sigma_threshold * rms
            detectable[start:end] = peak_fluxes >= threshold

        # save mock fluxes to a file for later use
        if show_progress:
            self.logger.info("Saving mock fluxes and detectability to file...")
        with open("mock_fluxes_detectability.txt", "w", encoding="utf-8") as f:
            f.write("Mock_Flux(mJy/beam)\tDetectable\n")
            for flux, detect in zip(mock_fluxes, detectable):
                f.write(f"{flux}\t{detect}\n")

        return mock_fluxes, detectable


    def _compute_completeness_per_bin(self,
                             int_flux_bins : np.ndarray,
                             mock_sources : pd.DataFrame,
                             show_progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the completeness by calculating the fraction of detectable sources in every integrated flux bin. Also
        produces y-axis errors for these values based on the Poisson 1-sigma confidence itnervals.

        Parameters
        ----------
        int_flux_bins : np.ndarray
            The integrated flux bins.
        mock_sources : pd.DataFrame
            A dataframe containing the mock fluxes and whether they were detected.
        show_progress : : bool, optional
            Whether to show progress bars for the different stages of the completeness estimation. Defaults to True.

        Returns
        -------
        completeness : np.ndarray
            The completeness values for each flux bin.
        yerr : np.ndarray
            The errors on the y-axis of the completeness points, based on Poisson statistics.
        """
        # Count detected sources in each bin and calculate completeness
        n_bins = len(int_flux_bins) - 1
        completeness = np.zeros(n_bins, dtype=float)  # to store completeness per bin
        total_counts = np.zeros(n_bins, dtype=int)  # optional: for diagnostics

        # For all bins
        for i in tqdm(range(n_bins), desc='Calculating completeness per flux bin', disable=not show_progress):
            # Select sources in this flux bin
            in_bin = (mock_sources['mock_flux'] >= int_flux_bins[i]) & (
                mock_sources['mock_flux'] < int_flux_bins[i+1])

            self.logger.info(f"Flux bin {int_flux_bins[i]:.3f} - {int_flux_bins[i+1]:.3f}"
                             f" mJy/beam: {np.sum(in_bin)} sources")

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
        conf_interval = astropy.stats.poisson_conf_interval(np.array(completeness) * total_counts,
                                                            interval='frequentist-confidence', sigma=1)
        conf_interval /= total_counts
        conf_interval[:, zero_counts] = 0
        yerr = np.array(conf_interval[1] - conf_interval[0])

        return completeness, yerr


    def estimate_completeness(self,
                              function: Callable = sigmoid,
                              initial_guess : list[float] | np.ndarray | None = None,
                              comp_output_file : str | Path | None = None,
                              func_output_file : str | Path | None = None,
                              plot_completeness : bool = True,
                              figure_save_name : str | None = None,
                              show_progress : bool = True
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate a completeness curve for a specified dataset.

        It does this by creating mock images, which are the original sources with added noise patches that are convolved
        with a kernel to simulate the beam-correlated noise in LOFAR images. These mock sources are then checked for
        detectability based on a peak flux threshold (e.g., 5 sigma). The completeness is calculated as the fraction of
        detectable sources in bins of flux, and confidence intervals are calculated using Poisson statistics.
        
        Parameters
        ----------
        function : Callable, optional
            The function to fit to the completeness curve. Defaults to sigmoid.
        initial_guess : list[float] | np.ndarray | None, optional
            Initial guess for the parameters of the function to be fitted. Defaults to None.
        comp_output_file : str | Path | None, optional
            The file to save the binned completeness estimates to. Defaults to None.
        func_output_file : str | Path | None, optional
            The file to save the fitted function parameters to. Defaults to None.
        plot_completeness : bool, optional
            Whether to plot the completeness curve and fitted function. Defaults to True.
        figure_save_name : str | None, optional
            The name of the file to save the completeness plot to. Defaults to None.
        show_progress : bool, optional
            Whether to show progress bars for the different stages of the completeness estimation. Defaults to True.

        Returns
        -------
        log_bin_centers : np.ndarray
            The log10 of the bin centers used for calculating completeness.
        completeness : np.ndarray
            The completeness values calculated for each flux bin.
        yerr : np.ndarray
            The errors on the y-axis of the completeness points, based on Poisson statistics.
        fitted_params : np.ndarray
            The fitted parameters for the given function.
        pcov : np.ndarray
            The covariance matrix of the fitted parameters.
        """
        # Get the mock fluxes and whether they are detectable for all the images in the dataset
        mock_fluxes, detectable = self._detect_mock_sources(self.data.model_images,
                                                            self.data.model_fluxes, show_progress)

        # Combine these into a dataframe for easier analysis
        mock_sources = pd.DataFrame()
        mock_sources['mock_flux'] = mock_fluxes
        mock_sources['detectable'] = detectable

        # Calculate completeness per bin and y errors, and save to a file if desired
        int_flux_bins = np.logspace(self.min_log_flux, self.max_log_flux, num=self.num_flux_bins)  # in mJy
        bin_centers = 0.5 * (int_flux_bins[1:] + int_flux_bins[:-1])
        completeness, yerr = self._compute_completeness_per_bin(int_flux_bins, mock_sources, show_progress)

        # Store in a file for later use
        if comp_output_file is not None:
            if show_progress:
                self.logger.info("Saving binned completeness estimates to file...")
            with open(comp_output_file, "w", encoding="utf-8") as f:
                f.write("Flux_bin_center(mJy/beam)\tCompleteness\tError\n")
                for center, comp, err in zip(bin_centers, completeness, yerr):
                    f.write(f"{center}\t{comp}\t{err}\n")

        # Fit a function to the completeness curve. Even if not plotting, if this fit fails something is wrong
        log_bin_centers = np.log10(bin_centers)
        fitted_params, pcov = self._fit_function(
            log_bin_centers,
            completeness,
            yerr,
            function=function,
            initial_guess=initial_guess,
            output_file=func_output_file,
            show_progress=show_progress,
        )

        if plot_completeness:
            self.plot_completeness(log_bin_centers, completeness, yerr, function, fitted_params, figure_save_name)

        return log_bin_centers, completeness, yerr, fitted_params, pcov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help=f"Which config to to use for Dataset/Generated subdirs, as defined in {paths.PROGRAM_CONFIG.name}",
        type=str)
    args = parser.parse_args()

    completeness_estim = CompletenessEstimator(args.config)
    completeness_estim.estimate_completeness()
