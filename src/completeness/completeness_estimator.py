import argparse
from scipy.optimize import curve_fit
import numpy as np
from tqdm import tqdm
import astropy.stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
from pathlib import Path
from typing import Callable
import inspect
from astropy.io import fits

from utils.catalogue_dist import RMSDistribution
from utils.img_data_arrays import ImageDataArrays
import configparser
import utils.paths as pth
import logging
import utils.logging
from utils.functions import sigmoid
from completeness.ang_size_finder import AngularSizeFinder
from utils.recursive_file_analyzer import RecursiveFileAnalyzer
from analysis.log_analyzer import get_model_flux


class CompletenessEstimator:
    
    def __init__(self,
                 config_str : str,
                 which_dataset : str | None = "GENERATED_SUBDIR",
                 override_data : bool = False):
        """
        A class to estimate the completeness of the dataset by creating mock images with noise and checking if they are detectable based on a peak-flux limit. 

        Args:
            config_str (str): The specific configuration in the config file to use.
            which_dataset (str, optional): Which of the two subdir to use in the analysis. Defaults to "GENERATED_SUBDIR"
            override_data (bool, optional): Whether to not use ImageDataArrays as the source of data. Defaults to False.
        """
        # Set up logging
        self.logger = utils.logging.get_logger("CompletenessEstimator", logging.DEBUG)

        assert which_dataset in ["GENERATED_SUBDIR", "DATASET_SUBDIR"], "which_dataset must be either 'GENERATED_SUBDIR' or 'DATASET_SUBDIR'"
        self.which_dataset = which_dataset.split("_")[0].lower()  # "generated" or "dataset"

        # Initialise the RMS distribution finder
        self.rms_dist = RMSDistribution()

        # Read parameters from the config.ini file
        _config = configparser.ConfigParser()
        _config.read(pth.PROGRAM_CONFIG)
        self.config = _config[config_str]

        # Get values from config
        self.sigma_threshold = int(self.config['DETECTION_SIGMA_THRESHOLD'])
        self.num_flux_bins = int(self.config['COMPLETENESS_FLUX_BINS'])
        self.min_log_flux = float(self.config['COMPLETENESS_MIN_LOG_FLUX'])
        self.max_log_flux = float(self.config['COMPLETENESS_MAX_LOG_FLUX'])
        self.num_noise_patches = int(self.config['N_NOISE_PATCHES'])
                
        if not override_data:
            # Extract all the relevant arrays from the specified dataset
            self.logger.info(f"Extracting data arrays for dataset")
            config_data_arrays = ImageDataArrays(self.config)
            self.data = config_data_arrays.__getattribute__(self.which_dataset + "_data")
        else:
            self.data = Data()

    # ---------- FITTING FUNCTION ----------
    def fit_function(self,
                     bin_centers : np.ndarray[float, np.dtype[np.float64]],
                     completeness : np.ndarray[float, np.dtype[np.float64]],
                     yerr : np.ndarray[float, np.dtype[np.float64]],
                     function: Callable = sigmoid,
                     initial_guess : list[float] | np.ndarray[float, np.dtype[np.float64]] | None = None,
                     output_file : str | Path | None = None,
                     show_progress : bool = True,
                     **kwargs) -> tuple[np.ndarray[float, np.dtype[np.float64]], np.ndarray[float, np.dtype[np.float64]]]:
        """
        Fit a function to the completeness curve.

        :param bin_centers: The centers of the flux bins used for calculating completeness.
        :param completeness: The completeness values calculated for each flux bin.
        :param yerr: The errors on the y-axis of the completeness points.
        :param function: The function to fit to the completeness curve. Defaults to sigmoid.
        :param initial_guess: Initial guess for the parameters of the function to be fitted. Defaults to [0.5, 7.0, 1.0, 0.0] for sigmoid.
        :param output_file: Where to save the results. Defaults to None.
        :param show_progress: Whether to show progress bars. Defaults to True.
        :returns: The fitted parameters for the given function.
        """
        # `bin_centers` are expected to already be in log10(flux) space.
        # Provide a sensible default initial guess based on the function signature.
        if initial_guess is None:
            try:
                sig = inspect.signature(function)
                params = list(sig.parameters.values())[1:]  # drop x
                param_names = [p.name for p in params]
                n_params = max(len(params), 0)
            except Exception:
                param_names = []
                n_params = 0

            # Guess the 50% point from the data if possible.
            if completeness.size > 0:
                x0_guess = float(bin_centers[int(np.argmin(np.abs(completeness - 0.5)))])
            else:
                x0_guess = float(np.median(bin_centers))

            span = float(np.ptp(bin_centers)) if bin_centers.size > 1 else 1.0
            span = span if span > 0 else 1.0
            k_guess = 5.0 / span
            width_guess = span / 5.0

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
            if show_progress:
                self.logger.info(f"Fitting {function.__name__} function to completeness curve...")

            if initial_guess is None:
                popt, pcov = curve_fit(function, bin_centers, completeness, sigma=yerr, maxfev=10000, **kwargs)
            else:
                popt, pcov = curve_fit(function, bin_centers, completeness, p0=initial_guess, sigma=yerr, maxfev=10000, **kwargs)

            # Save fitted parameters to a file for use in RLF
            if output_file:
                np.savetxt(output_file, popt, pcov, header="Fitted parameters and covariance for completeness function fit", comments='')

            return popt, pcov

        except Exception as e:
            self.logger.error(f"Error: {function.__name__} fit failed: {e}")
            return None, None

    def plot_completeness(self,
                          bin_centers: np.ndarray[float, np.dtype[np.float64]],
                          completeness: np.ndarray[float, np.dtype[np.float64]],
                          yerr : np.ndarray[float, np.dtype[np.float64]],
                          function: Callable = sigmoid,
                          popt : list[float] | np.ndarray | None = None,
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
        assert popt is not None, "You need a fitted completeness function to plot."
        if save_name is None:
            save_name = f"completeness_curve.png"
    
        # `bin_centers` are in log10(flux) throughout this module.
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
                            model_fluxes: np.ndarray,
                            show_progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        For a given set of input images and the model fluxes of those images from PyBDSF, creates mock images by adding
        noise patches to the input images, and checks if the mock sources are detectable based on a peak-flux threshold.

        :param images: The input images to which we will add noise patches
        :param model_fluxes: The fluxes of the sources in the input images
        :param show_progress: Whether to show progress bars for the different stages of the completeness estimation. Defaults to True.
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

        for i in tqdm(range(images.shape[0]), desc='Creating mock images and running detection logic', disable=not show_progress):
            # Use start/end indices so each image occupies a contiguous block of the arrays.
            start = i * self.num_noise_patches
            end = start + self.num_noise_patches

            # Randomly draw a RMS from the distribution of values present in the Hardcastle catalogue sources
            rms = self.rms_dist.sample()

            # Create and apply noise patches for every input image
            mock_fluxes[start:end] = np.full((self.num_noise_patches,), model_fluxes[i], dtype=float)
            noise_patches = self.create_noise_LOFAR(_filter_kernel_2d, rms=rms, shape=(self.num_noise_patches, 80, 80))

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

        # save mock fluxes to a file for later use        if show_progress:
        self.logger.info(f"Saving mock fluxes and detectability to file...")
        with open("mock_fluxes_detectability.txt", "w") as f:
            f.write("Mock_Flux(mJy/beam)\tDetectable\n")
            for flux, detect in zip(mock_fluxes, detectable):
                f.write(f"{flux}\t{detect}\n")

        return mock_fluxes, detectable

    def compute_completeness_per_bin(self,
                             int_flux_bins : np.ndarray,
                             mock_sources : pd.DataFrame,
                             show_progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the completeness by calculating the fraction of detectable sources in every integrated flux bin.
        Also produces y-axis errors for these values based on the Poisson 1-sigma confidence itnervals.

        Args:
            int_flux_bins (np.ndarray): The integrated flux bins.
            mock_sources (pd.DataFrame): A dataframe containing the mock fluxes and whether they were detected.
            show_progress (bool, optional): Whether to show progress bars for the different stages of the completeness estimation. Defaults to True.

        Returns:
            tuple(np.ndarray, nd.ndarray) : The completeness values and calculated y errors for each integrated flux bin.
        """
        # Count detected sources in each bin and calculate completeness
        n_bins = len(int_flux_bins) - 1
        completeness = np.zeros(n_bins, dtype=float)  # to store completeness per bin
        total_counts = np.zeros(n_bins, dtype=int)  # optional: for diagnostics

        # For all bins
        for i in tqdm(range(n_bins), desc='Calculating completeness per flux bin', disable=not show_progress):
            # Select sources in this flux bin
            in_bin = (mock_sources['mock_flux'] >= int_flux_bins[i]) & (mock_sources['mock_flux'] < int_flux_bins[i + 1])

            self.logger.info(f"Flux bin {int_flux_bins[i]:.3f} - {int_flux_bins[i + 1]:.3f} mJy/beam: {np.sum(in_bin)} sources")

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
                                                            interval='frequentist-confidence', sigma=1.0)
        conf_interval /= total_counts
        conf_interval[:, zero_counts] = 0
        yerr = np.array(conf_interval[1] - conf_interval[0])
        
        return completeness, yerr

    def estimate_completeness(self,
                              function: Callable = sigmoid,
                              initial_guess : list[float] | np.ndarray[float, np.dtype[np.float64]] | None = None,
                              comp_output_file : str | Path | None = None,
                              func_output_file : str | Path | None = None,
                              plot_completeness : bool = True,
                              figure_save_name : str | None = None,
                              show_progress : bool = True
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate a completeness curve for a specified dataset.

        It does this by creating mock images, which are the original sources with added noise patches that are convolved
        with a kernel to simulate the beam-correlated noise in LOFAR images. These mock sources are then checked for
        detectability based on a peak flux threshold (e.g., 5 sigma). The completeness is calculated as the fraction of
        detectable sources in bins of flux, and confidence intervals are calculated using Poisson statistics.
        
        :param function: The function to fit to the completeness curve. Defaults to sigmoid.
        :param initial_guess: Initial guess for the parameters of the function to be fitted. Defaults to [0.5, 7.0, 1.0, 0.0] for sigmoid.
        :param comp_output_file: The file in which to save completeness results to. Defaults to None.
        :param func_output_file: The file in which to save fitted parameters to. Defaults to None.
        :param plot_completeness: Whether the fitted completeness should be plotted. Defaults to True.
        :param figure_save_name: The name of the file to save the completeness plot to. Defaults to None.
        :param show_progress: Whether to show progress bars for the different stages of the completeness estimation. Defaults to True.
        :return: The bin centers in log space, estimated completeness for those bins, y error on the completeness,
        and the fitted parameters for the fitting function.
        """
        # Get the mock fluxes and whether they are detectable for all the images in the dataset
        mock_fluxes, detectable = self.detect_mock_sources(self.data.model_images, self.data.model_fluxes,  show_progress)

        # Combine these into a dataframe for easier analysis
        mock_sources = pd.DataFrame()
        mock_sources['mock_flux'] = mock_fluxes
        mock_sources['detectable'] = detectable

        # Calculate completeness per bin and y errors, and save to a file if desired
        int_flux_bins = np.logspace(self.min_log_flux, self.max_log_flux, num=self.num_flux_bins)  # in mJy
        bin_centers = 0.5 * (int_flux_bins[1:] + int_flux_bins[:-1])
        completeness, yerr = self.compute_completeness_per_bin(int_flux_bins, mock_sources, show_progress)

        # Store in a file for later use
        if comp_output_file is not None:
            if show_progress:
                self.logger.info(f"Saving binned completeness estimates to file...")
            with open(comp_output_file, "w") as f:
                f.write("Flux_bin_center(mJy/beam)\tCompleteness\tError\n")
                for center, comp, err in zip(bin_centers, completeness, yerr):
                    f.write(f"{center}\t{comp}\t{err}\n")

        # Fit a function to the completeness curve and plot
        log_bin_centers = np.log10(bin_centers)
        fitted_params, pcov = self.fit_function(log_bin_centers, completeness, function, initial_guess, func_output_file, show_progress)

        if plot_completeness:
            self.plot_completeness(log_bin_centers, completeness, yerr, function, fitted_params, figure_save_name)
        
        return log_bin_centers, completeness, yerr, fitted_params, pcov


class Data(object):
    pass

class SizeBinnedCompleteness(CompletenessEstimator):
    def __init__(self,
                 config_str : str,
                 which_dataset : str | None = "GENERATED_SUBDIR",
                 override_data : bool = False,
                 paths_to_use : list[Path] | None = None,
                 output_file : str | Path | None = None):
        """
        A class to generate completeness curves binned by angular size, to investigate how completeness varies with source size.

        Args:
            config_str (str): The specific configuration in the config file to use.
            which_dataset (str, optional): Which of the two subdir to use in the analysis. Defaults to "GENERATED_SUBDIR"
            output_file (str | Path | None, optional): The file to save the output to. Defaults to None.
        """
        super().__init__(config_str, which_dataset, override_data)
        self.logger = utils.logging.get_logger("SizeBinnedCompleteness", logging.DEBUG)
                
        # Add functionality to extract angular sizes and model images/fluxes if not using ImageDataArrays as source of data
        if override_data:
            assert paths_to_use is not None, "You must provide a list of paths to use if override_data is True"
            assert len(paths_to_use) == 3, "You must provide exactly 3 paths to use if override_data is True: [ang_size_path, model_image_path, model_flux_path]"
            assert output_file is not None, "You must provide an output file if override_data is True"
            
            rfa = RecursiveFileAnalyzer(paths_to_use[0])
            
            self.data = Data()
            self.orig_data = Data()
            
            # Get sizes
            self.logger.info(f"Getting angular sizes...")
            ang_size_finder = AngularSizeFinder(paths_to_use[0])
            indices, sizes = ang_size_finder.run(output_file=output_file)
            self.sizes = sizes
            
            # Get model images 
            def read_model_images(path: Path):
                return fits.getdata(path, 0)
            
            self.logger.info(f"Getting model images...")
            model_images, mi_indices = rfa.run_pipeline(read_model_images,
                                            return_nums=True,
                                            root_dir=paths_to_use[1],
                                            mode="file",
                                            pattern=r'.*?\D+(\d+)\.fits$',
                                            show_progress=False)
            self.model_images = np.array(model_images)
            self.model_images *= 1e3 # convert from Jy/beam to mJy/beam
            
            # Get model fluxes
            self.logger.info(f"Getting model fluxes...")
            model_fluxes, mf_indices = rfa.run_pipeline(get_model_flux,
                                                return_nums=True,
                                                pattern=r'.*?\D+(\d+)\.fits.pybdsf.log$',
                                                root_dir=paths_to_use[2],
                                                mode="file",
                                                show_progress=False)
            self.model_fluxes = np.array(model_fluxes)
            self.model_fluxes *= 1e3  # convert from Jy/beam to mJy/beam
            
            # Filter the sizes, model images, and model fluxes to only include those with matching indices across all three datasets
            common_indices = np.intersect1d(indices, mi_indices)
            common_indices = np.intersect1d(common_indices, mf_indices)
            print(f"Number of sources with matching indices across all datasets: {len(common_indices)}")
            self.sizes = self.sizes[common_indices]
            self.model_images = self.model_images[common_indices]
            self.model_fluxes = self.model_fluxes[common_indices]

            # Plot a histogram of model fluxes to check they look reasonable
            # plt.figure()
            # plt.hist(self.model_fluxes, bins=50, log=True)
            # plt.xlabel("Model Flux (mJy/beam)")
            # plt.xscale('log')
            # plt.ylabel("Number of sources")
            # plt.title("Distribution of Model Fluxes")
            # plt.savefig(dpi=1000, fname="model_flux_distribution.png")
            # plt.show()
        
        # Otherwise, pull these static values from ImageDataArrays as normal
        else:
            self.sizes = self.data.las_values
            self.model_images = self.data.model_images
            self.model_fluxes = self.data.model_fluxes
        
        self.max_size = 120 # arcseconds
        self.max_bins = 12
        # size_bins_edges = np.linspace(0, self.max_size, num=self.max_bins+1, endpoint=True)
        
        # hardcoded bins for snr5 transforms, to try and keep similar number of sources in each bin
        self.size_bins_edges = np.array([0, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 120])
        self.size_bins_edges = np.array([0, 10, 15, 20, 25, 30, 40, 50, 70, 120])
        self.max_bins = len(self.size_bins_edges) - 1
        
    
    def estimate_size_binned_completeness(self, show_progress: bool = True) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Estimate completeness curves binned by angular size.

        Returns:
            list[tuple]: A list of tuples containing the log bin centers, completeness values, y errors, and fitted parameters for each size bin.
        """
        # Store results
        completeness_results = []

        for i in range(len(self.size_bins_edges) - 1):
            # Select sources in this size bin
            size_min = self.size_bins_edges[i]
            size_max = self.size_bins_edges[i + 1]
            in_bin = (self.sizes >= size_min) & (self.sizes < size_max)
            self.logger.info(f"Size bin {size_min:.1f} - {size_max:.1f} arcseconds: {np.sum(in_bin)} sources")
            images_in_bin = self.model_images[in_bin]
            fluxes_in_bin = self.model_fluxes[in_bin]
            
            print(f"Model fluxes in this size bin: {fluxes_in_bin}")
            print(f"Max model flux in this size bin: {np.max(fluxes_in_bin)} mJy/beam")
            
            # Dynamically set the model images and fluxes in self.data to be the sources in this size bin, so that we can reuse the same completeness estimation code
            setattr(self.data, 'model_images', images_in_bin)
            setattr(self.data, 'model_fluxes', fluxes_in_bin)
            
            log_bin_centers, completeness, yerr, fitted_params = self.estimate_completeness(
                comp_output_file=f"completeness_size_bin_{i}.txt",
                func_output_file=f"completeness_fit_params_size_bin_{i}.txt",
                plot_completeness=False,
                show_progress=show_progress
            )
            completeness_results.append((log_bin_centers, completeness, yerr, fitted_params))

        return completeness_results
    
    def plot_size_binned_completeness(self, completeness_results: list[tuple]):
        """
        Plot the completeness curves for each size bin on the same plot, with fitted curves.

        Args:
            completeness_results (list[tuple]): A list of tuples containing the log bin centers, completeness values, y errors, and fitted parameters for each size bin.
        """
        # Build a 256-colour palette (HSV) with relatively low brightness so
        # colours are darker, then sample `self.max_bins` well-separated colours
        # from that palette regardless of how many bins are requested.
        n_palette = 256
        h = np.linspace(0, 1, n_palette, endpoint=False)
        s = np.full(n_palette, 0.85)
        v = np.full(n_palette, 0.75)
        hsv = np.column_stack((h, s, v))
        palette = mpl.colors.hsv_to_rgb(hsv)
        indices = np.round(np.linspace(0, n_palette - 1, self.max_bins)).astype(int)
        mcolors = palette[indices]

        plt.figure(figsize=(10, 6))
        # interval = self.max_size / self.max_bins
        for i, (log_bin_centers, completeness, yerr, fitted_params) in enumerate(completeness_results):
            # size_min = i * interval
            # size_max = (i + 1) * interval
            size_min = self.size_bins_edges[i]
            size_max = self.size_bins_edges[i + 1]
            label = f"{size_min:.1f} - {size_max:.1f} arcsecs"
            
            # Plot the measured completeness with error bars
            plt.errorbar(10 ** log_bin_centers, completeness, yerr=yerr, fmt='.', color=mcolors[i])
            
            # Plot the fitted curve for this size bin
            smooth_flux = np.logspace(self.min_log_flux, self.max_log_flux, 100)
            smooth_completeness = sigmoid(np.log10(smooth_flux), *fitted_params)
            plt.plot(smooth_flux, smooth_completeness, label=f"{label}", alpha=0.5, color=mcolors[i])
        
        plt.xscale('log')
        plt.ylim(-0.01, 1.05)
        plt.xlim(10**-1, 10**2)
        plt.xlabel("Integrated Flux Density (mJy/beam)")
        plt.ylabel("Completeness")
        plt.title("Completeness Curves Binned by Angular Size")
        plt.legend()
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.savefig(dpi=1000, fname="size_binned_completeness.png")
        plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", help=f"Which config to to use for Dataset/Generated subdirs, as defined in {pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()
       
    # completeness_estim = SizeBinnedCompleteness(args.config)
    do_size_binned_completeness = True
    if do_size_binned_completeness:
        root = pth.STORAGE_PARENT / "src/completeness/"
        folder_name = "snr5_exclusive_50k"
        completeness_estim = SizeBinnedCompleteness(folder_name, override_data=True,
            paths_to_use=[root / (folder_name + "_catalogs"),
                        root / (folder_name + "_images/gaus_model"),
                        root / (folder_name + "_logs")],
            output_file=f"estimated_angular_sizes_{folder_name}.csv"
        )
        completeness_results = completeness_estim.estimate_size_binned_completeness(show_progress=False)
        completeness_estim.plot_size_binned_completeness(completeness_results)
    else:
        completeness_estim = CompletenessEstimator(args.config)
        completeness_estim.estimate_completeness()
