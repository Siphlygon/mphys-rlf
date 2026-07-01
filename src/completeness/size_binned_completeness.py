import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import utils.logging
from completeness.completeness_estimator import CompletenessEstimator
from utils.recursive_file_analyzer import RecursiveFileAnalyzer
from completeness.angular_size_finder import AngularSizeFinder
from analysis.log_analyzer import get_model_flux
from utils.functions import sigmoid


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
            model_images, mi_indices = rfa.run_pipeline(function=read_model_images,
                                            return_nums=True,
                                            root_dir=paths_to_use[1],
                                            mode="file",
                                            pattern=r'.*?\D+(\d+)\.fits$',
                                            show_progress=False)
            self.model_images = np.array(model_images)
            self.model_images *= 1e3 # convert from Jy/beam to mJy/beam

            # Get model fluxes
            self.logger.info(f"Getting model fluxes...")
            model_fluxes, mf_indices = rfa.run_pipeline(function=get_model_flux,
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

    
    def estimate_size_binned_completeness(self, show_progress: bool = True) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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

            # Dynamically set the model images and fluxes in self.data to be the sources in this size bin, so that we
            # can reuse the same completeness estimation code
            setattr(self.data, 'model_images', images_in_bin)
            setattr(self.data, 'model_fluxes', fluxes_in_bin)

            log_bin_centers, completeness, yerr, fitted_params, pcov = self.estimate_completeness(
                comp_output_file=f"completeness_size_bin_{i}.txt",
                func_output_file=f"completeness_fit_params_size_bin_{i}.txt",
                plot_completeness=False,
                show_progress=show_progress
            )
            completeness_results.append((log_bin_centers, completeness, yerr, fitted_params, pcov))

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
