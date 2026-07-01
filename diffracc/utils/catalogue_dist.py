import configparser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_histogram

from hardcastle_catalogue import HardcastleCatalogue, Source

from . import paths
from .logger import LoggingLevels, get_logger


class CatalogueDistribution:
    """
    A class to represent the distribution of sources in a catalogue, including their sizes, fluxes, and model images.
    """
    def __init__(self,
                 logger_name: str = "CatalogueDistribution",
                 use_catalogue: bool = True,
                 data_path: Path | None = None,
                 bin_num: int = 100,
                 log_scale: bool = False,
                 value_name: str = "Value",
                 units: str = "Units",
                 resolved_only: bool = True):
        """
        Initialises the CatalogueDistribution by extracting values from the Hardcastle catalogue, fitting a histogram to
        these values, and creating a random variable distribution.

        Parameters
        ----------
        logger_name : str
            The name of the logger to use for logging messages. Defaults to "CatalogueDistribution".
        use_catalogue : bool
            Whether to use the Hardcastle catalogue for data extraction. If False, data will be loaded from a provided
            path. Defaults to True.
        data_path : Path | None
            The path to the preprocessed data file containing the cutout images. If None, a default path is used.
            Defaults to None.
        bin_num : int
            The number of bins to use for histogram fitting. Defaults to 100.
        log_scale : bool
            Whether to use logarithmic scale for the histogram fitting. Defaults to False.
        value_name : str
            The name of the value being analyzed (e.g., "RMS", "Flux"). Defaults to "Value".
        units : str
            The units of the value being analyzed (e.g., "mJy/beam", "Jy"). Defaults to "Units".
        resolved_only : bool
            Whether to consider only resolved sources from the catalogue. Defaults to True.
        """
        self.logger = get_logger(logger_name, LoggingLevels.DEBUG.value)

        self.use_catalogue = use_catalogue
        if not self.use_catalogue:
            assert data_path is not None, "You must provide a data path if use_catalogue is False"
            self.path = data_path

        # Initialise attributes
        self.value_name = value_name
        self.units = units
        self.log_scale = log_scale

        # Get the RMS values from the Hardcastle catalogue and filter out NaN values
        if use_catalogue:
            self.catalogue = HardcastleCatalogue(resolved_only=resolved_only)
        else:
            with h5py.File(self.path, "r") as f:
                self.catalogue = f["cat_info"][:]
        self.values = self.get_values()

        # Fit a histogram to the RMS values and create a random variable distribution
        self.bin_num = bin_num

        if self.log_scale:
            # If using log scale, we need log bins to properly fit the histogram
            self.logger.info("Using logarithmic bins for histogram fitting...")
            self.bins = np.logspace(np.log10(self.values.min()), np.log10(self.values.max()), self.bin_num)
        else:
            self.bins = self.bin_num

        self.hist_tr = np.histogram(self.values, bins=self.bins)

        self.model_dist = rv_histogram(self.hist_tr, density=False)


    def get_values(self) -> np.ndarray:
        """
        Extracts the specified values from the resolved items in the Hardcastle catalogue.
        Note: This method should be overridden by subclasses to specify which values to extract (e.g., RMS, flux, size).

        Returns
        -------
        np.ndarray
            An array of values extracted from the catalogue.
        """
        return np.array([])


    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generates random samples from the fitted distribution.

        Parameters
        ----------
        size : int
            The number of samples to generate. Defaults to 1.
        
        Returns
        -------
        np.ndarray
            An array of random samples drawn from the fitted distribution.
        """
        return self.model_dist.rvs(size=size)


    def plot(self,
             value_name: str | None = None,
             units: str | None = None):
        """
        Plots the histogram of values to visualize the distribution.
        
        Parameters
        ----------
        value_name : str | None
            The name of the value being plotted (e.g., "RMS", "Flux"). If None, the default value_name is used.
        units : str | None
            The units of the value being plotted (e.g., "mJy/beam", "Jy"). If None, the default units are used.
        """
        if value_name is not None:
            self.value_name = value_name
        if units is not None:
            self.units = units

        self.logger.info(f"Plotting the histogram of {self.value_name} values...")
        if self.log_scale:
            plt.xscale('log')
        plt.hist(self.values, bins=self.bins, density=False, log=self.log_scale)
        plt.xlabel(f'{self.value_name} ({self.units})')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {self.value_name} in the Hardcastle Catalogue')
        plt.grid(True)
        plt.savefig('values_histogram.png')
        plt.show()


    def get_statistics(self,
                       value_name: str | None = None,
                       units: str | None = None,
                       show_output: bool = True) -> dict:
        """
        Computes and returns basic statistics of the values.

        Parameters
        ----------
        value_name : str | None
            The name of the value for which statistics are being computed (e.g., 'RMS', 'Flux'). If None, the default
            value_name is used.
        units : str | None
            The units of the value for which statistics are being computed (e.g., 'mJy/beam', 'Jy'). If None, the
            default units are used.
        show_output : bool
            Whether to display the statistics in the logger.

        Returns
        -------
        dict
            A dictionary containing the mean, median, standard deviation, and percentiles of the values.
        """
        if value_name is None:
            value_name = self.value_name
        if units is None:
            units = self.units

        stats = {
            'mean': np.mean(self.values),
            'median': np.median(self.values),
            'std_dev': np.std(self.values),
            'percentiles': {
                '90th': np.percentile(self.values, 90),
                '95th': np.percentile(self.values, 95),
                '99th': np.percentile(self.values, 99)
            }
        }

        if show_output:
            self.logger.info(f"{value_name} Statistics:")
            for stat_name, stat_value in stats.items():
                if stat_name != 'percentiles':
                    self.logger.info(f"{stat_name.capitalize()}: {stat_value:.4f} {units}")
                else:
                    self.logger.info("Percentiles:")
                    for perc_name, perc_value in stat_value.items():
                        self.logger.info(f"  {perc_name}: {perc_value:.4f} {units}")

        return stats



class RMSDistribution(CatalogueDistribution):
    """
    A class to model the distribution of RMS values of resolved sources from the Hardcastle catalogue. It
    fits a histogram to the RMS values and creates a random variable distribution that can be used to generate new RMS
    values that follow the same distribution as the original data.
    """
    def __init__(self,
                 bin_num: int = 100,
                 resolved_only: bool = True,
                 use_catalogue: bool = True,
                 data_path: Path | None = None):
        """
        Initializes the RMSDistribution by extracting RMS values from the Hardcastle catalogue, fitting a histogram to
        these values, and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to load the catalogue data. Defaults to True.
        data_path : Path | None, optional
            The path to the preprocessed data file containing the cutout images. If None, a default path is used.
            Defaults to None.
        """
        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)

        # we are using sources generated in a loguniform way
        de_config = config['DEFAULT']

        # Get values from config
        self.percentage_threshold = float(de_config['RMS_PERCENTAGE_THRESHOLD'])

        super().__init__(
            logger_name="RMSDistribution",
            bin_num=bin_num,
            value_name="RMS",
            units="mJy/beam",
            resolved_only=resolved_only,
            use_catalogue=use_catalogue,
            data_path=data_path
        )


    def get_values(self) -> np.ndarray:
        """
        Gets the RMS values from the Hardcastle catalogue, filters out NaN values, and applies a threshold to retain a
        certain percentage of the data.

        Returns
        -------
        np.ndarray
            An array of RMS values that have been filtered to retain a specified percentage of the data.
        """
        # Get all RMS values from the catalogue
        if self.use_catalogue:
            self.logger.info("Extracting RMS values from the Hardcastle catalogue...")
            rms_values = self.catalogue.get_values(Source.RMS)
        else:
            self.logger.info("Extracting RMS values from the preprocessed data...")
            rms_values = self.catalogue["Isl_rms"]

        #  Remove NaN values from the RMS data
        rms_values = np.array(rms_values)
        rms_values = rms_values[~np.isnan(rms_values)]

        # Filter out extreme RMS values to focus on the main distribution, based on the specified percentage threshold
        upper_limit = np.percentile(rms_values, self.percentage_threshold * 100)
        self.logger.info(f"Filtering RMS values to retain {self.percentage_threshold * 100}% of the data...")
        rms_values = rms_values[rms_values < upper_limit]

        return rms_values



class LASDistribution(CatalogueDistribution):
    """
    A class to model the distribution of Largest Angular Size (LAS) values of resolved sources from the Hardcastle
    catalogue. It fits a histogram to the LAS values and creates a random variable distribution that can be used to
    generate new LAS values that follow the same distribution as the original data.
    """
    def __init__(self,
                 bin_num: int = 100,
                 resolved_only: bool = True,
                 use_catalogue: bool = True,
                 data_path: Path | None = None):
        """
        Initialises the LASDistribution by extracting LAS values from the Hardcastle catalogue, filtering out NaN and
        zero values, applying a threshold to retain a certain percentage of the data, fitting a histogram to these
        values, and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to load the catalogue data. Defaults to True.
        data_path : Path | None, optional
            The path to the preprocessed data file containing the cutout images. If None, a default path is used.
            Defaults to None.
        """
        # # Read parameters from the config.ini file
        # config = configparser.ConfigParser()
        # config.read(paths.PROGRAM_CONFIG)

        # # we are using sources generated in a loguniform way
        # de_config = config['DEFAULT']

        # Get values from config
        #todo: likely not needed to have this be a separate thing, this is just for visualisation purposes
        # self.percentage_threshold = float(de_config['RMS_PERCENTAGE_THRESHOLD'])
        self.percentage_threshold = 0.98

        super().__init__(
            logger_name="LASDistribution",
            bin_num=bin_num,
            value_name="LAS",
            units="arcseconds",
            resolved_only=resolved_only,
            use_catalogue=use_catalogue,
            data_path=data_path)


    def get_values(self) -> np.ndarray:
        """
        Gets the LAS values from the Hardcastle catalogue, filters out NaN values, and applies a threshold to retain a
        certain percentage of the data.

        Returns
        -------
        np.ndarray
            An array of LAS values that have been filtered to retain a specified percentage of the data.
        """
        # Get all LAS values from the catalogue
        if self.use_catalogue:
            self.logger.info("Extracting LAS values from the Hardcastle catalogue...")
            las_values = self.catalogue.get_values(Source.AngSize)
        else:
            self.logger.info("Extracting LAS values from the preprocessed data...")
            las_values = self.catalogue["LAS"]  # Assuming the preprocessed data has an "LAS" field

        #  Remove NaN values from the LAS data
        las_values = np.array(las_values)
        las_values = las_values[~np.isnan(las_values)]

        # Remove all 0 values, I'm unsure of how resolved sources can nonetheless be recorded as having 0 LAS but
        las_values = las_values[las_values > 0]

        # Filter out extreme LAS values to focus on the main distribution, based on the specified percentage threshold
        upper_limit = np.percentile(las_values, self.percentage_threshold * 100)
        self.logger.info(f"Filtering LAS values to retain {self.percentage_threshold * 100}% of the data...")
        las_values = las_values[las_values < upper_limit]

        return las_values



class PeakFluxDistribution(CatalogueDistribution):
    """
    A class to model the distribution of peak flux values of resolved sources from the Hardcastle catalogue. It fits a
    histogram to the peak flux values and creates a random variable distribution that can be used to generate new peak
    flux values that follow the same distribution as the original data.
    """
    def __init__(self,
                 bin_num: int = 100,
                 resolved_only: bool = True,
                 use_catalogue: bool = True,
                 data_path: Path | None = None):
        """
        Initialises the PeakFluxDistribution by extracting peak flux values from the Hardcastle catalogue, filtering out
        NaN values, applying a threshold to retain a certain percentage of the data, fitting a histogram to these
        values, and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to use the catalogue data. Defaults to True.
        data_path : Path | None, optional
            The path to the preprocessed data file. Defaults to None.
        """
        # Get values from config
        # self.percentage_threshold = float(de_config['PEAK_FLUX_PERCENTAGE_THRESHOLD'])
        # self.percentage_threshold = 0.80
        self.percentage_threshold = 1

        # todo: consider having fits to a log distribution instead, as flux values are lognormally distributed
        # todo: which results in the rv histogram being very poor

        super().__init__(
            logger_name="PeakFluxDistribution",
            bin_num=bin_num,
            value_name="Peak Flux",
            units="mJy/beam",
            resolved_only=resolved_only,
            use_catalogue=use_catalogue,
            data_path=data_path,
            log_scale=True)


    def get_values(self) -> np.ndarray:
        """
        Gets the peak flux values from the Hardcastle catalogue, filters out NaN values, and applies a threshold to
        retain a certain percentage of the data.

        Returns
        -------
        np.ndarray
            An array of peak flux values that have been filtered to retain a specified percentage of the data.
        """
        # Get all peak flux values from the catalogue
        if self.use_catalogue:
            self.logger.info("Extracting peak flux values from the Hardcastle catalogue...")
            peak_flux_values = self.catalogue.get_values(Source.PeakFlux)
        else:
            self.logger.info("Extracting peak flux values from the preprocessed data...")
            peak_flux_values = self.catalogue["Peak_flux"]  # Assuming the preprocessed data has a "Peak_flux" field

        #  Remove NaN values from the peak flux data
        peak_flux_values = np.array(peak_flux_values)
        peak_flux_values = peak_flux_values[~np.isnan(peak_flux_values)]

        # Filter out extreme peak flux values to focus on the main distribution, based on the specified percentage
        # threshold
        upper_limit = np.percentile(peak_flux_values, self.percentage_threshold * 100)
        self.logger.info(f"Filtering peak flux values to retain {self.percentage_threshold * 100}% of the data...")
        peak_flux_values = peak_flux_values[peak_flux_values < upper_limit]

        return peak_flux_values



class PeakPixDistribution(CatalogueDistribution):
    """
    A class to model the distribution of peak pixel values of resolved sources from the Hardcastle catalogue. It fits a
    histogram to the peak pixel values and creates a random variable distribution that can be used to generate new peak
    pixel values that follow the same distribution as the original data.
    """
    def __init__(self,
                 bin_num: int = 100,
                 path: Path | None = None):
        """
        Initialises the PeakPixDistribution by extracting peak pixel values from cutout images of the Hardcastle
        catalogue, applying a threshold to retain a certain percentage of the data, fitting a histogram to these values,
        and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        path : Path | None, optional
            The path to the preprocessed data file containing the cutout images. If None, a default path is used.
            Defaults to None.
        """
        if path is None:
            self.path = paths.DATASET_PARENT / "clean_hardcastle_catalogue.h5"
        else:
            self.path = path

        # Get values from config
        # self.percentage_threshold = float(de_config['PEAK_PIXEL_PERCENTAGE_THRESHOLD'])
        self.percentage_threshold = 0.80

        # todo: consider having fits to a log distribution instead, as flux values are lognormally distributed
        # todo: which results in the rv histogram being very poor

        super().__init__(
            logger_name="PeakPixDistribution",
            bin_num=bin_num,
            value_name="Peak Pixel",
            units="mJy/beam",
            use_catalogue=False,
            data_path=self.path)


    def get_values(self, data_path: Path | None = None) -> np.ndarray:
        """
        Gets the peak pixel values from cutout images of the Hardcastle catalogue, and applies a threshold to retain a
        certain percentage of the data.

        Parameters
        ----------
        data_path : Path, optional
            The path to the preprocessed data file containing the cutout images. If None, a default path is used.

        Returns
        -------
        np.ndarray
            An array of peak pixel values that have been filtered to retain a specified percentage of the data.
        """
        self.logger.info("Extracting peak pixel values from cutout images using preprocessed data...")
        if data_path is None:
            data_path = self.path

        with h5py.File(data_path, "r") as f:
            images = np.array(f["images"][:])
            peak_pixel_values = np.array(images.max(axis=(1, 2)) * 1000)  # Convert from Jy/beam to mJy/beam

        # Remove all 0 values, I'm unsure of how resolved sources can nonetheless be recorded as having 0 peak pixel but
        peak_pixel_values = peak_pixel_values[peak_pixel_values > 0]

        # Choose a threshold to contain a percentage of the data, e.g., 99.7% for a normal distribution (3 sigma)
        upper_limit = np.percentile(peak_pixel_values, self.percentage_threshold * 100)

        # Very few values above this range, but they affect our histogram fitting, so filtering
        self.logger.info(f"Filtering peak pixel values to retain {self.percentage_threshold * 100}% of the data...")
        peak_pixel_values = peak_pixel_values[peak_pixel_values < upper_limit]

        return peak_pixel_values



class RedshiftDistribution(CatalogueDistribution):
    """
    A class to model the distribution of redshift values of resolved sources from the Hardcastle catalogue. It fits a
    histogram to the redshift values and creates a random variable distribution that can be used to generate new
    redshift values that follow the same distribution as the original data.
    """
    def __init__(self,
                 bin_num: int = 100,
                 resolved_only: bool = True,
                 use_catalogue: bool = True,
                 data_path: Path | None = None):
        """
        Initialises the RedshiftDistribution by extracting redshift values from the Hardcastle catalogue, filtering out
        NaN values, applying a threshold to retain a certain percentage of the data, fitting a histogram to these
        values, and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to use the catalogue data. Defaults to True.
        data_path : Path | None, optional
            The path to the preprocessed data file. Defaults to None.
        """
        # Get values from config
        # self.percentage_threshold = float(de_config['REDSHIFT_PERCENTAGE_THRESHOLD'])
        self.percentage_threshold = 0.98

        super().__init__(
            logger_name="RedshiftDistribution",
            bin_num=bin_num,
            value_name="Redshift",
            units="z",
            resolved_only=resolved_only,
            use_catalogue=use_catalogue,
            data_path=data_path)


    def get_values(self) -> np.ndarray:
        """
        Gets the redshift values from the Hardcastle catalogue, filters out NaN values, and applies a threshold to
        retain a certain percentage of the data.

        Returns
        -------
        np.ndarray
            An array of redshift values that have been filtered to retain a specified percentage of the data.
        """
        # Get all redshift values from the catalogue
        if self.use_catalogue:
            self.logger.info("Extracting redshift values from the Hardcastle catalogue...")
            redshift_values = self.catalogue.get_values(Source.Redshift)
        else:
            self.logger.info("Extracting redshift values from the preprocessed data...")
            redshift_values = self.catalogue["z_best"]

        #  Remove NaN values from the redshift data
        redshift_values = np.array(redshift_values)
        redshift_values = redshift_values[~np.isnan(redshift_values)]

        # Filter out extreme redshift values to focus on the main distribution, based on the specified threshold
        upper_limit = np.percentile(redshift_values, self.percentage_threshold * 100)
        self.logger.info(f"Filtering redshift values to retain {self.percentage_threshold * 100}% of the data...")
        redshift_values = redshift_values[redshift_values < upper_limit]

        return redshift_values



if __name__ == "__main__":
    rms_dist = RMSDistribution()
    las_dist = LASDistribution()
    peak_flux_dist = PeakFluxDistribution()
    peak_pixel_dist = PeakPixDistribution()
    redshift_dist = RedshiftDistribution()

    rms_dist.plot()
    las_dist.plot()
    peak_flux_dist.plot()
    peak_pixel_dist.plot()
    redshift_dist.plot()

    rms_stats = rms_dist.get_statistics(show_output=True)
    las_stats = las_dist.get_statistics(show_output=True)
    peak_flux_stats = peak_flux_dist.get_statistics(show_output=True)
    peak_pixel_stats = peak_pixel_dist.get_statistics(show_output=True)
    redshift_stats = redshift_dist.get_statistics(show_output=True)
