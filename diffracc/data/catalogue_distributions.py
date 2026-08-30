"""
A module to model the distributions of various properties of sources in the Hardcastle catalogue, including RMS, LAS,
peak flux, peak pixel values, and redshift. Each distribution is represented by a class that extracts the relevant
values from the catalogue, fits a histogram to these values, and creates a random variable distribution that can be used
to generate new values following the same distribution as the original data. The module also provides methods to plot
the distributions and compute basic statistics of the values. 

Only RMS and LAS distributions are used elsewhere in the program, but the other distributions are provided for easy
visualisation.
"""
import configparser
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_histogram

from ..utils import paths
from ..utils.data_utils import Source
from ..utils.logger import LoggingLevels, get_logger
from .hardcastle_catalogue import HardcastleCatalogue


class CatalogueDistribution(ABC):
    """
    An abstract base class representing the distribution of a property of sources in a catalogue.

    Subclasses only need to specify where their raw values come from, by implementing `_extract_raw_values`. The shared
    cleaning pipeline (dropping NaN and non-positive values, then applying a percentile threshold) is provided here by
    `get_values` and is not intended to be overridden. For the common case of extracting a single catalogue column, the
    `_CatalogueColumnDistribution` intermediate class implements `_extract_raw_values` in terms of a `SOURCE_COLUMN`
    class attribute, so those subclasses need only set that attribute.
    """
    def __init__(self,
                 logger_name: str = "CatalogueDistribution",
                 use_catalogue: bool = True,
                 data_path: Path | str | None = None,
                 bin_num: int = 100,
                 log_scale: bool = False,
                 value_name: str = "Value",
                 units: str = "Units",
                 resolved_only: bool = True):
        """
        Initialises the class by extracting values from the Hardcastle catalogue, fitting a histogram to these values,
        and creating a random variable distribution.

        Parameters
        ----------
        logger_name : str
            The name of the logger to use for logging messages. Defaults to "CatalogueDistribution".
        use_catalogue : bool
            Whether to use the Hardcastle catalogue for data extraction. If False, data will be loaded from a provided
            path. Defaults to True.
        data_path : Path | str | None
            The path to an alternative data file, only accessed if `use_catalogue` is False. Must be specified if
            `use_catalogue` is False. Defaults to None.
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
            self.logger.info("Using logarithmic bins for histogram fitting...")
            self.bins = np.logspace(np.log10(self.values.min()), np.log10(self.values.max()), self.bin_num)
        else:
            self.bins = self.bin_num

        self.hist_tr = np.histogram(self.values, bins=self.bins)
        self.model_dist = rv_histogram(self.hist_tr, density=False)

    @abstractmethod
    def _extract_raw_values(self) -> np.ndarray:
        """
        Extracts the raw, unfiltered values for this distribution.

        It should return the values as read from their source, without any cleaning; NaN removal, non-positive removal,
        and percentile thresholding are applied centrally by `get_values`.

        Returns
        -------
        np.ndarray
            An array of raw values for this distribution.
        """

    def get_values(self) -> np.ndarray:
        """
        Extracts the raw values via `_extract_raw_values` and applies the shared cleaning pipeline: NaN values and
        non-positive values are removed, then values above the `percentage_threshold` percentile are discarded to focus
        on the main distribution.

        Returns
        -------
        np.ndarray
            An array of cleaned values filtered to retain the specified percentage of the data.
        """
        values = np.asarray(self._extract_raw_values(), dtype=float)

        # Remove NaN values and non-positive values, which are physically meaningless for the modelled properties
        values = values[~np.isnan(values)]
        values = values[values > 0]

        # Filter out extreme values to focus on the main distribution, based on the specified percentage threshold
        upper_limit = np.percentile(values, self.percentage_threshold * 100)
        self.logger.info(f"Filtering {self.value_name} values to retain {self.percentage_threshold * 100}% of the "
                         "data...")
        return values[values < upper_limit]

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
        return self.model_dist.rvs(size=size)  # type: ignore

    def plot(self,
             value_name: str | None = None,
             units: str | None = None,
             file_name: str | None = None) -> None:
        """
        Plots the histogram of values to visualize the distribution.
        
        Parameters
        ----------
        value_name : str | None
            The name of the value being plotted (e.g., "RMS", "Flux"). If None, the default value_name is used.
        units : str | None
            The units of the value being plotted (e.g., "mJy/beam", "Jy"). If None, the default units are used.
        file_name : str | None
            The name of the file to save the plot. If None, the plot will be saved as 'values_histogram.png'.
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
        if file_name is not None:
            plt.savefig(file_name)
        else:
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
            'count': len(self.values),
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


class _CatalogueColumnDistribution(CatalogueDistribution):
    """
    An intermediate base class for distributions whose raw values are a single column of the catalogue.

    Subclasses set the `SOURCE_COLUMN` class attribute to the relevant `Source` column; extraction (including the
    `use_catalogue` vs preprocessed-data branch) is then handled here, so they need not implement `_extract_raw_values`
    themselves.
    """
    SOURCE_COLUMN: Source

    def _extract_raw_values(self) -> np.ndarray:
        """
        Extracts the raw values for `SOURCE_COLUMN`, either from the Hardcastle catalogue or the preprocessed data
        depending on `use_catalogue`.

        Returns
        -------
        np.ndarray
            An array of raw values for the configured source column.
        """
        if self.use_catalogue:
            self.logger.info(f"Extracting {self.value_name} values from the Hardcastle catalogue...")
            return self.catalogue.get_value_column(self.SOURCE_COLUMN)

        self.logger.info(f"Extracting {self.value_name} values from the preprocessed data...")
        return self.catalogue[self.SOURCE_COLUMN]


class RMSDistribution(_CatalogueColumnDistribution):
    """
    A class to model the distribution of RMS values of sources from the Hardcastle catalogue.
    """
    SOURCE_COLUMN = Source.RMS

    def __init__(self,
                 bin_num: int = 100,
                 resolved_only: bool = True,
                 use_catalogue: bool = True,
                 data_path: Path | str | None = None):
        """
        Initializes the RMSDistribution by extracting RMS values from the Hardcastle catalogue, fitting a histogram to
        these values, and creating a random variable distribution.
        
        Note: percentage threshold is not an argument of the constructor as it's currently configured in the config.ini
        file, due to this class being used in the completeness pipeline.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to use the catalogue as the source of data. Defaults to True.
        data_path : Path | str | None, optional
            The path to an alternative data file. Must not be None if `use_catalogue` is False. Defaults to None.
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


class LASDistribution(_CatalogueColumnDistribution):
    """
    A class to model the distribution of Largest Angular Size (LAS) values of sources from the Hardcastle catalogue.
    """
    SOURCE_COLUMN = Source.AngSize

    def __init__(self,
                 bin_num: int = 100,
                 percentage_threshold: float = 0.98,
                 resolved_only: bool = True,
                 use_catalogue: bool = True,
                 data_path: Path | str | None = None):
        """
        Initialises the LASDistribution by extracting LAS values from the Hardcastle catalogue, filtering out NaN and
        zero values, applying a threshold to retain a certain percentage of the data, fitting a histogram to these
        values, and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        percentage_threshold : float, optional
            The percentage of data to retain after filtering. Defaults to 0.98.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to load the catalogue data. Defaults to True.
        data_path : Path | str | None, optional
            The path to the preprocessed data file containing the cutout images. Must not be None if `use_catalogue` is
            False. Defaults to None.
        """
        self.percentage_threshold = percentage_threshold

        super().__init__(
            logger_name="LASDistribution",
            bin_num=bin_num,
            value_name="LAS",
            units="arcseconds",
            resolved_only=resolved_only,
            use_catalogue=use_catalogue,
            data_path=data_path
        )


class PeakFluxDistribution(_CatalogueColumnDistribution):
    """
    A class to model the distribution of peak flux values of sources from the Hardcastle catalogue.
    """
    SOURCE_COLUMN = Source.PeakFlux

    def __init__(self,
                 bin_num: int = 100,
                 percentage_threshold: float = 0.98,
                 resolved_only: bool = True,
                 use_catalogue: bool = True,
                 data_path: Path | str | None = None):
        """
        Initialises the PeakFluxDistribution by extracting peak flux values from the Hardcastle catalogue, filtering out
        NaN values, applying a threshold to retain a certain percentage of the data, fitting a histogram to these
        values, and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        percentage_threshold : float, optional
            The percentage of data to retain after filtering. Defaults to 0.98.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to use the catalogue data. Defaults to True.
        data_path : Path | str | None, optional
            The path to the preprocessed data file. Must not be None if `use_catalogue` is False. Defaults to None.
        """
        self.percentage_threshold = percentage_threshold

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
            log_scale=True
        )


class PeakPixDistribution(CatalogueDistribution):
    """
    A class to model the distribution of peak pixel values of cutouts of resolved sources from the Hardcastle catalogue.
    """
    def __init__(self,
                 bin_num: int = 100,
                 path: Path | str = paths.DATASET_PATH_H5,
                 percentage_threshold: float = 0.80):
        """
        Initialises the PeakPixDistribution by extracting peak pixel values from cutout images of the Hardcastle
        catalogue, applying a threshold to retain a certain percentage of the data, fitting a histogram to these values,
        and creating a random variable distribution.

        Parameters
        ----------
        bin_num : int, optional
            The number of bins for the histogram. Defaults to 100.
        path : Path | str, optional
            The path to the preprocessed data file containing the cutout images. By default `paths.DATASET_PATH_H5`.
        percentage_threshold : float, optional
            The percentage of data to retain after filtering. Defaults to 0.80.
        """
        self.path = path
        self.percentage_threshold = percentage_threshold

        # todo: consider having fits to a log distribution instead, as flux values are lognormally distributed
        # todo: which results in the rv histogram being very poor
        super().__init__(
            logger_name="PeakPixDistribution",
            bin_num=bin_num,
            value_name="Peak Pixel",
            units="mJy/beam",
            use_catalogue=False,
            data_path=self.path
        )

    def _extract_raw_values(self) -> np.ndarray:
        """
        Extracts the peak pixel values from the cutout images in the preprocessed data.

        Unlike the other distributions, the raw values are not a catalogue column but the per-image maximum pixel value,
        converted from Jy/beam to mJy/beam. NaN removal, non-positive removal, and percentile thresholding are then
        applied by the inherited `get_values` pipeline.

        Returns
        -------
        np.ndarray
            An array of raw peak pixel values, one per cutout image.
        """
        self.logger.info("Extracting peak pixel values from cutout images using preprocessed data...")
        with h5py.File(self.path, "r") as f:
            images = np.array(f["images"][:])
            return np.array(images.max(axis=(1, 2)) * 1000)  # Convert from Jy/beam to mJy/beam


class RedshiftDistribution(_CatalogueColumnDistribution):
    """
    A class to model the distribution of redshift values of sources from the Hardcastle catalogue.
    """
    SOURCE_COLUMN = Source.Redshift

    def __init__(self,
                 bin_num: int = 100,
                 percentage_threshold: float = 0.98,
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
        percentage_threshold : float, optional
            The percentage of data to retain after filtering. Defaults to 0.98.
        resolved_only : bool, optional
            Whether to consider only resolved sources. Defaults to True.
        use_catalogue : bool, optional
            Whether to use the catalogue data. Defaults to True.
        data_path : Path | None, optional
            The path to the preprocessed data file. Defaults to None.
        """
        self.percentage_threshold = percentage_threshold

        super().__init__(
            logger_name="RedshiftDistribution",
            bin_num=bin_num,
            value_name="Redshift",
            units="z",
            resolved_only=resolved_only,
            use_catalogue=use_catalogue,
            data_path=data_path
        )


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
