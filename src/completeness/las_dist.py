import numpy as np
import matplotlib.pyplot as plt

import logging
import utils.logging
import hardcastle_catalogue as hdc
import configparser
import utils.paths as pths

class LASDistribution:
    """
    This is a class that models the distribution of largest angular size (LAS) values of resolved sources from the Hardcastle catalogue. It
    fits a histogram to the LAS values and creates a random variable distribution that can be used to generate new LAS
    values that follow the same distribution as the original data.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("LASDistribution", logging.DEBUG)

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(pths.PROGRAM_CONFIG)

        # we are using sources generated in a loguniform way
        de_config = config['DEFAULT']

        # Get values from config
        #todo: likely not needed to have this be a separate thing, this is just for visualisation purposes
        # self.percentage_threshold = float(de_config['RMS_PERCENTAGE_THRESHOLD'])
        self.percentage_threshold = 0.98

        # Get the LAS values from the Hardcastle catalogue and filter out NaN values
        self.las_values = self.get_las_values()

    def get_las_values(self) -> np.ndarray:
        # Get all LAS values from the catalogue
        catalogue = hdc.HardcastleCatalogue()

        self.logger.info("Extracting LAS values from the Hardcastle catalogue...")
        las_values = catalogue.get_values(hdc.Source.AngSize)

        #  Remove NaN values from the LAS data
        self.logger.info("Filtering any NaNs...")
        las_values = np.array(las_values)
        las_values = las_values[~np.isnan(las_values)]
        
        # Remove all 0 values, I'm unsure of how resolved sources can nonetheless be recorded as having 0 LAS but
        las_values = las_values[las_values > 0]

        # Choose a threshold to contain a certain percentage of the data, e.g., 99.7% for a normal distribution (3 sigma)
        # This filters from the highest first; that is, removes the largest LAS values until the desired percentage of the data is retained
        upper_limit = np.percentile(las_values, self.percentage_threshold * 100)

        # Very few values above this range, but they affect our histogram fitting, so filtering
        self.logger.info(f"Filtering LAS values to retain {self.percentage_threshold * 100}% of the data...")
        las_values = las_values[las_values < upper_limit]

        return las_values

    def plot(self):
        """
        Plots the histogram of LAS values to visualize the distribution.
        """
        self.logger.info("Plotting the histogram of LAS values...")
        plt.hist(self.las_values, bins=100, density=False)
        plt.xlabel('LAS Value (arcsec)')
        plt.ylabel('Density')
        plt.title('Distribution of LAS Values in the Hardcastle Catalogue')
        plt.grid(True)
        plt.savefig('las_histogram.png')
        plt.show()

    def get_statistics(self) -> dict:
        """
        Computes and returns basic statistics of the LAS values.

        :return: A dictionary containing the mean, median, standard deviation, and percentiles of the LAS values.
        """
        stats = {
            'mean': np.mean(self.las_values),
            'median': np.median(self.las_values),
            'std_dev': np.std(self.las_values),
            'percentiles': {
                '90th': np.percentile(self.las_values, 90),
                '95th': np.percentile(self.las_values, 95),
            },
            'max': np.max(self.las_values),
            'min': np.min(self.las_values)
        }

        self.logger.info("LAS Statistics:")
        self.logger.info(f"Mean: {stats['mean']:.4f} arcsec")
        self.logger.info(f"Median: {stats['median']:.4f} arcsec")
        self.logger.info(f"Standard Deviation: {stats['std_dev']:.4f} arcsec")
        self.logger.info(f"90th Percentile: {stats['percentiles']['90th']:.4f} arcsec")
        self.logger.info(f"95th Percentile: {stats['percentiles']['95th']:.4f} arcsec")
        self.logger.info(f"Max: {stats['max']:.4f} arcsec")
        self.logger.info(f"Min: {stats['min']:.4f} arcsec")

        return stats
    
    def get_counts_in_bins(self, bins: np.ndarray) -> np.ndarray:
        """
        Computes the counts of LAS values in specified bins.

        :param bins: An array of bin edges to compute the histogram.
        :return: An array of counts corresponding to each bin.
        """
        self.logger.info("Computing counts of LAS values in specified bins...")
        counts, _ = np.histogram(self.las_values, bins=bins)
        return counts

if __name__ == "__main__":
    LAS = LASDistribution()
    LAS.plot()
    LAS.get_statistics()
    
    # Consider binning
    bins = np.linspace(0, 150, 15)
    counts = LAS.get_counts_in_bins(bins)
    print("Bin edges:", bins)
    print("Counts in bins:", counts)
