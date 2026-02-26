import image_downloading.hardcastle_catalogue as hdc
import numpy as np
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt

class RMSDistribution:
    """
    This is a class that models the distribution of RMS values of resolved sources from the Hardcastle catalogue. It
    fits a histogram to the RMS values and creates a random variable distribution that can be used to generate new RMS
    values that follow the same distribution as the original data.
    """

    def __init__(self):
        self.rms_values = self.get_rms_values()

        # Fit a histogram to the RMS values and create a random variable distribution
        self.hist_tr = np.histogram(self.rms_values, bins=100)
        self.model_dist = rv_histogram(self.hist_tr, density=False)

    def get_rms_values(self) -> np.ndarray:
        # Get all RMS values from the catalogue
        catalogue = hdc.HardcastleCatalogue()
        rms_values = catalogue.get_values(hdc.Source.RMS)

        #  Remove NaN values from the RMS data
        rms_values = np.array(rms_values)
        rms_values = rms_values[~np.isnan(rms_values)]

        # Choose a threshold to contain a certain percentage of the data, e.g., 99.7% for a normal distribution (3 sigma)
        # This filters from the highest first; that is, removes the largest RMS values until the desired percentage of the data is retained
        percentage_threshold = 0.95
        upper_limit = np.percentile(rms_values, percentage_threshold * 100)

        # Very few values above this range, but they affect our histogram fitting, so filtering
        rms_values = rms_values[rms_values < upper_limit]

        return rms_values

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generates random samples from the fitted RMS distribution.

        :param size: The number of random samples to generate.
        :return: An array of random samples drawn from the fitted RMS distribution.
        """
        return self.model_dist.rvs(size=size)

    def plot(self):
        """
        Plots the histogram of RMS values to visualize the distribution.
        """
        plt.hist(self.rms_values, bins=100, density=False)
        plt.xlabel('RMS Value (mJy/beam)')
        plt.ylabel('Density')
        plt.title('Distribution of RMS Values in the Hardcastle Catalogue')
        plt.grid(True)
        plt.savefig('rms_histogram.png')
        plt.show()

    def get_statistics(self) -> dict:
        """
        Computes and returns basic statistics of the RMS values.

        :return: A dictionary containing the mean, median, standard deviation, and percentiles of the RMS values.
        """
        stats = {
            'mean': np.mean(self.rms_values),
            'median': np.median(self.rms_values),
            'std_dev': np.std(self.rms_values),
            'percentiles': {
                '90th': np.percentile(self.rms_values, 90),
                '95th': np.percentile(self.rms_values, 95),
            }
        }

        print("RMS Statistics:")
        print(f"Mean: {stats['mean']:.4f} mJy/beam")
        print(f"Median: {stats['median']:.4f} mJy/beam")
        print(f"Standard Deviation: {stats['std_dev']:.4f} mJy/beam")
        print(f"90th Percentile: {stats['percentiles']['90th']:.4f} mJy/beam")
        print(f"95th Percentile: {stats['percentiles']['95th']:.4f} mJy/beam")
        
        return stats

if __name__ == "__main__":
    RMS = RMSDistribution()
    RMS.plot()
    RMS.get_statistics()