import hardcastle_catalogue as hdc
import utils.paths as pths
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm, Normalize


def load_initial_dataset(
        dataset_file_path: Path = pths.DATASET_PARENT / 'hardcastle_catalogue_with_images.h5'):
    """
    Loads the initial dataset from a HDF5 file into a pandas dataframe for future use.

    :param dataset_file_path: The path to the HDF5 file containing the initial dataset with header information and pixel values.
    :return:
    """
    print("Loading Hardcastle catalogue from HDF5 file...")
    with h5py.File(dataset_file_path, 'r') as h5file:
                images = h5file['images'][:]
                cat_info = h5file['cat_info'][:]

    return images, cat_info

def get_values():
    # Load the initial dataset with pixel values and extract the peak pixel values
    dataset, cat_info = load_initial_dataset()
    peak_pixels = np.array(extract_peak_pixel_values(dataset)) * 1000
    peak_fluxes = np.array(extract_peak_fluxes(cat_info))
    source_sizes = np.array([item['LAS'] if 'LAS' in item.dtype.names else np.nan for item in cat_info])
    
    # Save peak_pixels and peak_fluxes to txt files for future use
    np.savetxt(pths.DATASET_PARENT / 'preprocessing/peak_pixels.txt', peak_pixels)
    np.savetxt(pths.DATASET_PARENT / 'preprocessing/peak_fluxes.txt', peak_fluxes)
    np.savetxt(pths.DATASET_PARENT / 'preprocessing/source_sizes.txt', source_sizes)

def extract_peak_pixel_values(dataset):    
    peak_pixels = [np.max(item) if isinstance(item, np.ndarray) else np.nan for item in dataset]
    return peak_pixels

def extract_peak_fluxes(cat_info):
    # Get the peak fluxes from the catalogue information
    peak_fluxes = [item['Peak_flux'] if 'Peak_flux' in item.dtype.names else np.nan for item in cat_info]
    return peak_fluxes

def filter_data(peak_fluxes, peak_pixels, source_sizes, percentage_threshold=0.85):
    # Remove any nan values from the peak fluxes and peak pixels
    indices = ~np.isnan(peak_fluxes) & ~np.isnan(peak_pixels)
    
    # Remove any entries due to a very large size
    indices &= source_sizes < 120
    
    # Remove any entries where peak_fluxes are close to peak_pixels
    # indices &= ~np.isclose(peak_fluxes, peak_pixels, rtol=0.05)

    # Filter out any entries where the peak fluxes are above a certain threshold to retain a certain percentage of the data
    upper_limit = np.percentile(peak_fluxes, percentage_threshold * 100)
    indices &= peak_fluxes < upper_limit
    
    #
    
    return peak_fluxes[indices], peak_pixels[indices], upper_limit

def get_statistics(array):
    print(f"Count: {len(array)}")
    print(f"Mean: {np.mean(array):.4f}")
    print(f"Median: {np.median(array):.4f}")
    print(f"Standard Deviation: {np.std(array):.4f}")
    print(f"90th Percentile: {np.percentile(array, 90):.4f}")
    print(f"95th Percentile: {np.percentile(array, 95):.4f}")
    print(f"Max: {np.max(array):.4f}")
    print(f"Min: {np.min(array):.4f}")

def plot_both(peak_fluxes, peak_pixels):
    plt.plot(peak_fluxes, peak_pixels, '.', alpha=0.1)
    plt.xlabel('Peak Flux from Catalogue (mJy/beam)')
    plt.ylabel('Peak Pixel Value from Dataset (mJy/beam)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Peak Flux from Catalogue vs Peak Pixel Value from Dataset')
    plt.grid(True)
    plt.show()

def plot_both_2d_hist(peak_fluxes, peak_pixels):
    plt.figure(figsize=(6.4 * 1.5, 4.8 * 1.5))
    x_bin_edges = np.logspace(-1, 2, 800)
    y_bin_edges = np.logspace(-1, 3, 800)
    # x_bin_edges = np.linspace(0, 2.5, 700)
    # y_bin_edges = np.linspace(0, 2.5, 700)
    h, _, _, _ = plt.hist2d(peak_fluxes, 
               peak_pixels, 
               bins=[x_bin_edges, y_bin_edges],
            #    range = [[np.min(peak_fluxes), np.max(peak_fluxes)], [np.min(peak_pixels), 10**3]],
               norm=LogNorm())
    print(f"Count of entries in 2D histogram: {np.sum(h)}")
    plt.colorbar(label='Counts')
    plt.plot([10**-1, 10**2], [10**-1, 10**2], color='red', linestyle='--', label='y = x Line')
    plt.xlabel('Peak Flux from Catalogue (mJy/beam)')
    plt.ylabel('Peak Pixel Value from Dataset (mJy/beam)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('2D Histogram of Peak Flux from Catalogue vs Peak Pixel Value from Dataset')
    plt.grid(True)
    plt.show()

def plot_residuals(peak_fluxes, peak_pixels):
    # Plot a histogram of the residual between the peak fluxes from the catalogue and the peak pixel values from the dataset
    residuals = np.abs(peak_fluxes - peak_pixels)
    upper_limit = 0.02
    residuals = residuals[residuals < upper_limit]
    print(f"Count of residuals below {upper_limit}: {len(residuals)}")
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Residual (Peak Flux from Catalogue - Peak Pixel Value from Dataset)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals between Peak Flux from Catalogue and Peak Pixel Value from Dataset')
    plt.grid(True)
    plt.savefig('residuals_histogram.png')
    plt.show()

def plot_percentage_error(percentage_error, peak_fluxes, upper_limit):
    # percentiles
    percentile_75 = np.percentile(peak_fluxes, 75)
    percentile_50 = np.percentile(peak_fluxes, 50)
    percentile_25 = np.percentile(peak_fluxes, 25)

    print(f"Count: {len(percentage_error)}")
    # plt.scatter(peak_fluxes, percentage_error, alpha=0.2, marker='.')
    plt.hexbin(peak_fluxes, percentage_error, gridsize=200, cmap='Blues', bins='log', xscale='log', yscale='log')
    plt.colorbar(label='log(N)')

    # from matplotlib.colors import LogNorm
    # plt.hist2d(peak_fluxes, percentage_error, bins=100, norm=LogNorm())    
    # plt.colorbar(label='Counts')
    
    # # Plot % thresholds for relative residuals
    plt.plot([0, upper_limit], [5, 5], color='blue', linestyle='--', label='5% Percentage Error Threshold')
    plt.plot([0, upper_limit], [10, 10], color='red', linestyle='--', label='10% Percentage Error Threshold')
    plt.plot([0, upper_limit], [20, 20], color='green', linestyle='--', label='20% Percentage Error Threshold')
    
    # Plot vertical lines to show percentiles of the peak fluxes
    plt.axvline(x=percentile_75, color='orange', linestyle='--', label='75th Percentile of Peak Fluxes')
    plt.axvline(x=percentile_50, color='purple', linestyle='--', label='50th Percentile of Peak Fluxes')
    plt.axvline(x=percentile_25, color='brown', linestyle='--', label='25th Percentile of Peak Fluxes')
    plt.legend()
    plt.xlabel('Peak Flux from Catalogue (mJy/beam)')
    plt.ylabel('Percentage Error (%)')
    # plt.xlim(xmin=-0.0001)
    plt.xlim(xmin=0.03)
    plt.xscale('log')
    plt.ylim(ymin=0.1, ymax=10**6)
    plt.yscale('log')
    plt.title('Percentage Error between Peak Flux and Peak Pixel Value')
    plt.grid(True)
    plt.savefig('percentage_error_scatter.png')
    plt.show()

if __name__ == "__main__":
    # Get values
    # get_values()
    print("Loading peak fluxes, peak pixels, and source sizes from txt files...")
    peak_pixels = np.loadtxt(pths.DATASET_PARENT / 'preprocessing/peak_pixels.txt')
    peak_fluxes = np.loadtxt(pths.DATASET_PARENT / 'preprocessing/peak_fluxes.txt')
    source_sizes = np.loadtxt(pths.DATASET_PARENT / 'preprocessing/source_sizes.txt')
    
    # Get percentiles of peak fluxes
    print("Calculating percentiles of peak fluxes...")
    per95 = np.percentile(peak_fluxes, 95)
    per90 = np.percentile(peak_fluxes, 90)
    per75 = np.percentile(peak_fluxes, 75)
    per50 = np.percentile(peak_fluxes, 50)
    per25 = np.percentile(peak_fluxes, 25)
    
    #Filter data
    print("Filtering data based on percentage threshold and source size...")
    peak_fluxes, peak_pixels, upper_limit = filter_data(peak_fluxes, peak_pixels, source_sizes, 0.95)
    get_statistics(peak_fluxes)

    # plot_both_2d_hist(peak_fluxes, peak_pixels)
    

    # Calculate percentage error between peak fluxes and peak pixel values
    print("Calculating percentage error between peak fluxes and peak pixel values...")   
    percentage_error = (np.abs(peak_fluxes - peak_pixels) / peak_fluxes) * 100
    
    # Print some statistics about the percentage error
    print(f"Mean Percentage Error: {np.mean(percentage_error):.2f}%")
    print(f"Median Percentage Error: {np.median(percentage_error):.2f}%")
    print(f"Standard Deviation of Percentage Error: {np.std(percentage_error):.2f}%")
    print(f"56.5th Percentile of Percentage Error: {np.percentile(percentage_error, 56.5):.2f}%")
    print(f"90th Percentile of Percentage Error: {np.percentile(percentage_error, 90):.2f}%")
    print(f"95th Percentile of Percentage Error: {np.percentile(percentage_error, 95):.2f}%")
    print(f"Max Percentage Error: {np.max(percentage_error):.2f}%")
    print(f"Min Percentage Error: {np.min(percentage_error):.2f}%")
    
    # Plot residuals
    plt.figure(figsize=(6.4 * 1.5, 4.8 * 1.5))
    x_bin_edges = np.logspace(-1.5, np.log10(np.max(peak_fluxes)), 700)
    y_bin_edges = np.logspace(-1, 5, 700)
    h, xedges, yedges, _ = plt.hist2d(peak_fluxes, percentage_error, 
                                      bins=[x_bin_edges, y_bin_edges],
                                      range = [[0.03, upper_limit], [0.1, 10**5]],
                                    norm=LogNorm())
    plt.colorbar(label='Counts')
    print(f"Number of data points in histogram: {np.sum(h)}")
    print(f"% of data points below 1 mJy/beam: {np.sum(peak_fluxes < 1) / len(peak_fluxes) * 100:.2f}%")
    # plt.plot([0, upper_limit], [10, 10], color='blue', linestyle='--', label='10% Percentage Error')
    # plt.plot([0, upper_limit], [25, 25], color='red', linestyle='--', label='25% Percentage Error')
    # plt.plot([0, upper_limit], [50, 50], color='orange', linestyle='--', label='50% Percentage Error')
    # plt.plot([0, upper_limit], [100, 100], color='orange', linestyle='--', label='100% Percentage Error')
    plt.axvline(x=per90, color='blue', linestyle='--', label='90th Percentile of Peak Fluxes')
    plt.axvline(x=per75, color='purple', linestyle='--', label='75th Percentile of Peak Fluxes')
    plt.axvline(x=per50, color='magenta', linestyle='--', label='50th Percentile of Peak Fluxes')
    plt.axvline(x=per25, color='orange', linestyle='--', label='25th Percentile of Peak Fluxes')
    plt.legend(loc='upper right')
    plt.xlabel('Peak Flux from Catalogue (mJy/beam)')
    plt.ylabel('Percentage Error (%)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.title('2D Histogram of Peak Flux vs Percentage Error')
    plt.show()

    # Highlight some problematic sources; choose 10 random ones with peak pixel value = 0 or NaN
    # problematic_sources = [item for item in dataset if (isinstance(item['pixel_values'], np.ndarray) and np.max(item['pixel_values']) < 0.001) or (not isinstance(item['pixel_values'], np.ndarray))]

    # problematic_sources = [item for item in dataset if np.max(item) < 0.001 and not np.isnan(np.max(item))]
    # random_problematic_sources = np.random.choice(problematic_sources, size=10, replace=False)

    # for idx, item in enumerate(random_problematic_sources):
    #     plt.imshow(item['pixel_values'], cmap='gray')
    #     plt.title(f"Problematic Source {idx+1}")
    #     plt.colorbar(label='Pixel Value')
    #     plt.savefig(f'problematic_source_{idx+1}.png')
    #     plt.show()

    # # Do the same but for sources with peak catalogue value = 0
    # problematic_sources_catalogue = [item for item in peak_fluxes if item < 0.1]
    # random_problematic_sources_catalogue = np.random.choice(problematic_sources_catalogue, size=10, replace=False)

    # for idx, item in enumerate(random_problematic_sources_catalogue):
    #     plt.imshow(item['pixel_values'], cmap='gray')
    #     plt.title(f"Problematic Source from Catalogue {idx+1}")
    #     plt.colorbar(label='Pixel Value')
    #     plt.savefig(f'problematic_source_catalogue_{idx+1}.png')
    #     plt.show()

    # todo: far better implementation involves index tracking, maybe flagging these sources?
