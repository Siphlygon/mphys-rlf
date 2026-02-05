# Because the creation of a new database in this manner is computationally intensive (possibly 100k+ iterations for
# each of 100k+ items, oof!), this script is designed to test the speed of various matching algorithms
# to find the best way to do this.

import numpy as np
from tqdm import tqdm
import h5py
from astropy.io import fits
import os

import utils.logging
import logging
import asyncio

from utils.distributed import distribute

# global const
num_images = 1000

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped


########
# DATABASE LOADING FUNCTIONS
########

def load_optical_catalogue(file_path='hardcastle_catalogue/combined-release-v1.2-LM_opt_mass.fits', folder_path='hardcastle_catalogue/dr2_cutouts_download/'):
    # Get the header information for the resolved items from the optical catalogue
    with fits.open(file_path) as hdul:
        catalogue_data = hdul[1].data  # Assuming the data is in the first extension
        resolved_items = catalogue_data[catalogue_data['Resolved'] == True]

        # Just for this test, shorten it to num_images items
        resolved_items = resolved_items[:num_images]

    # Turn resolved_items into a dictionary list for easier handling
    resolved_list = [{'header': item} for item in resolved_items]

    # Load the pixel values from the corresponding cutout files into memory. This is very intensive.
    for i in (tqdm(distribute(range(len(resolved_list))), desc="Loading pixel values")):
        cutout_file = f"{folder_path}cutout{i}.fits"

        if not os.path.exists(cutout_file):
            # logger.info(f"Cutout file {cutout_file} does not exist. Skipping.")
            resolved_list[i]['pixel_values'] = [0]
            continue

        # the database is currently incomplete on local computer, continue if file not found
        try:
            with fits.open(cutout_file) as cutout_hdul:
                pixel_values = cutout_hdul[0].data  # Assuming pixel values are in the primary HDU
                resolved_list[i]['pixel_values'] = pixel_values.flatten()  # Flatten to 1D array for easier comparison
        except Exception as e:
            logger.error(f"Error loading cutout file {cutout_file}: {e}")

        # For testing purposes, limit to first num_images items
        if i >= num_images-1:
            break

    # 0-clip to match the LOFAR dataset's preprocessing
    logger.info("0-clipping the optical pixel values...")
    for i in range(len(resolved_list)):
        try:
            resolved_list[i]['pixel_values'] = np.clip(resolved_list[i]['pixel_values'], 0, None)  # clip negative values to zero
        except Exception as e:
            logger.error(f"Error during 0-clipping for optical item {i}: {e}")

    return resolved_list

def load_lofar_data(file_path='hardcastle_catalogue/LOFAR_Dataset.h5'):
    with h5py.File(file_path, 'r') as h5file:
        lofar_item_values = h5file['images'][:]

        # For testing purposes, limit to first num_images items
        # lofar_item_values = lofar_item_values[:num_images]

        # Flatten each image to 1D array for easier comparison
        lofar_item_values = [item.flatten() for item in lofar_item_values]
    return lofar_item_values

#########
# MATCHING ALGORITHMS
#########

# def is_equal_to_matching(opt_cat_items, lofar_item_values):
#     matches = []
#     for i, opt_item in enumerate(tqdm(opt_cat_items)):
#         try:
#             for lofar_item in lofar_item_values:
#                 if np.array_equal(opt_item['pixel_values'], lofar_item):
#                     matches.append((opt_item, lofar_item))
#                     break
#         except Exception as e:
#             logger.error(f"Error during matching for optical item {i}: {e}")
#     return matches

def per_pixel_matching(opt_cat_items, lofar_item_values):
    matches = []
    for i, opt_item in enumerate(tqdm(distribute(opt_cat_items))):
        pixel_values = opt_item['pixel_values']
        try:
            for lofar_item in lofar_item_values:
                for j in range(len(pixel_values)):
                    if pixel_values[j] != lofar_item[j]:
                        break
                else:
                    matches.append((opt_item, lofar_item))
                    break
        except Exception as e:
            logger.error(f"Error during matching for optical item {i}: {e}")
    return matches

if __name__ == "__main__":
    logger = utils.logging.get_logger("speed testing", logging.INFO)

    logger.info("Loading optical data...")
    opt = load_optical_catalogue()

    logger.info("Loading LOFAR data...")
    lof = load_lofar_data()

    # logger.info("Starting brute-force matching...")
    # brute_force_matches = is_equal_to_matching(opt, lof)
    # logger.info(f"Brute-force matching found {len(brute_force_matches)} matches.")

    logger.info("Starting optimized matching...")
    optimized_matches = per_pixel_matching(opt, lof)
    logger.info(f"Optimized matching found {len(optimized_matches)} matches.")

    # test = np.array([[1,2,12], [3, 4, 34], [5, 6, 56]])
    # print(test.flatten())

