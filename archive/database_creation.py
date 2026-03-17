"""
So -- it is very unclear the relationship between the LOFAR data given (LOFAR_Dataset.h5) and the actual LOFAR data
releases. From information in the paper, they refer to Hardcastle et al. 2023, which contains a catalogue of radio
sources that are "good" in the optical from LOFAR). The paper also mentions that they use only
resolved sources.

We have established from testing that the header information provided in the h5 file seems to be inaccurate - the RA
and DEC values give a galaxy with completely different pixel values from the LOFAR cutout server. This file's original
goal was to verify that we can reach the same dataset as the paper, from official resources, but we have now adapted it
to generate a new dataset with completely correct header information.

Have just now discovered that the Hardcastle catalogue does not contain pixel values. I am adding code to download the cutouts
from the LOFAR cutout server for each resolved item in the Hardcastle catalogue. Note that this is not the full catalogue.
"""

from astropy.io import fits
import numpy as np
import h5py
from tqdm import tqdm

import utils.logging
import logging
from utils.distributed import distribute
import os

class DatabaseCreator:
    """
    A class to create a full LOFAR database using the Hardcastle catalogue as the source, rather than the given LOFAR data.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("database creator", logging.DEBUG)

    def load_hardcastle_catalogue(self, file_path="hardcastle_catalogue/hardcastle_catalogue_with_images.fits"):
        """
        Loads the Hardcastle catalogue from a FITS file 0-clips it for matching with the present LOFAR data.

        :param file_path: The path to the Hardcastle catalogue FITS file.
        :return: A list of 0-clipped items from the catalogue.
        """
        self.logger.info("Loading Hardcastle catalogue from FITS file...")

        # Get the information from the Hardcastle catalogue
        catalogue_data = []
        with fits.open(file_path) as hdul:
            # The first non-Primary table has all the header information. This'll be stored in a separate file with
            # the final matching dataset. No usage for now
            header_information = hdul[1].data

            # Remove the first two HDUs which are just Primary and the header table
            hdul = hdul[2:]

            # Extract the pixel values from each imageHDU
            for idx, hdu in enumerate(tqdm(hdul, desc="Extracting pixel values & 0-clipping from Hardcastle catalogue")):
                try:
                    if isinstance(hdu.data, np.ndarray):
                        catalogue_data.append({'index': idx, 'clipped_values': np.clip(hdu.data, 0, None)})
                    else:
                        catalogue_data.append({'index': idx, 'clipped_values': np.nan})
                except Exception as e:
                    self.logger.error(f"Error loading Hardcastle catalogue item {idx}: {e}")
                    catalogue_data.append({'index': idx, 'clipped_values': np.nan})

        # This is now done implicitly storing too much data in memory was crashing personal laptop so this is better

        # # 0-clip to match the LOFAR dataset's preprocessing
        # self.logger.info("0-clipping the Hardcastle pixel values...")
        # for i in tqdm(range(len(catalogue_data)), desc="0-clipping Hardcastle values"):
        #     try:
        #         if isinstance(catalogue_data[i]['pixel_values'], np.ndarray):
        #             catalogue_data[i]['clipped_values'] = np.clip(catalogue_data[i]['pixel_values'], 0, None)
        #         else:
        #             catalogue_data[i]['clipped_values'] = np.nan
        #     except Exception as e:
        #         self.logger.error(f"Error during 0-clipping for Hardcastle item {i}: {e}")

        return header_information, catalogue_data

    def load_given_LOFAR_data(self, file_path='hardcastle_catalogue/LOFAR_Dataset.h5'):
        self.logger.info("Loading LOFAR data from H5 file...")

        # Extract the data from the h5 file
        with h5py.File(file_path, 'r') as h5file:
            lofar_images = h5file['images'][:]

        return lofar_images

    def sort_hardcastle_data(self, hdc_images):
        """
        """
        # Had this idea -- you can sort the Hardcastle pixel values array by total pixel value sum, which then, by computing
        # the same for the LOFAR catalogue, would allow the usage of e.g., a binary search to find matches faster.
        self.logger.info("Sorting Hardcastle data for faster matching...")
        hdc_sums = np.array([(np.nansum(image['clipped_values']), i) for i, image in enumerate(hdc_images)])

        # Sort the array by pixel sum
        hdc_sorted = sorted(hdc_sums, key=lambda item: item[0], reverse=False)
        # hdc_sorted = np.sort(hdc_sums, axis=1)
        self.logger.info(f'Lowest Hardcastle pixel sum: {hdc_sorted[0][0]}')
        self.logger.info(f'Highest Hardcastle pixel sum: {hdc_sorted[-1][0]}')
        self.logger.info(f'Median Hardcastle pixel sum: {hdc_sorted[len(hdc_sorted)//2][0]}')

        return hdc_sorted

    def find_dataset_matches(self, lofar_images, hdc_images, hdc_sums):
        self.logger.info("Finding dataset matches...")
        matches = []
        for idx, lof_item in tqdm(enumerate(lofar_images), desc="Finding matches"):
            lof_sum = np.nansum(lof_item)

            # Keep track of the closest match found so far
            closest_match_idx = -1
            closest_match_diff = float('inf')

            # Use binary search to find potential matches in the sorted Hardcastle sums
            left, right = 0, len(hdc_sums) - 1 # establish search bounds
            while left <= right:  # exit condition i.e., if we ever get left > right, no match has been found
                mid = (left + right) // 2  # get the middle position in the current search range
                mid_value = hdc_sums[mid][0] # get the Hardcastle sum at that position

                # Update the closest match this is here in case no match is found, we still have the closest one
                if abs(mid_value - lof_sum) < closest_match_diff:
                    closest_match_diff = abs(mid_value - lof_sum)
                    closest_match_idx = mid # note this is the sorted index and does not correspond to the original Hardcastle index

                # Check for next steps
                if np.isclose(mid_value, lof_sum, atol=1e-10): # found a match!!
                    break
                elif mid_value < lof_sum:  # otherwise, mid value is less than target, search right half (asc so larger values are right)
                    left = mid + 1
                else: # mid value is greater than target, search left half (asc so smaller values are left)
                    right = mid - 1

            # After the binary search, we know the closest match index. Note that we are ignoring extra logic for exact
            # matches because there's no guarantee that an exact match in pixel sum means every pixel is identical.
            # Get a list of 10 indices around the closest match to check for actual pixel equality
            candidate_indices = []
            for offset in range(-10, 11):
                candidate_idx = closest_match_idx + offset # generate new index
                if 0 <= candidate_idx < len(hdc_sums): # ensure it's within bounds
                    candidate_indices.append(candidate_idx)

            # Convert sorted indices back to original LOFAR indices so we can access the corresponding images
            # candidate_indices = [int(np.argwhere(sorted_indices==idx)[0]) for idx in candidate_indices]
            candidate_indices = [int(hdc_sums[idx][1]) for idx in candidate_indices]

            # Now check these candidate indice s for actual pixel equality
            matched = False
            lof_pixels = lof_item.flatten()  # this is done for ease of access, the final product is not flattened
            for candidate_idx in candidate_indices:
                hdc_pixels = hdc_images[candidate_idx]['clipped_values'].flatten()
                try:
                    for j in range(len(lof_pixels)):
                        # if hdc_pixels[j] != lof_pixels[j]:  # encountered a pixel that doesn't match!
                        if not np.isclose(hdc_pixels[j], lof_pixels[j], atol=1e-6):
                            break
                    else:
                        matched = True
                        matching_idx = candidate_idx
                        break
                except Exception as e:
                    self.logger.error(f"Error during matching for Hardcastle item {candidate_idx}: {e}")

            # Exact per-pixel match found - all proof i need!
            if matched:
                self.logger.info(f'Match found for LOFAR item index {idx} with Hardcastle catalogue index {matching_idx}.')
                matches.append({'lofar_index': idx, 'hardcastle_index': matching_idx, 'per_pixel_match': True, 'pixel_difference': 0})

            else:
                self.logger.info(f'No exact match found for LOFAR item index {idx}. Finding closest pixel match...')

                # The closest match should not actually be the closest in pixel value sum, but in the sum of pixel differences
                # Avoids the situation where completely different images happen to have very close pixel value sums
                pixels_match_idx = -1
                pixels_diff = float('inf')
                for candidate_idx in candidate_indices:
                    hdc_pixels = hdc_images[candidate_idx]['clipped_values'].flatten()
                    try:
                        pixel_diff = np.nansum(np.abs(lof_pixels - hdc_pixels))
                        if pixel_diff < pixels_diff:  # found new lowest pixel diff
                            pixels_diff = pixel_diff
                            pixels_match_idx = candidate_idx
                    except Exception as e:
                        self.logger.error(f"Error calculating pixel difference for Hardcastle item {candidate_idx}: {e}")

                self.logger.info(f'Closest match for LOFAR item index {idx} is Hardcastle catalogue index {pixels_match_idx} with pixel difference {pixels_diff}.')
                matches.append({'lofar_index': idx, 'hardcastle_index': int(hdc_sums[pixels_match_idx][1]), 'per_pixel_match': False, 'pixel_difference': pixels_diff})

        return matches

    def save_matches_and_info(self, matches, folder_name='hardcastle_catalogue/final_database/'):
        """
        Saves the matched items to a directory. Rather than compile into one file, these will be individual fits files
        with an accompanying CSV file for the match information. This is because pyBDSF analysis requires individual
        FITS files anyway, so this saves us from having to split them later.

        :param matches: The list of matched items.
        :param folder_name: The folder to save the matched items to.
        """
        self.logger.info(f"Saving matches to {folder_name}...")

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        match_info_path = os.path.join(folder_name, 'match_info.csv')
        with open(match_info_path, 'w') as info_file:
            info_file.write("lofar_index,hardcastle_index,per_pixel_match\n")
            for match in tqdm(matches, desc="Saving matched items"):
                lofar_idx = match['lofar_index']
                hardcastle_idx = match['hardcastle_index']
                per_pixel = match['per_pixel_match']

                # The individual FITS file is a cutout from dr2_cutouts_download
                cutout_file = f"hardcastle_catalogue/dr2_cutouts_download/cutout{hardcastle_idx}.fits"
                save_path = os.path.join(folder_name, f"matched_lofar{lofar_idx}_cutout{hardcastle_idx}.fits")
                try:
                    with fits.open(cutout_file) as hdul:
                        hdul.writeto(save_path, overwrite=True)
                except Exception as e:
                    self.logger.error(f"Error saving matched FITS file for Hardcastle index {hardcastle_idx}: {e}")
                    continue

                # Write match info to CSV
                info_file.write(f"{lofar_idx},{hardcastle_idx},{per_pixel}\n")

        self.logger.info("Matches saved successfully.")

    def create_matching_dataset(self):
        """
        Creates a matching dataset between the Hardcastle catalogue and the given LOFAR data.
        """
        self.logger.info("Starting database creation process...")

        # NOTE - I GET MEMORY ERROR WHEN RUNNING THIS. MAY WORK ON GALAHAD AS MORE AVAILABLE MEMORY, BUT CAN'T SAVE
        # SUCH LARGE NUMPY FILES
        # # Check to see if a .npy file for the sorted Hardcastle data exists if so, load it to save time
        # if os.path.exists('hardcastle_catalogue/sorted_hardcastle_sums.npy') and os.path.exists('hardcastle_catalogue/hardcastle_clipped_images.npy'):
        #     self.logger.info("Loading Hardcastle images & sorted sums from .npy file...")
        #     hdc_catalogue = np.load('hardcastle_catalogue/hardcastle_clipped_images.npy')
        #     hdc_sums = np.load('hardcastle_catalogue/sorted_hardcastle_sums.npy')
        # else:
        #     self.logger.info("Missing .npy files for Hardcastle data loading from FITS file...")
        #     header_info, hdc_catalogue = self.load_hardcastle_catalogue()
        #     hdc_sums = self.sort_hardcastle_data(hdc_catalogue)
        #
        #     # Save to .npy files for future usage
        #     self.logger.info("Saving Hardcastle images & sorted sums to .npy files for future usage...")
        #     np.save('hardcastle_catalogue/hardcastle_clipped_images.npy', hdc_catalogue)
        #     np.save('hardcastle_catalogue/sorted_hardcastle_sums.npy', hdc_sums)

        # Load Hardcastle stuff and sort it
        header_info, hdc_catalogue = self.load_hardcastle_catalogue()
        hdc_sums = self.sort_hardcastle_data(hdc_catalogue)

        # Load the LOFAR data (super fast so no need to store in a huge file)
        lofar_values = self.load_given_LOFAR_data()

        # Find matches
        matched_items = self.find_dataset_matches(lofar_values, hdc_catalogue, hdc_sums)

        # Save the matches to a file
        self.save_matches_and_info(matched_items)

if __name__ == "__main__":
    # NOTE - PLEASE RUN hardcastle_catalogue/hardcastle_catalogue_downloader.py FIRST TO DOWNLOAD THE Hardcastle CATALOGUE FITS FILE
    # IT'S A LITTLE ANNOYING TO IMPLEMENT IN HERE BECAUSE IT USES GALAHAD DDP

    # The FITS file for the Hardcastle catalogue comes from the LoTSS-DR2 and is described by Hardcastle et al. 2023
    # It has the data with nice headers, and although it's 4.1 mil items we can narrow it done to approximately the
    # same dataset used in the paper.
    db_creator = DatabaseCreator()
    db_creator.create_matching_dataset()
