

# Write code that will import the "combined-release-v1.2-LM_opt_mass.fits" file from the hardcastle_catalogue directory
# and print out the number of items with the Resolved flag equal to True

from astropy.io import fits
import numpy as np
import h5py
from tqdm import tqdm

def load_hardcastle_catalogue(file_path="combined-release-v1.2-LM_opt_mass.fits"):
    # Load the FITS file
    with fits.open(file_path) as hdul:
        catalogue_data = hdul[1].data  # Assuming the data is in the first extension

    # We have verified below that the number of resolved flags is about expected 27 more images than the paper uses, but
    # this is how they went from LOFAR's >4.3 million images to 314k

    # Print total number of items
    print(f'Total number of items: {len(catalogue_data)}')

    # Extract the Resolved flag
    resolved_flags = catalogue_data['Resolved']
    # Count the number of items with Resolved flag equal to True
    num_resolved = np.sum(resolved_flags == True)
    print(f'Number of items with Resolved flag equal to True: {num_resolved}')

    # Keep only the items that have the Resolved flag equal to True
    resolved_items = catalogue_data[catalogue_data['Resolved'] == True]

    # Print some example entries and column names
    print("Catalogue columns: ", catalogue_data.columns.names)
    print(resolved_items[:10])

    return resolved_items


def load_given_LOFAR_data(file_path='LOFAR_Dataset.h5'):
    # Extract the data from the h5 file
    with h5py.File(file_path, 'r') as h5file:
        print("Length of axis1: ", len(h5file['catalog']['axis1'])) # this seems to be a list of indexes - the indexes of the 106k final images inside the wider 314k dataset
        # block 0 is literally 21 million mosaic ids. i have no idea it's use or function and that information is nowhere else in this code.
        print("Length of block1: ", len(h5file['catalog']['block1_values'])) # this is a bunch of physical values like ra, dec, flux, but has a lot of NaNs
        print("Length of block2: ", len(h5file['catalog']['block2_values'])) # this seems not useful. flags like prefilter, postfilter, assoc...
        print("Length of block3: ", len(h5file['catalog']['block3_values'])) # could be useful! has resolved, nans_cutout, problem-clip, broken...

        # Assuming the dataset is named 'images'
        #lofar_data = h5file['images']
        lofar_block1_labels = h5file['catalog']['block1_items'][:]
        lofar_block1_values = h5file['catalog']['block1_values'][:]

        return lofar_block1_labels, lofar_block1_values


def count_matches(hdc_cat_items, lofar_item_values):
    # Iterate through the Hardcastle catalogue items and count matches in the LOFAR dataset for the following fields
    specific_fields = ['E_RA', 'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux']
    match_count = 0

    # LOFAR item values is an array of arrays where each inner array corresponds to an item's value across all fields
    lofar_ra_values = lofar_item_values[:, 0]  # RA values
    lofar_dec_values = lofar_item_values[:, 1]  # DEC values

    # For every item in the Hardcastle catalogue, we're going to try and find an exact match for all specific fields
    y=0
    for idx, hdc_item in tqdm(enumerate(hdc_cat_items)):
        y+=1

        passed_checks = [] # this is a list of fields which are the same between the image

        # We're going to use RA and DEC kinda like a primary key to find matches
        opt_ra = hdc_item['RA']
        opt_dec = hdc_item['DEC']

        # Going to use the fact we know the index for 'RA' is 0 and 'DEC' is 1 in the LOFAR dataset to speed things up
        # First find wherever the RA matches, and then see if any of those also have matching DEC
        ra_match_indices = np.where(np.isclose(lofar_ra_values, opt_ra, atol=1e-10))[0]
        dec_match_indices = np.where(np.isclose(lofar_dec_values, opt_dec, atol=1e-10))[0]
        common_indices = np.intersect1d(ra_match_indices, dec_match_indices)
        if len(common_indices) == 0:
            continue  # No match found for RA and DEC, skip to the next Hardcastle catalogue item
        if len(common_indices) > 1:
            print(f'Warning: Multiple matches found for Hardcastle catalogue item index {idx} with RA={opt_ra} and DEC={opt_dec}')
            break
        common_index = common_indices[0] # this should just be a single number

        # Now check the other specific fields for these common indices
        for field_idx, field in enumerate(specific_fields):
            hdc_value = hdc_item[field]  # gets the value of the field for this item

            lofar_item = lofar_item_values[common_index]
            lofar_value = lofar_item[field_idx+2]  # +2 because RA and DEC are the first two fields
            if np.isclose(lofar_value, hdc_value, atol=1e-10):
                passed_checks.append(field)
            else:
                print(f'Field {field} does not match: Hardcastle={hdc_value}, LOFAR={lofar_value}')

        if len(passed_checks) == len(specific_fields):
            match_count += 1
            print(f'Match found for Hardcastle catalogue item index {idx}: {hdc_item}')

        # if y > 1000:
        #     break  # Limit to first 1000 items for performance during testing

    print(f'Number of matches found: {match_count}')
    return match_count


if __name__ == "__main__":
    # The FITS file for the Hardcastle catalogue comes from the LoTSS-DR2 and is described by Hardcastle et al. 2023
    # It has the data with nice headers, and although it's 4.1 mil items we can narrow it done to approximately the
    # same dataset used in the paper.
    resolved_items = load_hardcastle_catalogue()

    # Unfortunately, the same is not true for the h5 file provided by the paper we're using. It's messy, with a lot of
    # headers filled with NaN values. I am doing my best to try and handle all of that but YEESH.
    lofar_labels, lofar_values = load_given_LOFAR_data()

    count_matches(resolved_items, lofar_labels, lofar_values)
    print()
    # COULD RUN THE CUTOUTS SCRIPTS GIVN BY PAPER TO SEE IF SAME OUTPUT AS GIVEN
    # WANT TO ESTABLISH HOW WE ARRIVED AT THE STATE OF THE GIVEN H5 FILE