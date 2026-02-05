import os
import requests
import logging
import utils.logging
from astropy.io import fits
import utils.paths as paths
from tqdm import tqdm
from utils.distributed import distribute


class CutoutDownloader:
    """
    This is a class to handle downloading cutouts from the LOFAR cutout server based on the Hardcastle catalogue, and
    storing them in appropriately binned folders.
    """

    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("cutout downloader", logging.DEBUG)
        self.subdir_size = 10000  # Number of images per subdirectory

    def read_positions(self, file_path=paths.DATASET_PARENT/"image_downloading/resolved_positions.txt"):
        """
        Reads the RA and DEC positions from a text file.

        :param file_path: The path to the text file containing RA & DEC positions.
        :return: A list of tuples containing (RA, DEC) for each resolved item.
        """
        try:
            positions = []
            with open(file_path, 'r') as f:
                for line in f:
                    ra, dec = map(float, line.split())
                    positions.append((ra, dec))
            self.logger.info(f"Successfully read {len(positions)} positions from {file_path}.")
            return positions
        except Exception as e:
            self.logger.error(f"Error reading positions from text file: {e}.")
            return []

    # This method comes from the LOFAR API, with changes made to use the logger file.
    # For more information, see: https://github.com/mhardcastle/lotss-cutout-api/blob/main/cutout.py
    def get_cutout(self, outfile, pos, size=2, low=False, dr3=False, auth=None):
        '''Get a cutout at position pos with size size arcmin. If low is
        True, get the 20-arcsec cutout, else get the 6-arcsec one. If dr3
        is true, try to access the DR3 data instead. Save to outfile.

        '''
        base = 'dr3' if dr3 else 'dr2'
        url = 'https://lofar-surveys.org/'
        if low:
            page = base + '-low-cutout.fits'
        else:
            page = base + '-cutout.fits'

        self.logger.debug(f'Trying {url + page}?pos={pos}&size={size}')
        r = requests.get(url + page, params={'pos': pos, 'size': size}, auth=auth, stream=True)
        self.logger.debug(f'received response code {r.status_code} and content type {r.headers["content-type"]}')
        if r.status_code != 200:
            raise RuntimeError('Status code %i returned' % r.status_code)
        if r.headers['content-type'] != 'application/fits':
            raise RuntimeError('Server did not return FITS file, probably no coverage of this area')

        with open(outfile, 'wb') as o:
            o.write(r.content)
            r.close()

    def make_subdir(self, subdir_num):
        """
        It can be hard and take a long time to view 300k image files in a single directory. We have made the design choice
        to store images in individual subdirectories containing up to 10k images each. This file decides which subdir
        to store an image in based on its index number.

        :param subdir_num: The index number of the image
        :return: The path to the subdirectory to store the image in
        """
        subdir_name = f"{subdir_num*self.subdir_size}-{(subdir_num+1)*self.subdir_size-1}"
        subdir_path = paths.DATASET_PARENT / "dr2_cutouts_download" / subdir_name
        if not os.path.exists(subdir_path):
            self.logger.info(f'Creating directory {subdir_path}...')
            os.makedirs(subdir_path)
        return subdir_path

    def download_all_cutouts(self):
        """
        Downloads all cutouts from the LOFAR cutout server based on the Hardcastle catalogue positions.
        """
        self.logger.info(f'Loading positions...')
        hdc_positions = self.read_positions()

        # Check if target directory exists, create if not
        target_directory = paths.DATASET_PARENT / "dr2_cutouts_download"
        if not os.path.exists(target_directory):
            self.logger.info(f'Creating directory {target_directory}...')
            os.makedirs(target_directory)

        # Clean the error log file for this run
        error_log_path = paths.DATASET_PARENT / "download_errors.log"
        if os.path.exists(error_log_path):
            self.logger.info(f'Cleaning existing error log file {error_log_path}...')
            os.remove(error_log_path)

        # Create a list of image numbers corresponding to the number of positions, which will be used for naming the cutout files and logging
        # e.g., 0 to 314699
        # On a slurm cluster, this list will be sliced accrding to the number of nodes, and distributed accordingly
        image_nums = distribute(list(range(len(hdc_positions))))

        # Cutout images will be stored in subdir of 10k size, by image index
        # Rather than checking that the right subdirs exist every single cutout creation, we will check they exist
        # whenever a new subdir is needed i.e., index has crossed a 10k boundary
        # this means, by default, 30 os checks will happen, rather than 300k
        subdir_index = image_nums[0] // self.subdir_size
        target_directory = target_directory / self.make_subdir(subdir_index)

        self.logger.info('Starting download of cutouts for images %i to %i...', image_nums[0], image_nums[-1])
        for i in tqdm(image_nums, desc="Downloading cutouts"):
            # get the RA and DEC for this image number
            ra, dec = hdc_positions[i]

            # get the path for this cutout file depending on its index
            new_index = i // self.subdir_size
            if new_index != subdir_index:
                # if we've crossed a subdir boundary, check the new subdir exists and update the current subdir index
                subdir_index = new_index
                target_directory = self.make_subdir(subdir_index)
            cutout_path = target_directory / f"cutout{i}.fits"

            # check if file exists and don't download if so
            if os.path.exists(cutout_path):
                self.logger.info(f'Skipping cutout for existing image {i}...')
                continue
            print(f'Downloading image {i} for RA={ra}, DEC={dec} degrees')

            try:
                self.logger.info(f'Downloading image {i}...')
                self.get_cutout(cutout_path, f"{ra} {dec}")
            except Exception as e:
                self.logger.error(f"Error downloading cutout for image {i} (RA={ra}, DEC={dec}): {e}")
                with open(paths.DATASET_PARENT / "download_errors.log", "a") as log_file:
                    log_file.write(f"Image {i}: RA={ra}, DEC={dec}, Error: {e}\n")

if __name__ == "__main__":
    downloader = CutoutDownloader()
    downloader.download_all_cutouts()