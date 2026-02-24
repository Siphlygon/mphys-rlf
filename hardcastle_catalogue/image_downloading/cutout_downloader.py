import os
import requests
import logging
import utils.logging
import utils.paths as paths
from tqdm import tqdm
from utils.distributed import distribute
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt
import time
import configparser
from pathlib import Path


class CutoutDownloader:
    """
    This is a class to handle downloading cutouts from the LOFAR cutout server based on the Hardcastle catalogue, and
    storing them in appropriately binned folders.
    """

    # Requests params - note, do not overload the server - please be polite to LOFAR!
    MAX_WORKERS = 64  # polite concurrency
    RATE_DELAY = 0.03 # ~10 requests/sec max
    RETRIES = 6

    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("cutout downloader", logging.DEBUG)

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)

        # we are using sources generated in a loguniform way
        de_config = config['DEFAULT']

        # Get values from config
        self.folder_size = int(de_config['FOLDER_SIZE'])

        # Reusable session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.MAX_WORKERS,
            pool_maxsize=self.MAX_WORKERS
        )
        self.session.mount("https://", adapter)
        self.recent_errors = 0
        self.last_request = 0

    # ---------- SET UP ----------
    def read_positions(self,
                       file_path : Path = paths.IMAGE_DOWNLOADING/"resolved_positions.txt")\
            -> list[tuple[float, float]]:
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

    def make_folder(self,
                    folder_num : int)\
            -> Path:
        """
        It can be hard and take a long time to view 300k image files in a single directory. We have made the design choice
        to store images in individual folders containing up to 10k images each. This file decides which folder
        to store an image in based on its index number.

        :param folder_num: The index number of the image
        :return: The path to the folder to store the image in
        """
        folder_name = f"{folder_num*self.folder_size}-{(folder_num+1)*self.folder_size-1}"
        folder_path = paths.DATASET_PARENT / "dr2_cutouts_download" / folder_name
        if not os.path.exists(folder_path):
            self.logger.info(f'Creating directory {folder_path}...')
            os.makedirs(folder_path)
        return folder_path

    # ---------- NETWORK LAYER ----------

    # def _rate_limit(self):
    #     """Simple client-side pacing."""
    #     dt = time.time() - self.last_request
    #     if dt < self.RATE_DELAY:
    #         time.sleep(self.RATE_DELAY - dt)
    #     self.last_request = time.time()

    def _rate_limit(self):
        # If we start getting slow responses, back off automatically
        if getattr(self, "recent_errors", 0) > 3:
            time.sleep(0.5)
        elif getattr(self, "recent_errors", 0) > 0:
            time.sleep(0.15)

    # This method was comes from the LOFAR API, with major changes made to optimise it for large-batch requests.
    # For more information, see: https://github.com/mhardcastle/lotss-cutout-api/blob/main/cutout.py
    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(RETRIES))
    def get_cutout(self,
                   outfile : Path | str,
                   pos : str,
                   size : int = 2,
                   low=False, dr3=False, auth=None):
        """
        Get a cutout at position pos with size size arcmin. If low is True, get the 20-arcsec cutout, else get the
        6-arcsec one. If dr3 is true, try to access the DR3 data instead. Save to outfile

        NOTE: pos is in format "RA DEC"
        """
        base = 'dr3' if dr3 else 'dr2'
        url = 'https://lofar-surveys.org/'
        page = base + ('-low-cutout.fits' if low else '-cutout.fits')

        self._rate_limit()

        r = self.session.get(
            url + page,
            params={'pos': pos, 'size': size},
            auth=auth,
            stream=True,
            timeout=60
        )

        if r.status_code != 200:
            raise RuntimeError(f'Status {r.status_code}')

        if r.headers.get('content-type') != 'application/fits':
            raise RuntimeError('No FITS returned – probably no coverage')

        # # Stream write (memory safe)
        # with open(outfile, 'wb') as o:
        #     for chunk in r.iter_content(chunk_size=1024 * 1024):
        #         if chunk:
        #             o.write(chunk)

        # FITS files are ~54KB, so we can afford to load them into memory before writing to disk,
        # which is faster than streaming for small files
        with open(outfile, 'wb') as o:
            o.write(r.content)

        r.close()

    # ---------- PARALLEL ORCHESTRATION ----------

    def _download_one(self,
                      args : tuple[int, float, float, Path])\
            -> tuple[int, str | None]:
        """
        Downloads a single cutout image from the LOFAR cutout server based on the RA and DEC positions, and saves it to
        the specified path.

        :param args: A tuple containing the index of the image, RA, DEC, and the path to save the cutout file to.
        :return: A tuple containing the index of the image and any error message (or None if successful).
        """
        i, ra, dec, path = args

        if os.path.exists(path):
            return i, "exists"

        try:
            self.get_cutout(path, f"{ra} {dec}")
            self.recent_errors = max(0, self.recent_errors - 1)
            return i, None
        except Exception as e:
            self.recent_errors += 1
            return i, str(e)

    def download_all_cutouts(self,
                             custom_positions : list[tuple[float, float]] = None):
        """
        Downloads all cutouts from the LOFAR cutout server based on the Hardcastle catalogue positions.

        :param custom_positions: An optional list of RA and DEC positions to use instead of loading from the Hardcastle
        catalogue. This is useful for testing or if you want to download a specific subset of cutouts.
        """
        if custom_positions is not None:
            self.logger.info('Using custom positions provided as argument for downloading cutouts...')
            hdc_positions = custom_positions
        else:
            self.logger.info(f'No positions specified - loading all positions from Hardcastle catalogue...')
            hdc_positions = self.read_positions()

        # Check if target directory exists, create if not
        target_directory = paths.DATASET_PARENT / "dr2_cutouts_download"
        if not os.path.exists(target_directory):
            self.logger.info(f'Creating directory {target_directory}...')
            os.makedirs(target_directory)

        # Clean the error log file for this run
        error_log_path = paths.IMAGE_DOWNLOADING / "download_errors.log"
        if os.path.exists(error_log_path):
            self.logger.info(f'Cleaning existing error log file {error_log_path}...')
            os.remove(error_log_path)

        # Create a list of image numbers corresponding to the number of positions, which will be used for naming the cutout files and logging
        # e.g., 0 to 314699
        # On a slurm cluster, this list will be sliced according to the number of nodes, and distributed accordingly
        image_nums = distribute(list(range(len(hdc_positions))))

        # Cutout images will be stored in folders of configurable size, by image index
        # Rather than checking that the right folders exist every single cutout creation, we will check they exist
        # whenever a new folder is needed e.g., index has crossed a 10k boundary
        # this means, by default, 30 os checks will happen, rather than 300k
        folder_index = image_nums[0] // self.folder_size
        target_directory = target_directory / self.make_folder(folder_index)

        self.logger.info('Building task list for downloading... ')
        # Build task list for concurrent download
        tasks = []
        for i in tqdm(image_nums, desc="Building task list..."):
            ra, dec = hdc_positions[i]

            # get the path for this cutout file depending on its index
            new_index = i // self.folder_size
            if new_index != folder_index:
                # if we've crossed a folder boundary, check the new folder exists and update the current folder index
                folder_index = new_index
                target_directory = self.make_folder(folder_index)
            cutout_path = target_directory / f"cutout{i}.fits"

            tasks.append((i, ra, dec, cutout_path))

        # Concurrent downloads
        self.logger.info('Starting download of cutouts for images %i to %i...', image_nums[0], image_nums[-1])
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as ex:
            futures = {ex.submit(self._download_one, t): t[0] for t in tasks}

            for f in tqdm(as_completed(futures), total=len(futures)):
                i, err = f.result()

                if err and err != "exists":
                    with open(paths.IMAGE_DOWNLOADING / "download_errors.log", "a") as log:
                        log.write(f"{i}: {err}\n")

if __name__ == "__main__":
    downloader = CutoutDownloader()
    downloader.download_all_cutouts()
