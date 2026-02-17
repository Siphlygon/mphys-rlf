import os
import requests
import logging
import utils.logging
from astropy.io import fits
import utils.paths as paths
from tqdm import tqdm
from utils.distributed import distribute
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt
import time


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

        # Subdir params
        self.subdir_size = 10000  # Number of images per subdirectory

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
    def read_positions(self, file_path=paths.IMAGE_DOWNLOADING/"resolved_positions.txt"):
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
    def get_cutout(self, outfile, pos, size=2, low=False, dr3=False, auth=None):
        """
        Get a cutout at position pos with size size arcmin. If low is True, get the 20-arcsec cutout, else get the
        6-arcsec one. If dr3 is true, try to access the DR3 data instead. Save to outfile
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

    def _download_one(self, args):
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

    def download_all_cutouts(self, custom_positions=None):
        """
        Downloads all cutouts from the LOFAR cutout server based on the Hardcastle catalogue positions.
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
        # On a slurm cluster, this list will be sliced accrding to the number of nodes, and distributed accordingly
        image_nums = distribute(list(range(len(hdc_positions))))

        # Cutout images will be stored in subdir of 10k size, by image index
        # Rather than checking that the right subdirs exist every single cutout creation, we will check they exist
        # whenever a new subdir is needed i.e., index has crossed a 10k boundary
        # this means, by default, 30 os checks will happen, rather than 300k
        subdir_index = image_nums[0] // self.subdir_size
        target_directory = target_directory / self.make_subdir(subdir_index)

        self.logger.info('Building task list for downloading... ')
        # Build task list for concurrent download
        tasks = []
        for i in tqdm(image_nums, desc="Building task list..."):
            ra, dec = hdc_positions[i]

            # get the path for this cutout file depending on its index
            new_index = i // self.subdir_size
            if new_index != subdir_index:
                # if we've crossed a subdir boundary, check the new subdir exists and update the current subdir index
                subdir_index = new_index
                target_directory = self.make_subdir(subdir_index)
            cutout_path = target_directory / f"cutout{i}.fits"

            tasks.append((i, ra, dec, cutout_path))


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

    #todo: you want to write this up to be able to generate over a generic list of positions because this is how you
    #can easily tie in uh download verification functionality