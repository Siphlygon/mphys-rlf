import configparser
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from ..utils import paths
from ..utils.distributed import distribute
from ..utils.logger import LoggingLevels, get_logger


class CutoutDownloader:
    """
    This is a class to handle downloading cutouts from the LOFAR cutout server based on the Hardcastle catalogue, and
    storing them in appropriately binned folders.
    """
    # Requests params - note, do not overload the server - please be polite to LOFAR!
    MAX_WORKERS = 64
    RATE_DELAY = 0.03
    RETRIES = 6

    def __init__(self):
        """
        Initialises the CutoutDownloader by setting up logging, reading parameters from the config.ini file, and
        creating a reusable requests session with connection pooling. It also initializes a counter for recent errors
        and a timestamp for the last request to implement rate limiting.
        """
        self.logger = get_logger("CutoutDownloader", LoggingLevels.DEBUG.value)

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)
        de_config = config['DEFAULT']
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
                       file_path : Path = paths.INITIAL_DATASET/"resolved_positions.txt")\
            -> list[tuple[float, float]]:
        """
        Reads the RA and DEC positions of resolved sources from a text file and returns them as a list of tuples.
        
        Parameters
        ----------
        file_path : Path, optional
            The path to the text file containing the RA and DEC positions, by default
            paths.INITIAL_DATASET/"resolved_positions.txt"

        Returns
        -------
        list[tuple[float, float]]
            A list of tuples containing the RA and DEC positions of resolved sources.
        """
        try:
            positions = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ra, dec = map(float, line.split())
                    positions.append((ra, dec))
            self.logger.info(f"Successfully read {len(positions)} positions from {file_path}.")
            return positions
        except Exception as e:
            self.logger.error(f"Error reading positions from text file: {e}.")
            raise Exception(f"Failed to read positions from text file at {file_path}. "
                            f"Please check the file and try again") from e


    def make_folder(self,
                    folder_num : int,
                    directory_path : Path = paths.DATASET_PARENT / "dr2_cutouts_download")\
            -> Path:
        """
        Creates a folder for storing cutout files. The folder will be named in the format "start_index-end_index", where
        start_index and end_index are calculated based on the folder number and the configured folder size.

        Parameters
        ----------
        folder_num : int
            The folder number, which determines the range of cutout files that will be stored in this folder.
        directory_path : Path
            The path to the directory where the folder will be created.

        Returns
        -------
        Path
            The path to the created folder.
        """
        folder_name = f"{folder_num*self.folder_size}-{(folder_num+1)*self.folder_size-1}"
        folder_path = directory_path / folder_name
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


    # This method was comes from the LOFAR API, with changes made to optimise it for large-batch requests.
    # For more information, see: https://github.com/mhardcastle/lotss-cutout-api/blob/main/cutout.py
    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(RETRIES))
    def get_cutout(self,
                   outfile : Path | str,
                   pos : str,
                   size : int = 2,
                   low=False, dr3=False, auth=None):
        """
        Gets a cutout from the LOFAR cutout server and saves it to the specified path. The cutout is based on the RA and
        DEC positions provided in the `pos` parameter, and the size of the cutout is specified by the `size` parameter.
        The cutout can be either a standard cutout or a low-resolution cutout, and can be from either the DR2 or DR3
        data releases.

        Parameters
        ----------
        outfile : Path | str
            The path to save the downloaded cutout FITS file.
        pos : str
            The RA and DEC positions of the cutout.
        size : int, optional
            The size of the cutout in arcminutes, by default 2
        low : bool, optional
            Whether to download a low-resolution cutout, by default False
        dr3 : bool, optional
            Whether to download from the DR3 data release, by default False
        auth : _type_, optional
            The authentication credentials, by default None

        Raises
        ------
        RuntimeError
            If the server returns a status code other than 200, or if the content type of the response is not FITS.
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

        with open(outfile, 'wb') as o:
            o.write(r.content)

        r.close()


    # ---------- PARALLEL ORCHESTRATION ----------
    def _download_one(self,
                      args : tuple[int, float, float, Path])\
            -> tuple[int, str | None]:
        """
        Downloads a single cutout based on the provided RA and DEC positions, and saves it to the specified path. If the
        cutout file already exists, it skips the download. If the download fails, it returns an error message.
        
        Parameters
        ----------
        args : tuple[int, float, float, Path]
            A tuple containing the index of the cutout, the RA and DEC positions, and the path to save the downloaded
            cutout FITS file.

        Returns
        -------
        tuple[int, str | None]
            A tuple containing the index of the cutout and an error message if the download failed, or None if the
            download was successful or the file already exists.
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
                             directory_path : Path = paths.DATASET_PARENT / "dr2_cutouts_download",
                             custom_positions : list[tuple[float, float]] | None = None):
        """
        Downloads cutouts for all positions in the Hardcastle catalogue, or for a custom list of positions if provided.

        Parameters
        ----------
        directory_path : Path, optional
            The path to the directory where the cutout files will be saved, by default
            paths.DATASET_PARENT/"dr2_cutouts_download"
        custom_positions : list[tuple[float, float]] | None, optional
            A list of RA and DEC positions to use instead of loading from the Hardcastle catalogue. This is useful for
            testing or if you want to download a specific subset of cutouts, by default None
        """
        if custom_positions is not None:
            self.logger.info('Using custom positions provided as argument for downloading cutouts...')
            hdc_positions = custom_positions
        else:
            hdc_positions = self.read_positions()

        # Check if target directory exists, create if not
        target_directory = directory_path
        if not os.path.exists(target_directory):
            self.logger.info(f'Creating directory {target_directory}...')
            os.makedirs(target_directory)

        # Clean the error log file for this run
        error_log_path = paths.INITIAL_DATASET / "download_errors.log"
        if os.path.exists(error_log_path):
            self.logger.info(f'Cleaning existing error log file {error_log_path}...')
            os.remove(error_log_path)

        # Create a list of image numbers corresponding to the number of positions
        # On a slurm cluster, this list will be sliced according to the number of nodes, and distributed accordingly
        image_nums = distribute(list(range(len(hdc_positions))))

        # Cutout images will be stored in folders of configurable size, by image index
        folder_index = image_nums[0] // self.folder_size
        target_directory = target_directory / self.make_folder(folder_index)

        # Build a list of tasks for downloading cutouts, each task is a tuple of (index, RA, DEC, path)
        self.logger.info('Building task list for downloading... ')
        tasks = []
        for i in tqdm(image_nums, desc="Building task list..."):
            ra, dec = hdc_positions[i]
            new_index = i // self.folder_size

            # if we've crossed a folder boundary, check the new folder exists and update the current folder index
            if new_index != folder_index:
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
                    with open(paths.INITIAL_DATASET / "download_errors.log", "a", encoding="utf-8") as log:
                        log.write(f"{i}: {err}\n")


if __name__ == "__main__":
    downloader = CutoutDownloader()
    downloader.download_all_cutouts()
