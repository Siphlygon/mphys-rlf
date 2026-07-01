"""
Downloads the dataset from the provided link and saves it to a hardcoded path.
"""
import urllib.request

from ..utils.logger import show_dl_progress
from ..utils.paths import LOFAR_DATA_PATH


def download_dataset():
    """Downloads the dataset from the provided link and saves it to a hardcoded path."""
    files = {
        LOFAR_DATA_PATH: "https://cloud.hs.uni-hamburg.de/s/jPZdExPPmcZ48o5",
    }

    for file, link in files.items():
        if not file.exists():
            print("Downloading: ", file)
            urllib.request.urlretrieve(f"{link}/download", file, show_dl_progress)
            print("Done.")

if __name__ == '__main__':
    download_dataset()
