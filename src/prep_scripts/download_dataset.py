from utils.paths import LOFAR_DATA_PATH
import urllib.request
from utils.logging import show_dl_progress

def download_dataset():
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