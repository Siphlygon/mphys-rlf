"""
Downloads the pretrained models from the provided links and saves them to the specified paths.
"""
import shutil
import urllib.request

from ..utils import paths
from ..utils.logger import show_dl_progress


def download_models():
    """
    Downloads the pretrained models from the provided links and saves them to the specified paths.
    """
    models = [ paths.PRETRAINED_PARENT / f'parameters_{name}.pt' for name in paths.MODEL_NAMES ]

    # dict to make sure ordering is correct
    links_dict = {
        "LOFAR_model": "https://cloud.hs.uni-hamburg.de/s/KTAFWFnLByMgNRn", 
        "FIRST_model": "https://cloud.hs.uni-hamburg.de/s/xs7bbt99AMFf8gP"
    }
    links = [ links_dict[ name ] for name in paths.MODEL_NAMES ]

    for model, link, name in zip( models, links, paths.MODEL_NAMES ):
        if not model.exists():
            print("Downloading: ", model)
            urllib.request.urlretrieve(f"{link}/download", model, show_dl_progress)
            print("Done.")

        # Copy the model to the sampling directory if it doesn't exist
        sampling_file = paths.MODEL_PARENT / f"{name}" / f"parameters_{name}.pt"
        if not sampling_file.exists():
            shutil.copyfile( model, sampling_file )

if __name__ == '__main__':
    download_models()
