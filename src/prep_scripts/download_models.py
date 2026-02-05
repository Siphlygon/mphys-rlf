import utils.paths as pth
import urllib.request
from utils.logging import show_dl_progress
import shutil
from utils.distributed import DistributedUtils

def download_models():
    models = [ pth.PRETRAINED_PARENT / f'parameters_{name}_model.pt' for name in pth.NAMES ]

    # dict to make sure ordering is correct
    links_dict = {
        "LOFAR": "https://cloud.hs.uni-hamburg.de/s/KTAFWFnLByMgNRn", 
        "FIRST": "https://cloud.hs.uni-hamburg.de/s/xs7bbt99AMFf8gP"
    }
    links = [ links_dict[ name ] for name in pth.NAMES ]

    for model, link, name in zip( models, links, pth.NAMES ):
        if not model.exists():
            print("Downloading: ", model)
            urllib.request.urlretrieve(f"{link}/download", model, show_dl_progress)
            print("Done.")

        # Copy the model to the sampling directory if it doesn't exist
        sampling_file = pth.MODEL_PARENT / f"{name}_model" / f"parameters_{name}_model.pt"
        if not sampling_file.exists():
            shutil.copyfile( model, sampling_file )

if __name__ == '__main__':
    download_models()