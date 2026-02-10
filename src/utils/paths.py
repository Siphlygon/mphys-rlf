from pathlib import Path
from indexed import IndexedOrderedDict


# Base directories for code base & storage
BASE_PARENT = Path(__file__).parent.parent.parent

# CHANGE THIS IF DESIRED:
STORAGE_PARENT = BASE_PARENT  # Alternatively: Path("/your/desired/folder")

# Main storage folders.
MODEL_PARENT = STORAGE_PARENT / "model_results"
ANALYSIS_PARENT = STORAGE_PARENT / "analysis_results"
IMG_DATA_PARENT = STORAGE_PARENT / "image_data"
FITS_PARENT = STORAGE_PARENT / "fits_images"
PYBDSF_PARENT = STORAGE_PARENT / "pybdsf"
NP_ARRAY_PARENT = STORAGE_PARENT / "nparrays"
DATASET_PARENT = STORAGE_PARENT / "hardcastle_catalogue"

MAXVALS = NP_ARRAY_PARENT / "maxvals.npy"

# Folders for creating the training dataset
IMAGE_DOWNLOADING = DATASET_PARENT / "image_downloading"

# Model configuration presets
CONFIG_PARENT = BASE_PARENT / "src/model/configs"
MODEL_CONFIGS = IndexedOrderedDict({f.stem: f for f in CONFIG_PARENT.glob("*.json")})

# Config file
PROGRAM_CONFIG = BASE_PARENT / "src/program.config"

# Model Names
MODEL_NAMES = [ "LOFAR", "FIRST" ]

# Folders for different kinds of fits image data
SUBDIRS = [ "dataset", "generated_datadist", "generated_loguniform" ]
COLOURS = [ 'b', 'g', 'm' ]
PYBDSF_EXPORT_IMAGE_PARENT = PYBDSF_PARENT / "images"
PYBDSF_LOG_PARENT = PYBDSF_PARENT / "logs"
PYBDSF_CATALOG_PARENT = PYBDSF_PARENT / "catalogs"


# Pretrained models
PRETRAINED_PARENT = MODEL_PARENT / "pretrained"

# Train data subsets
LOFAR_SUBSETS = IndexedOrderedDict(
    {
        k: IMG_DATA_PARENT / "LOFAR" / v
        for k, v in {
            "0-clip": "0-clip.hdf5",
        }.items()
    }
)

# Paths for training data processing
LOFAR_DATA_PATH = IMG_DATA_PARENT / "LOFAR" / "LOFAR_Dataset.h5"
MOSAIC_DIR = IMG_DATA_PARENT / "LOFAR" / "mosaics"
CUTOUTS_DIR = IMG_DATA_PARENT / "LOFAR" / "cutouts"
LOFAR_RES_CAT = IMG_DATA_PARENT / "LOFAR" / "6-LoTSS_DR2-public-resolved_sources.csv"


def cast_to_Path(path):
    """
    Cast a string object to a Path object. If the input is already a Path object,
    return it as is. If not Path or str, raise a TypeError.

    Parameters
    ----------
    path : str or Path
        The path to be cast to a Path object.

    Returns
    -------
    Path
        The path as a Path object.

    Raises
    ------
    TypeError
        If the input is not a Path or a string.
    """
    match path:
        case Path():
            return path
        case str():
            return Path(path)
        case _:
            raise TypeError(f"Expected Path or str, got {type(path)}")


def rename_files(path, model_name_new, model_name_old=None):
    """
    Rename all files in the given directory and its subdirectories that contain
    the old model name to the new model name.

    Parameters
    ----------
    path : Path
        The directory containing the files to be renamed.
    model_name_new : str
        The new model name to replace the old model name.
    model_name_old : str, optional
        The old model name to be replaced, by default None.
        If None, the directory name is used as the old model name.
    """
    if model_name_old is None:
        model_name_old = path.name

    for file in path.iterdir():
        if file.is_file():
            name = file.stem.replace(model_name_old, model_name_new)
            file.rename(path / f"{name}{file.suffix}")
        elif file.is_dir():
            rename_files(file, model_name_new, model_name_old)


if __name__ == "__main__":

    print("Base directories for code base & storage")
    print(f"\tBASE_PARENT: {BASE_PARENT}")
    print(f"\tSTORAGE_PARENT: {STORAGE_PARENT}")

    print("\nThree main storage folders.")
    print(f"\tMODEL_PARENT: {MODEL_PARENT}")
    print(f"\tANALYSIS_PARENT: {ANALYSIS_PARENT}")
    print(f"\tIMG_DATA_PARENT: {IMG_DATA_PARENT}")

    print("\nFolders for different kinds of image data")
    print(f"\tLOFAR_DATA_PARENT: {IMG_DATA_PARENT}/LOFAR")
    print(f"\tFIRST_DATA_PARENT: {IMG_DATA_PARENT}/FIRST")

    print("\nTrain data subsets")
    for k, v in LOFAR_SUBSETS.items():
        print(f"\t{k}: {v}")
