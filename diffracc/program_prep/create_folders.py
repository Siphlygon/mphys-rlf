"""
Creates the necessary folders for the program to run.
"""

from ..utils import paths


def make_folders():
    """
    Creates the necessary folders for the program to run.
    """
    # Create folders and symlinks
    for p in [paths.MODEL_PARENT,
              paths.ANALYSIS_PARENT,
              paths.IMG_DATA_PARENT,
              paths.FITS_PARENT,
              paths.PYBDSF_PARENT,
              paths.NP_ARRAY_PARENT,
              paths.DATASET_PARENT,]:
        # Make folder if it doesn't exist
        if not p.exists():
            p.mkdir()

        # Create symlink if necessary
        if paths.STORAGE_PARENT != paths.BASE_PARENT:
            symlink = paths.BASE_PARENT / p.name
            if not symlink.exists():
                symlink.symlink_to(p)
            else:
                assert (
                    symlink.resolve() == p
                ), f"Broken folder structure: Symlink {symlink} points to {symlink.resolve()}."

    # per-model folders
    for name in paths.MODEL_NAMES:
        for f in [paths.IMG_DATA_PARENT]:
            if not (f/name).exists():
                (f/name).mkdir()

        # for model parent, the sub-folder needs to be f"{NAME}_model"
        for f in [paths.MODEL_PARENT]:
            if not (f/f"{name}_model").exists():
                (f/f"{name}_model").mkdir()

    # create the folders for the pretrained models and the mosaic/cutouts folders
    for f in [paths.PRETRAINED_PARENT]:
        if not f.exists():
            f.mkdir()

    # create the subfolders for the fits images, pybdsf outputs, and nparrays
    for f in [paths.FITS_PARENT,
              paths.PYBDSF_CATALOG_PARENT,
              paths.PYBDSF_EXPORT_IMAGE_PARENT,
              paths.PYBDSF_LOG_PARENT,
              paths.NP_ARRAY_PARENT]:
        if not f.exists():
            f.mkdir()
        for g in paths.SUBDIRS:
            if not (f/g).exists():
                (f/g).mkdir()

    # create the subfolders for the dataset preparation
    for f in [paths.CUTOUTS_PATH, paths.PREPROCESSING_PARENT]:
        if not f.exists():
            f.mkdir()


if __name__ == '__main__':
    make_folders()
