"""
Creates the necessary folders for the program to run.
"""

import utils.paths as pth


def make_folders():
    """
    Creates the necessary folders for the program to run.
    """
    # Create folders and symlinks
    for p in [pth.MODEL_PARENT,
              pth.ANALYSIS_PARENT,
              pth.IMG_DATA_PARENT,
              pth.FITS_PARENT,
              pth.PYBDSF_PARENT,
              pth.NP_ARRAY_PARENT]:
        # Make folder if it doesn't exist
        if not p.exists():
            p.mkdir()

        # Create symlink if necessary
        if not pth.STORAGE_PARENT == pth.BASE_PARENT:
            symlink = pth.BASE_PARENT / p.name
            if not symlink.exists():
                symlink.symlink_to(p)
            else:
                assert (
                    symlink.resolve() == p
                ), f"Broken folder structure: Symlink {symlink} points to {symlink.resolve()}."

    # per-model folders
    for name in pth.MODEL_NAMES:
        for f in [pth.IMG_DATA_PARENT]:
            if not (f/name).exists():
                (f/name).mkdir()

        # for model parent, the sub-folder needs to be f"{NAME}_model"
        for f in [pth.MODEL_PARENT]:
            if not (f/f"{name}_model").exists():
                (f/f"{name}_model").mkdir()

    # create the folders for the pretrained models and the mosaic/cutouts folders
    for f in [pth.PRETRAINED_PARENT,
              pth.MOSAIC_DIR,
              pth.CUTOUTS_DIR]:
        if not f.exists():
            f.mkdir()

    # create the subfolders for the fits images, pybdsf outputs, and nparrays
    for f in [pth.FITS_PARENT,
              pth.PYBDSF_CATALOG_PARENT,
              pth.PYBDSF_EXPORT_IMAGE_PARENT,
              pth.PYBDSF_LOG_PARENT,
              pth.NP_ARRAY_PARENT]:
        if not f.exists():
            f.mkdir()
        for g in pth.SUBDIRS:
            if not (f/g).exists():
                (f/g).mkdir()


if __name__ == '__main__':
    make_folders()
