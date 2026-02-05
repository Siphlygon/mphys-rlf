import utils.paths as pth
import shutil
from utils.distributed import DistributedUtils

def make_folders():
    # Create folders and symlinks
    for p in [pth.MODEL_PARENT,
              pth.ANALYSIS_PARENT, 
              pth.IMG_DATA_PARENT, 
              pth.FITS_PARENT, 
              pth.PYBDSF_PARENT, 
              pth.NP_ARRAY_PARENT, 
              pth.FLAGS_PARENT]:
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
    for name in pth.NAMES:
        for f in [pth.IMG_DATA_PARENT]:
            if not (f/name).exists():
                (f/name).mkdir()

        # for model parent, the sub-folder needs to be f"{NAME}_model"
        for f in [pth.MODEL_PARENT]:
            if not (f/f"{name}_model").exists():
                (f/f"{name}_model").mkdir()

    for f in [pth.PRETRAINED_PARENT, 
              pth.MOSAIC_DIR,
              pth.CUTOUTS_DIR]:
        if not f.exists():
            f.mkdir()

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