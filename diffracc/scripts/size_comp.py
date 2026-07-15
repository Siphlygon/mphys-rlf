import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from ..utils import paths


def main(args):
    img_directory = paths.FITS_PARENT / args.subdir

    lass = []
    lass_from_img = []

    for img_path in img_directory.rglob("*.fits"):
        with fits.open(img_path) as hdul:
            las = hdul[0].header['LASIZE']
            img = hdul[0].data
            # not actually largest angular size, just a basic proxy with no physical motivation
            las_from_img = np.sqrt((img > np.median(img) + 3 * np.std(img)).sum())
        lass.append(las)
        lass_from_img.append(las_from_img)

    if not lass:
        raise RuntimeError(f"No FITS files found in {img_directory}")

    plt.scatter(lass, lass_from_img, s=0.01)

    minval = min(np.min(lass), np.min(lass_from_img))
    maxval = max(np.max(lass), np.max(lass_from_img))

    x = np.geomspace(minval, maxval)
    plt.plot(x, x, color='r')
    plt.grid()

    plt.title('Size conditioning test')
    plt.xlabel('Largest Angular Size (arcsec)')
    plt.ylabel('Random Size Proxy (a.u.)')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('size_comp.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare conditioned peak fluxes with image peak fluxes.")
    parser.add_argument('--subdir', type=str, required=True, help="Subdirectory containing FITS files.")
    args = parser.parse_args()

    main(args)
