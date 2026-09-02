import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from ..utils import paths
from ..utils.power_transform import PeakFluxPowerTransformer


def main(args):
    """
    Main function to compare conditioned peak fluxes with image peak fluxes.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the subdirectory path.
    """
    img_directory = paths.FITS_PARENT / args.subdir
    power_transformer = PeakFluxPowerTransformer(args.subdir)

    peak_fluxes_conditioned = []
    peak_fluxes_from_imgs = []

    for img_path in img_directory.rglob("*.fits"):
        with fits.open(img_path) as hdul:
            flux_scaled = hdul[0].header['FXSCLD']
            peak_flux_from_img = np.max(hdul[0].data)
        peak_flux_conditioned = power_transformer.inverse_transform(np.array([flux_scaled]))

        peak_fluxes_from_imgs.append(peak_flux_from_img)
        peak_fluxes_conditioned.append(peak_flux_conditioned)

    if not peak_fluxes_from_imgs:
        raise RuntimeError(f"No FITS files found in {img_directory}")

    plt.scatter(peak_fluxes_conditioned, peak_fluxes_from_imgs, s=0.01, label=args.subdir)

    min_peakflux = min(np.min(peak_fluxes_conditioned), np.min(peak_fluxes_from_imgs))
    max_peakflux = max(np.max(peak_fluxes_conditioned), np.max(peak_fluxes_from_imgs))

    x = np.geomspace(min_peakflux, max_peakflux)
    plt.plot(x, x, color='r')
    plt.grid()

    plt.title('Peak flux from conditioning vs image')
    plt.xlabel('Conditioned peak flux, Jy/beam')
    plt.ylabel('Image peak flux, Jy/beam')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('peak_flux_conditioning.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare conditioned peak fluxes with image peak fluxes.")
    parser.add_argument('--subdir', type=str, required=True, help="Subdirectory containing FITS files.")
    args = parser.parse_args()

    main(args)
