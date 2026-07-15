import argparse
import configparser

import astropy.cosmology
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from ..rlf.agn_selection import get_catalogue_info
from ..rlf.rlf import RLF
from ..rlf.rlf_constants import colors
from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger


def main(args):
    """
    Main function to generate RLF plots.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing flux cut and plot selection options.
    """
    logger = get_logger("RLF Main", LoggingLevels.INFO.value)
    plt.rcParams['font.size'] = 18

    flux_cut_jy = args.flux_cut_jy

    # Read parameters from the config.ini file
    config = configparser.ConfigParser()
    config.read(paths.PROGRAM_CONFIG)
    default_config = config['DEFAULT']

    # Cosmological Parameters
    h = float(default_config['h']) # hubble constant = h * 100 km/s/Mpc
    Tcmb0 = float(default_config['Tcmb0']) # temp of the CMB at z=0 in K
    Om0 = float(default_config['Om0']) # matter density parameter at z=0
    cosmo = astropy.cosmology.FlatLambdaCDM(h * 100 * u.km / u.s / u.Mpc, Tcmb0=Tcmb0 * u.K, Om0=Om0)

    redshifts, fluxes, luminosities, resolved = get_catalogue_info(cosmo,
                                                                   flux_cut_jy,
                                                                   args.plot_rlagn_selection_contour)

    rlf0 = RLF(fluxes, redshifts, luminosities, resolved, cosmo, bias=0, flux_cut_jy=flux_cut_jy)
    rlf1 = RLF(fluxes, redshifts, luminosities, resolved, cosmo, bias=0, flux_cut_jy=flux_cut_jy, vmax_method=True)
    rlfs = [rlf0, rlf1]

    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))

    logger.debug(f"lum: {np.min(luminosities)}-{np.max(luminosities)}, "
                f"redsh: {np.min(redshifts)}-{np.max(redshifts)}, "
                f"flux: {np.min(fluxes)}-{np.max(fluxes)}")

    rlf_titles = ['H25 RLAGN RLF using Page & Carrera 2000', 'H25 RLAGN RLF using $1/V_a$']
    draw_ylabels = [True, False]

    for rlf, ax, title, draw_ylabel in zip(rlfs, axes, rlf_titles, draw_ylabels):
        rlf.calculate_rlf(plot_rlf=False)
        rlf.plot_rlf(title, colors, ax, draw_ylabel=draw_ylabel)

    plt.savefig('rlfs_vmax.png')
    plt.show()

    logger.info('done')


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Generate RLF plots.')
    parser.add_argument('--flux_cut_jy', type=float, default=1.1e-3,
                        help='Flux cut in Jy (default: 1.1e-3)')
    parser.add_argument('--plot_rlagn_selection_contour', action='store_true',
                        help='Plot RLAGN selection contour (default: False)')

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    main(args)
