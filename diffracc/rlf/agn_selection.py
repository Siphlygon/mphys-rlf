"""
A module for selecting RLAGN sources based on the criteria from Hardcastle et al. 2025, using WISE magnitudes,
luminosities, and redshifts.

This module provides functions to classify sources as RLAGN, SFG, or RQQ based on their WISE magnitudes, luminosities,
and redshifts. The selection criteria are based on the work of Hardcastle et al. 2025, which defines specific thresholds
for these parameters to identify RLAGN sources.

The module also includes functionality to handle sources with insufficient data for classification, allowing users to
choose whether to keep or discard such sources based on the `exclusive` parameter. It is designed to be used in
conjunction with the Hardcastle catalogue, which contains information about various astronomical sources, including
their WISE magnitudes, luminosities, and redshifts.
"""
import astropy.cosmology
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from ..data.hardcastle_catalogue import HardcastleCatalogue, Source
from ..utils.functions import k_corr_factor, mag_to_flux_w2, mag_to_flux_w3
from ..utils.logger import LoggingLevels, get_logger

RQQ_XPT = -27.923076923076923  # mag
RQQ_YPT = 25.563106796116504  # log10(lum
WISE_3_FREQ = 3e8 / 12e-6
WISE_2_FREQ = 3e8 / 4.6e-6


def _get_wise3_absmag(wise3_mag: np.ndarray | float,
                      wise2_mag: np.ndarray | float,
                      redshifts: np.ndarray | float,
                      cosmo: astropy.cosmology.Cosmology) -> np.ndarray | float:
    """
    Calculate the absolute magnitude in the WISE 3 band for a set of sources, given their apparent magnitudes in the
    WISE 2 and 3 bands, their redshifts, and a cosmology.

    Accepts either arrays (vectorised, over a whole catalogue) or plain floats (a single source) - the return type
    matches whatever was passed in for wise3_mag.

    Parameters
    ----------
    wise3_mag : np.ndarray or float
        The apparent magnitude(s) in the WISE 3 band.
    wise2_mag : np.ndarray or float
        The apparent magnitude(s) in the WISE 2 band.
    redshifts : np.ndarray or float
        The redshift(s) of the source(s).
    cosmo : astropy.cosmology.Cosmology
        The cosmology to use for calculating luminosity distances.

    Returns
    -------
    wise3_absmag : np.ndarray or float
        The absolute magnitude(s) in the WISE 3 band.
    """
    # select_rlagn is called both vectorised (arrays, over a whole catalogue) and per-source (plain floats, from
    # apply_preprocessing's iterative flag path) - accept either by working in at-least-1D arrays internally, then
    # unwrapping back to a scalar if that's what came in.
    scalar_input = np.ndim(wise3_mag) == 0
    wise3_mag = np.atleast_1d(wise3_mag)
    wise2_mag = np.atleast_1d(wise2_mag)
    redshifts = np.atleast_1d(redshifts)

    # Filter out sources with non-finite redshifts or WISE magnitudes, as these cannot be used for the log calculation.
    # Does NOT filter specific criteriaa e.g., flux > 1.1mJy; that's not the job of this function
    # This does not change the behaviour (as it would default to NaN anyway and be caught in select_agn), but it avoids
    # a runtime warning from numpy about log of non-finite values.
    valid = np.isfinite(redshifts) & (redshifts > -1) \
            & np.isfinite(wise3_mag) & (wise3_mag > 0) \
            & np.isfinite(wise2_mag) & (wise2_mag > 0)

    # Extract the WISE frequencies
    wise3_flux = np.full(wise3_mag.shape, np.nan)
    wise2_flux = np.full(wise2_mag.shape, np.nan)
    wise3_flux[valid] = mag_to_flux_w3(wise3_mag[valid])
    wise2_flux[valid] = mag_to_flux_w2(wise2_mag[valid])

    # Calculate the spectral indices for the sources for a k-correction
    spectral_inds = np.full(wise3_mag.shape, np.nan)
    spectral_inds[valid] = -np.log(wise3_flux[valid] / wise2_flux[valid]) / np.log(WISE_3_FREQ / WISE_2_FREQ)

    # Sources with a non-physical redshift (<=0, e.g. the catalogue's value for "missing" commonly z=-1) would send
    # astropy's luminosity_distance and the k-correction into invalid territory due to a divie by 0.
    wise3_absmag = np.full(wise3_mag.shape, np.nan)
    wise3_absmag[valid] = (
        wise3_mag[valid]
        - 5 * (np.log10(cosmo.luminosity_distance(redshifts[valid]).to(u.parsec).value) - 1)
        + k_corr_factor(redshifts[valid], mag_space=True, spectral_index=spectral_inds[valid])
    )
    return wise3_absmag[0] if scalar_input else wise3_absmag


def _plot_rlagn_selection_contour(wise2_mag: np.ndarray,
                                  wise3_mag: np.ndarray,
                                  redshifts: np.ndarray,
                                  luminosities: np.ndarray,
                                  cosmo: astropy.cosmology.Cosmology):
    """
    Plots the relationship between L144 and Abs W3 (Fig. 2, H25) for RLAGN selection.

    Parameters
    ----------
    wise2_mag : np.ndarray
        The magnitudes in the WISE 2 band.
    wise3_mag : np.ndarray
        The magnitudes in the WISE 3 band.
    redshifts : np.ndarray
        The redshifts of the sources.
    luminosities : np.ndarray
        The luminosities of the sources.
    cosmo : astropy.cosmology.Cosmology
        The cosmology to use for calculating luminosity distances.
    """
    wise3_absmag = _get_wise3_absmag(wise3_mag, wise2_mag, redshifts, cosmo)

    wise3_linspace_sfg = np.linspace(-27, -18, 1000)
    wise3_linspace_rqq = np.linspace(-34, -27, 1000)
    sfg_lum_cutoff = 10**(14 - wise3_linspace_sfg / 2.5)
    rqq_lum_cutoff = 10**(-(wise3_linspace_rqq - RQQ_XPT) / 3.4844629455909923 + RQQ_YPT)
    plt.figure(figsize=(8,8))
    hist2d, xedges, yedges = np.histogram2d(wise3_absmag, np.log10(luminosities), bins=50,
                                            range=[[-35, -17], [19+np.log10(4), 30]])
    xcenters = (xedges[1:] + xedges[:-1]) / 2
    ycenters = (yedges[1:] + yedges[:-1]) / 2
    plt.contourf(xcenters[::-1], 10**ycenters, np.sqrt(hist2d[::-1, :].T))
    plt.plot(wise3_linspace_sfg, sfg_lum_cutoff, color='r')
    plt.plot(wise3_linspace_rqq, rqq_lum_cutoff, color='m')
    plt.plot([-27, -27], [1, rqq_lum_cutoff[-1]], color='m')
    plt.xlabel('WISE-band 3 absolute magnitude')
    plt.ylabel('$L_144$')
    plt.title('H25 RLAGN Selection Plot')
    plt.yscale('log')
    plt.xlim(-18, -34)
    plt.ylim(4e20, 1e29)
    plt.savefig('lum_vs_w3.png')
    plt.show()


def get_catalogue_info(cosmo: astropy.cosmology.Cosmology,
                       flux_cut_jy: float,
                       plot_rlagn_selection_contour: bool = False) \
                         -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get information about the catalogue data.

    Parameters
    ----------
    cosmo : astropy.cosmology.Cosmology
        The cosmology to use for calculating luminosity distances and volumes
    flux_cut_jy : float
        The flux cut in Jy
    plot_rlagn_selection_contour : bool, optional
        Whether to plot the RLAGN selection contour, by default False

    Returns
    -------
    redshifts : np.ndarray
        The redshifts of the sources in the catalog
    fluxes : np.ndarray
        The fluxes of the sources in the catalog
    luminosities : np.ndarray
        The luminosities of the sources in the catalog
    resolved : np.ndarray
        The resolved status of the sources in the catalog
    """
    logger = get_logger("RLF Catalogue Info", LoggingLevels.INFO.value)

    catalogue = HardcastleCatalogue(resolved_only=False)

    logger.info("Getting redshifts, fluxes, magnitudes, and luminosities from the catalogue...")
    redshifts = catalogue.get_value_column(Source.Redshift)  # z, unitless
    fluxes = catalogue.get_value_column(Source.TotalFlux) / 1000  # catalogue stores mJy; RLF expects Jy throughout
    # The Hardcastle et al. 2025 survey cut is on PEAK flux, not integrated flux, so it needs its own column -- the two
    # differ for exactly the resolved sources this selection cares about.
    peak_fluxes = catalogue.get_value_column(Source.PeakFlux) / 1000  # catalogue stores mJy/beam; cut is applied in Jy
    luminosities = catalogue.get_value_column(Source.Luminosity)  # in W/Hz
    wise3_mag = catalogue.get_value_column(Source.WISE3Mag)  # in mag
    wise3_magerr = catalogue.get_value_column(Source.WISE3MagErr)  # in mag
    wise2_mag = catalogue.get_value_column(Source.WISE2Mag)  # in mag
    resolved = catalogue.get_value_column(Source.Resolved)  # boolean
    logger.info(f"Total sources in catalogue: {redshifts.shape[0]}")

    logger.debug(f'wise3_mag: mean={np.average(wise3_mag)}, '
                 f'std={np.std(wise3_mag)}, '
                 f'max={np.max(wise3_mag)}, '
                 f'min={np.min(wise3_mag) }, '
                 f'count={wise3_mag.shape[0]}')
    logger.debug(f'wise2_mag: mean={np.average(wise2_mag)}, '
                 f'std={np.std(wise2_mag)}, '
                 f'max={np.max(wise2_mag)}, '
                 f'min={np.min(wise2_mag) }, '
                 f'count={wise2_mag.shape[0]}')

    # Calculate the RLAGN, SFG, and RQQ masks using the selection criteria from Hardcastle et al. 2025
    rlagn_mask, sfg_mask, rqq_mask = select_rlagn(wise2_mag,
                                                  wise3_mag,
                                                  wise3_magerr,
                                                  luminosities,
                                                  redshifts,
                                                  peak_fluxes,
                                                  cosmo=cosmo,
                                                  exclusive=True,
                                                  peak_flux_threshold=flux_cut_jy)

    logger.info(f'# agn: {redshifts[rlagn_mask].shape[0]} - '
                f'# sfg: {redshifts[sfg_mask].shape[0]} - '
                f'# rqq: {redshifts[rqq_mask].shape[0]} - '
                f'total: {redshifts.shape[0]}')
    logger.debug(f'{np.isnan(wise3_magerr).sum()} wise 3 values are upper limits (NaN)')

    redshifts = redshifts[rlagn_mask]
    fluxes = fluxes[rlagn_mask]
    luminosities = luminosities[rlagn_mask]
    resolved = resolved[rlagn_mask]

    # plot the relationship between L144 and Abs W3 (Fig. 2, H25)
    if plot_rlagn_selection_contour:
        _plot_rlagn_selection_contour(wise2_mag, wise3_mag, redshifts, luminosities, cosmo)
        logger.info("saved lum_vs_w3.png")

    return redshifts, fluxes, luminosities, resolved


def select_rlagn(wise2_mag: np.ndarray,
                 wise3_mag: np.ndarray,
                 wise3_magerr: np.ndarray,
                 luminosities: np.ndarray,
                 redshifts: np.ndarray,
                 peak_flux: np.ndarray,
                 cosmo: astropy.cosmology.Cosmology,
                 exclusive: bool = False,
                 peak_flux_threshold: float = 1.1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selects RLAGN sources based on the criteria from Hardcastle et al. 2025, using WISE magnitudes, luminosities,
    and redshifts.

    Parameters
    ----------
    wise2_mag : np.ndarray
        The WISE W2 magnitudes.
    wise3_mag : np.ndarray
        The WISE W3 magnitudes.
    wise3_magerr : np.ndarray
        The errors in the WISE W3 magnitudes.
    luminosities : np.ndarray
        The luminosities of the sources.
    redshifts : np.ndarray
        The redshifts of the sources.
    peak_flux : np.ndarray
        The peak fluxes of the sources.
    cosmo : astropy.cosmology.Cosmology
        The cosmology to use for calculating luminosity distances and volumes.
    exclusive : bool, optional
        Whether to keep only sources with sufficient data to classify them as SFG or RQQ (True) or to keep sources
        lacking sufficient data (False), by default False.
    peak_flux_threshold : float, optional
        The peak flux threshold below which sources are automatically classified as RLAGN, by default 1.1e-3 Jy.

    Returns
    -------
    rlagn_mask : np.ndarray
        A boolean mask indicating which sources are RLAGN. Sources lacking the WISE/luminosity/redshift data needed to
        classify them as SFG or RQQ are kept by default (exclusive=False) or dropped (exclusive=True), except for the
        low peak-flux / low-redshift override, which always applies.
    sfg_mask : np.ndarray
        A boolean mask indicating which sources are SFG.
    rqq_mask : np.ndarray
        A boolean mask indicating which sources are RQQ.
    """
    wise3_absmag = _get_wise3_absmag(wise3_mag, wise2_mag, redshifts, cosmo)

    # Calculate the SFG exclusion mask based on Hardcastle et al. 2025
    sfg_mask = (luminosities < 10**(14 - wise3_absmag / 2.5)) \
        & (luminosities < 10**(24.8)) & ~np.isnan(wise3_magerr)

    # Calculate the RQQ exclusion criteria based on Hardcastle et al. 2025
    rqq_mask = (luminosities < 10**(-(wise3_absmag - RQQ_XPT) / 3.4844629455909923 + RQQ_YPT)) \
        & (wise3_absmag < -27) & ~np.isnan(wise3_magerr)

    # Everything left is RLAGN
    rlagn_mask = ~sfg_mask & ~rqq_mask

    # Sources without enough WISE/luminosity/redshift data to be classified as SFG or RQQ fall through to ~sfg_mask &
    # ~rqq_mask = True above, i.e. they are kept by default. exclusive flips this default: only sources with the data to
    # positively confirm they're not SFG/RQQ are kept, so undetermined sources are cut.
    if exclusive:
        insufficient_data = np.isnan(wise2_mag) | np.isnan(wise3_mag) | np.isnan(wise3_magerr) \
            | np.isnan(luminosities) | np.isnan(redshifts)
        rlagn_mask = rlagn_mask & ~insufficient_data

    # They also cut out peak fluxes less than or equal to 1.1mjy, and also redshifts lower than or equal to 0.01,
    # regardless of exclusivity, since this cut doesn't depend on the WISE-based classification above.
    survey_selection = (peak_flux > peak_flux_threshold) & (redshifts > 0.01)
    rlagn_mask = rlagn_mask & survey_selection
    sfg_mask = sfg_mask & survey_selection
    rqq_mask = rqq_mask & survey_selection

    return rlagn_mask, sfg_mask, rqq_mask
