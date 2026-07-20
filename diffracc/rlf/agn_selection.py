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

from ..data.hardcastle_catalogue import HardcastleCatalogue
from ..utils.data_utils import Source
from ..utils.functions import k_corr_factor, mag_to_flux_w2, mag_to_flux_w3
from ..utils.logger import LoggingLevels, get_logger
from ..utils.plotting import Boundary, Population, density_scatter, paper_style

WISE_3_FREQ = 3e8 / 12.1e-6
WISE_2_FREQ = 3e8 / 4.6e-6
logger = get_logger("RLF Catalogue Info", LoggingLevels.INFO.value)


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
    # Does NOT filter specific criteria e.g., flux > 1.1mJy; that's not the job of this function
    # This does not change the behaviour (as it would default to NaN anyway and be caught in select_agn), but it avoids
    # a runtime warning from numpy about log of non-finite values.
    valid = np.isfinite(redshifts) & (redshifts > -1) \
            & np.isfinite(wise3_mag) & (wise3_mag > 0) \
            & np.isfinite(wise2_mag) & (wise2_mag > 0)

    wise3_absmag = np.full(wise3_mag.shape, np.nan)

    # astropy's cosmology methods fall back to a np.vectorize-wrapped path whenever there's no closed-form comoving
    # distance (e.g. Tcmb0 != 0 adds a radiation term) - and np.vectorize explicitly refuses a zero-length input rather
    # than just returning an empty array. `valid` is all-False on every call whose single/every source fails the
    # redshift/magnitude checks above - guaranteed to happen from apply_preprocessing's per-source iterative flag path,
    # where each call passes exactly one source
    if not np.any(valid):
        # Can exit here as the next calculations would just return NaN anyway
        return wise3_absmag[0] if scalar_input else wise3_absmag

    # Extract the WISE frequencies
    wise3_flux = np.full(wise3_mag.shape, np.nan)
    wise2_flux = np.full(wise2_mag.shape, np.nan)
    wise3_flux[valid] = mag_to_flux_w3(wise3_mag[valid])
    wise2_flux[valid] = mag_to_flux_w2(wise2_mag[valid])

    # Calculate the spectral indices for the sources for a k-correction
    spectral_inds = np.full(wise3_mag.shape, np.nan)
    spectral_inds[valid] = -np.log(wise3_flux[valid] / wise2_flux[valid]) / np.log(WISE_3_FREQ / WISE_2_FREQ)
    logger.debug(f"average spectral index: {np.nanmean(spectral_inds):.4f}, std: {np.nanstd(spectral_inds):.4f}")

    # Note; the -ve here is necessary despite the negative in the mag_space return of k_corr, because we calculate
    # the factor as (1-alpha), whereas H26 uses (alpha-1)
    wise3_absmag[valid] = (
        wise3_mag[valid]
        - 5 * (np.log10(cosmo.luminosity_distance(redshifts[valid]).to(u.parsec).value) - 1)
        - k_corr_factor(redshifts[valid], mag_space=True, spectral_index=spectral_inds[valid])
    )
    return wise3_absmag[0] if scalar_input else wise3_absmag


def _plot_rlagn_selection_contour(wise2_mag: np.ndarray,
                                  wise3_mag: np.ndarray,
                                  redshifts: np.ndarray,
                                  luminosities: np.ndarray,
                                  cosmo: astropy.cosmology.Cosmology,
                                  rlagn_mask: np.ndarray | None = None,
                                  sfg_mask: np.ndarray | None = None,
                                  rqq_mask: np.ndarray | None = None,
                                  output: str = 'lum_vs_w3.png'):
    """
    Plot L144 vs absolute W3 magnitude (Fig. 2, H25) for RLAGN selection: a greyscale density of all sources, the SF
    and RQQ exclusion boundaries, and - when the classification masks are supplied - the excluded SFG and RQQ
    populations scattered on top. Rendered through diffracc.utils.plotting so it shares the project's paper style.

    Parameters
    ----------
    wise2_mag, wise3_mag : np.ndarray
        Apparent WISE W2 and W3 magnitudes.
    redshifts : np.ndarray
        Source redshifts.
    luminosities : np.ndarray
        144-MHz luminosities in W/Hz.
    cosmo : astropy.cosmology.Cosmology
        Cosmology for the absolute-magnitude conversion.
    rlagn_mask, sfg_mask, rqq_mask : np.ndarray | None
        Boolean masks (over the full input arrays) of the RLAGN, SFG- and RQQ-classified sources. Each supplied mask is
        overlaid as a coloured, sub-sampled scatter population; the greyscale hexbin behind them is the density of all
        sources.
    output : str
        Path to save the figure to.
    """
    logger.info("Plotting L144 vs absolute W3 for RLAGN selection...")

    cond = np.isfinite(wise3_mag) & np.isfinite(wise2_mag) & np.isfinite(redshifts) & np.isfinite(luminosities)
    wise3_absmag = np.full(wise3_mag.shape, np.nan)
    wise3_absmag[cond] = _get_wise3_absmag(wise3_mag[cond], wise2_mag[cond], redshifts[cond], cosmo)
    valid = np.isfinite(wise3_absmag) & np.isfinite(luminosities)

    # Analytic selection boundaries (see select_rlagn). The SF line spans the whole panel (it re-intersects the RQQ
    # line towards the right); the RQQ line runs left from W3 = -27; and a vertical dotted edge at W3 = -27 drops to
    # the x-axis, marking where the RQQ exclusion zone begins.
    y_bottom = 4e20
    w3_sf = np.linspace(-18, -34, 500)
    w3_rqq = np.linspace(-27, -34, 500)
    boundaries = [
        Boundary(w3_sf, 10**(14 - w3_sf / 2.5), label='SF exclusion', linestyle='-'),
        Boundary(w3_rqq, 10**(25.3 + (-w3_rqq - 27) * 2.0/7), label='RQQ exclusion', linestyle=':'),
        Boundary(np.array([-27.0, -27.0]), np.array([y_bottom, 10**25.3]), linestyle=':'), # connecting RQQ to x-axis
    ]

    # Each population is drawn as a smooth filled density contour in its own colour (the H25 "blobs") with only a
    # sparse scatter of individual sources on top, so the dense centre stays legible.
    populations = []
    if rlagn_mask is not None:
        m = valid & rlagn_mask
        populations.append(Population('RLAGN', wise3_absmag[m], luminosities[m],
                                      color='#5e3c99', shade=True, max_scatter=400, size=10, alpha=0.9))
    if sfg_mask is not None:
        m = valid & sfg_mask
        populations.append(Population('SFG (excluded)', wise3_absmag[m], luminosities[m],
                                      color='#2c7fb8', shade=True, max_scatter=400, size=10, alpha=0.9))
    if rqq_mask is not None:
        m = valid & rqq_mask
        populations.append(Population('RQQ (excluded)', wise3_absmag[m], luminosities[m],
                                      color='#d95f02', shade=True, max_scatter=400, size=10, alpha=0.9))

    with paper_style():
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        density_scatter(
            ax, wise3_absmag[valid], luminosities[valid], gridsize=100,
            populations=populations, boundaries=boundaries, density='hexbin',
            xlabel='Absolute $W3$ magnitude', ylabel='$L_{144}$ (W Hz$^{-1}$)',
            xlim=(-18, -34), ylim=(y_bottom, 1e29), ylog=True,
            legend_loc='lower right', title='RLAGN selection (H25)',
        )
        fig.savefig(output)
        plt.show()
        plt.close(fig)
    logger.info(f"saved {output}")


def get_catalogue_info(cosmo: astropy.cosmology.Cosmology,
                       plot_rlagn_selection_contour: bool = False) \
                         -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get information about the catalogue data.

    Parameters
    ----------
    cosmo : astropy.cosmology.Cosmology
        The cosmology to use for calculating luminosity distances and volumes
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
    catalogue = HardcastleCatalogue(resolved_only=False)

    logger.info("Getting redshifts, fluxes, magnitudes, and luminosities from the catalogue...")
    redshifts = catalogue.get_value_column(Source.Redshift)  # z, unitless
    tot_fluxes = catalogue.get_value_column(Source.TotalFlux) / 1000  # catalogue stores mJy; RLF expects Jy throughout
    luminosities = catalogue.get_value_column(Source.Luminosity)  # in W/Hz
    wise1_mag = catalogue.get_value_column(Source.WISE1Mag)  # in mag
    wise2_mag = catalogue.get_value_column(Source.WISE2Mag)  # in mag
    wise3_mag = catalogue.get_value_column(Source.WISE3Mag)  # in mag
    wise3_magerr = catalogue.get_value_column(Source.WISE3MagErr)  # in mag
    resolved = catalogue.get_value_column(Source.Resolved)  # boolean

    logger.info(f"Total sources in catalogue: {redshifts.shape[0]}")

    logger.debug(f'wise3_mag: mean={np.average(wise3_mag[np.isfinite(wise3_mag)]):.4f}, '
                 f'std={np.std(wise3_mag[np.isfinite(wise3_mag)]):.4f}, '
                 f'max={np.max(wise3_mag[np.isfinite(wise3_mag)]):.4f}, '
                 f'min={np.min(wise3_mag[np.isfinite(wise3_mag)]):.4f}, '
                 f'count={wise3_mag.shape[0]}')
    logger.debug(f'wise2_mag: mean={np.average(wise2_mag[np.isfinite(wise2_mag)]):.4f}, '
                 f'std={np.std(wise2_mag[np.isfinite(wise2_mag)]):.4f}, '
                 f'max={np.max(wise2_mag[np.isfinite(wise2_mag)]):.4f}, '
                 f'min={np.min(wise2_mag[np.isfinite(wise2_mag)]):.4f}, '
                 f'count={wise2_mag.shape[0]}')

    # Calculate the RLAGN, SFG, and RQQ masks using the selection criteria from Hardcastle et al. 2025
    rlagn_mask, sfg_mask, rqq_mask = select_rlagn(wise1_mag,
                                                  wise2_mag,
                                                  wise3_mag,
                                                  wise3_magerr,
                                                  luminosities,
                                                  redshifts,
                                                  tot_fluxes,
                                                  cosmo=cosmo,
                                                  exclusive=False)
    num_rlagn = np.sum(rlagn_mask)
    num_sfg = np.sum(sfg_mask)
    num_rqq = np.sum(rqq_mask)
    logger.info(f'# agn: {num_rlagn} - '
                f'# sfg: {num_sfg} - '
                f'# rqq: {num_rqq} - '
                f'total: {num_rlagn + num_sfg + num_rqq} -')
    logger.debug(f'{(np.isnan(wise3_magerr) & np.isfinite(wise3_mag)).sum()} wise 3 values are upper limits')

    # Plot L144 vs absolute W3 (Fig. 2, H25) with the excluded SFG/RQQ populations overlaid on the selection cuts.
    if plot_rlagn_selection_contour:
        _plot_rlagn_selection_contour(wise2_mag, wise3_mag, redshifts, luminosities, cosmo,
                                      rlagn_mask=rlagn_mask, sfg_mask=sfg_mask, rqq_mask=rqq_mask)

    redshifts = redshifts[rlagn_mask]
    tot_fluxes = tot_fluxes[rlagn_mask]
    luminosities = luminosities[rlagn_mask]
    resolved = resolved[rlagn_mask]

    return redshifts, tot_fluxes, luminosities, resolved


def select_rlagn(wise1_mag: np.ndarray | float,
                 wise2_mag: np.ndarray | float,
                 wise3_mag: np.ndarray | float,
                 wise3_magerr: np.ndarray | float,
                 luminosities: np.ndarray | float,
                 redshifts: np.ndarray | float,
                 tot_fluxes: np.ndarray | float,
                 cosmo: astropy.cosmology.Cosmology,
                 exclusive: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selects RLAGN sources based on the criteria from Hardcastle et al. 2025, using WISE magnitudes, luminosities,
    and redshifts.

    Parameters
    ----------
    wise1_mag : np.ndarray
        The WISE W1 magnitudes.
    wise2_mag : np.ndarray
        The WISE W2 magnitudes.
    wise3_mag : np.ndarray
        The WISE W3 magnitudes.
    wise3_magerr : np.ndarray
        The errors in the WISE W3 magnitudes.
    luminosities : np.ndarray
        The luminosities of the sources, in W/Hz.
    redshifts : np.ndarray
        The redshifts of the sources.
    tot_fluxes : np.ndarray
        The total fluxes of the sources, in Jy.
    cosmo : astropy.cosmology.Cosmology
        The cosmology to use for calculating luminosity distances and volumes.
    exclusive : bool, optional
        Whether to keep only sources with sufficient data (i.e., non-NaN wise3_magerr) to classify them as SFG or RQQ
        (True) or to keep sources lacking sufficient data (i.e., NaN wise3_magerr) (False), by default False.

    Returns
    -------
    rlagn_mask : np.ndarray
        A boolean mask indicating which sources are RLAGN. Sources lacking the WISE-band 3 magnitude error data needed
        to classify them as SFG or RQQ are kept by default (exclusive=False) or dropped (exclusive=True). Removes
        sources with NaNs in other values, or below the flux/redshift thresholds, regardless of exclusivity.
    sfg_mask : np.ndarray
        A boolean mask indicating which sources are SFG.
    rqq_mask : np.ndarray
        A boolean mask indicating which sources are RQQ.
    """
    # float guard
    wise1_mag = np.atleast_1d(wise1_mag)
    wise2_mag = np.atleast_1d(wise2_mag)
    wise3_mag = np.atleast_1d(wise3_mag)
    wise3_magerr = np.atleast_1d(wise3_magerr)
    luminosities = np.atleast_1d(luminosities)
    redshifts = np.atleast_1d(redshifts)
    tot_fluxes = np.atleast_1d(tot_fluxes)

    # Perform all initial filters H25 uses before they apply the WISE-based classification i.e., obtain their sample
    sample = (tot_fluxes > 1.1e-3) & (redshifts > 0.01) & (np.isfinite(luminosities)) \
        & (np.isfinite(wise1_mag)) & (np.isfinite(wise2_mag)) & (np.isfinite(wise3_mag))

    wise3_absmag = np.full(wise3_mag.shape, np.nan)
    wise3_absmag[sample] = _get_wise3_absmag(wise3_mag[sample],
                                             wise2_mag[sample],
                                             redshifts[sample],
                                             cosmo)
    logger.debug("number of sources with finite wise3_absmag "
                 f": {np.isfinite(wise3_absmag).sum()} / {wise3_absmag.shape[0]}")

    # Calculate the SFG exclusion mask based on Hardcastle et al. 2025
    sfg_mask = np.full(wise3_absmag.shape, False)
    sfg_mask[sample] = (luminosities[sample] < 10**(14 - wise3_absmag[sample] / 2.5)) \
        & (luminosities[sample] < 10**(24.8))
    logger.debug(f"number of sources initially classified as SFG: {sfg_mask.sum()} / {sfg_mask.shape[0]}")

    # Calculate the RQQ exclusion criteria based on Hardcastle et al. 2025
    rqq_mask = np.full(wise3_absmag.shape, False)
    rqq_mask[sample] = (luminosities[sample] < 10**(25.3 + (-wise3_absmag[sample] - 27) * 2.0/7)) \
        & (wise3_absmag[sample] < -27) & (luminosities[sample] >= 10**(24.8))
    logger.debug(f"number of sources initially classified as RQQ: {rqq_mask.sum()} / {rqq_mask.shape[0]}")

    # Sources with insufficient data to classify as SFG or RQQ (i.e., NaN wise3_magerr) are kept by default, but can be
    # dropped if exclusive=True. This will remove them from the SFG and RQQ masks and therefore classify them as RLAGN
    # by default therefore increasing the number of RLAGN sources.
    if not exclusive:
        insufficient_data = ~np.isfinite(wise3_magerr)
        logger.debug("number of sources with insufficient data to classify as SFG or RQQ: "
                     f"{insufficient_data.sum()} / {insufficient_data.shape[0]}")
        sfg_mask = sfg_mask & ~insufficient_data
        rqq_mask = rqq_mask & ~insufficient_data

    # Everything left is RLAGN
    rlagn_mask = np.full(wise3_absmag.shape, False)
    rlagn_mask[sample] = ~sfg_mask[sample] & ~rqq_mask[sample]

    return rlagn_mask, sfg_mask, rqq_mask
