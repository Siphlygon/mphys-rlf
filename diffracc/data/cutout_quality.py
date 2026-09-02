"""
Uses the component catalogue for LoTSS-DR2 to flag cutouts that are contaminated by foreign components or cropped by the
frame edge.

A fixed-size cutout centred on one catalogued source frequently contains other catalogued LoTSS sources. Because the
peak-flux S/N only measures the single brightest pixel, such neighbours can influence the S/N cut and also teach a
generative model unphysical, off-target structure; empirically they explain ~81% of the cases where the image-max peak
and the catalogue peak strongly disagree.

Separately, because the cutout is centred on the catalogue position - the fitted-Gaussian peak for a single (`S`) source
but the flux-weighted centroid for multi-component (`M`/`Z`) sources - an asymmetric source can sit off its own
geometric midpoint and spill past the frame edge even though its `LAS` is below the cutout width. We would also want to
avoid teaching a generative model that the source is cropped, so we flag those cases too.

Both checks hang off the component catalogue's `Parent_Source` column, which names the source that each radio component
belongs to:

* :func:`flag_foreign_components` - collects the components inside the cutout footprint, discards those whose parent is
  the cutout's own source, and flags any remaining (foreign) component detected above a S/N threshold.
* :func:`flag_cropped_sources` - takes the cutout source's own components and tests, exactly from each component's
  fitted ellipse (`Maj`/`Min`/`PA`), whether it crosses the frame; it also returns the source's own-emission bounding
  box so a re-centring step can salvage a croppable source.
"""
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

from ..utils import paths
from ..utils.data_utils import Source
from ..utils.logger import LoggingLevels, get_logger

logger = get_logger("Contamination", LoggingLevels.INFO.value)

# Cutout geometry (matches the 80x80, 1.5"/pixel dr2 cutouts).
PIXEL_SCALE_ARCSEC = 1.5
CUTOUT_PIXELS = 80
CUTOUT_SIZE_ARCSEC = CUTOUT_PIXELS * PIXEL_SCALE_ARCSEC  # 120"
# FWHM (in units of Gaussian sigma) - used to turn a peak/rms ratio into an iso-contour extent.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _as_str_array(column) -> np.ndarray:
    """
    Return a whitespace-stripped unicode array for a FITS string column, whether it comes back as bytes (`S`) or unicode
    (`U`).

    Parameters
    ----------
    column : array-like
        A FITS character column (e.g. `Source_Name` or `Parent_Source`).

    Returns
    -------
    np.ndarray
        A stripped unicode (`<U`) array.
    """
    arr = np.asarray(column)
    if arr.dtype.kind == "S":
        arr = np.char.decode(arr, "utf-8")
    return np.char.strip(arr.astype("U"))


def _unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """
    Convert sky coordinates (degrees) to 3D unit vectors so a Euclidean KD-tree gives correct angular nearest neighbours
    (no RA wrap or cos-dec distortion).

    Parameters
    ----------
    ra, dec : np.ndarray
        Right ascension and declination in degrees.

    Returns
    -------
    np.ndarray
        An `(N, 3)` array of unit vectors.
    """
    r = np.radians(ra)
    cos_dec = np.cos(np.radians(dec))
    return np.column_stack([cos_dec * np.cos(r), cos_dec * np.sin(r), np.sin(np.radians(dec))])


def _ellipse_halfwidths(maj: np.ndarray, minr: np.ndarray, pa_deg: np.ndarray):
    """
    Axis-aligned half-widths of an ellipse projected onto the RA and Dec axes.

    For an ellipse with full axes `maj`/`minr` at position angle `pa_deg` (North through East), the extents of its
    bounding box are `d_ra = sqrt((a sin PA)^2 + (b cos PA)^2)` and `d_dec = sqrt((a cos PA)^2 + (b sin PA)^2)`, with
    semi-axes `a = maj/2`, `b = minr/2`. Works element-wise on arrays or scalars.

    Parameters
    ----------
    maj, minr : np.ndarray or float
        Full ellipse axes (e.g. FWHM `Maj`/`Min`) in arcsec.
    pa_deg : np.ndarray or float
        Position angle in degrees, North through East.

    Returns
    -------
    d_ra, d_dec : np.ndarray or float
        The half-widths of the ellipse's axis-aligned bounding box, in the same units as `maj`/`minr`.
    """
    a = maj / 2.0
    b = minr / 2.0
    pa = np.radians(pa_deg)
    d_ra = np.sqrt((a * np.sin(pa)) ** 2 + (b * np.cos(pa)) ** 2)
    d_dec = np.sqrt((a * np.cos(pa)) ** 2 + (b * np.sin(pa)) ** 2)
    return d_ra, d_dec


def load_component_catalogue(component_catalogue_path=paths.COMPONENT_CATALOGUE_PATH) -> dict:
    """
    Load the positions, peak fluxes, fitted-ellipse shapes and parent-source names of every radio component.

    Parameters
    ----------
    component_catalogue_path : Path, optional
        Path to the value-added component catalogue, by default `paths.COMPONENT_CATALOGUE_PATH`.

    Returns
    -------
    dict
        With keys `ra` / `dec` (deg), `peak` (mJy/beam), `maj` / `min` (arcsec FWHM), `pa` (deg, North through East) and
        `parent` (the owning source's `Source_Name`).
    """
    logger.info(f"Loading component catalogue from {component_catalogue_path}...")
    with fits.open(component_catalogue_path, memmap=True) as hdul:
        data = hdul[1].data
        components = {
            "ra": np.asarray(data[Source.RA.value], dtype=float),
            "dec": np.asarray(data[Source.DEC.value], dtype=float),
            "peak": np.asarray(data[Source.PeakFlux.value], dtype=float),  # mJy/beam
            "maj": np.asarray(data["Maj"], dtype=float),                   # arcsec (FWHM)
            "min": np.asarray(data["Min"], dtype=float),                   # arcsec (FWHM)
            "pa": np.asarray(data["PA"], dtype=float),                     # deg, N through E
            "parent": _as_str_array(data["Parent_Source"]),
        }
    logger.info(f"Loaded {len(components['ra'])} components.")
    return components


def _ensure_components(components, component_catalogue_path) -> dict:
    """Load the component catalogue only if a pre-loaded dict was not supplied."""
    return components if components is not None else load_component_catalogue(component_catalogue_path)


def flag_foreign_components(source_ra: np.ndarray,
                            source_dec: np.ndarray,
                            source_name: np.ndarray,
                            isl_rms: np.ndarray,
                            component_catalogue_path=paths.COMPONENT_CATALOGUE_PATH,
                            cutout_size_arcsec: float = CUTOUT_SIZE_ARCSEC,
                            sigma_threshold: float = 5.0,
                            components: dict = None) -> pd.DataFrame:
    """
    Flag each cutout that contains a foreign (neighbour) radio component detected above a S/N threshold, using the
    component catalogue's `Parent_Source` association.

    The cutout is treated as a `cutout_size_arcsec` square, axis-aligned to RA/Dec and centred on the source position
    (verified centred to ~0.6" for the dr2 cutouts). A component contaminates the cutout if any part of its fitted
    ellipse (`Maj`/`Min`/`PA`) overlaps the frame - not merely if its centre is inside it. This matters because a bright
    neighbour whose centre lies just outside the frame can still spill emission across the edge; testing centres alone
    (as an earlier version did) missed those, leaving the pixel-based `edge_max` guard to catch them. A component counts
    as foreign if its `Parent_Source` differs from the cutout source's `Source_Name`, and as detected if its peak flux
    exceeds `sigma_threshold` times the source's island rms.

    Uncatalogued edge features (imaging artefacts, hot pixels, sources missing from the catalogue) are, by construction,
    invisible to this catalogue-based test; a pixel-based guard such as `edge_max` remains the backstop for those.

    Parameters
    ----------
    source_ra, source_dec : np.ndarray
        Cutout-source positions in degrees.
    source_name : np.ndarray
        Cutout-source `Source_Name` values (used to recognise a component's own emission).
    isl_rms : np.ndarray
        Island rms of each source in mJy/beam, used as the local noise for the S/N test.
    component_catalogue_path : Path, optional
        Path to the component catalogue, by default `paths.COMPONENT_CATALOGUE_PATH`.
    cutout_size_arcsec : float, optional
        Side length of the (square) cutout in arcsec, by default 120.0.
    sigma_threshold : float, optional
        Foreign-component detection threshold in units of the source island rms, by default 5.0.
    components : dict, optional
        A pre-loaded component catalogue (see :func:`load_component_catalogue`) to avoid re-reading the file; loaded on
        demand when None.

    Returns
    -------
    pd.DataFrame
        One row per input source (same order), with columns:
        `n_foreign_components` (all foreign components whose ellipse overlaps the frame),
        `n_foreign_detected` (foreign components >= sigma_threshold),
        `brightest_foreign_flux` (mJy/beam),
        `brightest_foreign_snr` (peak flux / island rms),
        `foreign_contaminant` (bool, True if any foreign component is detected).
    """
    source_name = _as_str_array(source_name)
    n = len(source_ra)
    components = _ensure_components(components, component_catalogue_path)
    comp_ra, comp_dec = components["ra"], components["dec"]
    comp_peak, comp_parent = components["peak"], components["parent"]
    comp_maj, comp_min, comp_pa = components["maj"], components["min"], components["pa"]

    logger.info("Building component KD-tree...")
    tree = cKDTree(_unit_vectors(comp_ra, comp_dec))
    half = cutout_size_arcsec / 2.0
    # Search out to the circle circumscribing the cutout PLUS the largest component's half-extent, so no component whose
    # ellipse could reach the frame is missed; the exact ellipse-vs-square test is applied per candidate below.
    max_halfext = float(comp_maj.max()) / 2.0 if len(comp_maj) else 0.0
    radius = np.radians((half * np.sqrt(2.0) + max_halfext) / 3600.0)
    logger.info(f"Querying components within {half * np.sqrt(2.0) + max_halfext:.0f}\" of {n} sources...")
    candidates = tree.query_ball_point(_unit_vectors(source_ra, source_dec), radius)

    n_foreign = np.zeros(n, dtype=np.int32)
    n_detected = np.zeros(n, dtype=np.int32)
    brightest_flux = np.zeros(n, dtype=float)
    brightest_snr = np.zeros(n, dtype=float)

    for i, cand in enumerate(candidates):
        if not cand:
            continue
        cand = np.asarray(cand)
        # Keep components whose fitted ellipse's bounding box overlaps the square cutout (centre may be outside it).
        x0 = (comp_ra[cand] - source_ra[i]) * np.cos(np.radians(source_dec[i])) * 3600.0
        y0 = (comp_dec[cand] - source_dec[i]) * 3600.0
        d_ra, d_dec = _ellipse_halfwidths(comp_maj[cand], comp_min[cand], comp_pa[cand])
        overlaps = ((x0 - d_ra <= half) & (x0 + d_ra >= -half)
                    & (y0 - d_dec <= half) & (y0 + d_dec >= -half))
        cand = cand[overlaps]
        if cand.size == 0:
            continue
        foreign = cand[comp_parent[cand] != source_name[i]]
        if foreign.size == 0:
            continue
        foreign_flux = comp_peak[foreign]
        foreign_snr = foreign_flux / isl_rms[i] if isl_rms[i] > 0 else np.zeros_like(foreign_flux)
        detected = foreign_snr >= sigma_threshold
        n_foreign[i] = foreign.size
        n_detected[i] = int(detected.sum())
        j = int(np.argmax(foreign_flux))
        brightest_flux[i] = foreign_flux[j]
        brightest_snr[i] = foreign_snr[j]

    result = pd.DataFrame({
        "n_foreign_components": n_foreign,
        "n_foreign_detected": n_detected,
        "brightest_foreign_flux": brightest_flux,
        "brightest_foreign_snr": brightest_snr,
        "foreign_contaminant": n_detected > 0,
    })
    logger.info(f"Flagged {int(result['foreign_contaminant'].sum())} / {n} cutouts "
                f"({result['foreign_contaminant'].mean() * 100:.1f}%) as containing a "
                f">={sigma_threshold:g} sigma foreign component.")
    return result


def _group_by_parent(parent: np.ndarray):
    """
    Group component indices by their parent-source name for O(1) lookup of a source's own components.

    Parameters
    ----------
    parent : np.ndarray
        The `Parent_Source` of every component (stripped unicode).

    Returns
    -------
    names : np.ndarray
        The sorted unique parent names.
    starts : np.ndarray
        Start offset of each name's block within `order` (length `len(names) + 1`).
    order : np.ndarray
        Component indices sorted by parent name; `order[starts[k]:starts[k+1]]` are the components of `names[k]`.
    """
    order = np.argsort(parent, kind="stable")
    names, first = np.unique(parent[order], return_index=True)
    starts = np.append(first, len(order))
    return names, starts, order


def flag_cropped_sources(source_ra: np.ndarray,
                         source_dec: np.ndarray,
                         source_name: np.ndarray,
                         isl_rms: np.ndarray,
                         component_catalogue_path=paths.COMPONENT_CATALOGUE_PATH,
                         cutout_size_arcsec: float = CUTOUT_SIZE_ARCSEC,
                         boundary_sigma: float = None,
                         components: dict = None) -> pd.DataFrame:
    """
    Flag cutouts in which the source's own emission is cropped by the frame, from the exact fitted ellipse of each of
    its components - no tunable extent factor.

    Each own component is an elliptical Gaussian of semi-axes `Maj/2`, `Min/2` rotated by `PA`. Its emission is taken
    out to the `boundary_sigma` iso-contour: for a component of peak `P` and local noise `sigma` the Gaussian falls to
    `boundary_sigma * sigma` at a radius that scales the FWHM semi-axes by `sqrt(ln(P/(boundary_sigma * sigma))/ln 2)`.
    This is parameter-free once the (inherited) detection level is fixed - a bright lobe's contour reaches further than
    a faint one's, as it should. Set `boundary_sigma=None` to use the bare FWHM ellipse instead.

    The axis-aligned half-extents of that rotated ellipse are
    `dRA = sqrt((a*sin PA)^2 + (b*cos PA)^2)` and `dDec = sqrt((a*cos PA)^2 + (b*sin PA)^2)`; the component crosses the
    frame when `|x0| + dRA` or `|y0| + dDec` exceeds the cutout half-width.

    Own components are found by `Parent_Source` grouping (not a spatial query), so a component lying outside the current
    frame is still counted - that is exactly the cropped case.

    Parameters
    ----------
    source_ra, source_dec : np.ndarray
        Cutout-source (current) centre positions in degrees.
    source_name : np.ndarray
        Cutout-source `Source_Name` values (matched against components' `Parent_Source`).
    isl_rms : np.ndarray
        Island rms of each source in mJy/beam (the local noise for the iso-contour level).
    component_catalogue_path : Path, optional
        Path to the component catalogue, by default `paths.COMPONENT_CATALOGUE_PATH`.
    cutout_size_arcsec : float, optional
        Side length of the (square) cutout in arcsec, by default 120.0.
    boundary_sigma : float, optional
        The number of standard deviations above the local noise level to use as the iso-contour. By default None, which
        tests the bare FWHM ellipse (`Maj`/``Min`) - the fitted component footprint as PyBDSF reports it. Pass a value
        (e.g. 3.0) to instead extend each component to its peak/rms iso-contour, which additionally captures the faint
        wings of bright components.
    components : dict, optional
        A pre-loaded component catalogue to avoid re-reading the file; loaded on demand when None.

    Returns
    -------
    pd.DataFrame
        One row per input source (same order), with columns:
        `n_own_components`, `cropped` (bool, any own component crosses the frame),
        `own_extent` (arcsec, the larger side of the own-emission bounding box).
    """
    source_name = _as_str_array(source_name)
    n = len(source_ra)
    components = _ensure_components(components, component_catalogue_path)
    comp_ra, comp_dec, comp_peak = components["ra"], components["dec"], components["peak"]
    comp_maj, comp_min, comp_pa, comp_parent = (components["maj"], components["min"],
                                                components["pa"], components["parent"])
    half = cutout_size_arcsec / 2.0

    logger.info("Grouping components by parent source...")
    names, starts, order = _group_by_parent(comp_parent)
    slot = np.searchsorted(names, source_name)

    n_own = np.zeros(n, dtype=np.int32)
    cropped = np.zeros(n, dtype=bool)
    own_extent = np.zeros(n, dtype=float)

    logger.info(f"Testing own-component containment for {n} sources...")
    for i in range(n):
        k = slot[i]
        if k >= len(names) or names[k] != source_name[i]:
            continue  # no catalogued components for this source name
        own = order[starts[k]:starts[k + 1]]
        n_own[i] = own.size

        cos_dec = np.cos(np.radians(source_dec[i]))
        x0 = (comp_ra[own] - source_ra[i]) * cos_dec * 3600.0
        y0 = (comp_dec[own] - source_dec[i]) * 3600.0
        maj = comp_maj[own]
        minr = comp_min[own]
        if boundary_sigma is not None and isl_rms[i] > 0:
            ratio = comp_peak[own] / (boundary_sigma * isl_rms[i])
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.sqrt(np.clip(np.log(ratio) / np.log(2.0), 0.0, None))
            maj = maj * scale
            minr = minr * scale
        d_ra, d_dec = _ellipse_halfwidths(maj, minr, comp_pa[own])

        ra_min, ra_max = (x0 - d_ra).min(), (x0 + d_ra).max()
        dec_min, dec_max = (y0 - d_dec).min(), (y0 + d_dec).max()
        cropped[i] = (ra_min < -half) or (ra_max > half) or (dec_min < -half) or (dec_max > half)
        own_extent[i] = max(ra_max - ra_min, dec_max - dec_min)

    result = pd.DataFrame({
        "n_own_components": n_own,
        "cropped": cropped,
        "own_extent": own_extent,
    })
    logger.info(f"Flagged {int(cropped.sum())} / {n} cutouts ({cropped.mean() * 100:.1f}%) as cropped.")
    return result


def compute_from_catalogues(source_catalogue_path=paths.STRIPPED_CATALOGUE_PATH,
                            component_catalogue_path=paths.COMPONENT_CATALOGUE_PATH,
                            resolved_only: bool = True,
                            cutout_size_arcsec: float = CUTOUT_SIZE_ARCSEC,
                            sigma_threshold: float = 5.0,
                            boundary_sigma: float = None) -> pd.DataFrame:
    """
    Convenience entry point: load the source catalogue once, run both `flag_foreign_components` and
    `flag_cropped_sources`, and return the combined per-source flags aligned to the (resolved) source order used by the
    preprocessing.

    Parameters
    ----------
    source_catalogue_path : Path, optional
        Path to the (raw) source catalogue carrying `Source_Name`, by default `paths.STRIPPED_CATALOGUE_PATH`.
    component_catalogue_path : Path, optional
        Path to the component catalogue, by default `paths.COMPONENT_CATALOGUE_PATH`.
    resolved_only : bool, optional
        Restrict to `Resolved == True` sources (the preprocessing's population), by default True.
    cutout_size_arcsec : float, optional
        Side length of the square cutout in arcsec, by default 120.0.
    sigma_threshold : float, optional
        Foreign-component detection threshold, by default 5.0.
    boundary_sigma : float, optional
        Iso-contour level for the cropping test; by default None (bare FWHM ellipse).

    Returns
    -------
    pd.DataFrame
        The merged foreign and cropping flags with a leading `index` column giving the position within the (resolved)
        source order.
    """
    logger.info(f"Loading source catalogue from {source_catalogue_path}...")
    with fits.open(source_catalogue_path, memmap=True) as hdul:
        data = hdul[1].data
        mask = np.asarray(data[Source.Resolved.value]).astype(bool) if resolved_only \
            else np.ones(len(data), dtype=bool)
        source_ra = np.asarray(data[Source.RA.value][mask], dtype=float)
        source_dec = np.asarray(data[Source.DEC.value][mask], dtype=float)
        isl_rms = np.asarray(data[Source.RMS.value][mask], dtype=float)
        source_name = np.asarray(data["Source_Name"][mask])
    logger.info(f"Loaded {len(source_ra)} sources (resolved_only={resolved_only}).")

    components = load_component_catalogue(component_catalogue_path)
    foreign = flag_foreign_components(source_ra, source_dec, source_name, isl_rms,
                                      cutout_size_arcsec=cutout_size_arcsec,
                                      sigma_threshold=sigma_threshold, components=components)
    cropped = flag_cropped_sources(source_ra, source_dec, source_name, isl_rms,
                                   cutout_size_arcsec=cutout_size_arcsec,
                                   boundary_sigma=boundary_sigma, components=components)
    flags = pd.concat([foreign, cropped], axis=1)
    flags.insert(0, "index", np.arange(len(flags)))
    return flags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=5.0,
                        help="Foreign-component detection threshold (default 5).")
    parser.add_argument("--boundary-sigma", type=float, default=None,
                        help="Iso-contour level for the cropping test; omitted uses the FWHM ellipse.")
    parser.add_argument("--cutout-arcsec", type=float, default=CUTOUT_SIZE_ARCSEC,
                        help=f"Square cutout side length in arcsec (default {CUTOUT_SIZE_ARCSEC:g}).")
    parser.add_argument("--output", type=str,
                        default=str(paths.PREPROCESSING_PARENT / "cutout_quality_flags.csv"),
                        help="Where to write the per-source flag CSV.")
    args = parser.parse_args()

    result = compute_from_catalogues(cutout_size_arcsec=args.cutout_arcsec,
                                     sigma_threshold=args.sigma,
                                     boundary_sigma=args.boundary_sigma)
    result.to_csv(args.output, index=False)
    logger.info(f"Saved contamination + cropping flags to {args.output}")
