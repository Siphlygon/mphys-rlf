"""
Pixel-based flood-fill source sizing, a faithful port of the LoTSS-DR2 / LoMorph size measurement (Mingo et al. 2019;
Hardcastle et al. 2023). It is the pixel-geometry equivalent of `MakeShape` in `angular_size_finder`: where `MakeShape`
measures a size from component ellipses, this measures one from the image pixels of a source above the local noise.

The algorithm is adapted from the `sizeflux` code in the LoTSS catalogue repository while remaining functionally
identical (https://github.com/mhardcastle/lotss-catalogue, `sizeflux/sizeflux_tools.py` and
`sizeflux/lgz-sizeflux-dr2.py`), with the adoption rule from `dr2_catalogue/merge_sizeflux.py`. The LoTSS code is in
turn an adaptation of the original `LoMorph` code (Mingo et al. 2019), which can be found at
(https://github.com/bmingo/LoMorph/).

Per source, `measure_source` thresholds the image, forces the source's own component ellipses in, masks foreign
components out, keeps the connected pixel islands that overlap the source, and reports the largest angular size (max
pairwise pixel separation) and integrated flux of that flood-filled region. Deciding when this pixel size supersedes the
component-based size to become the final `LAS` is decided in `AngularSizeFinder.select_angular_size`.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import label
from scipy.spatial.distance import pdist

# A full 3x3 structuring element counts diagonal neighbours as connected (8-connectivity) when labelling islands.
_EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=int)

# Included component pixels are forced to this (arbitrary, finite, above-threshold) value so the source core always
# survives into the connectivity step even where it dips below the rms threshold. It never enters the flux/size, which
# are taken from a copy of the image made before this substitution.
_INCLUDE_FILL = 0.02


def _mask_ellipse(shape: tuple[int, int],
                  centre_x: float,
                  centre_y: float,
                  major_axis_px: float,
                  minor_axis_px: float,
                  angle_deg: float) -> np.ndarray:
    """
    Return an integer indicator array (1 inside, 0 outside) for a single ellipse.
    
    Note that the LoTSS normalisation of the ellipse is non-standard: the major axis is scaled by the geometric mean of
    the two half-lengths, so the effective major half-axis is sqrt(major*minor)/2. This is preserved from the original
    code so that the pixel selection matches the published measurement.

    Parameters
    ----------
    shape : tuple[int, int]
        The `(n_rows, n_cols)` shape of the image.
    centre_x, centre_y : float
        The ellipse centre in pixel coordinates (column, row), zero-based.
    major_axis_px, minor_axis_px : float
        The ellipse full axes in pixels.
    angle_deg : float
        The ellipse orientation in degrees.

    Returns
    -------
    np.ndarray
        An `(n_rows, n_cols)` array of 0/1 `int`.
    """
    n_rows, n_cols = shape
    pixel_x, pixel_y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    cos_angle = np.cos(angle_deg * np.pi / 180.0)
    sin_angle = np.sin(angle_deg * np.pi / 180.0)

    # The ellipse is a normalised squared radius: each principal-axis offset squared over that axis' squared
    # half-length, summing to <= 1 inside. The major normalisation uses the geometric mean of the two half-lengths (so
    # the effective major half-axis is sqrt(major*minor)/2); this idiosyncrasy is preserved from the original so the
    # pixel selection matches the published measurement.
    major_norm = (major_axis_px / 2.0) * (minor_axis_px / 2.0)
    minor_norm = (minor_axis_px / 2.0) * (minor_axis_px / 2.0)
    major_offset_sq = (cos_angle * (centre_x - pixel_x) + sin_angle * (centre_y - pixel_y)) ** 2.0
    minor_offset_sq = (sin_angle * (centre_x - pixel_x) - cos_angle * (centre_y - pixel_y)) ** 2.0
    normalised_radius_sq = (major_offset_sq / major_norm) + (minor_offset_sq / minor_norm)
    return (normalised_radius_sq <= 1).astype(int)


def _ellipse_coverage(shape: tuple[int, int], wcs: WCS, components: np.ndarray, pixel_scale_deg: float) -> np.ndarray:
    """
    Count, for every pixel, how many component ellipses cover it.

    Parameters
    ----------
    shape : tuple[int, int]
        The `(n_rows, n_cols)` image shape.
    wcs : WCS
        The (2D) world coordinate system of the image, used to place each component by its sky position.
    components : np.ndarray
        An `(n, 5)` array of `(ra_deg, dec_deg, maj_deg, min_deg, pa_deg)` rows. May be empty.
    pixel_scale_deg : float
        The pixel scale (degrees/pixel) used to convert the component axes to pixels.

    Returns
    -------
    np.ndarray
        An `(n_rows, n_cols)` integer array of per-pixel ellipse counts.
    """
    coverage = np.zeros(shape, dtype=int)
    if components is None or len(components) == 0:
        return coverage
    components = np.asarray(components, dtype=float)
    centre_x, centre_y = wcs.wcs_world2pix(components[:, 0], components[:, 1], 1)
    for i in range(len(components)):
        major_axis_px = 2.0 * components[i, 2] / pixel_scale_deg
        minor_axis_px = 2.0 * components[i, 3] / pixel_scale_deg
        coverage += _mask_ellipse(shape, centre_x[i], centre_y[i], major_axis_px, minor_axis_px, components[i, 4]+90.0)
    return coverage


def flood_mask(image: np.ndarray,
               header: fits.Header,
               include: np.ndarray,
               exclude: np.ndarray,
               threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the flood-filled pixel mask for one source.

    The image is thresholded; the source's own (`include`) component ellipses are forced in; foreign (`exclude`)
    component ellipses are masked out; the surviving pixels are grouped into connected islands, and the islands
    overlapping the source's own ellipses are kept.

    Parameters
    ----------
    image : np.ndarray
        The 2D image data (Jy/beam).
    header : astropy.io.fits.Header
        The 2D FITS header, providing the WCS and pixel scale.
    include : np.ndarray
        The source's own components as `(ra, dec, maj_deg, min_deg, pa_deg)` rows.
    exclude : np.ndarray
        Foreign components (same columns) to mask out; may be empty.
    threshold : float
        The pixel threshold (Jy/beam); pixels below it are discarded before flood-filling.

    Returns
    -------
    flooded_mask : np.ndarray
        An `(n_rows, n_cols)` array, 1 inside the flood-filled region and 0 outside.
    flooded_array : np.ndarray
        The thresholded image with everything outside the flood-filled region set to NaN. Kept-region pixels retain
        their original values (the forced-in fill is never written here), so this carries the flux and the pixels the
        size is measured from.
    """
    working_image = np.array(image, dtype=float, copy=True)
    working_image[working_image < threshold] = np.nan

    # Snapshot the thresholded image before any forcing/exclusion: this is what carries the flux and size.
    flooded_array = working_image.copy()

    shape = working_image.shape
    wcs = WCS(header).celestial
    pixel_scale_deg = header['CDELT2']

    include_mask = _ellipse_coverage(shape, wcs, include, pixel_scale_deg)
    working_image[include_mask >= 1] = _INCLUDE_FILL

    exclude_mask = _ellipse_coverage(shape, wcs, exclude, pixel_scale_deg)
    exclude_mask[exclude_mask > 1] = 1
    include_mask[include_mask > 1] = 1

    # Blank pixels covered by a foreign component so a neighbouring source cannot leak into the measurement.
    working_image[exclude_mask == 1] = np.nan

    above_threshold = np.isfinite(working_image).astype(int)
    island_labels, _ = label(above_threshold, structure=_EIGHT_CONNECTIVITY)

    # Keep only the islands that overlap the source's own ellipses (its contiguous emission), discarding the rest.
    source_overlap = island_labels * include_mask
    keep_labels = np.unique(source_overlap[np.nonzero(source_overlap)])

    flooded_mask = np.isin(island_labels, keep_labels).astype(int)
    flooded_array[flooded_mask < 1] = np.nan
    return flooded_mask, flooded_array


def largest_pixel_separation(region: np.ndarray) -> float:
    """
    Largest separation, in pixels, between any two non-NaN pixels of `region`.

    Parameters
    ----------
    region : np.ndarray
        A 2D array whose non-NaN entries are the flood-filled region.

    Returns
    -------
    float
        The maximum pairwise pixel distance, or 0.0 if fewer than two non-NaN pixels are present.
    """
    in_region = ~np.isnan(region)
    if np.count_nonzero(in_region) < 2:
        return 0.0
    n_rows, n_cols = region.shape
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, n_cols - 1, n_cols), np.linspace(0, n_rows - 1, n_rows))
    return float(np.max(pdist(np.column_stack([pixel_x[in_region], pixel_y[in_region]]))))


def _beam_area_pixels(header) -> float:
    """
    Beam area in pixels, `2*pi*beam_major*beam_minor / (2*sqrt(2*ln2))^2` with the axes converted to pixels.

    Parameters
    ----------
    header : astropy.io.fits.Header
        A header carrying `BMAJ`, `BMIN` (degrees) and `CDELT1`.

    Returns
    -------
    float
        The beam area in pixels, used to normalise the summed flux to Jy.
    """
    pixel_scale_deg = abs(header['CDELT1'])
    beam_major_px = header['BMAJ'] / pixel_scale_deg
    beam_minor_px = header['BMIN'] / pixel_scale_deg
    fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
    return 2.0 * np.pi * beam_major_px * beam_minor_px / (fwhm_to_sigma * fwhm_to_sigma)


def _bad_image(flooded_array: np.ndarray) -> bool:
    """
    Flag a flood-fill region as unreliable when fewer than five of its pixels have all four neighbours present and
    positive (i.e. the region is too small or too thin to trust).

    Parameters
    ----------
    flooded_array : np.ndarray
        The flood-filled image (NaN outside the region).

    Returns
    -------
    bool
        True if the region is too small/thin to be a reliable measurement.
    """
    up = np.full_like(flooded_array, np.nan)
    down = np.full_like(flooded_array, np.nan)
    left = np.full_like(flooded_array, np.nan)
    right = np.full_like(flooded_array, np.nan)
    up[1:, :] = flooded_array[0:-1, :]
    down[0:-1, :] = flooded_array[1:, :]
    left[:, 0:-1] = flooded_array[:, 1:]
    right[:, 1:] = flooded_array[:, 0:-1]
    # NaN in any neighbour propagates, so this is only positive where the pixel and all four neighbours are present.
    neighbourhood_mean = (4 * flooded_array + up + down + left + right) / 4
    return bool(np.sum(neighbourhood_mean > 0) < 5)


def measure_source(image: np.ndarray,
                   header: fits.Header,
                   include: np.ndarray,
                   exclude: np.ndarray,
                   rms: float,
                   peak: float,
                   badflux: float) -> dict:
    """
    Measure one source's flood-fill size and flux from an already-extracted cutout.

    Parameters
    ----------
    image : np.ndarray
        The 2D cutout image (Jy/beam).
    header : astropy.io.fits.Header
        The 2D header (WCS, `CDELT1/2`, `BMAJ`, `BMIN`).
    include : np.ndarray
        The source's own components as `(ra, dec, maj_deg, min_deg, pa_deg)` rows.
    exclude : np.ndarray
        Foreign components (same columns) to mask out; may be empty.
    rms : float
        The source island rms in Jy/beam.
    peak : float
        The source peak flux in Jy/beam.
    badflux : float
        The minimum credible flood-fill flux in Jy; below it `Bad_flux` is set.

    Returns
    -------
    dict
        With keys `LM_size` (arcsec), `LM_flux` (Jy), `Bad_flux` (bool) and `Bad_image` (bool).
    """
    # Detect down to 4 sigma, but no deeper than 1/50 of the peak so a bright source's dynamic range is not counted in.
    threshold = max(4.0 * rms, peak / 50.0)
    _, flooded_array = flood_mask(image, header, include, exclude, threshold)

    total_pixel_flux = float(np.nansum(flooded_array))
    size_pixels = largest_pixel_separation(flooded_array)
    lm_flux = total_pixel_flux / _beam_area_pixels(header)
    lm_size = size_pixels * abs(header['CDELT1']) * 3600.0

    return {
        "LM_size": lm_size,
        "LM_flux": lm_flux,
        "Bad_flux": bool(lm_flux < badflux),
        "Bad_image": _bad_image(flooded_array),
    }
