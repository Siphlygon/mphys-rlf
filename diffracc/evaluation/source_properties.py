"""
Per-image physical source properties for radio cut-outs.

A fast, dependency-light source finder used to turn an image (in physical Jy/beam) into a vector of interpretable
quantities: peak/total flux, background rms, S/N, number of components, size and concentration. These feed both the 1-D
distribution comparisons and the multivariate physical FID/KID.

The default backend is pure numpy/scipy (plus astropy's sigma-clipped stats if available, to match the paper's S/N
definition). A PyBDSF backend can be substituted for publication-grade source finding via :func:`extract_batch`'s
``extractor`` argument.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

# Order of scalar properties returned per image.
PROPERTY_KEYS = [
    "peak", "total_flux", "rms", "snr", "n_components", "source_area", "extent", "concentration", "bg_median",
]

# Features used for the multivariate physical FID/KID. Flux-like quantities are log-scaled (they span OoMs);
# counts/sizes/ratios are used linearly. See :func:`feature_matrix`.
FEATURE_KEYS = ["peak", "total_flux", "rms", "snr", "n_components", "extent", "concentration"]
_LOG_FEATURES = {"peak", "total_flux", "rms", "snr"}



# ---------- UTILITIES FOR EXTRACTING PROPERTIES ----------
def as_image_stack(images: np.ndarray) -> np.ndarray:
    """
    Coerce input to a float32 stack of shape (N, H, W), squeezing a channel axis if present.
    
    Parameters
    ----------
    images : array_like
        Input images, either a single (H, W) image, or a stack of shape (N, H, W) or (N, 1, H, W).
    
    Returns
    -------
    np.ndarray
        Stack of images with shape (N, H, W) and dtype float32.
    """
    arr = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:  # single image
        arr = arr[None]
    if arr.ndim == 4 and arr.shape[1] == 1:  # (N, 1, H, W) -> (N, H, W)
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image stack of shape (N,H,W) or (N,1,H,W); got {arr.shape}.")
    return arr


def _background_stats(img: np.ndarray) -> tuple[float, float]:
    """
    Robust background (median, sigma). Uses astropy sigma-clipping if appropriate, else median absolute deviation.
    
    Parameters
    ----------
    img : np.ndarray
        Input image array.
    
    Returns
    -------
    tuple[float, float]
        Background median and sigma.
    """
    try:
        _, med, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
        return float(med), float(std)
    except Exception:
        med = np.median(img)
        return float(med), float(1.4826 * np.median(np.abs(img - med)))


@dataclass
class SourceProperties:
    """
    Container for per-image source properties.
    
    Attributes
    ----------
    peak : float
        Peak flux in Jy/beam.
    total_flux : float
        Total integrated flux in Jy.
    rms : float
        Background RMS in Jy/beam.
    snr : float
        Signal-to-noise ratio (peak - background median) / rms.
    n_components : int
        Number of connected components in the source mask.
    source_area : int
        Area of the source mask in pixels.
    extent : float
        Flux-weighted second-moment size (LAS-like extent) in pixels.
    concentration : float
        Peak-to-integrated flux ratio, a measure of compactness.
    bg_median : float
        Background median in Jy/beam.
    """
    peak: float
    total_flux: float
    rms: float
    snr: float
    n_components: int
    source_area: int
    extent: float
    concentration: float
    bg_median: float


    def to_dict(self) -> dict[str, float]:
        """
        Convert the source properties to a dictionary.
        
        Returns
        -------
        dict[str, float]
            Dictionary of source properties keyed by :data:`PROPERTY_KEYS`.
        """
        return {key: getattr(self, key) for key in PROPERTY_KEYS}


    def to_array(self, keys: list[str] | None = None) -> np.ndarray:
        """
        Convert the source properties to a numpy array.
        
        Parameters
        ----------
        keys : list[str], optional
            Subset of keys to include in the array. Defaults to :data:`PROPERTY_KEYS`.
        
        Returns
        -------
        np.ndarray
            Array of source properties in the order of the specified keys.
        """
        keys = keys or PROPERTY_KEYS
        return np.array([getattr(self, key) for key in keys], dtype=float)


    @staticmethod
    def stack(props: Sequence[SourceProperties], keys: list[str] | None = None) -> dict[str, np.ndarray]:
        """
        Collate a batch of instances into a dict of arrays, one per key.

        This is the array-oriented view used for per-property vectorised statistics (e.g. W1/KS tests across a batch);
        for a per-image feature matrix use :func:`feature_matrix` instead.

        Parameters
        ----------
        props : sequence of SourceProperties
            Per-image properties, e.g. as returned by :func:`extract_batch`.
        keys : list[str], optional
            Subset of keys to collate. Defaults to :data:`PROPERTY_KEYS`.

        Returns
        -------
        dict[str, np.ndarray]
            Arrays of shape (N,), one per requested key.
        """
        keys = keys or PROPERTY_KEYS
        return {key: np.array([getattr(p, key) for p in props], dtype=float) for key in keys}


    @classmethod
    def from_image(cls, image: np.ndarray, nsigma: float = 5.0) -> SourceProperties:
        """
        Extract source properties from a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image array.
        nsigma : float, optional
            Detection threshold in units of the background sigma above the background median, by default 5.0.

        Returns
        -------
        SourceProperties
            Instance containing the extracted source properties.
        """
        return extract_properties(image, nsigma=nsigma)

    @classmethod
    def from_batch(
        cls,
        images: np.ndarray,
        nsigma: float = 5.0,
        extractor: Callable[..., SourceProperties] | None = None
        ) -> list[SourceProperties]:
        """
        Extract source properties from a batch of images.

        Parameters
        ----------
        images : np.ndarray
            Stack of images of shape (N, H, W) or (N, 1, H, W).
        nsigma : float, optional
            Detection threshold in units of the background sigma above the background median, by default 5.0.
        extractor : callable, optional
            Per-image extractor returning a :class:`SourceProperties` instance. Defaults to
            :func:`extract_properties`. Provide a PyBDSF-backed callable here for publication-grade source finding.

        Returns
        -------
        list[SourceProperties]
            One instance per image in the batch, in input order.
        """
        return extract_batch(images, nsigma=nsigma, extractor=extractor)


# ---------- PER-IMAGE PROPERTY EXTRACTION ----------
def extract_properties(image: np.ndarray, nsigma: float = 5.0) -> SourceProperties:
    """
    Extract physical properties from a single image (in physical Jy/beam).

    Parameters
    ----------
    image : array_like
        A single (H, W) image, or (1, H, W).
    nsigma : float, optional
        Detection threshold in units of the background sigma above the background median, by default 5.0. Pixels above
        ``median + nsigma * sigma`` form the source mask.

    Returns
    -------
    SourceProperties
        A SourceProperties instance containing the extracted properties.
    """
    img = as_image_stack(image)[0]
    med, std = _background_stats(img)
    peak = float(img.max())
    rms = float(std)
    snr = float(peak - med) / rms if rms > 0 else np.nan

    mask = img > (med + nsigma * std)
    # If no pixels are above the threshold, return zeroed properties with NaN concentration.
    if not mask.any():
        return SourceProperties(peak=peak, total_flux=0.0, rms=rms, snr=snr, n_components=0,
                                source_area=0, extent=0.0, concentration=np.nan, bg_median=med)

    total_flux = float((img[mask] - med).sum())  # background-subtracted integrated flux
    _, n_components = ndimage.label(mask)  # number of connected components in the source mask
    source_area = int(mask.sum())

    # Flux-weighted second-moment size (a LAS-like extent, in pixels).
    # Compute the flux-weighted centroid and variance, then take the square root of the variance to get the extent.
    ys, xs = np.nonzero(mask)
    w = np.clip(img[mask] - med, 0.0, None)
    wsum = float(w.sum()) + 1e-30
    cy = float((w * ys).sum() / wsum)
    cx = float((w * xs).sum() / wsum)
    var = float((w * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum() / wsum)
    extent = float(2.0 * np.sqrt(max(var, 0.0)))
    concentration = float((peak - med) / (total_flux + 1e-30))  # peak / integrated ~ compactness

    return SourceProperties(peak=peak, total_flux=total_flux, rms=rms, snr=snr, n_components=n_components,
                            source_area=source_area, extent=extent, concentration=concentration, bg_median=med)


def extract_batch(
    images: np.ndarray, nsigma: float = 5.0, extractor: Callable[..., SourceProperties] | None = None
) -> list[SourceProperties]:
    """
    Extract properties for a stack of images.

    Parameters
    ----------
    images : array_like
        Stack of shape (N, H, W) or (N, 1, H, W), in physical Jy/beam.
    nsigma : float, optional
        Detection threshold, by default 5.0.
    extractor : callable, optional
        Per-image extractor returning a :class:`SourceProperties` instance. Defaults to :func:`extract_properties`.
        Provide a PyBDSF-backed callable here for LOFAR appropriate source finding.

    Returns
    -------
    list[SourceProperties]
        One instance per image in the batch, in input order.
    """
    extractor = extractor or extract_properties
    stack = as_image_stack(images)
    return [extractor(img, nsigma=nsigma) for img in stack]


def feature_matrix(props: Sequence[SourceProperties], keys: list[str] | None = None) -> np.ndarray:
    """
    Assemble the (N, D) feature matrix for the multivariate physical FID/KID.

    Flux-like features are log10-scaled (with a small positive floor, since detections are positive) so their OoMs of
    range don't dominate; other features are used linearly. Rows with any non-finite feature are dropped.

    Parameters
    ----------
    props : sequence of SourceProperties
        Per-image properties, e.g. as returned by :func:`extract_batch`.
    keys : list of str, optional
        Subset of keys to include in the feature matrix. Defaults to :data:`FEATURE_KEYS`.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, D) with finite rows only.
    """
    keys = keys or FEATURE_KEYS
    if not props:
        return np.empty((0, len(keys)))

    mat = np.stack([p.to_array(keys) for p in props])
    log_cols = [i for i, k in enumerate(keys) if k in _LOG_FEATURES]
    if log_cols:
        mat[:, log_cols] = np.log10(np.clip(mat[:, log_cols], 1e-12, None))
    return mat[np.isfinite(mat).all(axis=1)]
