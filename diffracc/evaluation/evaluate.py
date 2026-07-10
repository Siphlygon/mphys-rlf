"""
Scientific evaluation reports for the diffusion model's outputs and performance.

Three headline checks:

* :func:`physical_distribution_report` -- do the physical summary statistics of the generated set match a held-out real
  set (per-quantity W1/KS + a multivariate physical FID/KID)?
* :func:`calibration_report` -- does the peak-flux conditioning do what it claims (recovered vs prompted), and over what
  range is it reliable?
* :func:`memorization_report` -- is the model generating, or reproducing training images?

:func:`full_report` runs all three. All functions take image stacks in **physical Jy/beam** (invert any global flux
transform before calling; see :mod:`diffracc.data.flux_transforms`).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..utils.logger import get_logger
from . import metrics
from .source_properties import (
    FEATURE_KEYS,
    PROPERTY_KEYS,
    SourceProperties,
    as_image_stack,
    extract_batch,
    feature_matrix,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------------------
# 1. Physical distribution match
# --------------------------------------------------------------------------------------------------
def physical_distribution_report(
    generated,
    real,
    nsigma: float = 5.0,
    feature_keys: list[str] | None = None,
    kid_degree: int = 3,
) -> dict:
    """
    Compare the physical-property distributions of generated vs real images.

    Returns per-property Wasserstein-1 and KS statistics, plus a multivariate physical Fréchet distance (FID) and kernel
    distance (KID) computed on the standardised physical feature vector.
    
    Parameters
    ----------
    generated : array_like
        Generated image stack in physical Jy/beam.
    real : array_like
        Real image stack in physical Jy/beam.
    nsigma : float, optional
        Detection threshold for the source finder, by default 5.0.
    feature_keys : list[str] | None, optional
        List of physical feature keys to include in the multivariate FID/KID computation. If ``None``, all features in
        :data:`~diffracc.evaluation.source_properties.FEATURE_KEYS` are used, by default None.
    kid_degree : int, optional
        Degree of the polynomial kernel for the KID computation, by default 3.
    
    Returns
    -------
    dict
        Dictionary containing the number of generated and real images, per-property statistics, and the multivariate
        physical FID and KID.
    """
    gen_props = extract_batch(generated, nsigma=nsigma)
    real_props = extract_batch(real, nsigma=nsigma)
    gen_stats = SourceProperties.stack(gen_props)
    real_stats = SourceProperties.stack(real_props)

    per_property = {}
    for key in PROPERTY_KEYS:
        g, r = gen_stats[key], real_stats[key]
        ks_stat, ks_p = metrics.ks_2samp(g, r)
        per_property[key] = {
            "w1": metrics.wasserstein_1d(g, r),
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "gen_median": float(np.nanmedian(g)),
            "real_median": float(np.nanmedian(r)),
        }

    keys = feature_keys or FEATURE_KEYS
    gen_feat = feature_matrix(gen_props, keys)
    real_feat = feature_matrix(real_props, keys)
    real_std, gen_std = metrics.standardise(real_feat, gen_feat)  # scaler fit on real

    return {
        "n_generated": int(as_image_stack(generated).shape[0]),
        "n_real": int(as_image_stack(real).shape[0]),
        "per_property": per_property,
        "physical_fid": metrics.frechet_distance(gen_std, real_std),
        "physical_kid": metrics.kernel_distance(gen_std, real_std, degree=kid_degree),
        "feature_keys": list(keys),
    }


# --------------------------------------------------------------------------------------------------
# 2. Conditioning calibration (recovered vs prompted)
# --------------------------------------------------------------------------------------------------
def calibration_report(
    generated,
    prompted_peak,
    nsigma: float = 5.0,
    n_bins: int = 12,
) -> dict:
    """
    Compare the recovered peak flux of generated images against the prompted (conditioned) value.
    
    Acts as a calibration check for the conditioning mechanism, and returns a binned reliability curve to locate the
    range over which the conditioning is reliable.

    Parameters
    ----------
    generated : array_like
        Generated image stack in physical Jy/beam.
    prompted_peak : array_like
        The physical peak flux (Jy/beam) each image was conditioned on, length N.
    nsigma : float, optional
        Detection threshold for the source finder, by default 5.0.
    n_bins : int, optional
        Number of logarithmic prompted-flux bins for the binned reliability curve, by default 12.

    Returns
    -------
    dict
        Global fit (slope/intercept/R^2 in log-log), multiplicative bias, dex scatter, and a binned median/scatter curve
        for locating where the relation diverges (your reliable range).
    """
    props = extract_batch(generated, nsigma=nsigma)
    recovered = np.array([p.peak for p in props], dtype=float)
    prompted = np.asarray(prompted_peak, float).ravel()
    if prompted.shape[0] != recovered.shape[0]:
        raise ValueError(f"prompted_peak length {prompted.shape[0]} != n images {recovered.shape[0]}.")

    ok = np.isfinite(recovered) & np.isfinite(prompted) & (recovered > 0) & (prompted > 0)
    lp, lr = np.log10(prompted[ok]), np.log10(recovered[ok])

    # Global log-log linear fit; ideal is slope 1, intercept 0.
    slope, intercept = np.polyfit(lp, lr, 1)
    resid = lr - (slope * lp + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((lr - lr.mean()) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot

    log_ratio = lr - lp  # log10(recovered / prompted)
    bias = float(10 ** np.median(log_ratio))     # multiplicative bias (1.0 = unbiased)
    scatter_dex = float(np.std(log_ratio))       # scatter in dex

    # Binned reliability curve.
    edges = np.linspace(lp.min(), lp.max(), n_bins + 1)
    idx = np.clip(np.digitize(lp, edges) - 1, 0, n_bins - 1)
    centers, med_ratio, scat, counts = [], [], [], []
    for b in range(n_bins):
        sel = idx == b
        centers.append(10 ** (0.5 * (edges[b] + edges[b + 1])))
        counts.append(int(sel.sum()))
        if sel.any():
            med_ratio.append(float(10 ** np.median(log_ratio[sel])))
            scat.append(float(np.std(log_ratio[sel])))
        else:
            med_ratio.append(np.nan); scat.append(np.nan)

    return {
        "n_used": int(ok.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "bias": bias,
        "scatter_dex": scatter_dex,
        "prompted": prompted,
        "recovered": recovered,
        "bin_centers": np.array(centers),
        "bin_median_ratio": np.array(med_ratio),
        "bin_scatter_dex": np.array(scat),
        "bin_counts": np.array(counts),
    }


# --------------------------------------------------------------------------------------------------
# 3. Memorisation check
# --------------------------------------------------------------------------------------------------
def _memorisation_vectors(images, downsample: int) -> np.ndarray:
    """
    Peak-normalise each image (so structural copying is detected regardless of amplitude), optionally block-average to
    (downsample, downsample), and flatten.
    
    Used for the nearest-neighbour memorisation check, where we want to detect whether generated images are 
    near-duplicates of training images. Downsampling is used to reduce the dimensionality of the pixel space and to make
    the nearest-neighbour search more robust to small shifts or rotations.
    
    Parameters
    ----------
    images : array_like
        Image stack in physical Jy/beam.
    downsample : int
        Downsample each image to (downsample, downsample) by block-averaging.
    
    Returns
    -------
    np.ndarray
        Array of shape (N, downsample**2) containing the processed image vectors.
    """
    stack = as_image_stack(images)
    peaks = stack.max(axis=(1, 2), keepdims=True)
    stack = stack / np.where(peaks > 0, peaks, 1.0)
    if downsample and downsample < stack.shape[1]:
        n, h, w = stack.shape
        f = h // downsample
        if f > 1:
            stack = stack[:, : f * downsample, : f * downsample]
            stack = stack.reshape(n, downsample, f, downsample, f).mean(axis=(2, 4))
    return stack.reshape(stack.shape[0], -1)


def memorization_report(
    generated,
    train,
    val = None,
    downsample: int = 20,
) -> dict:
    """
    Check whether generated images are near-duplicates of training images.

    For each generated image, find its nearest training image (Euclidean distance in peak-normalised, down-sampled pixel
    space). If a held-out ``val`` set is given, the same nearest-training distance is computed for it as a baseline: if
    generated images sit systematically *closer* to the training set than genuinely-unseen validation images do, that
    indicates memorisation / leakage.
    
    Parameters
    ----------
    generated : array_like
        Generated image stack in physical Jy/beam.
    train : array_like
        Training image stack in physical Jy/beam.
    val : array_like | None, optional
        Optional held-out validation image stack in physical Jy/beam, by default None. If provided, the nearest-training
        distances of the validation set are computed as a baseline for comparison.
    downsample : int, optional
        Downsample each image to (downsample, downsample) by block-averaging, by default 20.

    Returns
    -------
    dict
        Dictionary containing the downsample factor, nearest-neighbour distances for generated images, and if ``val`` is
        provided, the nearest-neighbour distances for validation images and the ratio of medians between generated and
        validation distances.
    """
    train_vec = _memorisation_vectors(train, downsample)
    gen_vec = _memorisation_vectors(generated, downsample)
    nn = NearestNeighbors(n_neighbors=1).fit(train_vec)

    gen_dist = nn.kneighbors(gen_vec, return_distance=True)[0].ravel()
    out = {
        "downsample": downsample,
        "gen_nn_median": float(np.median(gen_dist)),
        "gen_nn_p05": float(np.percentile(gen_dist, 5)),
        "gen_nn_distances": gen_dist,
    }
    if val is not None:
        val_dist = nn.kneighbors(_memorisation_vectors(val, downsample), return_distance=True)[0].ravel()
        out["val_nn_median"] = float(np.median(val_dist))
        out["val_nn_distances"] = val_dist
        # Ratio < 1 and a small W1 offset in the wrong direction flags memorisation.
        out["median_ratio_gen_over_val"] = float(np.median(gen_dist) / (np.median(val_dist) + 1e-30))
        out["w1_gen_vs_val"] = metrics.wasserstein_1d(gen_dist, val_dist)
    return out


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def full_report(
    generated,
    real,
    prompted_peak=None,
    train=None,
    val=None,
    nsigma: float = 5.0,
) -> dict:
    """
    Run the full evaluation suite. ``prompted_peak`` enables the calibration report; ``train`` enables
    the memorisation report (with optional ``val`` baseline).
    
    Parameters
    ----------
    generated : array_like
        Generated image stack in physical Jy/beam.
    real : array_like
        Real image stack in physical Jy/beam.
    prompted_peak : array_like | None, optional
        The physical peak flux (Jy/beam) each image was conditioned on, length N. If ``None``, the calibration report is
        skipped, by default None.
    train : array_like | None, optional
        Training image stack in physical Jy/beam. If ``None``, the memorisation report is skipped, by default None.
    val : array_like | None, optional
        Optional held-out validation image stack in physical Jy/beam, by default None. If provided, the nearest-training
        distances of the validation set are computed as a baseline for comparison.
    nsigma : float, optional
        Detection threshold for the source finder, by default 5.0.
    
    Returns
    -------
    dict
        Dictionary containing the results of the physical distribution report, and if requested, the calibration and
        memorisation reports.
    """
    report = {"physical_distribution": physical_distribution_report(generated, real, nsigma=nsigma)}
    if prompted_peak is not None:
        report["calibration"] = calibration_report(generated, prompted_peak, nsigma=nsigma)
    if train is not None:
        report["memorization"] = memorization_report(generated, train, val=val)
    logger.info("Completed Tier-1 evaluation report.")
    return report


def summarise(report: dict) -> str:
    """
    Human-readable one-screen summary of a :func:`full_report` result.
    
    Parameters
    ----------
    report : dict
        The report dictionary returned by :func:`full_report`.
    
    Returns
    -------
    str
        A multi-line string summarising the key metrics of the report.
    """
    lines = ["==================== Tier-1 evaluation ===================="]
    pd = report.get("physical_distribution")
    if pd:
        lines.append(f"Physical distribution  (N_gen={pd['n_generated']}, N_real={pd['n_real']})")
        lines.append(f"  physical FID = {pd['physical_fid']:.4f}   physical KID = {pd['physical_kid']:.4e}")
        lines.append(f"  {'property':<13}{'W1':>12}{'KS':>8}{'KS p':>9}   gen/real median")
        for k, v in pd["per_property"].items():
            lines.append(f"  {k:<13}{v['w1']:>12.4g}{v['ks_stat']:>8.3f}{v['ks_pvalue']:>9.2g}"
                         f"   {v['gen_median']:.3g} / {v['real_median']:.3g}")
    cal = report.get("calibration")
    if cal:
        lines.append(f"Calibration (recovered vs prompted peak, N={cal['n_used']})")
        lines.append(f"  slope={cal['slope']:.3f}  intercept={cal['intercept']:.3f}  R^2={cal['r2']:.4f}"
                     f"  bias={cal['bias']:.3f}x  scatter={cal['scatter_dex']:.3f} dex")
    mem = report.get("memorization")
    if mem:
        lines.append("Memorisation (nearest training image, peak-normalised)")
        line = f"  gen NN median = {mem['gen_nn_median']:.4f}"
        if "val_nn_median" in mem:
            line += (f"   val NN median = {mem['val_nn_median']:.4f}"
                     f"   ratio = {mem['median_ratio_gen_over_val']:.3f} (<1 => possible memorisation)")
        lines.append(line)
    lines.append("===========================================================")
    return "\n".join(lines)
