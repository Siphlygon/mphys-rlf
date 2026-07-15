"""
A set of embedding-agnostic statistical distances for comparing two sets of samples.

These operate on plain numpy arrays: 1-D distances on scalar distributions (e.g. peak flux), and multivariate distances
on feature matrices (N, D) -- where D may be a physical feature vector (the "physical FID/KID" of this package) or, in
future, neural embeddings (the "neural FID/KID" of the computer-vision literature). The distances are all interpretable
in physical units, and do not require a pretrained image network or any other learned embedding.

Glossary:
- FID: Fréchet Inception Distance, the Wasserstein-2 distance between two Gaussians fitted to feature matrices.
- KID: Kernel Inception Distance, the unbiased polynomial-kernel MMD^2 between two feature matrices.
- MMD: Maximum Mean Discrepancy, a general class of kernel-based distances between distributions. KID is a specific MMD
  with a polynomial kernel.
- KS: Kolmogorov-Smirnov test, a nonparametric test of the equality of two distributions
- W1: Wasserstein-1 distance, also known as the earth-mover distance, a measure of the distance between two probability
  distributions on a given metric space.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, stats


def _clean_samples(*samples: np.ndarray) -> list[np.ndarray]:
    """
    Convert to float64 and remove NaN/Inf values from each sample.
    
    Parameters
    ----------
    *samples : np.ndarray
        One or more samples of scalar values.
    
    Returns
    -------
    list[np.ndarray]
        The cleaned samples, each as a 1-D array of finite float64 values.
    """
    return [np.asarray(s, np.float64)[np.isfinite(s)] for s in samples]



def wasserstein_1d(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    1-D Wasserstein-1 (earth-mover) distance between two samples.

    In the same physical units as the inputs, so directly interpretable (e.g. "the generated peak-flux distribution
    differs from real by W1 = 3 mJy").
    
    Parameters
    ----------
    sample1 : np.ndarray
        First sample of scalar values.
    sample2 : np.ndarray
        Second sample of scalar values.
    
    Returns
    -------
    float
        The Wasserstein-1 distance between the two samples.
    """
    sample1, sample2 = _clean_samples(sample1, sample2)
    return float(stats.wasserstein_distance(sample1, sample2))


def ks_2samp(sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test. Returns (statistic, p-value).

    A large p-value means the two samples are consistent with the same distribution.
    
    Parameters
    ----------
    sample1 : np.ndarray
        First sample of scalar values.
    sample2 : np.ndarray
        Second sample of scalar values.
    
    Returns
    -------
    tuple[float, float]
        The KS statistic and p-value.
    """
    sample1, sample2 = _clean_samples(sample1, sample2)

    res = stats.ks_2samp(sample1, sample2)
    return float(res.statistic), float(res.pvalue)


def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """
    Fréchet (Wasserstein-2 between Gaussians) distance between two feature matrices.

    This is the FID computation, ``||mu_x - mu_y||^2 + Tr(Sx + Sy - 2 (Sx Sy)^1/2)``, applied to arbitrary feature
    matrices ``x``, ``y`` of shape (N, D). With a physical feature vector it is a "physical FID". Lower is better; 0
    means identical feature means and covariances.
    
    Parameters
    ----------
    x : np.ndarray
        First feature matrix of shape (N, D).
    y : np.ndarray
        Second feature matrix of shape (M, D).
    eps : float, optional
        Small value to add to the diagonal of the covariance matrices for numerical stability, by default 1e-6.
    
    Returns
    -------
    float
        The Fréchet distance between the two feature matrices.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    mu_x, mu_y = x.mean(0), y.mean(0)
    diff = mu_x - mu_y

    # Compute covariances and ensure they are 2-D arrays
    sx = np.cov(x, rowvar=False)
    sy = np.cov(y, rowvar=False)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)

    # Product of covariances may be numerically non-PSD; regularise before the matrix sqrt. We don't use the
    # errest that disp=False would add, and scipy>=1.16 deprecates disp (dropping it entirely in 1.18) while
    # also special-casing 1x1 inputs to ignore disp's tuple contract anyway - so leave disp at its default
    # (True), which returns a plain array on every scipy version, old and new.
    covmean = linalg.sqrtm(sx @ sy)
    if not np.isfinite(covmean).all():
        offset = np.eye(sx.shape[0]) * eps
        covmean = linalg.sqrtm((sx + offset) @ (sy + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sx + sy - 2.0 * covmean))


def _polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int, gamma: float | None, coef0: float) -> np.ndarray:
    """
    Polynomial kernel matrix between two feature matrices.
    
    Used internally for the unbiased polynomial-kernel MMD^2 (KID) computation. The kernel is defined as
    ``K(x, y) = (gamma * <x, y> + coef0)^degree``. If ``gamma`` is ``None``, it defaults to ``1 / D`` where ``D`` is the
    number of features (columns) in ``x``.

    Parameters
    ----------
    x : np.ndarray
        The first feature matrix of shape (N, D).
    y : np.ndarray
        The second feature matrix of shape (M, D).
    degree : int
        The degree of the polynomial kernel.
    gamma : float | None
        The gamma parameter for the polynomial kernel. If ``None``, defaults to 1 / D.
    coef0 : float
        The coef0 parameter for the polynomial kernel.

    Returns
    -------
    np.ndarray
        The polynomial kernel matrix.
    """
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef0) ** degree


def kernel_distance(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> float:
    """
    Unbiased polynomial-kernel MMD^2 between two feature matrices -- the KID computation.

    Unlike :func:`frechet_distance` it makes no Gaussian assumption and is unbiased, so it is more reliable at the
    modest sample sizes typical in astronomy. Lower is better; can be slightly negative due to the unbiased estimator
    (report as-is or clip at 0).
    
    Parameters
    ----------
    x : np.ndarray
        First feature matrix of shape (N, D).
    y : np.ndarray
        Second feature matrix of shape (M, D).
    degree : int, optional
        Degree of the polynomial kernel, by default 3.
    gamma : float | None, optional
        Gamma parameter for the polynomial kernel. If ``None``, defaults to 1 / D, by default None.
    coef0 : float, optional
        Coef0 parameter for the polynomial kernel, by default 1.0.
    
    Returns
    -------
    float
        The unbiased polynomial-kernel MMD^2 (KID) between the two feature matrices.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m, n = x.shape[0], y.shape[0]

    kxx = _polynomial_kernel(x, x, degree, gamma, coef0)
    kyy = _polynomial_kernel(y, y, degree, gamma, coef0)
    kxy = _polynomial_kernel(x, y, degree, gamma, coef0)

    # Remove self-similarity (diagonal) for the unbiased estimator.
    sum_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    sum_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    sum_xy = kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def standardise(reference: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """
    Whiten feature matrices using the mean/std of ``reference`` (the real set).

    Fitting the scaler on the real data and applying it to both sets puts every feature on a comparable scale so no
    single quantity dominates the multivariate distances. Returns the standardised ``reference`` followed by each of
    ``others``. Useful for computing FID/KID on physical feature vectors with very different scales (e.g. fluxes in Jy,
    sizes in arcsec, S/N ratios, etc.).
    
    Parameters
    ----------
    reference : np.ndarray
        The reference feature matrix of shape (N, D) to compute the mean and std from.
    *others : np.ndarray
        One or more feature matrices of shape (M, D) to be standardised using the mean and std of ``reference``.
    
    Returns
    -------
    list[np.ndarray]
        A list containing the standardised ``reference`` followed by each of the standardised ``others``.
    """
    reference = np.asarray(reference, np.float64)
    mu = reference.mean(0)
    sigma = reference.std(0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    return [((np.asarray(a, np.float64) - mu) / sigma) for a in (reference, *others)]
