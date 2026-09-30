"""
A number of statistical distances for comparing two sets of samples.

These operate on plain numpy arrays: 1-D distances on scalar distributions (e.g. peak flux), and multivariate distances
on feature matrices (N, D) -- where D, for our project, is a physical feature vector. 

Glossary:
- KID: Kernel Inception Distance, the unbiased polynomial-kernel MMD^2 between two feature matrices.
- MMD: Maximum Mean Discrepancy, a general class of kernel-based distances between distributions. KID is a specific MMD
  with a polynomial kernel.
- W1: Wasserstein-1 distance, also known as the earth-mover distance, a measure of the distance between two probability
  distributions on a given metric space.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from scipy.spatial.distance import cdist


# ----- UTILITY FUNCTIONS -----
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


def _knn_radii(x: np.ndarray, k: int) -> np.ndarray:
    """
    Distance from each row of `x` to its k-th nearest neighbour within `x` (excluding itself).

    These radii define the sample manifold used by methods below: the manifold of a set is the union of the balls
    centred on each sample with its k-th-NN radius.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix of shape (N, D).
    k : int
        Neighbourhood size (1 = nearest neighbour).

    Returns
    -------
    np.ndarray
        The k-th nearest-neighbour distance for each row, shape (N,).
    """
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)  # exclude self so the k-th neighbour is a genuine other point
    return np.partition(d, k - 1, axis=1)[:, k - 1]  # k-th smallest per row (1-indexed k -> index k-1)


def standardise(reference: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """
    Whiten feature matrices using the mean/std of `reference` (the real set).

    Fitting the scaler on the real data and applying it to both sets puts every feature on a comparable scale so no
    single quantity dominates the multivariate distances. Returns the standardised `reference` followed by each of
    `others`. Useful for computing FID/KID on physical feature vectors with very different scales (e.g. fluxes in Jy,
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


# --- DISTANCES BETWEEN SCALAR DISTRIBUTIONS ---
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


# --- DISTANCES BETWEEN FEATURE MATRICES (N, D) ---
def _polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int, gamma: float | None, coef0: float) -> np.ndarray:
    """
    Polynomial kernel matrix between two feature matrices.
    
    Used internally for the unbiased polynomial-kernel MMD^2 (KID) computation. The kernel is defined as
    `K(x, y) = (gamma * <x, y> + coef0)^degree`. If `gamma` is `None`, it defaults to `1 / D` where `D` is the number of
    features (columns) in `x`.

    Parameters
    ----------
    x : np.ndarray
        The first feature matrix of shape (N, D).
    y : np.ndarray
        The second feature matrix of shape (M, D).
    degree : int
        The degree of the polynomial kernel.
    gamma : float | None
        The gamma parameter for the polynomial kernel. If `None`, defaults to 1 / D.
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
    coef0: float = 1.0) -> float:
    """
    Unbiased polynomial-kernel MMD^2 between two feature matrices -- the KID computation.

    Unlike :func:`frechet_distance` it makes no Gaussian assumption and is unbiased, so it is more reliable at modest
    sample sizes (e.g., in our brief evaluation). Lower is better; can be slightly negative due to the unbiased
    estimator.

    Parameters
    ----------
    x : np.ndarray
        First feature matrix of shape (N, D).
    y : np.ndarray
        Second feature matrix of shape (M, D).
    degree : int, optional
        Degree of the polynomial kernel, by default 3.
    gamma : float | None, optional
        Gamma parameter for the polynomial kernel. If `None`, defaults to 1 / D, by default None.
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


def density_coverage(real: np.ndarray, generated: np.ndarray, k: int = 5) -> tuple[float, float]:
    """
    Density and coverage (Naeem et al. 2020) between two feature matrices -- more outlier-robust cousins of precision
    and recall.

    Density counts, for each generated sample, how many real k-NN balls it falls into, normalised by `k`; unlike
    precision it is not saturated by a single real outlier with a huge ball, and can exceed 1 (values near 1 indicate
    the generated samples sit where real samples are dense).
    
    Coverage is the fraction of real samples that have at least one generated sample inside their own k-NN ball; unlike
    recall it is not inflated by generated outliers, so it is a cleaner "did the model cover this real region?" measure.
    Density is >= 0; coverage is in [0, 1]; higher is better.

    Expects comparably-scaled features (e.g. from :func:`standardise`).

    Parameters
    ----------
    real : np.ndarray
        Real feature matrix of shape (N_r, D).
    generated : np.ndarray
        Generated feature matrix of shape (N_g, D).
    k : int, optional
        Nearest-neighbour count defining the real balls, by default 5 (the value used in the paper).

    Returns
    -------
    tuple[float, float]
        The `(density, coverage)`.
    """
    if real.shape[0] < k + 1:
        raise ValueError(
            f"Real set has {real.shape[0]} finite samples; need at least k+1={k+1} for k={k} nearest neighbours.")
    if generated.shape[0] < k + 1:
        raise ValueError(
            f"Generated set has {generated.shape[0]} finite samples; need at least k+1={k+1} for k={k} nearest neighbours.")

    real_radii = _knn_radii(real, k)          # (N_r,)
    d = cdist(generated, real)                 # (N_g, N_r): d[j, i] = ||g_j - r_i||
    within = d <= real_radii[None, :]          # within[j, i]: g_j lies inside real sample i's k-NN ball

    density = float(within.sum() / (k * generated.shape[0]))
    coverage = float(np.mean(within.any(axis=0)))  # per real i: is any generated sample inside its ball?
    return density, coverage
