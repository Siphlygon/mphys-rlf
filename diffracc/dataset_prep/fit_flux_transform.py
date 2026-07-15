"""
Fit a global, invertible flux transform to a training dataset, verify it is non-destructive, and save it for use during
training and sampling.

Example
-------
Fit the recommended asinh transform (sigma_data = 0.5) to the cleaned Hardcastle set and save ``flux_transform.json``
next to it::

    python -m diffracc.dataset_prep.fit_flux_transform \
        --dataset datasets/clean_hardcastle_catalogue.h5 --kind asinh --sigma_data 0.5

The saved file can then be passed to ``TrainDatasetNoScale(path, flux_transform=<dir or json>)`` and to the ``Sampler``
via the ``flux_transform`` setting (both accept the same argument).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from ..data import flux_transforms as ft
from ..utils.logger import get_logger

logger = get_logger("fit_flux_transform")


def _load_images(dataset_path: Path, key: str, sample_images: int, seed: int) -> np.ndarray:
    """
    Load (a random subset of) the images from an HDF5 dataset.
    
    Parameters
    ----------
    dataset_path : Path
        Path to the HDF5 file containing the training images.
    key : str
        HDF5 dataset key for the images.
    sample_images : int
        Number of images to subsample for fitting. If 0, use all images.
    seed : int
        Random seed for reproducibility of the subsample.
    
    Returns
    -------
    np.ndarray
        Array of float32 images loaded from the dataset, subsampled if requested.
    """
    with h5py.File(dataset_path, "r") as f:
        n = f[key].shape[0]
        if sample_images and n > sample_images: # Subsample if requested and if there are enough images
            idx = np.sort(np.random.default_rng(seed).choice(n, size=sample_images, replace=False))
            imgs = f[key][idx]
        else:
            imgs = f[key][:]
    return np.asarray(imgs, dtype=np.float32) # Ensure the images are in float32 for consistency


def fit_and_report(
    dataset_path: Path,
    kind: str = "asinh",
    sigma_data: float = 0.5,
    beta: float | None = None,
    beta_scale: float = 3.0,
    key: str = "images",
    sample_images: int = 4000,
    output: Path | None = None,
    seed: int = 0,
) -> ft._GlobalFluxTransform:
    """
    Fit a flux transform, print a diagnostic report, verify invertibility, and save it.
    
    Parameters
    ----------
    dataset_path : Path
        Path to the HDF5 file containing the training images.
    kind : str
        Type of flux transform to fit. Options are "asinh" or "linear". Default is "asinh".
    sigma_data : float
        Target standard deviation of the transformed data. Default is 0.5.
    beta : float | None
        Transition scale for the asinh transform in Jy/beam. If ``None``, it will be set to beta_scale times the robust
        noise. Default is ``None``.
    beta_scale : float
        Scale factor for determining beta if beta is not provided. Default is 3.0.
    key : str
        HDF5 dataset key for the images. Default is "images".
    sample_images : int
        Number of images to subsample for fitting. If 0, use all images. Default is 4000.
    output : Path | None
        Path to save the fitted flux transform JSON file. If ``None``, it will be saved next to the dataset. Default is
        ``None``.
    seed : int
        Random seed for reproducibility. Default is 0.
    
    Returns
    -------
    ft._GlobalFluxTransform
        The fitted flux transform object.
    
    Raises
    ------
    ValueError
        If an unknown transform kind is specified.
    """
    logger.info(f"Fitting a {kind} flux transform to the dataset at {dataset_path}...")
    logger.info(f"Loading images from key '{key}' with sample size {sample_images} and seed {seed}...")
    imgs = _load_images(dataset_path, key, sample_images, seed)
    finite = imgs[np.isfinite(imgs)]

    # Compute the raw standard deviation of the finite pixels and estimate the robust background noise
    logger.info(f"Computing raw std and robust noise estimate for {finite.size} finite pixels...")
    raw_std = float(finite.std())
    noise = ft.robust_noise(ft._flat_sample(imgs, 100_000, seed))
    logger.info(f"Loaded {imgs.shape[0]} images from {dataset_path}.")
    logger.info(f"Raw pooled std = {raw_std:.4e} Jy/beam,  robust background noise ~ {noise:.4e} Jy/beam "
                f"({noise * 1e6:.1f} uJy/beam).")

    if kind == "asinh":
        transform = ft.GlobalAsinhScale.fit(
            imgs, sigma_data=sigma_data, beta=beta, noise=noise, beta_scale=beta_scale, seed=seed
        )
    elif kind == "linear":
        transform = ft.GlobalLinearScale.fit(imgs, sigma_data=sigma_data, seed=seed)
    else:
        raise ValueError(f"Unknown kind '{kind}'. Choose 'asinh' or 'linear'.")

    # Diagnostics: achieved std, round-trip error, and (for asinh) how much of the data is in the quasi-linear regime
    # |x| < beta.
    logger.info("Computing diagnostics for the fitted transform...")
    transformed = transform.forward(finite)
    achieved_std = float(np.asarray(transformed).std())
    roundtrip_err = transform.max_abs_roundtrip_error(imgs, seed=seed)
    rel_err = roundtrip_err / (np.abs(finite).max() + 1e-30)

    logger.info("\n================ flux transform report ================")
    logger.info(f"dataset            : {dataset_path}")
    logger.info(f"kind               : {transform.name}")
    logger.info(f"params             : {transform.to_dict()}")
    logger.info(f"raw pooled std     : {raw_std:.4e} Jy/beam")
    logger.info(f"robust noise sigma : {noise:.4e} Jy/beam ({noise * 1e6:.1f} uJy/beam)")
    logger.info(f"target sigma_data  : {sigma_data}")
    logger.info(f"achieved std       : {achieved_std:.4f}   (should be ~= sigma_data)")
    if transform.name == "asinh":
        frac_linear = float(np.mean(np.abs(finite) < transform.beta))
        logger.info(f"beta               : {transform.beta:.4e} Jy/beam ({transform.beta * 1e3:.3f} mJy/beam)")
        logger.info(f"frac |x| < beta    : {frac_linear:.3f}   (fraction of pixels in the ~linear regime)")
    logger.info(f"max round-trip err : {roundtrip_err:.3e} Jy/beam  (relative {rel_err:.2e})")
    logger.info("-------------------------------------------------------")
    logger.info("Recommended config values (in transformed units):")
    logger.info(f'    "sigma_data": {sigma_data},  "sigma_min": 0.002,  "sigma_max": 80,')
    logger.info('    "p_mean": -2.5,  "p_std": 1.8,  "ema_rate": 0.9999')  # this is the same as the config defaults
    logger.info("=======================================================\n")

    # Check that the round-trip error is within a reasonable tolerance. If it exceeds 0.1% of the maximum absolute value
    # of the finite pixels, log a warning.
    if roundtrip_err > 1e-3 * (np.abs(finite).max() + 1e-30): 
        logger.warning("Round-trip error is larger than expected; check the transform / dtype.")

    out_path = output if output is not None else dataset_path.parent / f"{dataset_path.stem}_flux_transform.json"
    saved = transform.save(out_path)
    logger.info(f"Saved transform to: {saved}")
    return transform


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the training HDF5 file.")
    parser.add_argument("--kind", type=str, default="asinh", choices=["asinh", "linear"],
                        help="Transform type. Default 'asinh'.")
    parser.add_argument("--sigma_data", type=float, default=0.5,
                        help="Target std of the transformed data (== EDM sigma_data). Default 0.5.")
    parser.add_argument("--beta", type=float, default=None,
                        help="asinh transition scale in Jy/beam. Default: beta_scale * robust noise.")
    parser.add_argument("--beta_scale", type=float, default=3.0,
                        help="Multiple of the robust noise used for beta if --beta is not given. Default 3.")
    parser.add_argument("--key", type=str, default="images", help="HDF5 dataset key for images. Default 'images'.")
    parser.add_argument("--sample_images", type=int, default=4000,
                        help="Number of images to subsample for fitting. Default 4000. Use 0 for all.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to save flux_transform.json (file or directory). Default: next to the dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default 0.")
    return parser


if __name__ == "__main__":
    args = _build_argument_parser().parse_args()
    fit_and_report(
        dataset_path=args.dataset,
        kind=args.kind,
        sigma_data=args.sigma_data,
        beta=args.beta,
        beta_scale=args.beta_scale,
        key=args.key,
        sample_images=args.sample_images,
        output=args.output,
        seed=args.seed,
    )
