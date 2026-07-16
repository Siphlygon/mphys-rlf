"""
Conditioning-response sanity check: does the model actually listen to its prompt?

sample_snapshot_grid.py answers "does this model produce structure at all?". This script answers a narrower, equally
important question: "does that structure respond to the conditioning it was given, or is the model ignoring context
and generating an unconditional 'blob soup'?" That distinction matters because an unresponsive model would produce
plausible-looking sources whose statistics are still wrong for the physical claims used downstream (peak-flux and
LAS calibration).

Method: sample a row of n images that all share the SAME initial noise latent, varying only one conditioning
variable (LAS extent, or peak flux) across a physical range while holding the other fixed at a typical value. Because
the latent is identical across the row, any visible trend (source getting larger / brighter) is attributable to the
conditioning, not to sampling variance - a fair test that a shared-latent grid (like sample_snapshot_grid.py's) can't
give you.

The peak-flux and LAS standardisation transforms are refit here directly from the training h5 (mirroring
TrainDatasetNoScale.transform_max_vals / transform_las_vals, and diffracc.sampling.generate_fits_files's
_get_las_transformer), rather than importing generate_fits_files - that module pulls in ImageAnalyzer, which needs the
real `bdsf` package and isn't installable on Windows.

Usage (from the repo root):
    python -m diffracc.scripts.sample_conditioning_sweep --model-name snr15_inclusive_las \\
        --train-data-path /path/to/snr_15_peak_500_inclusive.h5 --sweep las
    python -m diffracc.scripts.sample_conditioning_sweep --model-name snr15_inclusive_las \\
        --train-data-path /path/to/snr_15_peak_500_inclusive.h5 --sweep peak --n 8

The grid is written to <model_dir>/conditioning_sweep_<sweep>_iter<N>.png by default.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer

from ..data import flux_transforms as ft
from ..model import diffusion
from ..utils import paths
from .sample_snapshot_grid import _find_snapshot, _load_model

# Physical LAS bounds (arcsec) used at generation time - kept in sync with
# diffracc.sampling.generate_fits_files._LAS_LOWER_BOUND / _LAS_UPPER_BOUND.
_LAS_LOWER_BOUND = 6
_LAS_UPPER_BOUND = 120

_CONTEXT_KEY = {"las": "las_values_tr", "peak": "max_values_tr"}


def _fit_peak_flux_transformer(train_data_path: str) -> tuple[PowerTransformer, np.ndarray]:
    """
    Fit the peak-flux standardisation transform the model was trained with, from the training h5's images.

    Mirrors ``ImagePathDataset.set_max_values`` + ``transform_max_vals``: per-image max pixel value, Box-Cox
    standardised. float64 fitting matches how sklearn treats the torch float32 tensor training actually fits on
    (verified to reproduce identical lambdas to the training-time fit - see the LAS-transform analogue in
    generate_fits_files._get_las_transformer for the same float32->float64 rationale).

    Parameters
    ----------
    train_data_path : str
        Path to the training h5 file containing an "images" dataset.

    Returns
    -------
    tuple[PowerTransformer, np.ndarray]
        The fitted transformer, and the raw per-image peak-flux values it was fit on (for choosing sweep bounds).
    """
    with h5py.File(train_data_path, "r") as f:
        max_vals = np.max(f["images"][:], axis=(1, 2)).astype(np.float32).astype(np.float64)
    pt = PowerTransformer(method="box-cox")
    pt.fit(max_vals.reshape(-1, 1))
    return pt, max_vals


def _fit_las_transformer(train_data_path: str) -> PowerTransformer:
    """
    Fit the LAS standardisation transform the model was trained with, from the training h5's catalogue.

    Identical to ``diffracc.sampling.generate_fits_files._get_las_transformer`` - duplicated locally rather than
    imported, since that module pulls in ``ImageAnalyzer`` (requires the real `bdsf` package, not installable on
    Windows) purely as a side effect of module import.

    Parameters
    ----------
    train_data_path : str
        Path to the training h5 file containing ``cat_info['LAS']``.

    Returns
    -------
    PowerTransformer
        The fitted transformer.
    """
    with h5py.File(train_data_path, "r") as f:
        las_values = np.ascontiguousarray(f["cat_info"][:]["LAS"], dtype=np.float32).astype(np.float64)
    method = "yeo-johnson" if (las_values <= 0).any() else "box-cox"
    pt = PowerTransformer(method=method)
    pt.fit(las_values.reshape(-1, 1))
    return pt


def _extent_proxy(img: np.ndarray) -> float:
    """
    A crude, physically-unmotivated proxy for source extent: sqrt of the count of pixels above median + 3*std.
    Matches the proxy used in diffracc.scripts.size_comp, so this sweep's numbers are comparable to that script's
    real-vs-conditioned scatter.

    Parameters
    ----------
    img : np.ndarray
        A single 2D image.

    Returns
    -------
    float
        The extent proxy value.
    """
    return float(np.sqrt((img > np.median(img) + 3 * np.std(img)).sum()))


@torch.no_grad()
def sweep_conditioning(model_name: str,
                      train_data_path: str,
                      sweep: str,
                      n: int = 6,
                      snapshot_iter: int | None = None,
                      key: str = "ema_model",
                      timesteps: int = 25,
                      invert: str = "auto",
                      las_bounds: tuple[float, float] = (_LAS_LOWER_BOUND, _LAS_UPPER_BOUND),
                      peak_bounds: tuple[float, float] | None = None,
                      out_path: Path | None = None) -> Path:
    """
    Sample a row of images sweeping one conditioning variable, with a shared initial noise latent, and save as a PNG.

    Parameters
    ----------
    model_name : str
        Name of the model directory under paths.MODEL_PARENT.
    train_data_path : str
        Path to the training h5, used to refit the peak-flux / LAS standardisation transforms.
    sweep : str
        Which condition to vary: "las" or "peak". The other condition (if present in the model) is held fixed at its
        typical (standardised-zero) value.
    n : int, optional
        Number of panels in the sweep, by default 6.
    snapshot_iter : int or None, optional
        Which snapshot iteration to load, or None (default) for the latest.
    key : str, optional
        Which weights to sample from: "ema_model" (default) or "model".
    timesteps : int, optional
        Number of sampling steps, by default 25.
    invert : str, optional
        Whether to invert the model's flux transform to Jy/beam before plotting/measuring: "auto" (default), "yes",
        or "no". See sample_snapshot_grid.sample_grid for the same option.
    las_bounds : tuple[float, float], optional
        Physical LAS range (arcsec) to sweep, by default (6, 120) - the same range used at generation time.
    peak_bounds : tuple[float, float] or None, optional
        Physical peak-flux range (Jy) to sweep, by default None (uses the 5th-95th percentile of the training set's
        peak fluxes, so the sweep stays within the training distribution).
    out_path : Path or None, optional
        Where to write the PNG, or None to auto-name it in the model directory.

    Returns
    -------
    Path
        The path the PNG was written to.

    Raises
    ------
    ValueError
        If `sweep` is not "las" or "peak", or the requested condition is not in this model's context.
    """
    if sweep not in _CONTEXT_KEY:
        raise ValueError(f"sweep must be 'las' or 'peak', got {sweep!r}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = paths.MODEL_PARENT / model_name

    snapshot_path = _find_snapshot(model_dir, snapshot_iter)
    print(f"Loading '{key}' weights from {snapshot_path} onto {device}...")
    model, config = _load_model(model_dir, snapshot_path, key)
    model = model.to(device)

    context_list = list(getattr(config, "context", []))
    sweep_context_key = _CONTEXT_KEY[sweep]
    if sweep_context_key not in context_list:
        raise ValueError(
            f"This model's context is {context_list}, which does not include {sweep_context_key!r} - "
            f"it was not trained with {sweep!r} conditioning."
        )

    # Fit the standardisation transforms from the training set (whichever ones this model's context actually uses).
    peak_pt = max_vals = las_pt = None
    if "max_values_tr" in context_list:
        peak_pt, max_vals = _fit_peak_flux_transformer(train_data_path)
    if "las_values_tr" in context_list:
        las_pt = _fit_las_transformer(train_data_path)

    # Physical values for the swept variable; the fixed condition (if any) stays at 0.0 - the standardised space's
    # typical value (PowerTransformer standardises to ~N(0,1), so 0.0 is the central/median prompt).
    if sweep == "las":
        lo, hi = las_bounds
        physical = np.geomspace(lo, hi, n)
        swept_standardised = las_pt.transform(physical.reshape(-1, 1))[:, 0]
        unit = "arcsec"
    else:
        if peak_bounds is None:
            lo, hi = np.percentile(max_vals, [5, 95])
        else:
            lo, hi = peak_bounds
        physical = np.geomspace(lo, hi, n)
        swept_standardised = peak_pt.transform(physical.reshape(-1, 1))[:, 0]
        unit = "Jy"

    # Build the (n, context_dim) tensor in the order the model expects (config.context), filling the swept column
    # and holding any other column fixed at 0.0.
    columns = []
    for feature in context_list:
        if feature == sweep_context_key:
            columns.append(swept_standardised)
        else:
            columns.append(np.zeros(n))
    context = torch.tensor(np.stack(columns, axis=1), dtype=torch.float32, device=device)

    # Same latent noise for every panel - isolates the conditioning's effect from sampling variance.
    shared_latent = torch.randn(1, 1, 80, 80, device=device)
    latents = shared_latent.repeat(n, 1, 1, 1)

    print(f"Sampling {n} images, sweeping {sweep!r} over {physical} {unit} (shared latent noise)...")
    steps = diffusion.edm_sampling(
        model, context_batch=context, latents=latents, batch_size=n, image_size=80, timesteps=timesteps)
    imgs = steps[-1][:, 0].cpu().numpy()

    recorded = getattr(config, "flux_transform", None)
    do_invert = {"yes": True, "no": False, "auto": recorded is not None}[invert]
    if do_invert:
        if recorded is None:
            raise ValueError("invert='yes' but this model's config records no flux transform to invert.")
        imgs = np.asarray(ft.load(recorded).inverse(imgs))
        space = "Jy/beam"
    else:
        space = "Jy/beam" if recorded is None else "model output space (asinh)"

    # Quantitative readout: does the measured property actually track the prompt? Eyeballing a sweep is easy to
    # kid yourself on; these numbers are the real check.
    peaks = imgs.reshape(n, -1).max(axis=1)
    extents = np.array([_extent_proxy(img) for img in imgs])
    print(f"\n{'prompted ' + sweep:>16} ({unit}) | measured peak ({space}) | extent proxy")
    for p, pk, ex in zip(physical, peaks, extents):
        print(f"{p:16.4g} | {pk:20.4g} | {ex:11.2f}")
    tracked = extents if sweep == "las" else peaks
    corr = float(np.corrcoef(physical, tracked)[0, 1]) if n > 1 else float("nan")
    print(f"\nCorrelation(prompted {sweep}, measured {'extent' if sweep == 'las' else 'peak'}) = {corr:.3f}"
          " (near 1 = model responds strongly to this condition; near 0 = model appears to ignore it)")

    # Plot: a single row, one panel per swept value.
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4))
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        img = imgs[idx]
        vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
        ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax if vmax > vmin else None, origin="lower")
        ax.set_title(f"{physical[idx]:.3g} {unit}", fontsize=9)
        ax.axis("off")

    iter_tag = int(_find_snapshot(model_dir, snapshot_iter).stem.split("_")[-1])
    fig.suptitle(f"{model_name}  |  sweeping {sweep}  |  iter {iter_tag}  |  {space}", fontsize=10)
    fig.tight_layout()

    out_path = out_path or (model_dir / f"conditioning_sweep_{sweep}_iter{iter_tag}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved sweep grid to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Model directory name under model_results/.")
    parser.add_argument("--train-data-path", required=True,
                        help="Path to the training h5, used to refit the peak-flux/LAS standardisation transforms.")
    parser.add_argument("--sweep", choices=["las", "peak"], required=True, help="Which condition to vary.")
    parser.add_argument("--n", type=int, default=6, help="Number of panels in the sweep (default 6).")
    parser.add_argument("--snapshot", type=int, default=None, help="Snapshot iteration to load (default: latest).")
    parser.add_argument("--key", choices=["ema_model", "model"], default="ema_model",
                        help="Which weights to sample from (default: ema_model).")
    parser.add_argument("--timesteps", type=int, default=25, help="Sampling steps (default 25).")
    parser.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                        help="Invert the flux transform to Jy/beam: auto (if recorded), yes (force), no (raw output).")
    parser.add_argument("--las-min", type=float, default=_LAS_LOWER_BOUND, help="LAS sweep lower bound (arcsec).")
    parser.add_argument("--las-max", type=float, default=_LAS_UPPER_BOUND, help="LAS sweep upper bound (arcsec).")
    parser.add_argument("--peak-min", type=float, default=None,
                        help="Peak-flux sweep lower bound (Jy); default 5th percentile of training peaks.")
    parser.add_argument("--peak-max", type=float, default=None,
                        help="Peak-flux sweep upper bound (Jy); default 95th percentile of training peaks.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: auto in model dir).")
    args = parser.parse_args()

    peak_bounds = (args.peak_min, args.peak_max) if args.peak_min is not None or args.peak_max is not None else None

    sweep_conditioning(
        args.model_name, args.train_data_path, args.sweep, n=args.n, snapshot_iter=args.snapshot, key=args.key,
        timesteps=args.timesteps, invert=args.invert, las_bounds=(args.las_min, args.las_max),
        peak_bounds=peak_bounds, out_path=args.out,
    )
