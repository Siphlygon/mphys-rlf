"""
A script to evaluate the convergence/training of a diffusion model by sampling images from several snapshots an
comparing them to a real image set.

Experience shows (and this is a maxim in ML training anyway) that validation loss is not always a good progress metre.
Especially for our diffusion models, we found that although validation loss may barely decrease between e.g., 34k and
100k iterations, the quality of the generated samples can improve dramatically - from static noise to clean, physical
radio galaxies. So "the loss is flat" cannot be used to argue convergence, and the only way to decide whether more
iterations are worth the GPU-hours is to evaluate sample quality directly as a function of iteration.

This script samples N images from each of several snapshots and runs the evaluation against a real image set
(diffracc.evaluation.evaluate.full_report) emitting:

* a metric-vs-iteration table (and CSV) - physical FID/KID, plus calibration and memorisation scalars when available
* a multi-row PNG contact sheet, one row per snapshot, for the qualitative view alongside the numbers

Note:
* with ema_rate=0.9999 the EMA averages over ~10,000 iterations, so adjacent late snapshots are smoothed toward each
other by construction and will look near-identical regardless of real progress. Compare e.g. 50k/70k/85k/100k, not
96k/98k/100k.
* everything is in physical units (Jy/beam) - the evaluation suite requires it, and it is the only way a
transform-trained and a raw-trained model can be put on the same axis. Inversion is handled automatically via the flux
transform recorded in each model's config (see sample_snapshot_grid's --invert).

Usage (from the repo root):
    python -m diffracc.scripts.snapshot_convergence_sweep --model-name snr15_inclusive_las_old \\
        --real-data-path /path/to/snr_15_peak_500_inclusive.h5 --snapshots 50000 70000 85000 100000

    # auto-pick N evenly-spaced snapshots across the whole run instead of listing them
    python -m diffracc.scripts.snapshot_convergence_sweep --model-name snr15_inclusive_las_old \\
        --real-data-path /path/to/data.h5 --n-snapshots 5 --n 256
"""
import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import flux_transforms as ft
from ..evaluation.evaluate import full_report
from ..model import diffusion
from ..utils import paths
from ..utils.logger import get_logger
from .sample_snapshot_grid import _find_snapshot, _load_model

logger = get_logger(__name__)

# Scalars pulled out of the nested full_report dict for the summary table, as (column name, report section, key within
# that section). Sections that were skipped (e.g. calibration without prompted peaks) are simply absent from the table.
_TABLE_METRICS = [
    ("physical_fid", "physical_distribution", "physical_fid"),
    ("physical_kid", "physical_distribution", "physical_kid"),
    ("calib_slope", "calibration", "slope"),
    ("calib_r2", "calibration", "r2"),
    ("calib_scatter_dex", "calibration", "scatter_dex"),
    ("memo_nn_median", "memorisation", "gen_nn_median"),
    ("memo_ratio_vs_val", "memorisation", "median_ratio_gen_over_val"),
]


def _available_snapshots(model_dir: Path) -> list[int]:
    """
    List every snapshot iteration available for a model, sorted ascending.

    Parameters
    ----------
    model_dir : Path
        The model results directory (containing a 'snapshots' subdirectory).

    Returns
    -------
    list[int]
        Sorted snapshot iteration numbers.

    Raises
    ------
    FileNotFoundError
        If the snapshots directory or any snapshot files are missing.
    """
    snap_dir = model_dir / "snapshots"
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"No snapshots directory at {snap_dir}")
    iters = sorted(
        int(m.group(1)) for p in snap_dir.glob("snapshot_iter_*.pt")
        if (m := re.search(r"snapshot_iter_(\d+)", p.name))
    )
    if not iters:
        raise FileNotFoundError(f"No snapshot_iter_*.pt files in {snap_dir}")
    return iters


def _pick_evenly_spaced(available: list[int], n_snapshots: int) -> list[int]:
    """
    Choose n_snapshots evenly-spaced iterations from those available, always including the latest.

    The EMA's ~10k-iteration averaging window makes adjacent late snapshots look near-identical regardless of genuine
    progress, so a convergence check needs snapshots far enough apart to outrun that smoothing.

    Parameters
    ----------
    available : list[int]
        Sorted available snapshot iterations.
    n_snapshots : int
        How many to pick.

    Returns
    -------
    list[int]
        The chosen iterations, sorted ascending and de-duplicated.
    """
    if n_snapshots >= len(available):
        return available
    idx = np.linspace(0, len(available) - 1, n_snapshots).round().astype(int)
    return sorted({available[i] for i in idx})


def _load_real_images(real_data_path: str, n_real: int | None, seed: int = 0) -> np.ndarray:
    """
    Load a real image stack (physical Jy/beam) from the training/held-out h5 to compare against.

    Parameters
    ----------
    real_data_path : str
        Path to an h5 file with an "images" dataset.
    n_real : int | None
        Randomly subsample this many images (without replacement) for speed; None uses all of them.
    seed : int, optional
        RNG seed for the subsample, by default 0, so repeated runs compare against the identical real set.

    Returns
    -------
    np.ndarray
        Real image stack, shape (n, H, W).
    """
    with h5py.File(real_data_path, "r") as f:
        images = f["images"]
        total = len(images)
        if n_real is None or n_real >= total:
            return np.asarray(images[:], dtype=np.float32)
        # Sorted indices: h5py fancy-indexing requires increasing order.
        idx = np.sort(np.random.default_rng(seed).choice(total, size=n_real, replace=False))
        return np.asarray(images[idx], dtype=np.float32)


@torch.no_grad()
def _sample_snapshot(model_dir: Path,
                     snapshot_iter: int,
                     n: int,
                     key: str,
                     timesteps: int,
                     device: torch.device,
                     seed: int,
                     sample_batch: int = 256) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Sample n images from one snapshot, returned in physical Jy/beam.

    The same latent seed is used for every snapshot in a sweep, so differences between rows reflect the model's training
    progress rather than sampling variance.

    Parameters
    ----------
    model_dir : Path
        The model results directory.
    snapshot_iter : int
        Which snapshot iteration to sample from.
    n : int
        Number of images to sample.
    key : str
        Which weights to use: "ema_model" or "model".
    timesteps : int
        Number of sampling steps.
    device : torch.device
        Device to sample on.
    seed : int
        Seed for the initial latents, shared across snapshots in a sweep.
    sample_batch : int, optional
        Maximum images pushed through the U-Net at once, by default 256. The fixed pool of latents is processed in
        chunks of this size to avoid massive memory usage. Lower as required.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        The sampled images in Jy/beam with shape (n, H, W), and the physical prompted peak fluxes if the model has
        peak-flux conditioning (else None, which disables the calibration report).
    """
    snapshot_path = _find_snapshot(model_dir, snapshot_iter)
    model, config = _load_model(model_dir, snapshot_path, key)
    model = model.to(device)

    context_dim = model.model.context_dim

    # Draw every latent up front (cheap - this is just noise) so the shared-seed guarantee is identical regardless of
    # chunking; only the U-Net forward pass is memory-bound, so the latents are then sampled in sample_batch-sized
    # chunks and the resulting images moved to the CPU before the next chunk.
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(n, 1, 80, 80, device=device, generator=generator)

    chunks = []
    for start in range(0, n, sample_batch):
        lat = latents[start:start + sample_batch]
        bs = lat.shape[0]
        # Central (standardised-zero) prompt for every image, matching sample_snapshot_grid's convention.
        context = torch.zeros(bs, context_dim, device=device) if context_dim else None
        steps = diffusion.edm_sampling(
            model, context_batch=context, latents=lat, batch_size=bs, image_size=80, timesteps=timesteps)
        chunks.append(steps[-1][:, 0].cpu().numpy())
    imgs = np.concatenate(chunks, axis=0)

    # The evaluation suite requires physical Jy/beam, so always invert a recorded transform here (unlike
    # sample_snapshot_grid, where leaving samples in model space is a valid viewing choice).
    recorded = getattr(config, "flux_transform", None)
    if recorded is not None:
        imgs = np.asarray(ft.load(recorded).inverse(imgs))

    # The calibration report needs the *physical* peak flux each image was conditioned on. A standardised-zero
    # prompt maps back to some fixed physical peak, but recovering it needs the training-set power transform, which
    # this script deliberately does not refit (see sample_conditioning_sweep.py for that). So calibration is left
    # disabled here - this sweep is about distribution match over training time, not conditioning fidelity.
    return imgs, None


def sweep_snapshots(model_name: str,
                    real_data_path: str,
                    snapshots: list[int] | None = None,
                    n_snapshots: int = 5,
                    n: int = 256,
                    n_real: int | None = 2000,
                    key: str = "ema_model",
                    timesteps: int = 25,
                    seed: int = 0,
                    check_memorisation: bool = False,
                    sample_batch: int = 256,
                    out_dir: Path | None = None) -> dict:
    """
    Sample and evaluate several snapshots of one model, to see whether quality is still improving with training.

    Parameters
    ----------
    model_name : str
        Model directory name under paths.MODEL_PARENT.
    real_data_path : str
        Path to an h5 with an "images" dataset of real images (physical Jy/beam) to compare against.
    snapshots : list[int] | None, optional
        Explicit snapshot iterations to evaluate. If None (default), n_snapshots evenly-spaced ones are chosen.
    n_snapshots : int, optional
        How many evenly-spaced snapshots to auto-pick when `snapshots` is None, by default 5.
    n : int, optional
        Images to sample per snapshot, by default 256. FID/KID are biased at small n, so keep this constant across a
        sweep (it is) and treat absolute values as comparable only within one sweep.
    n_real : int | None, optional
        Subsample this many real images for the comparison, by default 2000. None uses all.
    key : str, optional
        Which weights to sample from, by default "ema_model".
    timesteps : int, optional
        Sampling steps, by default 25.
    seed : int, optional
        Latent seed, shared across all snapshots so rows differ by training progress only, by default 0.
    check_memorisation : bool, optional
        Also run the memorisation report against the real set, by default False. Off by default because it is a
        nearest-neighbour search over the full real stack and is the slowest part of the report.
    sample_batch : int, optional
        Maximum images pushed through the U-Net at once, by default 256. Only caps peak GPU memory; the result is
        unchanged. Lower it if sampling OOMs, raise it if there is headroom.
    out_dir : Path | None, optional
        Where to write outputs, by default the model directory.

    Returns
    -------
    dict
        Mapping of snapshot iteration -> the full_report dict for that snapshot.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = paths.MODEL_PARENT / model_name
    out_dir = out_dir or model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    available = _available_snapshots(model_dir)
    chosen = sorted(set(snapshots)) if snapshots else _pick_evenly_spaced(available, n_snapshots)
    missing = [it for it in chosen if it not in available]
    if missing:
        raise FileNotFoundError(f"Requested snapshots not available: {missing}. Available: {available}")

    logger.info(f"Evaluating snapshots {chosen} of {model_name} on {device}.")
    if len(chosen) > 1 and min(np.diff(chosen)) < 10000:
        logger.warning(
            "Some chosen snapshots are <10k iterations apart, which is within the EMA's ~10k averaging window at "
            "ema_rate=0.9999 - those rows will look similar by construction, not because progress has stopped. "
            "Prefer wider spacing for a convergence check."
        )

    logger.info(f"Loading real images from {real_data_path}...")
    real = _load_real_images(real_data_path, n_real, seed=seed)

    reports, sampled = {}, {}
    for snapshot_iter in chosen:
        logger.info(f"Sampling {n} images from iteration {snapshot_iter}...")
        imgs, prompted_peak = _sample_snapshot(
            model_dir, snapshot_iter, n, key, timesteps, device, seed, sample_batch=sample_batch)
        sampled[snapshot_iter] = imgs

        logger.info(f"Running Tier-1 evaluation for iteration {snapshot_iter}...")
        try:
            reports[snapshot_iter] = full_report(
                imgs, real,
                prompted_peak=prompted_peak,
                train=real if check_memorisation else None,
            )
        except Exception as exc:  # noqa: BLE001 - a bad early snapshot must not abort the whole sweep
            # The usual cause is an early/undertrained snapshot whose images contain no sources the finder can detect,
            # leaving the property arrays empty (scipy then raises). That is a real, informative result for a
            # convergence sweep - the model was not yet producing detectable sources at this iteration - so record it as
            # such and carry on rather than losing the later snapshots we actually care about.
            logger.warning(
                f"Evaluation failed at iteration {snapshot_iter} ({type(exc).__name__}: {exc}). "
                "This usually means no sources were detected in the generated images - recording as NaN."
            )
            reports[snapshot_iter] = {"evaluation_error": f"{type(exc).__name__}: {exc}"}

    _write_table(reports, out_dir / f"convergence_{model_name}_{key}.csv")
    _write_report_json(reports, out_dir / f"convergence_{model_name}_{key}.json")
    _plot_contact_sheet(sampled, reports, model_name, key, out_dir / f"convergence_{model_name}_{key}.png")
    return reports


def _extract_table(reports: dict) -> tuple[list[str], list[list]]:
    """
    Flatten the nested per-snapshot reports into table columns/rows, keeping only metrics actually present.

    Parameters
    ----------
    reports : dict
        Mapping of iteration -> full_report dict.

    Returns
    -------
    tuple[list[str], list[list]]
        Column headers (starting with "iteration") and one row per snapshot.
    """
    present = [
        (col, section, key) for col, section, key in _TABLE_METRICS
        if any(key in report.get(section, {}) for report in reports.values())
    ]
    headers = ["iteration"] + [col for col, _, _ in present]
    rows = [
        [it] + [reports[it].get(section, {}).get(key, float("nan")) for _, section, key in present]
        for it in sorted(reports)
    ]
    return headers, rows


def _write_table(reports: dict, path: Path) -> None:
    """
    Write the metric-vs-iteration table to CSV and log it, so the trend is readable straight from the job output.

    Parameters
    ----------
    reports : dict
        Mapping of iteration -> full_report dict.
    path : Path
        CSV destination.
    """
    headers, rows = _extract_table(reports)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    widths = [max(len(h), 14) for h in headers]
    logger.info("Metric vs iteration (lower physical_fid / physical_kid = closer to the real distribution):")
    logger.info(" | ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for row in rows:
        logger.info(" | ".join(
            (f"{v:.5g}" if isinstance(v, float) else str(v)).rjust(w) for v, w in zip(row, widths)))
    logger.info(f"Wrote metric table to {path}")


def _write_report_json(reports: dict, path: Path) -> None:
    """
    Persist the full nested reports as JSON, dropping the bulky raw per-image arrays that some sections carry.

    Parameters
    ----------
    reports : dict
        Mapping of iteration -> full_report dict.
    path : Path
        JSON destination.
    """
    def _clean(obj):
        if isinstance(obj, dict):
            # Per-image arrays (nn distances, prompted/recovered vectors) are large and not useful in a summary file.
            return {k: _clean(v) for k, v in obj.items()
                    if not (isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 50)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): _clean(v) for k, v in reports.items()}, f, indent=4)
    logger.info(f"Wrote full reports to {path}")


def _plot_contact_sheet(sampled: dict, reports: dict, model_name: str, key: str, path: Path,
                        n_show: int = 6) -> None:
    """
    Plot one row of example images per snapshot, annotated with that snapshot's physical FID.

    Parameters
    ----------
    sampled : dict
        Mapping of iteration -> sampled image stack.
    reports : dict
        Mapping of iteration -> full_report dict, for the FID annotation.
    model_name : str
        Model name, for the figure title.
    key : str
        Which weights were sampled, for the figure title.
    path : Path
        PNG destination.
    n_show : int, optional
        Example images per row, by default 6.
    """
    iterations = sorted(sampled)
    ncols = min(n_show, min(len(v) for v in sampled.values()))
    fig, axes = plt.subplots(len(iterations), ncols, figsize=(2.0 * ncols, 2.2 * len(iterations)), squeeze=False)

    for row, snapshot_iter in enumerate(iterations):
        imgs = sampled[snapshot_iter]
        fid = reports[snapshot_iter].get("physical_distribution", {}).get("physical_fid", float("nan"))
        for col in range(ncols):
            ax = axes[row][col]
            img = imgs[col]
            vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
            ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax if vmax > vmin else None, origin="lower")
            ax.axis("off")
        axes[row][0].set_ylabel(f"{snapshot_iter}", fontsize=9)
        axes[row][0].axis("on")
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        axes[row][0].set_title(f"iter {snapshot_iter}  (FID {fid:.3g})", fontsize=9, loc="left")

    fig.suptitle(f"{model_name}  |  {key}  |  convergence sweep  |  Jy/beam", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote contact sheet to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Model directory name under model_results/.")
    parser.add_argument("--real-data-path", required=True,
                        help="Path to an h5 with an 'images' dataset of real images (physical Jy/beam).")
    parser.add_argument("--snapshots", type=int, nargs="+", default=None,
                        help="Explicit snapshot iterations to evaluate (default: auto-pick evenly spaced).")
    parser.add_argument("--n-snapshots", type=int, default=5,
                        help="How many evenly-spaced snapshots to auto-pick when --snapshots is not given.")
    parser.add_argument("--n", type=int, default=256, help="Images to sample per snapshot (default 256).")
    parser.add_argument("--n-real", type=int, default=2000,
                        help="Real images to compare against (default 2000; use 0 for all).")
    parser.add_argument("--key", choices=["ema_model", "model"], default="ema_model",
                        help="Which weights to sample from (default: ema_model).")
    parser.add_argument("--timesteps", type=int, default=25, help="Sampling steps (default 25).")
    parser.add_argument("--seed", type=int, default=0, help="Latent seed, shared across snapshots (default 0).")
    parser.add_argument("--check-memorisation", action="store_true",
                        help="Also run the memorisation report (slow: nearest-neighbour over the real stack).")
    parser.add_argument("--sample-batch", type=int, default=256,
                        help="Max images sampled through the U-Net at once (default 256); lower this if sampling OOMs. "
                             "Only affects peak GPU memory, not the result.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: the model dir).")
    args = parser.parse_args()

    sweep_snapshots(
        args.model_name, args.real_data_path, snapshots=args.snapshots, n_snapshots=args.n_snapshots,
        n=args.n, n_real=(None if args.n_real == 0 else args.n_real), key=args.key, timesteps=args.timesteps,
        seed=args.seed, check_memorisation=args.check_memorisation, sample_batch=args.sample_batch,
        out_dir=args.out_dir,
    )
