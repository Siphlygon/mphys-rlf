"""
A script to evaluate the convergence/training of a diffusion model by sampling images from several snapshots and
comparing them to a held-out real image set.

This script samples from each of several snapshots and runs the evaluation (diffracc.evaluation.evaluate.full_report)
against a held-out real image set, emitting:

* a metric-vs-iteration table (and CSV) - physical KID, plus calibration and memorisation scalars
* a multi-row PNG contact sheet, one row per snapshot, for the qualitative view alongside the numbers

Note:
* with ema_rate=0.9999 the EMA averages over ~10,000 iterations, so adjacent late snapshots are smoothed toward each
other by construction and will look near-identical regardless of real progress. Compare e.g. 50k/70k/85k/100k, not
96k/98k/100k.
* everything is in physical units (Jy/beam) - the evaluation suite requires it, and it is the only way a
transform-trained and a raw-trained model can be put on the same axis. Inversion is handled automatically via the flux
transform recorded in each model's config (see sample_snapshot_grid's --invert).
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
from ..data.datasets import ImagePathDataset
from ..data.splits import load_split
from ..evaluation.evaluate import full_report
from ..model import diffusion
from ..model.config import ModelConfig
from ..utils import paths
from ..utils.logger import get_logger
from .sample_snapshot_grid import _find_snapshot, _load_model

logger = get_logger(__name__)

# Context column names, mirroring diffracc.training.trainer / diffracc.evaluation.compare_models.
_PEAK_KEY = "max_values_tr"
_LAS_KEY = "las_values_tr"

# sampling always uses 25 EDM steps and the deployed EMA weights (the default of diffracc.model.model_utils.load_model).
_TIMESTEPS = 25
_WEIGHTS_KEY = "ema_model"

# Scalars pulled out of the nested full_report dict for the summary table, as (column name, report section, key within
# that section). Sections that were skipped (e.g. calibration without prompted peaks) are simply absent from the table.
_TABLE_METRICS = [
    ("physical_kid", "physical_distribution", "physical_kid"),
    ("density", "physical_distribution", "density"),
    ("coverage", "physical_distribution", "coverage"),
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


def _load_reference_cutouts(dataset_path: Path,
                    idx: np.ndarray,
                    n: int | None,
                    seed: int,
                    images_key: str = "images") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a held-out real reference cutout (physical Jy/beam) plus its per-image peak flux and LAS, for the given dataset
    rows.

    Parameters
    ----------
    dataset_path : Path
        Path to the HDF5 dataset holding `images` (Jy/beam) and a `cat_info` table with an `LAS` column.
    idx : np.ndarray
        Dataset row indices to load (a split partition).
    n : int | None
        Cap the reference/prompt set to this many images (a fixed seeded subsample, shared across the sweep). `None`
        uses every row in `idx`.
    seed : int
        Seed for the subsample, so every snapshot in a sweep sees the identical reference and prompts.
    images_key : str, optional
        The images dataset key, by default "images".

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        `(images, peak, las, indices)`: images (n, H, W) float32 in Jy/beam, per-image peak flux (n,) Jy/beam, LAS
        (n,) arcsec, and the dataset row indices drawn (n,).
    """
    idx = np.sort(idx if n is None or n >= idx.size
                  else np.random.default_rng(seed).choice(idx, size=n, replace=False))

    with h5py.File(dataset_path, "r") as f:
        images = np.asarray(f[images_key][idx], dtype=np.float32)
        las = np.ascontiguousarray(f["cat_info"][idx]["LAS"], dtype=np.float64)
    peak = images.reshape(images.shape[0], -1).max(axis=1).astype(np.float64)
    logger.info("Loaded %i held-out real images from %s (peak median %.3g Jy/beam, LAS median %.3g arcsec).",
                images.shape[0], dataset_path.name, np.median(peak), np.median(las))
    return images, peak, las, idx


def _build_context(config,
                   dataset_path: Path,
                   peak_phys: np.ndarray,
                   las_phys: np.ndarray,
                   images_key: str = "images") -> np.ndarray | None:
    """
    Map physical (peak, LAS) prompts into the model's standardised context space, returning the (N, context_dim) matrix
    in the column order the model's config records, or `None` for an unconditional model (no context columns).

    The peak-flux and LAS standardisation transforms are fit on the full dataset, exactly as training did, so the
    prompts land in the same ~N(0, 1) space as the model was trained on. Reusing that static method keeps the fit
    identical without pulling in the bdsf-backed generation pipeline.

    Parameters
    ----------
    config : ModelConfig
        The trained model's config; `config.context` gives the context column order.
    dataset_path : Path
        Path to the HDF5 dataset whose full-dataset peak/LAS the transforms are fit on.
    peak_phys : np.ndarray
        Physical peak-flux prompts (Jy/beam), length N.
    las_phys : np.ndarray
        Physical LAS prompts (arcsec), length N.
    images_key : str, optional
        The images dataset key, by default "images".

    Returns
    -------
    np.ndarray | None
        The (N, context_dim) float32 context, or `None` if the model has no context columns.
    """
    context_cols = list(getattr(config, "context", []) or [])
    if not context_cols:
        return None

    columns = []
    with h5py.File(dataset_path, "r") as f:
        for col in context_cols:
            if col == _PEAK_KEY:
                # Per-image physical max over the full dataset, chunked so the fit never holds the whole ~GB stack.
                imgs = f[images_key]
                total = imgs.shape[0]
                peaks = np.empty(total, dtype=np.float64)
                step = 4096
                for s in range(0, total, step):
                    block = np.asarray(imgs[s:s + step], dtype=np.float32)
                    peaks[s:s + block.shape[0]] = block.reshape(block.shape[0], -1).max(axis=1)
                _, transform = ImagePathDataset._power_transform(torch.tensor(peaks))
                prompt = peak_phys
            elif col == _LAS_KEY:
                las_fit = np.ascontiguousarray(f["cat_info"][:]["LAS"], dtype=np.float64)
                _, transform = ImagePathDataset._power_transform(torch.tensor(las_fit))
                prompt = las_phys
            else:
                raise ValueError(f"Unhandled context column {col!r}; known columns are {_PEAK_KEY!r} and {_LAS_KEY!r}.")
            columns.append(transform.transform(np.asarray(prompt, np.float64).reshape(-1, 1))[:, 0])
    return np.stack(columns, axis=1).astype(np.float32)


@torch.no_grad()
def _sample_snapshot(model_dir: Path,
                     snapshot_iter: int,
                     context: np.ndarray | None,
                     n: int,
                     device: torch.device,
                     seed: int,
                     sample_batch: int = 256) -> np.ndarray:
    """
    Sample one snapshot's EMA weights on a fixed set of prompts, returned in physical Jy/beam.

    Both the latents (via `seed`) and the prompts (via `context`) are fixed across every snapshot in a sweep, so
    differences between rows reflect the model's training progress rather than sampling or prompt variance.

    Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = paths.MODEL_PARENT / model_name

    available = _available_snapshots(model_dir)
    chosen = sorted(set(snapshots))
    ----------
    model_dir : Path
        The model results directory.
    snapshot_iter : int
        Which snapshot iteration to sample from.
    context : np.ndarray | None
        The (n, context_dim) standardised context, or `None` for an unconditional model. Prompt `i` pairs with latent `i`.
    n : int
        Number of images to sample (must equal `len(context)` when conditioning).
    device : torch.device
        Device to sample on.
    seed : int
        Seed for the initial latents, shared across snapshots in a sweep.
    sample_batch : int, optional
        Maximum images pushed through the U-Net at once, by default 256. The fixed pool of latents is processed in
        chunks of this size to avoid massive memory usage. Lower as required.

    Returns
    -------
    np.ndarray
        The sampled images in Jy/beam with shape (n, H, W).
    """
    snapshot_path = _find_snapshot(model_dir, snapshot_iter)
    model, config = _load_model(model_dir, snapshot_path, _WEIGHTS_KEY)
    model = model.to(device)

    if context is not None:
        context_t = torch.as_tensor(context, dtype=torch.float32, device=device)

    # Draw every latent up front (cheap - this is just noise) so the shared-seed guarantee is identical regardless of
    # chunking; only the U-Net forward pass is memory-bound, so the latents are then sampled in sample_batch-sized
    # chunks and the resulting images moved to the CPU before the next chunk.
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(n, 1, 80, 80, device=device, generator=generator)

    chunks = []
    for start in range(0, n, sample_batch):
        lat = latents[start:start + sample_batch]
        bs = lat.shape[0]
        ctx = context_t[start:start + bs] if context is not None else None
        steps = diffusion.edm_sampling(
            model, context_batch=ctx, latents=lat, batch_size=bs, image_size=80, timesteps=_TIMESTEPS)
        chunks.append(steps[-1][:, 0].cpu().numpy())
    imgs = np.concatenate(chunks, axis=0)

    # The evaluation suite requires physical Jy/beam, so always invert a recorded transform here (unlike
    # sample_snapshot_grid, where leaving samples in model space is a valid viewing choice).
    recorded = getattr(config, "flux_transform", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = paths.MODEL_PARENT / model_name

    available = _available_snapshots(model_dir)
    chosen = sorted(set(snapshots))
    if recorded is not None:
        imgs = np.asarray(ft.load(recorded).inverse(imgs))
    return imgs


def sweep_snapshots(model_name: str,
                    snapshots: list[int],
                    n: int | None = None,
                    seed: int = 0,
                    check_memorisation: bool = False,
                    sample_batch: int = 256) -> dict:
    """
    Sample and evaluate several snapshots of one model against a held-out dataset, to see how sample quality changes
    with training iteration.

    At training, a number of images and their related conditioning information (peak/LAS) are held-out from the training
    set and recorded in a split file (split_<model_name>.npz) in the model directory. This function loads that split,
    draws the held-out reference and its physical prompts, and samples each requested snapshot using those prompts. 
    This provides a test with fixed latents and fixed prompts, with the latter having never been seen during training,
    ensuring that differences between snapshots reflect the model's training progress rather than sampling or prompt
    variance.

    Parameters
    ----------
    model_name : str
        Model directory name under `paths.MODEL_PARENT`.
    snapshots : list[int]
        Snapshot iterations to evaluate, chosen explicitly. Under EMA the right snapshots to compare are a deliberate
        choice (e.g. 50k/70k/85k/100k), not an evenly-spaced formula.
    n : int | None, optional
        Cap the reference/prompt set to this many images (a fixed seeded subsample, shared across the sweep), by default
        None (use the whole val partition).
    seed : int, optional
        Latent/subsample seed, shared across all snapshots so rows differ by training progress only, by default 0.
    check_memorisation : bool, optional
        Also run the memorisation report (nearest-neighbour of each generated image against the train partition, with
        the held-out reference as baseline), by default False. Off by default because it is a nearest-neighbour search
        over the full train stack and is the slowest part of the report.
    sample_batch : int, optional
        Maximum images pushed through the U-Net at once, by default 256. Only caps peak GPU memory; the result is
        unchanged. Lower it if sampling OOMs, raise it if there is headroom for speed.

    Returns
    -------
    dict
        Mapping of snapshot iteration -> the full_report dict for that snapshot.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = paths.MODEL_PARENT / model_name

    available = _available_snapshots(model_dir)
    chosen = sorted(set(snapshots))
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

    train_idx, val_idx, recorded_dataset = load_split(model_dir / f"split_{model_name}.npz")
    dataset_path = Path(recorded_dataset)
    real, peak, las, _ = _load_reference_cutouts(dataset_path, val_idx, n, seed)

    # The context columns are a property of the model, not the snapshot, so build the shared prompts (fitting the
    # standardisation transforms on the full dataset) once, then reuse the same prompts and latents for every snapshot.
    config = ModelConfig.from_preset(model_dir)
    context = _build_context(config, dataset_path, peak, las)
    prompted_peak = peak if context is not None else None  # calibration needs the physical peak we conditioned on
    if context is None:
        logger.info("Model is unconditional (no context columns) - sampling without prompts; calibration disabled.")

    # Memorisation compares generated images against the images the model actually trained on (the train partition).
    train_images = _load_reference_cutouts(dataset_path, train_idx, None, seed)[0] if check_memorisation else None

    n_sample = real.shape[0]
    reports, sampled = {}, {}
    for snapshot_iter in chosen:
        logger.info(f"Sampling {n_sample} images from iteration {snapshot_iter}...")
        imgs = _sample_snapshot(model_dir, snapshot_iter, context, n_sample, device, seed, sample_batch=sample_batch)
        sampled[snapshot_iter] = imgs

        logger.info(f"Running evaluation for iteration {snapshot_iter}...")
        try:
            reports[snapshot_iter] = full_report(
                imgs, real,
                prompted_peak=prompted_peak,
                train=train_images,
                val=real if check_memorisation else None,
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

    _write_table(reports, model_dir / f"convergence_{model_name}.csv")
    _write_report_json(reports, model_dir / f"convergence_{model_name}.json")
    _plot_contact_sheet(sampled, reports, model_name, model_dir / f"convergence_{model_name}.png")
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
    logger.info("Metric vs iteration (lower physical_kid = closer to the real distribution):")
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


def _plot_contact_sheet(sampled: dict, reports: dict, model_name: str, path: Path, n_show: int = 6) -> None:
    """
    Plot one row of example images per snapshot, annotated with that snapshot's physical KID.

    Parameters
    ----------
    sampled : dict
        Mapping of iteration -> sampled image stack.
    reports : dict
        Mapping of iteration -> full_report dict, for the KID annotation.
    model_name : str
        Model name, for the figure title.
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
        kid = reports[snapshot_iter].get("physical_distribution", {}).get("physical_kid", float("nan"))
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
        axes[row][0].set_title(f"iter {snapshot_iter}  (KID {kid:.3g})", fontsize=9, loc="left")

    fig.suptitle(f"{model_name}  |  EMA weights  |  convergence sweep  |  Jy/beam", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote contact sheet to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Model directory name under model_results/.")
    parser.add_argument("--snapshots", type=int, nargs="+", required=True,
                        help="Snapshot iterations to evaluate, chosen explicitly (e.g. 50000 70000 85000 100000).")
    parser.add_argument("--n", type=int, default=0,
                        help="Cap the reference/prompt set to this many images (default 0 = the whole val partition).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Latent/subsample seed, shared across snapshots (default 0).")
    parser.add_argument("--check-memorisation", action="store_true",
                        help="Also run the memorisation report (slow: nearest-neighbour over the train partition).")
    parser.add_argument("--sample-batch", type=int, default=256,
                        help="Max images sampled through the U-Net at once (default 256); lower this if sampling OOMs. "
                             "Only affects peak GPU memory, not the result.")
    args = parser.parse_args()

    sweep_snapshots(
        args.model_name, args.snapshots, n=(None if args.n == 0 else args.n),
        seed=args.seed, check_memorisation=args.check_memorisation, sample_batch=args.sample_batch,
    )
