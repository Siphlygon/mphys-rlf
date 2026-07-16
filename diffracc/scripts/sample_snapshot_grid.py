"""
Quick visual sanity check for a training run: load a model snapshot, sample a handful of images, and save them as a
PNG grid so as to eyeball whether the model is producing structure (galaxies) or just static noise.

This is deliberately lightweight - it bypasses the full FITS/PyBDSF pipeline. Images are shown in the model's own output
space (the asinh-transformed space if flux transform was used at training time), NOT inverted back to Jy/beam -
structure vs noise is equally visible there, and skipping the inverse avoids the huge dynamic range of physical fluxes
washing out faint structure.

The final grid is written to <model_dir>/sample_grid_<key>_iter<N>.png by default.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import flux_transforms as ft
from ..model import diffusion, unet
from ..model.config import ModelConfig
from ..utils import paths


def _find_snapshot(model_dir: Path, snapshot_iter: int | None) -> Path:
    """
    Locate a snapshot .pt file in <model_dir>/snapshots. If snapshot_iter is None, return the highest-iteration
    snapshot; otherwise return the one matching that iteration.

    Parameters
    ----------
    model_dir : Path
        The model results directory (contains a 'snapshots' subdirectory).
    snapshot_iter : int or None
        The iteration to load, or None for the latest.

    Returns
    -------
    Path
        Path to the chosen snapshot file.
    """
    snap_dir = model_dir / "snapshots"
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"No snapshots directory at {snap_dir}")

    # Snapshots are named snapshot_iter_<8-digit iteration>.pt (see OutputManager.save_snapshot).
    snaps = {int(m.group(1)): p for p in snap_dir.glob("snapshot_iter_*.pt")
             if (m := re.search(r"snapshot_iter_(\d+)", p.name))}
    if not snaps:
        raise FileNotFoundError(f"No snapshot_iter_*.pt files in {snap_dir}")

    if snapshot_iter is None:
        chosen = max(snaps)
    elif snapshot_iter in snaps:
        chosen = snapshot_iter
    else:
        raise FileNotFoundError(
            f"No snapshot for iteration {snapshot_iter} in {snap_dir}; available: {sorted(snaps)}")
    return snaps[chosen]


def _load_model(model_dir: Path, snapshot_path: Path, key: str) -> tuple[torch.nn.Module, ModelConfig]:
    """
    Build the model from its saved config and load a snapshot's weights.

    Parameters
    ----------
    model_dir : Path
        The model results directory (contains config_<name>.json).
    snapshot_path : Path
        The snapshot .pt to load weights from.
    key : str
        Which state dict to load: "ema_model" (the deployed weights) or "model" (raw training weights).

    Returns
    -------
    tuple[torch.nn.Module, ModelConfig]
        The loaded model (in eval mode, on CPU) and its config.
    """
    config = ModelConfig.from_preset(model_dir)
    model = unet.EDMPrecond.from_config(config)

    # weights_only=False: our snapshots also carry optimizer/EMA-bookkeeping objects, not just plain tensors.
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)

    # The snapshot may not hold an EMA state dict if it was saved before EMA initialisation; fall back to "model".
    if key == "ema_model" and checkpoint.get("ema_model") is None:
        print(f"[warn] '{key}' not available in snapshot; falling back to 'model' weights.")
        key = "model"

    state_dict = checkpoint[key]
    # The EMA (AveragedModel) state dict prefixes every parameter with "module." and adds an "n_averaged" buffer;
    # strip the prefix and drop non-module keys so it loads into the bare model. The plain "model" dict needs neither.
    if key != "model":
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k.startswith("module.")}

    model.load_state_dict(state_dict)
    return model.eval(), config


@torch.no_grad()
def sample_grid(model_name: str,
                n: int = 16,
                snapshot_iter: int | None = None,
                key: str = "ema_model",
                timesteps: int = 25,
                invert: str = "auto",
                out_path: Path | None = None) -> Path:
    """
    Sample n images from a model snapshot and save them as a PNG grid.

    Every image is drawn with the same, central (median) conditioning prompt - a zero vector in the standardised
    context space, which maps back to the median peak flux / LAS - so the panels differ only by their initial noise.
    That isolates the question "does the model turn noise into structure?" from any conditioning effects.

    Parameters
    ----------
    model_name : str
        Name of the model directory under paths.MODEL_PARENT.
    n : int, optional
        Number of images to sample, by default 16.
    snapshot_iter : int or None, optional
        Which snapshot iteration to load, or None (default) for the latest.
    key : str, optional
        Which weights to sample from: "ema_model" (default) or "model".
    timesteps : int, optional
        Number of sampling steps, by default 25.
    invert : str, optional
        Whether to invert the model's global flux transform back to Jy/beam before plotting: "auto" (default) inverts
        iff the model config records a flux transform, "yes" forces it (errors if none recorded), "no" leaves samples
        in the model's raw output space. Inverting puts a transform-trained model on the same Jy/beam axis as a
        raw-trained one for a fair side-by-side.
    out_path : Path or None, optional
        Where to write the PNG, or None to auto-name it in the model directory.

    Returns
    -------
    Path
        The path the PNG grid was written to.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = paths.MODEL_PARENT / model_name

    snapshot_path = _find_snapshot(model_dir, snapshot_iter)
    print(f"Loading '{key}' weights from {snapshot_path} onto {device}...")
    model, config = _load_model(model_dir, snapshot_path, key)
    model = model.to(device)

    # Central conditioning prompt (zeros = median in the standardised context space). context_dim==0 -> unconditional.
    context_dim = model.model.context_dim
    context = torch.zeros(n, context_dim, device=device) if context_dim else None

    latents = torch.randn(n, 1, 80, 80, device=device)
    print(f"Sampling {n} images ({timesteps} steps, context_dim={context_dim})...")
    steps = diffusion.edm_sampling(
        model, context_batch=context, latents=latents, batch_size=n, image_size=80, timesteps=timesteps)
    imgs = steps[-1][:, 0].cpu().numpy()  # (n, 80, 80), final denoised step

    # Optionally invert the training-time flux transform so samples are back in physical Jy/beam. The trainer records
    # the transform in the model config; a raw-trained model has none (its output is already Jy/beam).
    recorded = getattr(config, "flux_transform", None)
    do_invert = {"yes": True, "no": False, "auto": recorded is not None}[invert]
    if do_invert:
        if recorded is None:
            raise ValueError("invert='yes' but this model's config records no flux transform to invert.")
        transform = ft.load(recorded)
        imgs = np.asarray(transform.inverse(imgs))
        space = "Jy/beam (transform inverted)"
    else:
        space = "Jy/beam (no transform)" if recorded is None else "model output space (asinh)"
    print(f"Image space: {space}")

    # Report basic stats - a near-constant image (tiny std, no bright core) is the signature of a noise-only model.
    print(f"Sample pixel stats: min={imgs.min():.3g} max={imgs.max():.3g} "
          f"mean={imgs.mean():.3g} std={imgs.std():.3g}")
    per_image_peak = imgs.reshape(n, -1).max(axis=1)
    print(f"Per-image peak: mean={per_image_peak.mean():.3g} (a real source should stand well above the background)")

    # Grid layout
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for idx, ax in zip(range(n), np.atleast_1d(axes).ravel()):
        img = imgs[idx]
        # Robust per-image stretch (1st-99th percentile) so a single hot pixel doesn't wash out faint structure.
        vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
        ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax if vmax > vmin else None, origin="lower")

    iter_tag = re.search(r"snapshot_iter_(\d+)", snapshot_path.name).group(1)
    fig.suptitle(f"{model_name}  |  {key}  |  iter {int(iter_tag)}  |  {space}", fontsize=10)
    fig.tight_layout()

    space_tag = "jy" if "Jy/beam" in space else "asinh"
    out_path = out_path or (model_dir / f"sample_grid_{key}_iter{int(iter_tag)}_{space_tag}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Model directory name under model_results/.")
    parser.add_argument("--n", type=int, default=16, help="Number of images to sample (default 16).")
    parser.add_argument("--snapshot", type=int, default=None,
                        help="Snapshot iteration to load (default: latest).")
    parser.add_argument("--key", choices=["ema_model", "model"], default="ema_model",
                        help="Which weights to sample from (default: ema_model).")
    parser.add_argument("--timesteps", type=int, default=25, help="Sampling steps (default 25).")
    parser.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                        help="Invert the flux transform to Jy/beam: auto (if recorded), yes (force), no (raw output).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: auto in model dir).")
    args = parser.parse_args()

    sample_grid(args.model_name, n=args.n, snapshot_iter=args.snapshot, key=args.key,
                timesteps=args.timesteps, invert=args.invert, out_path=args.out)
