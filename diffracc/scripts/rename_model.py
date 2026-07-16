"""
Safely rename/move a model's results directory, so a later pickup can actually find everything.

DiffusionTrainer.from_pickup / ModelConfig.from_preset both derive the config filename from the *current* directory
name (config_<dirname>.json), and OutputManager rebuilds the results folder from the "model_name" field *stored
inside* that JSON. Renaming or moving a model directory with a plain `mv` breaks both: the internal files are still
named after the old model name, and the JSON still claims the old name, so pickup either can't find the config at
all, or finds it but then looks for parameters/logs/losses in the wrong (old) location.

This script fixes both problems in one step: it renames every file inside the model directory that embeds the model
name, and patches the JSON's "model_name" field to match. It also handles the directory move itself, so it's safe to
run either before or after you've already `mv`-ed the folder.

Snapshot files under snapshots/snapshot_iter_*.pt and the wandb_run_id.txt marker do NOT embed the model name, so
they need no renaming and are left untouched.

Usage (from the repo root):
    python -m diffracc.scripts.rename_model --old-name snr15_inclusive_las_5 --new-name snr15_inclusive_las
    python -m diffracc.scripts.rename_model --old-name foo --new-name bar --dry-run   # preview only
"""
import argparse
import json
from pathlib import Path

from ..utils import paths
from ..utils.logger import get_logger

logger = get_logger(__name__)

# (prefix, suffix) for every file OutputManager templates on the model name - see OutputManager._setup_files.
# Snapshot files (snapshots/snapshot_iter_*.pt) and wandb_run_id.txt are deliberately absent: neither embeds the
# model name, so neither needs renaming.
_TEMPLATED_FILES = [
    ("config_", ".json"),
    ("parameters_", ".pt"),
    ("losses_train_", ".csv"),
    ("losses_val_", ".csv"),
    ("training_log_", ".log"),
    ("power_ema_", ".pt"),
]


def _rename_file(old_file: Path, new_file: Path, dry_run: bool) -> None:
    """
    Rename a single file if it exists, refusing to clobber an existing destination.

    Parameters
    ----------
    old_file : Path
        The file to rename, if it exists.
    new_file : Path
        The destination path.
    dry_run : bool
        If True, report what would happen without touching the filesystem.

    Raises
    ------
    FileExistsError
        If both old_file and new_file already exist - renaming would silently overwrite new_file, which is very
        likely a leftover from a previous partial rename attempt and needs a human to look at it, not a silent clobber.
    """
    if not old_file.exists():
        return
    if new_file.exists():
        raise FileExistsError(
            f"Both {old_file} and {new_file} exist - refusing to overwrite. "
            "Resolve this manually (likely a leftover from a previous partial rename)."
        )
    logger.info(f"{'[dry-run] Would rename' if dry_run else 'Renaming'} {old_file.name} -> {new_file.name}")
    if not dry_run:
        old_file.rename(new_file)


def rename_model(old_name: str,
                 new_name: str,
                 parent_dir: Path = paths.MODEL_PARENT,
                 dry_run: bool = False) -> Path:
    """
    Rename a model's results directory and every internal file/field that embeds its name.

    Safe to call whether the directory move has already happened (e.g. via a manual `mv` on the cluster) or not:

    - If parent_dir/old_name exists and parent_dir/new_name does not, the directory itself is renamed first.
    - If parent_dir/new_name already exists (the move already happened), the directory is left alone and only the
      internal files/JSON field are fixed.

    Parameters
    ----------
    old_name : str
        The model name currently embedded in the internal filenames and the config JSON's "model_name" field.
    new_name : str
        The model name to rename everything to.
    parent_dir : Path, optional
        Parent directory holding model result folders, by default paths.MODEL_PARENT.
    dry_run : bool, optional
        If True, report every action that would be taken without touching the filesystem, by default False.

    Returns
    -------
    Path
        The model directory, at parent_dir/new_name, with every internal name now consistent with new_name.

    Raises
    ------
    ValueError
        If old_name equals new_name (nothing to rename).
    FileNotFoundError
        If neither parent_dir/old_name nor parent_dir/new_name exists.
    FileExistsError
        If both parent_dir/old_name and parent_dir/new_name already exist (ambiguous which is the real one), or if
        renaming an internal file would overwrite an existing one of the same target name.
    """
    if old_name == new_name:
        raise ValueError(f"old_name and new_name are both {old_name!r} - nothing to rename.")

    old_dir = parent_dir / old_name
    new_dir = parent_dir / new_name

    if old_dir.exists() and new_dir.exists():
        raise FileExistsError(
            f"Both {old_dir} and {new_dir} exist - ambiguous which is the real model directory. "
            "Remove or merge one manually before rerunning."
        )
    if old_dir.exists():
        logger.info(f"{'[dry-run] Would move' if dry_run else 'Moving'} {old_dir} -> {new_dir}")
        if not dry_run:
            old_dir.rename(new_dir)
    elif not new_dir.exists():
        raise FileNotFoundError(f"Neither {old_dir} nor {new_dir} exists.")
    else:
        logger.info(f"{new_dir} already exists (directory already moved) - fixing internal file names only.")

    for prefix, suffix in _TEMPLATED_FILES:
        _rename_file(new_dir / f"{prefix}{old_name}{suffix}", new_dir / f"{prefix}{new_name}{suffix}", dry_run)

    config_file = new_dir / f"config_{new_name}.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            param_dict = json.load(f)
        if param_dict.get("model_name") != new_name:
            logger.info(
                f"{'[dry-run] Would set' if dry_run else 'Setting'} "
                f"model_name: {param_dict.get('model_name')!r} -> {new_name!r} in {config_file.name}"
            )
            if not dry_run:
                param_dict["model_name"] = new_name
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(param_dict, f, indent=4)
    else:
        logger.warning(f"No {config_file.name} found in {new_dir} - nothing to patch (or it was never created).")

    logger.info(f"{'[dry-run] Done (no changes made).' if dry_run else f'Done - model is now {new_name!r} at {new_dir}.'}")
    return new_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-name", required=True, help="The model name currently embedded in filenames/JSON.")
    parser.add_argument("--new-name", required=True, help="The model name to rename everything to.")
    parser.add_argument("--parent-dir", type=Path, default=None,
                        help="Parent directory holding model folders (default: the configured MODEL_PARENT).")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without touching any files.")
    args = parser.parse_args()

    rename_model(args.old_name, args.new_name,
                parent_dir=args.parent_dir or paths.MODEL_PARENT, dry_run=args.dry_run)
