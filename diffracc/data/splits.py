"""Read and write the train/val partition a training run used, so evaluation can reuse the exact held-out set."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_split(path: str | Path, train_idx, val_idx, dataset_path: str | Path) -> None:
    """
    Save the train/val row indices (sorted) and the dataset they index to an `.npz` file.

    Parameters
    ----------
    path : str or Path
        The path to the `.npz` file to save the split to.
    train_idx : array-like
        The row indices of the training set.
    val_idx : array-like
        The row indices of the validation set.
    dataset_path : str or Path
        The path to the dataset the indices index into.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train=np.sort(np.asarray(train_idx, dtype=np.int64)),
        val=np.sort(np.asarray(val_idx, dtype=np.int64)),
        dataset_path=str(dataset_path),
    )


def load_split(path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load a split saved by :func:`save_split`, returning `(train_idx, val_idx, dataset_path)`.

    Parameters
    ----------
    path : str or Path
        The path to the `.npz` file to load the split from.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, str]
        The training indices, validation indices, and dataset path.
    """
    with np.load(path, allow_pickle=False) as data:
        return data["train"], data["val"], str(data["dataset_path"])
