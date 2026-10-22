"""Unit tests for diffracc/data/splits.py's train/val split read/write."""
import numpy as np

from diffracc.data import splits


def test_roundtrip(tmp_path):
    """Test that save_split then load_split returns the same indices (sorted) and dataset path."""
    splits.save_split(tmp_path / "s.npz", train_idx=[3, 1, 2], val_idx=[9, 5], dataset_path="/data/d.h5")
    train, val, dataset_path = splits.load_split(tmp_path / "s.npz")
    np.testing.assert_array_equal(train, [1, 2, 3])
    np.testing.assert_array_equal(val, [5, 9])
    assert dataset_path == "/data/d.h5"


def test_indices_are_int64(tmp_path):
    """Test that indices load back as int64, valid for h5py fancy-indexing."""
    splits.save_split(tmp_path / "s.npz", train_idx=np.arange(10), val_idx=[10, 11], dataset_path="d.h5")
    train, val, _ = splits.load_split(tmp_path / "s.npz")
    assert train.dtype == np.int64 and val.dtype == np.int64
