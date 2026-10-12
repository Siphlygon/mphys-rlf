"""
Unit tests for diffracc/scripts/snapshot_convergence_sweep.py.

Covered here: snapshot discovery/selection, the metric-vs-iteration table, JSON serialisation, and the held-out
reference loading / real-marginal conditioning (against a synthetic dataset). Actually running sweep_snapshots()
end-to-end needs a real model, GPU sampling and the source-finder-backed evaluation suite, so that path is exercised on
the cluster rather than in unit tests (mirroring how tests/test_trainer.py treats the training loop).
"""
import json
import types

import h5py
import numpy as np
import pytest

from diffracc.data.datasets import ImagePathDataset
from diffracc.scripts import snapshot_convergence_sweep as scs


def _make_dataset(path, n=200, h=8, w=8, seed=0):
    """Write a synthetic h5 with positive images and a cat_info['LAS'] column, plus a 90/10 split; return them."""
    rng = np.random.default_rng(seed)
    images = np.abs(rng.standard_normal((n, h, w)).astype(np.float32)) + 0.01  # strictly positive -> box-cox path
    las = (rng.uniform(2.0, 120.0, size=n)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("cat_info", data=np.array(list(las), dtype=[("LAS", "f4")]))
    perm = rng.permutation(n)
    train_idx, val_idx = np.sort(perm[: int(0.9 * n)]), np.sort(perm[int(0.9 * n):])
    return path, train_idx, val_idx


def _make_snapshots(tmp_path, iterations, name="mymodel"):
    """Helper to create a model directory containing snapshot files at the given iterations."""
    snap_dir = tmp_path / name / "snapshots"
    snap_dir.mkdir(parents=True)
    for it in iterations:
        (snap_dir / f"snapshot_iter_{it:08d}.pt").write_text("dummy", encoding="utf-8")
    return tmp_path / name


class TestAvailableSnapshots:
    """Tests for _available_snapshots()'s discovery of snapshot iterations."""

    def test_lists_iterations_sorted(self, tmp_path):
        """Testing snapshot iterations are parsed out of the filenames and returned ascending."""
        model_dir = _make_snapshots(tmp_path, [100000, 2000, 40000])
        assert scs._available_snapshots(model_dir) == [2000, 40000, 100000]

    def test_missing_snapshots_dir_raises(self, tmp_path):
        """Testing a model with no snapshots directory raises rather than silently sweeping nothing."""
        (tmp_path / "bare").mkdir()
        with pytest.raises(FileNotFoundError):
            scs._available_snapshots(tmp_path / "bare")

    def test_empty_snapshots_dir_raises(self, tmp_path):
        """Testing an existing but empty snapshots directory raises."""
        model_dir = _make_snapshots(tmp_path, [])
        with pytest.raises(FileNotFoundError):
            scs._available_snapshots(model_dir)


class TestExtractTable:
    """Tests for _extract_table()'s flattening of nested reports into table rows."""

    def test_includes_only_metrics_present(self):
        """Testing sections that were skipped (e.g. no calibration/memorisation) don't become columns."""
        reports = {1000: {"physical_distribution": {"physical_kid": 0.03, "density": 0.9}}}
        headers, _ = scs._extract_table(reports)
        assert headers == ["iteration", "physical_kid", "density"]

    def test_rows_are_sorted_by_iteration(self):
        """Testing rows come out in ascending iteration order regardless of dict insertion order."""
        reports = {
            100000: {"physical_distribution": {"physical_kid": 1.0}},
            2000: {"physical_distribution": {"physical_kid": 9.0}},
        }
        _, rows = scs._extract_table(reports)
        assert [r[0] for r in rows] == [2000, 100000]

    def test_failed_snapshot_becomes_nan_row(self):
        """Testing a snapshot whose evaluation failed still appears, as NaN - the sweep must not silently drop it."""
        reports = {
            2000: {"evaluation_error": "ValueError: empty"},
            50000: {"physical_distribution": {"physical_kid": 4.0}},
        }
        headers, rows = scs._extract_table(reports)
        assert headers == ["iteration", "physical_kid"]
        assert rows[0][0] == 2000 and np.isnan(rows[0][1])
        assert rows[1] == [50000, 4.0]

    def test_metric_missing_from_only_some_snapshots_is_nan(self):
        """Testing a metric present in one snapshot but not another yields NaN for the one lacking it."""
        reports = {
            1000: {"physical_distribution": {"physical_kid": 5.0}},
            2000: {"physical_distribution": {"physical_kid": 4.0},
                   "memorisation": {"gen_nn_median": 0.5}},
        }
        headers, rows = scs._extract_table(reports)
        assert "memo_nn_median" in headers
        col = headers.index("memo_nn_median")
        assert np.isnan(rows[0][col])
        assert rows[1][col] == 0.5


class TestWriteOutputs:
    """Tests for the CSV/JSON writers."""

    def test_csv_matches_extracted_table(self, tmp_path):
        """Testing the CSV's header and rows are exactly what _extract_table produced."""
        reports = {1000: {"physical_distribution": {"physical_kid": 0.03, "density": 0.9}}}
        path = tmp_path / "t.csv"
        scs._write_table(reports, path)
        lines = [line for line in path.read_text().splitlines() if line]
        assert lines[0] == "iteration,physical_kid,density"
        assert lines[1] == "1000,0.03,0.9"

    def test_json_drops_bulky_per_image_arrays(self, tmp_path):
        """Testing large per-image arrays (nn distances etc.) are stripped, keeping the summary file small."""
        reports = {1000: {"memorisation": {"gen_nn_median": 0.4, "gen_nn_distances": np.arange(500.0)}}}
        path = tmp_path / "t.json"
        scs._write_report_json(reports, path)
        saved = json.loads(path.read_text())
        assert "gen_nn_distances" not in saved["1000"]["memorisation"]
        assert saved["1000"]["memorisation"]["gen_nn_median"] == 0.4

    def test_json_keeps_short_arrays(self, tmp_path):
        """Testing short arrays (e.g. per-bin calibration summaries) are preserved as lists, not dropped."""
        reports = {1000: {"calibration": {"slope": 0.9, "bin_centers": np.arange(5.0)}}}
        path = tmp_path / "t.json"
        scs._write_report_json(reports, path)
        saved = json.loads(path.read_text())
        assert saved["1000"]["calibration"]["bin_centers"] == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestLoadReference:
    """_load_reference draws the held-out reference and its physical prompts for a partition's rows."""

    def test_loads_partition_rows_with_peak_and_las(self, tmp_path):
        """Images/peak/LAS come from exactly the partition's rows; peak is the per-image max."""
        path, _, val_idx = _make_dataset(tmp_path / "d.h5", n=300, seed=1)
        images, peak, las, idx = scs._load_reference_cutouts(path, val_idx, n=None, seed=0)
        np.testing.assert_array_equal(idx, val_idx)
        assert images.shape[0] == val_idx.size == peak.size == las.size
        with h5py.File(path, "r") as f:
            expected_img = np.asarray(f["images"][val_idx], dtype=np.float32)
            expected_las = np.asarray(f["cat_info"][val_idx]["LAS"], dtype=np.float64)
        np.testing.assert_array_equal(images, expected_img)
        np.testing.assert_allclose(peak, expected_img.reshape(expected_img.shape[0], -1).max(axis=1))
        np.testing.assert_allclose(las, expected_las)

    def test_n_cap_is_a_fixed_seeded_subsample(self, tmp_path):
        """--n caps the set to a subsample of the partition, identical for a given seed (shared across the sweep)."""
        path, _, val_idx = _make_dataset(tmp_path / "d.h5", n=1000, seed=2)  # val ~100, comfortably above the cap
        _, _, _, idx_a = scs._load_reference_cutouts(path, val_idx, n=30, seed=7)
        _, _, _, idx_b = scs._load_reference_cutouts(path, val_idx, n=30, seed=7)
        assert idx_a.size == 30
        np.testing.assert_array_equal(idx_a, idx_b)
        assert set(idx_a.tolist()).issubset(val_idx.tolist())


class TestBuildContext:
    """_build_context maps physical prompts into the model's standardised context, fit on the full dataset."""

    def test_matches_full_dataset_fit_in_column_order(self, tmp_path):
        """Columns equal ImagePathDataset._power_transform (fit on the full dataset, as training does), in config order."""
        import torch
        path, _, _ = _make_dataset(tmp_path / "d.h5", n=250, seed=4)
        peak = np.array([0.5, 1.0, 5.0])
        las = np.array([10.0, 30.0, 90.0])
        cfg = types.SimpleNamespace(context=[scs._LAS_KEY, scs._PEAK_KEY])
        ctx = scs._build_context(cfg, path, peak, las)

        with h5py.File(path, "r") as f:
            imgs = np.asarray(f["images"][:], dtype=np.float32)
            las_full = np.ascontiguousarray(f["cat_info"][:]["LAS"], dtype=np.float64)
        peaks_full = imgs.reshape(imgs.shape[0], -1).max(axis=1).astype(np.float64)
        _, peak_ref = ImagePathDataset._power_transform(torch.tensor(peaks_full))
        _, las_ref = ImagePathDataset._power_transform(torch.tensor(las_full))

        assert ctx.shape == (3, 2) and ctx.dtype == np.float32
        np.testing.assert_allclose(ctx[:, 0], las_ref.transform(las.reshape(-1, 1))[:, 0], rtol=1e-6)
        np.testing.assert_allclose(ctx[:, 1], peak_ref.transform(peak.reshape(-1, 1))[:, 0], rtol=1e-6)

    def test_unconditional_returns_none(self, tmp_path):
        """A model with no context columns samples unconditionally (context None)."""
        path, _, _ = _make_dataset(tmp_path / "d.h5", n=120, seed=5)
        assert scs._build_context(types.SimpleNamespace(context=[]), path, np.array([1.0]), np.array([1.0])) is None

    def test_unknown_column_raises(self, tmp_path):
        """An unrecognised context column is a hard error, not a silent miscalibration."""
        path, _, _ = _make_dataset(tmp_path / "d.h5", n=120, seed=6)
        cfg = types.SimpleNamespace(context=["something_else"])
        with pytest.raises(ValueError, match="Unhandled context column"):
            scs._build_context(cfg, path, np.array([1.0]), np.array([1.0]))
