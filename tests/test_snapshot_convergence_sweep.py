"""
Unit tests for diffracc/scripts/snapshot_convergence_sweep.py.

Only the pure logic is covered here - snapshot discovery/selection, the metric-vs-iteration table, and JSON
serialisation. Actually running sweep_snapshots() needs a real model, GPU sampling and the source-finder-backed
evaluation suite, so that end-to-end path is exercised on the cluster rather than in unit tests (mirroring how
tests/test_trainer.py treats the training loop).
"""
import json

import numpy as np
import pytest

from diffracc.scripts import snapshot_convergence_sweep as scs


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


class TestPickEvenlySpaced:
    """Tests for _pick_evenly_spaced()'s auto-selection of snapshots to evaluate."""

    def test_picks_requested_count_spanning_the_run(self):
        """Testing the selection spans from the earliest to the latest snapshot."""
        available = list(range(2000, 102000, 2000))
        chosen = scs._pick_evenly_spaced(available, 5)
        assert len(chosen) == 5
        assert chosen[0] == 2000
        assert chosen[-1] == 100000

    def test_returns_all_when_fewer_available_than_requested(self):
        """Testing asking for more snapshots than exist just returns all of them, rather than erroring."""
        available = [2000, 4000]
        assert scs._pick_evenly_spaced(available, 5) == [2000, 4000]

    def test_result_is_sorted_and_deduplicated(self):
        """Testing rounding onto duplicate indices can't produce repeats or out-of-order entries."""
        available = [1000, 2000, 3000]
        chosen = scs._pick_evenly_spaced(available, 3)
        assert chosen == sorted(set(chosen))


class TestExtractTable:
    """Tests for _extract_table()'s flattening of nested reports into table rows."""

    def test_includes_only_metrics_present(self):
        """Testing sections that were skipped (e.g. no calibration/memorisation) don't become columns."""
        reports = {1000: {"physical_distribution": {"physical_fid": 5.0, "physical_kid": 0.03}}}
        headers, _ = scs._extract_table(reports)
        assert headers == ["iteration", "physical_fid", "physical_kid"]

    def test_rows_are_sorted_by_iteration(self):
        """Testing rows come out in ascending iteration order regardless of dict insertion order."""
        reports = {
            100000: {"physical_distribution": {"physical_fid": 1.0}},
            2000: {"physical_distribution": {"physical_fid": 9.0}},
        }
        _, rows = scs._extract_table(reports)
        assert [r[0] for r in rows] == [2000, 100000]

    def test_failed_snapshot_becomes_nan_row(self):
        """Testing a snapshot whose evaluation failed still appears, as NaN - the sweep must not silently drop it."""
        reports = {
            2000: {"evaluation_error": "ValueError: empty"},
            50000: {"physical_distribution": {"physical_fid": 4.0}},
        }
        headers, rows = scs._extract_table(reports)
        assert headers == ["iteration", "physical_fid"]
        assert rows[0][0] == 2000 and np.isnan(rows[0][1])
        assert rows[1] == [50000, 4.0]

    def test_metric_missing_from_only_some_snapshots_is_nan(self):
        """Testing a metric present in one snapshot but not another yields NaN for the one lacking it."""
        reports = {
            1000: {"physical_distribution": {"physical_fid": 5.0}},
            2000: {"physical_distribution": {"physical_fid": 4.0},
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
        reports = {1000: {"physical_distribution": {"physical_fid": 5.0, "physical_kid": 0.03}}}
        path = tmp_path / "t.csv"
        scs._write_table(reports, path)
        lines = [line for line in path.read_text().splitlines() if line]
        assert lines[0] == "iteration,physical_fid,physical_kid"
        assert lines[1] == "1000,5.0,0.03"

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
