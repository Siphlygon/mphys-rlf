"""Unit tests for diffracc/utils/distributed.py."""
import numpy as np
import pytest

from diffracc.utils.distributed import DistributedUtils, distribute


@pytest.fixture(autouse=True)
def _clean_slurm_env(monkeypatch):
    """Ensure no ambient SLURM_ARRAY_* env vars leak in from the real environment/other tests."""
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)


class TestDefaults:
    """Tests for the default behavior of DistributedUtils when no Slurm environment variables are set."""

    def test_defaults_to_task_0_of_1_when_no_slurm_env(self):
        """
        Test that DistributedUtils defaults to task ID 0 and task count 1 when no Slurm environment variables are set.
        """
        du = DistributedUtils()
        assert du.get_task_id() == 0
        assert du.get_task_count() == 1
        assert du.is_distributed() is False

    def test_default_bin_covers_the_whole_range(self):
        """Test that the default bin covers the whole range when no Slurm environment variables are set."""
        du = DistributedUtils()
        assert du.get_bin_start(10) == 0
        assert du.get_bin_end(10) == 10


class TestSlurmEnv:
    """Tests for the behavior of DistributedUtils when Slurm environment variables are set."""

    def test_reads_task_id_and_count_from_environment(self, monkeypatch):
        """Test that DistributedUtils correctly reads the task ID and task count from the environment variables."""
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
        du = DistributedUtils()
        assert du.get_task_id() == 2
        assert du.get_task_count() == 4

    def test_is_distributed_true_when_task_count_greater_than_one(self, monkeypatch):
        """Test that is_distributed returns True when the task count is greater than one."""
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
        assert DistributedUtils().is_distributed() is True

    def test_is_distributed_false_when_task_count_is_explicitly_one(self, monkeypatch):
        """Test that is_distributed returns False when the task count is explicitly set to one."""
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "1")
        assert DistributedUtils().is_distributed() is False


class TestBinSplitting:
    """Tests for the bin splitting logic in DistributedUtils."""

    def test_evenly_divides_bins_across_tasks(self, monkeypatch):
        """Test that the bin splitting logic evenly divides bins across tasks when n is divisible by task_count."""
        # n=8 split across 4 tasks -> each task gets exactly 2
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
        bins = []
        for task_id in range(4):
            monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task_id))
            du = DistributedUtils()
            bins.append((du.get_bin_start(8), du.get_bin_end(8)))
        assert bins == [(0, 2), (2, 4), (4, 6), (6, 8)]

    def test_bins_are_contiguous_and_cover_the_full_range_when_not_evenly_divisible(self, monkeypatch):
        """
        Test that the bin splitting logic produces contiguous bins that cover the full range when n is not divisible by
        task_count.
        """
        n = 10
        task_count = 3
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", str(task_count))
        bins = []
        for task_id in range(task_count):
            monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task_id))
            du = DistributedUtils()
            bins.append((du.get_bin_start(n), du.get_bin_end(n)))

        assert bins[0][0] == 0
        assert bins[-1][1] == n
        for (start, end), (next_start, _) in zip(bins, bins[1:]):
            assert end == next_start  # no gaps, no overlap


class TestDistribute:
    """Tests for the distribute function, which slices an array according to the current task's bin."""

    def test_returns_full_array_when_not_distributed(self):
        """Test that distribute returns the full array when not running in a distributed environment."""
        arr = np.arange(10)
        np.testing.assert_array_equal(distribute(arr), arr)

    def test_returns_only_this_tasks_slice(self, monkeypatch):
        """Test that distribute returns only the slice of the array corresponding to this task's bin."""
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
        arr = np.arange(8)
        np.testing.assert_array_equal(distribute(arr), arr[2:4])

    def test_every_tasks_slice_together_reconstructs_the_full_array_without_overlap(self, monkeypatch):
        """
        Test that the slices returned by distribute for each task together reconstruct the full array without overlap.
        """
        arr = np.arange(10)
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "3")
        pieces = []
        for task_id in range(3):
            monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task_id))
            pieces.append(distribute(arr))
        np.testing.assert_array_equal(np.concatenate(pieces), arr)
