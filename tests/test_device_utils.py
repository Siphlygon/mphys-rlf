"""Unit tests for diffracc/utils/device_utils.py. No real GPU/nvidia_smi hardware is needed - nvidia_smi and DDP
are monkeypatched with fakes throughout, since this module's job is orchestration logic (parsing memory info,
picking device IDs, dispatching to .to()/DDP), not the GPU calls themselves."""
import os

import numpy as np
import pytest
import torch

from diffracc.utils import device_utils as du


class _FakeMemInfo:
    """A fake object to simulate the memory info returned by nvidia_smi, with used, free, and total memory in bytes."""
    def __init__(self, used, free, total):
        self.used = used
        self.free = free
        self.total = total


class _FakeNvidiaSmi:
    """
    mem_infos_mib: list of (used, free, total) in MiB, one per fake GPU - internally stored/returned in bytes to
    match the real nvidia_smi API, which device_utils.py converts back to MiB via /1024**2.
    """
    def __init__(self, mem_infos_mib):
        self.mem_infos_mib = mem_infos_mib
        self.init_called = False

    def nvmlInit(self):
        self.init_called = True

    def nvmlDeviceGetCount(self):
        return len(self.mem_infos_mib)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetMemoryInfo(self, handle):
        used, free, total = self.mem_infos_mib[handle]
        return _FakeMemInfo(used * 1024**2, free * 1024**2, total * 1024**2)


@pytest.fixture
def fake_gpus(monkeypatch):
    """Fixture to monkeypatch nvidia_smi with a fake that returns specified memory info for each GPU."""
    def _patch(mem_infos_mib):
        fake = _FakeNvidiaSmi(mem_infos_mib)
        monkeypatch.setattr(du, "nvidia_smi", fake)
        return fake
    return _patch


class TestPhysicalGpuDf:
    """Tests for the physical_gpu_df function, which returns a DataFrame of GPU memory info."""

    def test_returns_memory_info_in_mib_per_device(self, fake_gpus):
        """Test that physical_gpu_df returns a DataFrame with the correct memory info in MiB for each device."""
        fake_gpus([(1000, 3000, 4000), (500, 1500, 2000)])
        df = du.physical_gpu_df()
        assert list(df["memory.used (MiB)"]) == [1000.0, 500.0]
        assert list(df["memory.free (MiB)"]) == [3000.0, 1500.0]
        assert list(df["memory.total (MiB)"]) == [4000.0, 2000.0]

    def test_calls_nvml_init(self, fake_gpus):
        """Test that physical_gpu_df calls nvmlInit to initialize the NVML library."""
        fake = fake_gpus([(0, 100, 100)])
        du.physical_gpu_df()
        assert fake.init_called is True

    def test_no_gpus_raises_runtime_error(self, fake_gpus):
        """Test that physical_gpu_df raises a RuntimeError when no GPUs are detected."""
        fake_gpus([])
        with pytest.raises(RuntimeError):
            du.physical_gpu_df()


class TestVisibleGpusBySpace:
    """Tests for the visible_gpus_by_space function, which returns a list of visible GPUs sorted by available memory."""

    def test_sorted_by_free_memory_descending(self, fake_gpus):
        """Test that visible_gpus_by_space returns a list of GPU indices sorted by free memory in descending order."""
        # device 0 has 2 GiB free, device 1 has 8 GiB free, device 2 has 4 GiB free
        fake_gpus([(0, 2000, 4000), (0, 8000, 8000), (0, 4000, 4000)])
        assert du.visible_gpus_by_space() == [1, 2, 0]

    def test_filters_and_renumbers_by_cuda_visible_devices(self, monkeypatch, fake_gpus):
        """
        Test that visible_gpus_by_space filters and renumbers GPUs based on the CUDA_VISIBLE_DEVICES environment
        variable.
        """
        fake_gpus([(0, 2000, 4000), (0, 8000, 8000), (0, 4000, 4000), (0, 1000, 4000)])
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
        # only physical devices 1 and 3 are visible; renumbered to 0,1 in visibility order, then sorted by free mem
        result = du.visible_gpus_by_space(renumber=True)
        assert set(result) == {0, 1}

    def test_no_renumber_keeps_physical_ids_when_filtered(self, monkeypatch, fake_gpus):
        """
        Test that visible_gpus_by_space keeps physical GPU IDs when renumber=False, even when filtered by
        CUDA_VISIBLE_DEVICES.
        """
        fake_gpus([(0, 2000, 4000), (0, 8000, 8000), (0, 4000, 4000), (0, 1000, 4000)])
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
        result = du.visible_gpus_by_space(renumber=False)
        assert set(result) == {1, 3}

    def test_no_cuda_visible_devices_returns_all_sorted(self, monkeypatch, fake_gpus):
        """
        Test that visible_gpus_by_space returns all GPUs sorted by free memory when CUDA_VISIBLE_DEVICES is not set.
        """
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_gpus([(0, 1000, 4000), (0, 5000, 8000)])
        assert du.visible_gpus_by_space() == [1, 0]


class _FakeModel:
    """A fake model class that simulates a PyTorch model with a .to() method for testing device distribution."""
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class _FakeDDP:
    """A fake DistributedDataParallel class that simulates wrapping a model for testing device distribution."""
    instances = []

    def __init__(self, model, device_ids=None):
        self.model = model
        self.device_ids = device_ids
        _FakeDDP.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_fake_ddp():
    """Reset the _FakeDDP.instances list before and after each test to ensure isolation."""
    _FakeDDP.instances = []
    yield
    _FakeDDP.instances = []


class TestDistributeModel:
    """
    Tests for the distribute_model function, which handles moving a model to the appropriate device(s) and wrapping in
    DDP if needed.
    """

    def test_single_device_moves_model_without_wrapping_in_ddp(self, monkeypatch):
        """Test that distribute_model moves the model to a single device without wrapping it in DDP."""
        monkeypatch.setattr(du, "DDP", _FakeDDP)
        model = _FakeModel()

        result_model, device_ids = du.distribute_model(model, n_devices=1, device_ids=[2])

        assert result_model is model
        assert model.to_calls == [torch.device("cuda", 2)]
        assert device_ids == [2]
        assert _FakeDDP.instances == []

    def test_multi_device_wraps_in_ddp(self, monkeypatch):
        """Test that distribute_model wraps the model in DDP when multiple devices are specified."""
        monkeypatch.setattr(du, "DDP", _FakeDDP)
        model = _FakeModel()

        result_model, device_ids = du.distribute_model(model, n_devices=2, device_ids=[0, 1])

        assert model.to_calls == [torch.device("cuda", 0)]
        assert device_ids == [0, 1]
        assert len(_FakeDDP.instances) == 1
        assert _FakeDDP.instances[0].model is model
        assert _FakeDDP.instances[0].device_ids == [0, 1]
        assert result_model is _FakeDDP.instances[0]

    def test_falls_back_to_visible_gpus_by_space_when_device_ids_not_given(self, monkeypatch, fake_gpus):
        """Test that distribute_model falls back to visible_gpus_by_space when device_ids is None."""
        monkeypatch.setattr(du, "DDP", _FakeDDP)
        fake_gpus([(0, 1000, 4000), (0, 5000, 8000)])  # device 1 has more free memory
        model = _FakeModel()

        _, device_ids = du.distribute_model(model, n_devices=1, device_ids=None)

        assert device_ids == [1]


class TestSetVisibleDevices:
    """Tests for the set_visible_devices function, which sets the CUDA_VISIBLE_DEVICES environment variable."""

    def test_int_spec_selects_n_gpus_by_free_space(self, monkeypatch, fake_gpus):
        """Test that set_visible_devices selects the top n GPUs by free memory when given an integer."""
        fake_gpus([(0, 1000, 4000), (0, 5000, 8000), (0, 3000, 4000)])
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        dev_ids = du.set_visible_devices(2)

        assert dev_ids == [1, 2]  # sorted by free memory descending, top 2
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"

    def test_list_spec_uses_given_ids_directly(self, monkeypatch, fake_gpus):
        """Test that set_visible_devices uses the given list of device IDs directly without sorting."""
        fake_gpus([(0, 1000, 4000), (0, 5000, 8000), (0, 3000, 4000)])
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        dev_ids = du.set_visible_devices([0, 2])

        assert dev_ids == [0, 2]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,2"

    def test_clears_existing_cuda_visible_devices_before_setting(self, monkeypatch, fake_gpus):
        """Test that set_visible_devices clears any existing CUDA_VISIBLE_DEVICES before setting new values."""
        fake_gpus([(0, 1000, 4000)])
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

        du.set_visible_devices([0])

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    def test_n_gpu_out_of_range_raises_assertion_error(self, monkeypatch, fake_gpus):
        """
        Test that set_visible_devices raises an AssertionError when the requested number of GPUs exceeds available
        GPUs.
        """
        fake_gpus([(0, 1000, 4000)])  # only 1 physical GPU
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        with pytest.raises(AssertionError):
            du.set_visible_devices(5)

    def test_invalid_type_raises_type_error(self, fake_gpus):
        """Test that set_visible_devices raises a TypeError when given an invalid type (not int or list)."""
        fake_gpus([(0, 1000, 4000)])
        with pytest.raises(TypeError):
            du.set_visible_devices("not an int or list")
