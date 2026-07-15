"""
Unit tests for diffracc/sampling/generate_fits_files.py.

Imports ImageAnalyzer (which needs the real `bdsf` package - not installable on Windows). See
tests/test_image_analyzer.py's docstring for how to run this file under WSL/conda.

sample()'s heavy dependencies (the diffusion Sampler, ImageAnalyzer) are always monkeypatched out with fakes below
- no real model loading, diffusion sampling, or PyBDSF processing runs in these tests. Only get_path_from_index's
pure path logic and sample()'s own orchestration logic (distribution selection, skip-if-nothing-to-do, image
scaling, LAS conditioning, index-collision avoidance) are under test.
"""
from pathlib import PurePath

import numpy as np
import pytest
import torch

pytest.importorskip("bdsf", reason="bdsf (PyBDSF) is not installable on Windows; run this file under WSL/conda.")

from diffracc.sampling import generate_fits_files as gff
from diffracc.utils import paths


class TestGetPathFromIndex:
    """Tests for get_path_from_index()'s binning and path construction logic."""

    def test_bin_naming_matches_index_range(self):
        """Test that the bin naming in the path matches the expected range of indices for a given bin size."""
        full_path, postfix = gff.get_path_from_index(5, "mysubdir", bin_size=10)
        assert postfix == PurePath("0-9", "image5.fits")
        assert full_path == paths.FITS_PARENT / "mysubdir" / "0-9" / "image5.fits"

    def test_bin_boundaries(self):
        """Test that the bin boundaries are calculated correctly for various indices and bin sizes."""
        assert gff.get_path_from_index(0, "s", bin_size=10)[1] == PurePath("0-9", "image0.fits")
        assert gff.get_path_from_index(9, "s", bin_size=10)[1] == PurePath("0-9", "image9.fits")
        assert gff.get_path_from_index(10, "s", bin_size=10)[1] == PurePath("10-19", "image10.fits")

    def test_respects_custom_bin_size(self):
        """Test that the function respects a custom bin size."""
        assert gff.get_path_from_index(4, "s", bin_size=2)[1] == PurePath("4-5", "image4.fits")


class _FakeSampler:
    """Stand-in for diffracc.model.sampler.Sampler - no real model loading or diffusion sampling."""
    instances = []

    def __init__(self, n_samples, timesteps):
        self.init_args = (n_samples, timesteps)
        self.get_fpeak_calls = []
        self.quick_sample_calls = []
        _FakeSampler.instances.append(self)

    def get_fpeak_model_dist(self, train_set_path, max_vals=None):
        """Return a dummy function for the peak flux model distribution."""
        self.get_fpeak_calls.append((train_set_path, max_vals))
        return lambda n: np.full(n, 0.5)

    def quick_sample(self, model_name, context=None, n_samples=None, distribute_model=None):
        """Return a dummy array of samples with shape (batch, T, C, H, W), matching Sampler.quick_sample's real
        numpy.ndarray return type."""
        self.quick_sample_calls.append((model_name, context, n_samples, distribute_model))
        batch = context.shape[0]
        # shape (batch, T, C, H, W); sample() reads samples[i, -1, 0, :, :] - each image has internal variation
        # (0..63) offset per-sample so im_max != im_min for the 0-1 scaling branch, and images are distinguishable.
        return np.stack([np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8) + i * 1000 for i in range(batch)])


class _FakeImageAnalyzer:
    """Stand-in for ImageAnalyzer - avoids real PyBDSF/FITS I/O, just records save_image_to_fits calls."""
    instances = []

    def __init__(self, subdir):
        self.subdir = subdir
        self.save_calls = []
        _FakeImageAnalyzer.instances.append(self)

    def save_image_to_fits(self, image, postfix, **kwargs):
        """Record the call to save_image_to_fits, including the image, postfix, and any keyword arguments."""
        self.save_calls.append((image.clone() if isinstance(image, torch.Tensor) else np.array(image), postfix, kwargs))


@pytest.fixture(autouse=True)
def _clear_fake_instances():
    """Clear the instances of _FakeSampler and _FakeImageAnalyzer before and after each test to ensure isolation."""
    _FakeSampler.instances = []
    _FakeImageAnalyzer.instances = []
    yield
    _FakeSampler.instances = []
    _FakeImageAnalyzer.instances = []


@pytest.fixture
def patched_sample_deps(monkeypatch, tmp_path, np_array_parent):
    """
    Monkeypatch out the heavy dependencies of sample() with fakes, and set up a valid maxvals file for
    PeakFluxPowerTransformer.
    """
    monkeypatch.setattr(gff, "ImageAnalyzer", _FakeImageAnalyzer)
    monkeypatch.setattr(gff.sampler, "Sampler", _FakeSampler)
    monkeypatch.setattr(paths, "FITS_PARENT", tmp_path / "fits")
    # a valid, strictly-positive maxvals file so PeakFluxPowerTransformer's box-cox fit succeeds
    (np_array_parent / "generated").mkdir(parents=True, exist_ok=True)
    np.save(np_array_parent / "generated" / paths.MAXVALS, np.abs(np.random.default_rng(0).normal(10, 2, 50)) + 1.0)
    return tmp_path


def _base_args(**overrides) -> gff.SampleArgs:
    """Return a base set of arguments for sample(), with optional overrides for specific tests."""
    defaults = dict(
        batch_size=4, timesteps=10, use_cpu=True, n_samples=3, generated_subdir="generated",
        distribution="dataset", lower_bound=0.0, upper_bound=1.0, las_conditioning_enabled=False,
        preserve_values=True, model_name="LOFAR_model", folder_size=100,
    )
    defaults.update(overrides)
    return gff.SampleArgs(**defaults)


class TestSample:
    """Tests for the sample() function in generate_fits_files.py."""

    def test_skips_when_nothing_to_generate(self, patched_sample_deps):
        """Test that sample() logs and returns early when n_samples_to_generate <= 0, without calling quick_sample()."""
        gff.sample(_base_args(n_samples=0))
        assert _FakeSampler.instances[0].quick_sample_calls == []

    def test_unknown_distribution_raises_value_error(self):
        """Test that constructing SampleArgs with an unknown distribution raises ValueError - sample() itself
        trusts args.distribution is already valid and no longer re-checks it."""
        with pytest.raises(ValueError):
            _base_args(distribution="not_a_real_distribution")

    def test_dataset_distribution_uses_get_fpeak_model_dist(self, patched_sample_deps):
        """Test that sample() calls get_fpeak_model_dist() with the correct path when distribution is 'dataset'."""
        gff.sample(_base_args(distribution="dataset"))
        calls = _FakeSampler.instances[0].get_fpeak_calls
        assert len(calls) == 1
        assert calls[0][1] == paths.NP_ARRAY_PARENT / "generated" / paths.MAXVALS

    def test_generates_and_saves_expected_number_of_samples(self, patched_sample_deps):
        """Test that sample() generates and saves the expected number of samples, respecting batch_size."""
        gff.sample(_base_args(n_samples=3, batch_size=4))
        analyzer = _FakeImageAnalyzer.instances[0]
        assert len(analyzer.save_calls) == 3

    def test_preserve_values_true_leaves_image_unscaled(self, patched_sample_deps):
        """Test that sample() leaves the image values unscaled when preserve_values is True."""
        gff.sample(_base_args(n_samples=1, batch_size=1, preserve_values=True))
        image, postfix, kwargs = _FakeImageAnalyzer.instances[0].save_calls[0]
        np.testing.assert_allclose(np.asarray(image), np.arange(64).reshape(8, 8))

    def test_preserve_values_false_scales_image_to_zero_one(self, patched_sample_deps):
        """Test that sample() scales the image values to the range [0, 1] when preserve_values is False."""
        gff.sample(_base_args(n_samples=1, batch_size=1, preserve_values=False))
        image, postfix, kwargs = _FakeImageAnalyzer.instances[0].save_calls[0]
        arr = np.asarray(image)
        assert arr.min() == pytest.approx(0.0)
        assert arr.max() == pytest.approx(1.0)

    def test_las_conditioning_adds_lasize_kwarg(self, patched_sample_deps):
        """Test that sample() adds the LASIZE keyword argument when las_conditioning_enabled is True."""
        gff.sample(_base_args(n_samples=1, batch_size=1, las_conditioning_enabled=True))
        image, postfix, kwargs = _FakeImageAnalyzer.instances[0].save_calls[0]
        assert "LASIZE" in kwargs
        assert "FXSCLD" in kwargs

    def test_no_las_conditioning_omits_lasize_kwarg(self, patched_sample_deps):
        """Test that sample() omits the LASIZE keyword argument when las_conditioning_enabled is False."""
        gff.sample(_base_args(n_samples=1, batch_size=1, las_conditioning_enabled=False))
        image, postfix, kwargs = _FakeImageAnalyzer.instances[0].save_calls[0]
        assert "LASIZE" not in kwargs

    def test_skips_index_collision_with_existing_file(self, patched_sample_deps, tmp_path):
        """
        Test that sample() skips over an existing file at the starting index and saves to the next available index.
        """
        # Pre-create the file that would normally be assigned to sample_index=0 (bin_start), forcing the loop to
        # advance to the next index instead of overwriting it. n_samples=2 so one new sample is still needed after
        # the pre-existing one is counted (n_samples=1 here would mean the bin is already full and nothing runs).
        existing_path, _ = gff.get_path_from_index(0, "generated", 100)
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.touch()

        gff.sample(_base_args(n_samples=2, batch_size=1))

        _, postfix, _ = _FakeImageAnalyzer.instances[0].save_calls[0]
        assert postfix == PurePath("0-99", "image1.fits")
