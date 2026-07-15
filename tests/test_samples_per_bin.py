"""
Unit tests for diffracc/plotting/samples_per_bin.py.

This module imports ImageAnalyzer's ProcessArgs (which needs the real `bdsf` package - not installable on Windows).
See tests/test_image_analyzer.py's docstring for how to run this file under WSL/conda.
"""
import numpy as np
import pytest

pytest.importorskip("bdsf", reason="bdsf (PyBDSF) is not installable on Windows; run this file under WSL/conda.")

from diffracc.plotting import samples_per_bin as spb
from diffracc.utils import paths


class TestGetNoise:
    """Tests for the get_noise function, which estimates the noise level of an image using a robust method."""
    
    def test_recovers_true_sigma_for_clean_gaussian_data(self):
        """Test that get_noise accurately recovers the standard deviation of a clean Gaussian distribution."""
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 5.0, 100_000)
        assert spb.get_noise(data) == pytest.approx(5.0, rel=0.05)

    def test_robust_to_a_small_fraction_of_extreme_outliers(self):
        """Test that get_noise is robust to a small fraction of extreme outliers in the data."""
        rng = np.random.default_rng(0)
        core = rng.normal(0.0, 1.0, 10_000)
        outliers = np.full(20, 1000.0)
        data = np.concatenate([core, outliers])
        assert spb.get_noise(data) == pytest.approx(1.0, rel=0.1)

    def test_ignores_values_at_or_below_the_masking_threshold(self):
        """Test that get_noise ignores values at or below the masking threshold (1e-7)."""
        rng = np.random.default_rng(1)
        core = rng.normal(0.0, 2.0, 5_000)
        # Values within +-1e-7 of zero are masked out by maskSup before any statistics are computed.
        data = np.concatenate([core, np.zeros(5_000)])
        assert spb.get_noise(data) == pytest.approx(spb.get_noise(core), rel=1e-6)


class TestMasking:
    """Tests for the masking function, which applies a threshold to an image to identify source pixels."""

    def test_source_pixel_is_kept_and_background_is_zeroed(self):
        """Test that the masking function keeps a source pixel above the threshold and zeroes out background pixels."""
        rng = np.random.default_rng(0)
        background = rng.normal(0.0, 1.0, 500)
        data = np.concatenate([background, [50.0]])

        masked = spb.masking(data, threshold_level=5.0)

        assert masked[-1] == pytest.approx(50.0)
        np.testing.assert_allclose(masked[:-1], 0.0)

    def test_higher_threshold_level_masks_more_aggressively(self):
        """Test that a higher threshold level results in more aggressive masking of the data."""
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(0.0, 1.0, 500), [6.0]])

        lenient = spb.masking(data, threshold_level=1.0)
        strict = spb.masking(data, threshold_level=10.0)

        assert np.count_nonzero(lenient) >= np.count_nonzero(strict)

    def test_does_not_mutate_input_array(self):
        """Test that the masking function does not mutate the input array in place."""
        data = np.array([0.1, 0.2, 10.0])
        original = data.copy()
        spb.masking(data)
        np.testing.assert_array_equal(data, original)


class TestCreateNoiseLOFAR:
    """Tests for the create_noise_lofar function, which generates synthetic LOFAR-like noise images."""

    def test_returns_requested_shape(self):
        """Test that create_noise_lofar returns an array with the requested shape."""
        noise = spb.create_noise_lofar(shape=(10, 20))
        assert noise.shape == (10, 20)

    def test_default_shape_is_80x80(self):
        """Test that the default shape of the noise image is 80x80 pixels if no shape is specified."""
        noise = spb.create_noise_lofar()
        assert noise.shape == (80, 80)

    def test_std_matches_requested_rms(self):
        """Test that the standard deviation of the generated noise matches the requested RMS value."""
        rng_state = np.random.get_state()
        np.random.seed(0)
        try:
            noise = spb.create_noise_lofar(shape=(500, 500), rms=2.5)
        finally:
            np.random.set_state(rng_state)
        assert np.std(noise) == pytest.approx(2.5, rel=0.02)

    def test_default_rms_is_rms_lofar_constant(self):
        """Test that the default RMS of the generated noise matches the RMS_LOFAR constant."""
        rng_state = np.random.get_state()
        np.random.seed(0)
        try:
            noise = spb.create_noise_lofar(shape=(500, 500))
        finally:
            np.random.set_state(rng_state)
        assert np.std(noise) == pytest.approx(spb.RMS_LOFAR, rel=0.05)


class _FakeSubdirData:
    """A fake object to simulate the data arrays for a subdirectory (dataset or generated)."""
    def __init__(self, images: np.ndarray, model_fluxes: np.ndarray):
        self.images = images
        self.model_fluxes = model_fluxes


class _FakeImageDataArrays:
    """A fake object to simulate the ImageDataArrays class."""
    def __init__(self, config_name: str):
        self.generated_data = _FakeSubdirData(
            images=np.zeros((5, 4, 4)),
            model_fluxes=np.array([0.05, 1.0, 10.0, 100.0, 1000.0]),
        )


class TestGetCompletenessEstim:
    """Unit tests for the get_completeness_estim function."""

    def test_runs_without_error_and_saves_a_figure(self, monkeypatch, tmp_path):
        """Test that the function runs without error and saves a figure."""
        monkeypatch.setattr(paths, "config", {"my_config": {"generated_subdir": "generated"}})
        monkeypatch.setattr(spb, "ImageDataArrays", _FakeImageDataArrays)
        monkeypatch.chdir(tmp_path)

        spb.get_completeness_estim("my_config")

        assert (tmp_path / "sources_per_bin.png").exists()
