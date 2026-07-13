"""Unit tests for diffracc/data/flux_transforms.py's global invertible flux-space transforms."""
import numpy as np
import pytest
import torch

from diffracc.data import flux_transforms as ft


class TestRobustNoise:
    """Tests that robust_noise correctly computes the robust noise estimate from an array of pixel values."""

    def test_matches_1_4826_times_median_absolute_deviation(self):
        """Test that robust_noise returns 1.4826 times the median absolute deviation of the input array."""
        pixels = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # median=3, deviations=[2,1,0,1,97], median dev=1
        assert ft.robust_noise(pixels) == pytest.approx(1.4826 * 1.0)

    def test_zero_for_constant_array(self):
        """Test that robust_noise returns 0.0 for a constant array."""
        assert ft.robust_noise(np.full(10, 5.0)) == pytest.approx(0.0)


class TestFlatSample:
    """Tests that _flat_sample correctly samples from an array of pixel values."""

    def test_drops_non_finite_values(self):
        """Test that _flat_sample drops non-finite values (NaN, inf, -inf) from the input array before sampling."""
        arr = np.array([1.0, np.nan, 2.0, np.inf, 3.0, -np.inf])
        sampled = ft._flat_sample(arr, sample_size=100, seed=0)
        assert sampled.shape[0] == 3
        np.testing.assert_allclose(np.sort(sampled), [1.0, 2.0, 3.0])

    def test_subsamples_to_sample_size(self):
        """Test that _flat_sample subsamples the input array to the specified sample size."""
        arr = np.arange(1000, dtype=np.float32)
        sampled = ft._flat_sample(arr, sample_size=100, seed=0)
        assert sampled.shape[0] == 100

    def test_reproducible_with_same_seed(self):
        """Test that _flat_sample returns the same sampled values when called with the same seed."""
        arr = np.arange(1000, dtype=np.float32)
        a = ft._flat_sample(arr, sample_size=100, seed=42)
        b = ft._flat_sample(arr, sample_size=100, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_accepts_torch_tensor(self):
        """Test that _flat_sample accepts a torch.Tensor as input and returns a torch.Tensor."""
        arr = torch.arange(10, dtype=torch.float32)
        sampled = ft._flat_sample(arr, sample_size=100, seed=0)
        assert sampled.shape[0] == 10


class TestGlobalLinearScale:
    """Tests that GlobalLinearScale correctly implements a global linear scaling transform."""

    def test_rejects_non_positive_k(self):
        """Test that GlobalLinearScale raises an AssertionError if k is not positive."""
        with pytest.raises(AssertionError):
            ft.GlobalLinearScale(k=0)
        with pytest.raises(AssertionError):
            ft.GlobalLinearScale(k=-1)

    def test_forward_inverse_round_trip_numpy(self):
        """Test that GlobalLinearScale correctly implements the forward and inverse transforms for numpy arrays."""
        transform = ft.GlobalLinearScale(k=2.0)
        x = np.array([1.0, -2.0, 3.5])
        y = transform.forward(x)
        np.testing.assert_allclose(y, x * 2.0)
        np.testing.assert_allclose(transform.inverse(y), x)
        assert isinstance(y, np.ndarray)

    def test_forward_inverse_round_trip_torch(self):
        """Test that GlobalLinearScale correctly implements the forward and inverse transforms for torch tensors."""
        transform = ft.GlobalLinearScale(k=2.0)
        x = torch.tensor([1.0, -2.0, 3.5])
        y = transform.forward(x)
        assert isinstance(y, torch.Tensor)
        torch.testing.assert_close(transform.inverse(y), x)

    def test_call_matches_forward(self):
        """Test that calling the transform instance directly is equivalent to calling its forward method."""
        transform = ft.GlobalLinearScale(k=3.0)
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(transform(x), transform.forward(x))

    def test_fit_achieves_target_sigma_data(self):
        """Test that GlobalLinearScale.fit computes a k that achieves the target sigma_data for the transformed data."""
        rng = np.random.default_rng(0)
        images = rng.normal(loc=0.0, scale=0.02, size=10000)
        transform = ft.GlobalLinearScale.fit(images, sigma_data=0.5)
        transformed = transform.forward(images)
        assert transformed.std() == pytest.approx(0.5, rel=0.05)

    def test_to_dict_round_trips_via_from_dict(self):
        """Test that GlobalLinearScale.to_dict and from_dict correctly serialize and deserialize the transform."""
        transform = ft.GlobalLinearScale(k=2.5)
        restored = ft.from_dict(transform.to_dict())
        assert isinstance(restored, ft.GlobalLinearScale)
        assert restored.k == pytest.approx(2.5)

    def test_max_abs_roundtrip_error_is_near_zero(self):
        """Test that GlobalLinearScale.max_abs_roundtrip_error returns a value near zero for a range of input images."""
        transform = ft.GlobalLinearScale(k=2.0)
        images = np.linspace(-1, 1, 1000)
        assert transform.max_abs_roundtrip_error(images) < 1e-5


class TestGlobalAsinhScale:
    """Tests that GlobalAsinhScale correctly implements a global inverse hyperbolic sine scaling transform."""

    def test_rejects_non_positive_params(self):
        """Test that GlobalAsinhScale raises an AssertionError if beta or k are not positive."""
        with pytest.raises(AssertionError):
            ft.GlobalAsinhScale(beta=0, k=1.0)
        with pytest.raises(AssertionError):
            ft.GlobalAsinhScale(beta=1.0, k=0)

    def test_forward_inverse_round_trip(self):
        """Test that GlobalAsinhScale correctly implements the forward and inverse transforms for numpy arrays."""
        transform = ft.GlobalAsinhScale(beta=0.05, k=2.0)
        x = np.array([-0.5, -0.01, 0.0, 0.01, 0.5])
        y = transform.forward(x)
        np.testing.assert_allclose(transform.inverse(y), x, atol=1e-6)

    def test_approximately_linear_for_small_x(self):
        """Test that GlobalAsinhScale.forward is approximately linear for small |x| compared to beta."""
        # For |x| << beta, asinh(x/beta) ~ x/beta, so forward(x) ~ (k/beta) * x.
        beta, k = 1.0, 1.0
        transform = ft.GlobalAsinhScale(beta=beta, k=k)
        x = 1e-4
        np.testing.assert_allclose(transform.forward(np.array([x]))[0], (k / beta) * x, rtol=1e-3)

    def test_fit_achieves_target_sigma_data(self):
        """
        Test that GlobalAsinhScale.fit computes beta and k that achieve the target sigma_data for the transformed data.
        """
        rng = np.random.default_rng(0)
        images = rng.normal(loc=0.0, scale=0.02, size=10000)
        transform = ft.GlobalAsinhScale.fit(images, sigma_data=0.5)
        transformed = transform.forward(images)
        assert transformed.std() == pytest.approx(0.5, rel=0.05)

    def test_fit_uses_beta_scale_times_robust_noise_by_default(self):
        """Test that GlobalAsinhScale.fit uses beta_scale times the robust noise estimate of the images by default."""
        rng = np.random.default_rng(1)
        images = rng.normal(loc=0.0, scale=0.02, size=20000)
        transform = ft.GlobalAsinhScale.fit(images, sigma_data=0.5, beta_scale=3.0)
        expected_beta = 3.0 * ft.robust_noise(images)
        assert transform.beta == pytest.approx(expected_beta, rel=0.1)

    def test_to_dict_round_trips_via_from_dict(self):
        """Test that GlobalAsinhScale.to_dict and from_dict correctly serialize and deserialize the transform."""
        transform = ft.GlobalAsinhScale(beta=0.1, k=1.5)
        restored = ft.from_dict(transform.to_dict())
        assert isinstance(restored, ft.GlobalAsinhScale)
        assert restored.beta == pytest.approx(0.1)
        assert restored.k == pytest.approx(1.5)


class TestFromDict:
    """Tests that from_dict correctly reconstructs flux transforms from their dictionary representations."""

    def test_unknown_name_raises_value_error(self):
        """Test that from_dict raises a ValueError if the 'name' key does not correspond to a known transform."""
        with pytest.raises(ValueError):
            ft.from_dict({"name": "not_a_real_transform"})


class TestSaveLoad:
    """Tests that flux transforms can be saved to and loaded from JSON files correctly."""

    def test_save_writes_json_and_load_round_trips(self, tmp_path):
        """Test that saving a flux transform to a JSON file and loading it back reconstructs the same transform."""
        transform = ft.GlobalLinearScale(k=1.7)
        path = transform.save(tmp_path / "my_transform.json")
        assert path.exists()

        loaded = ft.load(path)
        assert isinstance(loaded, ft.GlobalLinearScale)
        assert loaded.k == pytest.approx(1.7)

    def test_save_to_directory_uses_default_filename(self, tmp_path):
        """Test that saving a flux transform to a directory uses the default filename FLUX_TRANSFORM_FILE."""
        transform = ft.GlobalAsinhScale(beta=0.05, k=2.0)
        path = transform.save(tmp_path)
        assert path.name == ft.FLUX_TRANSFORM_FILE
        assert path.exists()

    def test_load_from_directory(self, tmp_path):
        """Test that loading a flux transform from a directory reads the default filename FLUX_TRANSFORM_FILE."""
        transform = ft.GlobalLinearScale(k=0.9)
        transform.save(tmp_path)
        loaded = ft.load(tmp_path)
        assert isinstance(loaded, ft.GlobalLinearScale)
        assert loaded.k == pytest.approx(0.9)

    def test_load_none_returns_none(self):
        """Test that ft.load(None) returns None."""
        assert ft.load(None) is None

    def test_load_instance_returns_as_is(self):
        """Test that ft.load(transform_instance) returns the same instance without modification."""
        transform = ft.GlobalLinearScale(k=1.0)
        assert ft.load(transform) is transform

    def test_load_dict_reconstructs(self):
        """Test that ft.load(dict) reconstructs the transform from its dictionary representation."""
        loaded = ft.load({"name": "linear", "k": 3.0})
        assert isinstance(loaded, ft.GlobalLinearScale)
        assert loaded.k == pytest.approx(3.0)

    def test_load_missing_file_raises(self, tmp_path):
        """Test that ft.load raises a FileNotFoundError when the specified JSON file does not exist."""
        with pytest.raises(FileNotFoundError):
            ft.load(tmp_path / "does_not_exist.json")
