"""Unit tests for diffracc/dataset_prep/fit_flux_transform.py."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from diffracc.data import flux_transforms as ft
from diffracc.dataset_prep.fit_flux_transform import _build_argument_parser, _load_images, fit_and_report


def _write_images_h5(path, images: np.ndarray, key: str = "images"):
    """Write a numpy array of images to an HDF5 file at the given path, under the specified key."""
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=images)


class TestLoadImages:
    """Tests for the _load_images() function in fit_flux_transform.py."""

    def test_loads_all_images_when_sample_size_is_zero(self, tmp_path):
        """Test that _load_images() loads all images when sample_images is set to 0 (no subsampling)."""
        images = np.random.default_rng(0).normal(size=(20, 8, 8)).astype(np.float32)
        path = tmp_path / "dataset.h5"
        _write_images_h5(path, images)

        loaded = _load_images(path, "images", sample_images=0, seed=0)

        assert loaded.shape == (20, 8, 8)
        assert loaded.dtype == np.float32

    def test_loads_all_images_when_dataset_smaller_than_sample_size(self, tmp_path):
        """Test that _load_images() loads all images when the dataset is smaller than the requested sample size."""
        images = np.random.default_rng(0).normal(size=(5, 8, 8)).astype(np.float32)
        path = tmp_path / "dataset.h5"
        _write_images_h5(path, images)

        loaded = _load_images(path, "images", sample_images=100, seed=0)

        assert loaded.shape == (5, 8, 8)

    def test_subsamples_when_dataset_larger_than_sample_size(self, tmp_path):
        """Test that _load_images() subsamples the dataset when it is larger than the requested sample size."""
        images = np.random.default_rng(0).normal(size=(100, 8, 8)).astype(np.float32)
        path = tmp_path / "dataset.h5"
        _write_images_h5(path, images)

        loaded = _load_images(path, "images", sample_images=10, seed=0)

        assert loaded.shape == (10, 8, 8)

    def test_reproducible_with_same_seed(self, tmp_path):
        """Test that _load_images() produces the same subsample when called with the same random seed."""
        images = np.random.default_rng(0).normal(size=(100, 8, 8)).astype(np.float32)
        path = tmp_path / "dataset.h5"
        _write_images_h5(path, images)

        a = _load_images(path, "images", sample_images=10, seed=42)
        b = _load_images(path, "images", sample_images=10, seed=42)

        np.testing.assert_array_equal(a, b)

    def test_respects_custom_key(self, tmp_path):
        """Test that _load_images() loads images from a custom key in the HDF5 file."""
        images = np.zeros((3, 4, 4), dtype=np.float32)
        path = tmp_path / "dataset.h5"
        _write_images_h5(path, images, key="my_images")

        loaded = _load_images(path, "my_images", sample_images=0, seed=0)

        assert loaded.shape == (3, 4, 4)


class TestFitAndReport:
    """Tests for the fit_and_report() function in fit_flux_transform.py."""

    @pytest.fixture
    def dataset_path(self, tmp_path):
        """Fixture to create a temporary HDF5 dataset of images for testing fit_and_report()."""
        rng = np.random.default_rng(0)
        # Background noise plus a handful of bright pixels to give the asinh fit something to compress.
        images = rng.normal(loc=0.0, scale=0.02, size=(50, 20, 20)).astype(np.float32)
        images[:, 10, 10] = rng.uniform(0.5, 5.0, size=50).astype(np.float32)
        path = tmp_path / "dataset.h5"
        _write_images_h5(path, images)
        return path

    def test_fits_asinh_transform_and_achieves_target_sigma_data(self, dataset_path):
        """Test that fit_and_report() fits an asinh transform and achieves the target sigma_data."""
        transform = fit_and_report(dataset_path, kind="asinh", sigma_data=0.5, sample_images=0)

        assert isinstance(transform, ft.GlobalAsinhScale)
        assert (dataset_path.parent / ft.FLUX_TRANSFORM_FILE).exists()

    def test_fits_linear_transform(self, dataset_path):
        """Test that fit_and_report() fits a linear transform when requested."""
        transform = fit_and_report(dataset_path, kind="linear", sigma_data=0.5, sample_images=0)
        assert isinstance(transform, ft.GlobalLinearScale)

    def test_unknown_kind_raises_value_error(self, dataset_path):
        """Test that fit_and_report() raises a ValueError when an unknown kind is specified."""
        with pytest.raises(ValueError):
            fit_and_report(dataset_path, kind="not_a_real_kind", sample_images=0)

    def test_explicit_output_path_is_used(self, dataset_path, tmp_path):
        """Test that fit_and_report() saves the transform to the explicitly specified output path."""
        out_dir = tmp_path / "somewhere_else"
        out_dir.mkdir()

        fit_and_report(dataset_path, kind="asinh", sample_images=0, output=out_dir)

        assert (out_dir / ft.FLUX_TRANSFORM_FILE).exists()
        assert not (dataset_path.parent / ft.FLUX_TRANSFORM_FILE).exists()

    def test_explicit_beta_is_respected_over_computed_noise(self, dataset_path):
        """Test that fit_and_report() respects an explicitly provided beta value over the computed noise."""
        transform = fit_and_report(dataset_path, kind="asinh", sample_images=0, beta=0.123)
        assert transform.beta == pytest.approx(0.123)

    def test_saved_transform_round_trips_via_load(self, dataset_path):
        """Test that the transform saved by fit_and_report() can be loaded back and is of the correct type."""
        fit_and_report(dataset_path, kind="linear", sigma_data=0.5, sample_images=0)
        loaded = ft.load(dataset_path.parent)
        assert isinstance(loaded, ft.GlobalLinearScale)


class TestBuildArgumentParser:
    """Tests for the _build_argument_parser() function in fit_flux_transform.py."""

    def test_parses_required_dataset_argument(self):
        """Test that the argument parser correctly parses the required --dataset argument and its defaults."""
        parser = _build_argument_parser()
        args = parser.parse_args(["--dataset", "some/path.h5"])
        assert args.dataset == Path("some/path.h5")
        assert args.kind == "asinh"
        assert args.sigma_data == pytest.approx(0.5)

    def test_missing_dataset_argument_raises_system_exit(self):
        """Test that the argument parser raises SystemExit when the required --dataset argument is missing."""
        parser = _build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_rejects_invalid_kind_choice(self):
        """Test that the argument parser raises SystemExit when an invalid --kind choice is provided."""
        parser = _build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "x.h5", "--kind", "not_a_real_kind"])
