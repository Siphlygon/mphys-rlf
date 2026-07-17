"""
Unit tests for diffracc/scripts/sample_conditioning_sweep.py.

sweep_conditioning is exercised end-to-end against a real (but tiny) EDMPrecond+Unet model, same rationale as
tests/test_sample_snapshot_grid.py. The shared-latent invariant (every panel starts from the same initial noise)
is verified by monkeypatching diffusion.edm_sampling to capture its actual `latents` argument, since a real
sampling pass with genuinely different conditioning per panel would make the final images differ regardless.
"""
import h5py
import numpy as np
import pytest
import torch

from diffracc.model import diffusion
from diffracc.scripts import sample_conditioning_sweep as scs
from diffracc.utils import paths


class TestFitPeakFluxTransformer:
    """
    Unit tests for the `_fit_peak_flux_transformer` function in sample_conditioning_sweep.py, which fits a transformer
    to the peak-flux values in a training dataset.
    """

    def test_max_vals_match_per_image_max(self, tmp_path):
        """
        Test that the maximum values returned by _fit_peak_flux_transformer match the per-image maximums in the dataset.
        """
        images = np.stack([np.full((4, 4), 1.0), np.full((4, 4), 5.0), np.full((4, 4), 2.0)])
        path = tmp_path / "train.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=images)

        _, max_vals = scs._fit_peak_flux_transformer(str(path))

        np.testing.assert_allclose(max_vals, [1.0, 5.0, 2.0])

    def test_transform_round_trips_via_inverse(self, tmp_path):
        """
        Test that the transformer returned by _fit_peak_flux_transformer can round-trip the maximum values via transform
        and inverse_transform.
        """
        rng = np.random.default_rng(0)
        images_max = np.abs(rng.normal(5, 1, 30)) + 1.0
        images = np.stack([np.full((4, 4), v) for v in images_max])
        path = tmp_path / "train.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=images)

        pt, max_vals = scs._fit_peak_flux_transformer(str(path))
        transformed = pt.transform(max_vals.reshape(-1, 1))
        recovered = pt.inverse_transform(transformed)[:, 0]

        np.testing.assert_allclose(recovered, max_vals, rtol=1e-4)


class TestFitLasTransformer:
    """
    Unit tests for the `_fit_las_transformer` function in sample_conditioning_sweep.py, which fits a transformer to the
    LAS values in a training dataset.
    """

    def _write_las(self, path, las_values):
        """This helper method writes a training HDF5 file with the given LAS values."""
        with h5py.File(path, "w") as f:
            f.create_dataset("cat_info", data=np.array([(v,) for v in las_values], dtype=[("LAS", "f4")]))

    def test_uses_box_cox_when_all_values_positive(self, tmp_path):
        """Test that _fit_las_transformer uses Box-Cox transformation when all LAS values are positive."""
        path = tmp_path / "train.h5"
        self._write_las(path, np.abs(np.random.default_rng(0).normal(20, 5, 30)) + 1.0)

        pt = scs._fit_las_transformer(str(path))

        assert pt.method == "box-cox"

    def test_uses_yeo_johnson_when_non_positive_values_present(self, tmp_path):
        """Test that _fit_las_transformer uses Yeo-Johnson transformation when non-positive LAS values are present."""
        path = tmp_path / "train.h5"
        values = np.concatenate([np.abs(np.random.default_rng(0).normal(20, 5, 29)), [-1.0]])
        self._write_las(path, values)

        pt = scs._fit_las_transformer(str(path))

        assert pt.method == "yeo-johnson"


class TestExtentProxy:
    """
    Unit tests for the `_extent_proxy` function in sample_conditioning_sweep.py, which computes a proxy for the extent
    of an image.
    """

    def test_matches_hand_computed_formula(self):
        """Test that _extent_proxy matches a hand-computed formula for a simple image."""
        img = np.zeros((10, 10))
        img[0, 0] = 1000.0
        expected = float(np.sqrt((img > np.median(img) + 3 * np.std(img)).sum()))
        assert scs._extent_proxy(img) == pytest.approx(expected)

    def test_zero_for_a_uniform_image(self):
        """Test that _extent_proxy returns zero for a uniform image."""
        img = np.full((5, 5), 3.0)
        assert scs._extent_proxy(img) == pytest.approx(0.0)


@pytest.fixture
def patched_sweep_env(monkeypatch, tmp_path):
    """Fixture that patches the MODEL_PARENT path to point to a temporary directory for testing sweep_conditioning."""
    monkeypatch.setattr(paths, "MODEL_PARENT", tmp_path)
    return tmp_path


def _save_snapshot(model_dir, model, iteration=10):
    """Helper function to save a model snapshot in the specified model directory."""
    snap_dir = model_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"snapshot_iter_{iteration:08d}.pt"
    torch.save({"model": model.state_dict(), "ema_model": model.state_dict()}, path)
    return path


def _write_train_h5(path, n=30, seed=0):
    """Helper function to write a training HDF5 file with synthetic images and LAS values."""
    rng = np.random.default_rng(seed)
    max_vals = np.abs(rng.normal(5, 1, n)) + 1.0
    images = np.stack([np.full((80, 80), v) for v in max_vals])
    las_values = np.abs(rng.normal(30, 10, n)) + 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("cat_info", data=np.array([(v,) for v in las_values], dtype=[("LAS", "f4")]))


class TestSweepConditioning:
    """
    Unit tests for the `sweep_conditioning` function in sample_conditioning_sweep.py, which performs a sweep over
    conditioning values and generates a grid of images.
    """

    def test_invalid_sweep_raises_value_error(self, make_tiny_model_dir, patched_sweep_env, tmp_path):
        """Test that providing an invalid sweep name raises a ValueError."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        with pytest.raises(ValueError):
            scs.sweep_conditioning("tinymodel", str(train_path), sweep="not_a_real_sweep")

    def test_sweep_not_in_model_context_raises_value_error(self, make_tiny_model_dir, patched_sweep_env, tmp_path):
        """Test that requesting a sweep for a context variable not present in the model raises a ValueError."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr"])  # no las_values_tr
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        with pytest.raises(ValueError):
            scs.sweep_conditioning("tinymodel", str(train_path), sweep="las")

    def test_runs_and_saves_expected_default_filename(self, make_tiny_model_dir, patched_sweep_env, tmp_path):
        """Test that sweep_conditioning runs without errors and saves the output to the expected default filename."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model, iteration=10)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        out_path = scs.sweep_conditioning("tinymodel", str(train_path), sweep="las", n=3, timesteps=2)

        assert out_path.exists()
        assert out_path.name == "conditioning_sweep_las_iter10.png"

    def test_peak_sweep_also_runs(self, make_tiny_model_dir, patched_sweep_env, tmp_path):
        """Test that sweep_conditioning runs without errors for the "peak" sweep and saves the output."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        out_path = scs.sweep_conditioning("tinymodel", str(train_path), sweep="peak", n=3, timesteps=2)

        assert out_path.exists()

    def test_every_panel_shares_the_same_initial_latent(self, make_tiny_model_dir, patched_sweep_env, tmp_path,
                                                         monkeypatch):
        """
        Test that every panel in the sweep starts from the same initial latent noise, ensuring consistency across
        panels.
        """
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        captured = {}
        real_edm_sampling = diffusion.edm_sampling

        def _capturing_edm_sampling(model, context_batch=None, label_batch=None, latents=None, **kwargs):
            captured["latents"] = latents.clone()
            captured["context_batch"] = context_batch.clone() if context_batch is not None else None
            return real_edm_sampling(model, context_batch=context_batch, label_batch=label_batch,
                                     latents=latents, **kwargs)

        monkeypatch.setattr(scs.diffusion, "edm_sampling", _capturing_edm_sampling)

        scs.sweep_conditioning("tinymodel", str(train_path), sweep="las", n=4, timesteps=2)

        latents = captured["latents"]
        assert latents.shape[0] == 4
        for i in range(1, 4):
            torch.testing.assert_close(latents[i], latents[0])

    def test_swept_context_column_varies_while_others_stay_fixed(self, make_tiny_model_dir, patched_sweep_env,
                                                                  tmp_path, monkeypatch):
        """Test that the swept context column varies across panels while other context columns remain fixed."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        captured = {}
        real_edm_sampling = diffusion.edm_sampling

        def _capturing_edm_sampling(model, context_batch=None, label_batch=None, latents=None, **kwargs):
            captured["context_batch"] = context_batch.clone()
            return real_edm_sampling(model, context_batch=context_batch, label_batch=label_batch,
                                     latents=latents, **kwargs)

        monkeypatch.setattr(scs.diffusion, "edm_sampling", _capturing_edm_sampling)

        scs.sweep_conditioning("tinymodel", str(train_path), sweep="las", n=4, timesteps=2)

        context = captured["context_batch"]
        # context columns are ordered by config.context = ["max_values_tr", "las_values_tr"]; "las" sweep varies
        # column 1 (las_values_tr) and holds column 0 (max_values_tr, the peak-flux condition) fixed at 0.
        torch.testing.assert_close(context[:, 0], torch.zeros(4, device=context.device))
        assert len(torch.unique(context[:, 1])) == 4  # 4 distinct swept values

    def test_invert_no_leaves_raw_output(self, make_tiny_model_dir, patched_sweep_env, tmp_path, capsys):
        """
        Test that setting invert="no" leaves the raw output in Jy/beam units without applying any inverse
        transformation.
        """
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        scs.sweep_conditioning("tinymodel", str(train_path), sweep="las", n=2, timesteps=2, invert="no")

        assert "Jy/beam" in capsys.readouterr().out

    def test_invert_yes_without_recorded_transform_raises_value_error(self, make_tiny_model_dir, patched_sweep_env,
                                                                       tmp_path):
        """Test that setting invert="yes" without a recorded transform raises a ValueError."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        with pytest.raises(ValueError):
            scs.sweep_conditioning("tinymodel", str(train_path), sweep="las", n=2, timesteps=2, invert="yes")

    def test_invert_auto_applies_recorded_transform_without_error(self, make_tiny_model_dir, patched_sweep_env,
                                                                   tmp_path):
        """Test that setting invert="auto" applies the recorded transform without raising an error."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"],
                                               flux_transform={"name": "linear", "k": 2.0})
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        out_path = scs.sweep_conditioning("tinymodel", str(train_path), sweep="las", n=2, timesteps=2,
                                          invert="auto")

        assert out_path.exists()

    def test_explicit_peak_bounds_are_used(self, make_tiny_model_dir, patched_sweep_env, tmp_path, monkeypatch):
        """Test that providing explicit peak bounds results in the context batch having values within those bounds."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)

        captured = {}
        real_edm_sampling = diffusion.edm_sampling

        def _capturing_edm_sampling(model, context_batch=None, label_batch=None, latents=None, **kwargs):
            captured["context_batch"] = context_batch.clone()
            return real_edm_sampling(model, context_batch=context_batch, label_batch=label_batch,
                                     latents=latents, **kwargs)

        monkeypatch.setattr(scs.diffusion, "edm_sampling", _capturing_edm_sampling)

        scs.sweep_conditioning("tinymodel", str(train_path), sweep="peak", n=3, timesteps=2,
                               peak_bounds=(1.0, 2.0))

        assert captured["context_batch"].shape == (3, 2)

    def test_explicit_out_path_is_respected(self, make_tiny_model_dir, patched_sweep_env, tmp_path):
        """Test that providing an explicit out_path results in the output being saved to that path."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr", "las_values_tr"])
        _save_snapshot(model_dir, model)
        train_path = tmp_path / "train.h5"
        _write_train_h5(train_path)
        custom_path = tmp_path / "custom_sweep.png"

        result = scs.sweep_conditioning("tinymodel", str(train_path), sweep="las", n=2, timesteps=2,
                                        out_path=custom_path)

        assert result == custom_path
        assert custom_path.exists()
