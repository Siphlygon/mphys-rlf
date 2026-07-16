"""
Unit tests for diffracc/model/sampler.py.

Sampler.sample()/quick_sample() heavy dependencies (model_utils.load_model, device_utils.distribute_model,
diffusion.edm_sampling) are monkeypatched out with fakes below - these tests exercise the Sampler's own orchestration
logic (settings merging, batch-size/reshaping math, model loading dispatch, output assembly), not real diffusion
sampling or GPU distribution. The fake edm_sampling deliberately matches the real function's keyword-argument names,
since quick_sample introspects diffusion.edm_sampling's signature via `inspect.signature` to decide which of its own
settings to forward as solver parameters.
"""
import h5py
import numpy as np
import pytest
import torch

from diffracc.model import diffusion, model_utils
from diffracc.model.sampler import Sampler
from diffracc.utils import device_utils


class TestInit:
    """Tests for Sampler.__init__() and its handling of default settings and kwargs overrides."""

    def test_default_settings(self):
        """Test that the default settings are set correctly when no kwargs are provided."""
        s = Sampler()
        assert s.settings["n_samples"] == 1000
        assert s.settings["timesteps"] == 25
        assert s.settings["sigma_min"] == pytest.approx(2e-3)

    def test_kwargs_override_defaults(self):
        """Test that the settings are correctly overridden when kwargs are provided."""
        s = Sampler(n_samples=5, timesteps=10)
        assert s.settings["n_samples"] == 5
        assert s.settings["timesteps"] == 10
        assert s.settings["sigma_min"] == pytest.approx(2e-3)  # untouched default

    def test_settings_not_save_list(self):
        """Test that the settings_not_save list contains the expected keys."""
        s = Sampler()
        assert set(s.settings_not_save) == {"n_samples", "n_devices", "samples_per_device", "flux_transform"}


class TestGetLabels:
    """
    Tests for Sampler.get_labels(), which generates a label array for a given number of labels and samples per label.
    """

    def test_default_samples_per_label_from_n_samples(self):
        """Test that the default samples_per_label is calculated correctly from n_samples and n_labels."""
        s = Sampler(n_samples=8)
        labels = s.get_labels(n_labels=4)
        assert len(labels) == 8
        assert list(labels[:2]) == [0, 0]
        assert list(labels[2:4]) == [1, 1]
        assert list(labels[-2:]) == [3, 3]

    def test_explicit_samples_per_label(self):
        """Test that the samples_per_label parameter is used correctly."""
        s = Sampler()
        labels = s.get_labels(n_labels=3, samples_per_label=5)
        assert len(labels) == 15
        assert list(labels[:5]) == [0] * 5
        assert list(labels[5:10]) == [1] * 5
        assert list(labels[10:]) == [2] * 5


class TestGetFpeakModelDist:
    """
    Tests for Sampler.get_fpeak_model_dist(), which returns a callable distribution function for generating fpeak
    values.
    """

    def test_from_max_vals_array(self):
        """Test that the distribution can be created from a numpy array of max_vals."""
        s = Sampler()
        rng = np.random.default_rng(0)
        max_vals = np.abs(rng.normal(5, 1, 200)) + 1.0
        dist = s.get_fpeak_model_dist(train_set_path=None, max_vals=max_vals)
        samples = dist(50)
        assert samples.shape == (50,)
        assert np.isfinite(samples).all()

    def test_from_max_vals_path(self, tmp_path):
        """Test that the distribution can be created from a path to a numpy array of max_vals."""
        s = Sampler()
        rng = np.random.default_rng(0)
        max_vals = np.abs(rng.normal(5, 1, 200)) + 1.0
        path = tmp_path / "maxvals.npy"
        np.save(path, max_vals)
        dist = s.get_fpeak_model_dist(train_set_path=None, max_vals=path)
        samples = dist(10)
        assert samples.shape == (10,)

    def test_from_train_set_h5_file(self, tmp_path):
        """Test that the distribution can be created from a path to an HDF5 file containing images."""
        s = Sampler()
        rng = np.random.default_rng(0)
        # A single bright pixel per image, with peak values spread widely enough for a stable box-cox fit -
        # a narrow/near-constant max-value distribution (e.g. plain uniform noise) makes scipy's box-cox lambda
        # search fail to bracket a minimum.
        peaks = np.abs(rng.normal(5, 1, 20)) + 1.0
        images = np.stack([np.full((8, 8), 0.01, dtype=np.float32) for _ in range(20)])
        for i, peak in enumerate(peaks):
            images[i, 0, 0] = peak
        path = tmp_path / "train.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=images)
        dist = s.get_fpeak_model_dist(train_set_path=path)
        samples = dist(5)
        assert samples.shape == (5,)


class TestGetLasModelDist:
    """
    Tests for Sampler.get_las_model_dist(), which returns a callable distribution function for generating LAS values.
    """

    def test_uses_box_cox_for_strictly_positive_values(self):
        """Test that the distribution uses a Box-Cox transformation when all LAS values are strictly positive."""
        s = Sampler()
        rng = np.random.default_rng(0)
        las = np.abs(rng.normal(10, 2, 100)) + 1.0
        dist = s.get_las_model_dist(train_set_path=None, las_values=las)
        samples = dist(10)
        assert samples.shape == (10,)

    def test_uses_yeo_johnson_when_non_positive_values_present(self):
        """Test that the distribution uses a Yeo-Johnson transformation when any LAS values are non-positive."""
        s = Sampler()
        las = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        dist = s.get_las_model_dist(train_set_path=None, las_values=las)
        samples = dist(5)
        assert samples.shape == (5,)

    def test_from_train_set_h5_file(self, tmp_path):
        """Test that the distribution can be created from a path to an HDF5 file containing LAS values."""
        s = Sampler()
        las = np.abs(np.random.default_rng(0).normal(10, 2, 50)) + 1.0
        path = tmp_path / "train.h5"
        with h5py.File(path, "w") as f:
            cat_info = np.zeros(50, dtype=[("LAS", "f8")])
            cat_info["LAS"] = las
            f.create_dataset("cat_info", data=cat_info)
        dist = s.get_las_model_dist(train_set_path=path)
        samples = dist(5)
        assert samples.shape == (5,)


class TestH5Saving:
    """
    Tests for Sampler._save_batch_h5(), which saves a batch of images to an HDF5 file, optionally with attributes and
    additional input datasets.
    """

    def test_save_batch_h5_creates_dataset_with_images_and_attrs(self, tmp_path):
        """Test that _save_batch_h5 creates a dataset with the given images and attributes."""
        s = Sampler()
        out_file = tmp_path / "out.h5"
        imgs = np.random.default_rng(0).normal(size=(4, 1, 8, 8)).astype(np.float32)
        s._save_batch_h5(out_file, imgs, dataset_name="samples", attrs={"timesteps": 25})

        with h5py.File(out_file, "r") as f:
            np.testing.assert_allclose(f["samples"][:], imgs)
            assert f["samples"].attrs["timesteps"] == 25

    def test_save_batch_h5_saves_inputs_as_separate_datasets(self, tmp_path):
        """Test that _save_batch_h5 saves additional input arrays as separate datasets in the HDF5 file."""
        s = Sampler()
        out_file = tmp_path / "out.h5"
        imgs = np.zeros((2, 1, 4, 4), dtype=np.float32)
        context = np.array([1.0, 2.0])
        s._save_batch_h5(out_file, imgs, dataset_name="samples", inputs={"context": context})

        with h5py.File(out_file, "r") as f:
            np.testing.assert_allclose(f["samples_context"][:], context)

    def test_append_to_existing_dataset_with_same_attrs(self, tmp_path):
        """Test that _save_batch_h5 appends to an existing dataset if the attributes match."""
        s = Sampler()
        out_file = tmp_path / "out.h5"
        imgs1 = np.ones((2, 1, 4, 4), dtype=np.float32)
        imgs2 = np.full((3, 1, 4, 4), 2.0, dtype=np.float32)
        attrs = {"timesteps": 10}
        s._save_batch_h5(out_file, imgs1, dataset_name="samples", attrs=attrs)
        s._save_batch_h5(out_file, imgs2, dataset_name="samples", attrs=attrs)

        with h5py.File(out_file, "r") as f:
            assert f["samples"].shape[0] == 5
            np.testing.assert_allclose(f["samples"][:2], imgs1)
            np.testing.assert_allclose(f["samples"][2:], imgs2)

    def test_different_attrs_creates_a_renamed_dataset_instead_of_appending(self, tmp_path):
        """Test that _save_batch_h5 creates a renamed dataset instead of appending when attributes differ."""
        s = Sampler()
        out_file = tmp_path / "out.h5"
        imgs1 = np.ones((2, 1, 4, 4), dtype=np.float32)
        imgs2 = np.full((2, 1, 4, 4), 2.0, dtype=np.float32)
        s._save_batch_h5(out_file, imgs1, dataset_name="samples", attrs={"timesteps": 10})
        s._save_batch_h5(out_file, imgs2, dataset_name="samples", attrs={"timesteps": 99})

        with h5py.File(out_file, "r") as f:
            assert "samples" in f
            assert "samples_1" in f
            assert f["samples"].shape[0] == 2  # not appended to
            assert f["samples_1"].shape[0] == 2
            np.testing.assert_allclose(f["samples_1"][:], imgs2)

    def test_settings_not_save_keys_are_ignored_when_comparing_attrs(self, tmp_path):
        """Test that keys in settings_not_save are ignored when comparing attributes for appending vs renaming."""
        s = Sampler()
        out_file = tmp_path / "out.h5"
        imgs1 = np.ones((2, 1, 4, 4), dtype=np.float32)
        imgs2 = np.full((2, 1, 4, 4), 2.0, dtype=np.float32)
        # "n_samples" is in settings_not_save, so differing values here should NOT trigger a rename.
        s._save_batch_h5(out_file, imgs1, dataset_name="samples", attrs={"n_samples": 10})
        s._save_batch_h5(out_file, imgs2, dataset_name="samples", attrs={"n_samples": 99})

        with h5py.File(out_file, "r") as f:
            assert "samples_1" not in f
            assert f["samples"].shape[0] == 4

    def test_second_rename_increments_the_suffix(self, tmp_path):
        """Test that a second rename increments the suffix correctly (samples -> samples_1 -> samples_2)."""
        s = Sampler()
        out_file = tmp_path / "out.h5"
        imgs = np.ones((2, 1, 4, 4), dtype=np.float32)
        s._save_batch_h5(out_file, imgs, dataset_name="samples", attrs={"timesteps": 1})
        s._save_batch_h5(out_file, imgs, dataset_name="samples", attrs={"timesteps": 2})  # -> samples_1
        s._save_batch_h5(out_file, imgs, dataset_name="samples", attrs={"timesteps": 3})  # -> samples_2

        with h5py.File(out_file, "r") as f:
            assert "samples" in f
            assert "samples_1" in f
            assert "samples_2" in f


class _FakeLoadedModel(torch.nn.Module):
    """
    This is a fake model that has a single parameter and returns itself when eval() is called. It is used to test the
    Sampler class without needing to load a real model.
    """

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self


def _fake_edm_sampling(model, context_batch=None, label_batch=None, latents=None, *,
                       image_size=80, batch_size=16, timesteps=25, guidance_strength=0.1,
                       sigma_min=2e-3, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float("inf"), S_noise=1):
    """
    Hook for diffusion.edm_sampling() that records its call arguments and returns a dummy tensor of the right shape.
    """
    _fake_edm_sampling.calls.append(dict(context_batch=context_batch, label_batch=label_batch, latents=latents,
                                         image_size=image_size, batch_size=batch_size, timesteps=timesteps))
    return [torch.full((batch_size, 1, image_size, image_size), float(i)) for i in range(timesteps + 1)]


@pytest.fixture(autouse=True)
def _reset_fake_edm_calls():
    """Fixture to reset the call history of _fake_edm_sampling before each test."""
    _fake_edm_sampling.calls = []
    yield


@pytest.fixture
def patched_sampling_deps(monkeypatch):
    """Fixture to monkeypatch out the heavy dependencies of Sampler.sample()/quick_sample() with fakes."""
    monkeypatch.setattr(diffusion, "edm_sampling", _fake_edm_sampling)
    monkeypatch.setattr(model_utils, "load_model", lambda model_name, **kwargs: _FakeLoadedModel())
    monkeypatch.setattr(device_utils, "distribute_model",
                        lambda model, n_devices, device_ids=None: (model, device_ids or [0]))


class TestQuickSample:
    """
    Tests for Sampler.quick_sample(), which orchestrates model loading, batch splitting, and calls to edm_sampling.
    """

    def test_single_batch_return_steps_true(self, patched_sampling_deps):
        """Test that quick_sample returns the correct shape when return_steps is True and a single batch is used."""
        s = Sampler(n_samples=4, samples_per_device=4, n_devices=1, timesteps=3, image_size=8)
        imgs = s.quick_sample("mymodel")
        assert imgs.shape == (4, 4, 1, 8, 8)  # (n_samples, timesteps+1, 1, image_size, image_size)

    def test_return_steps_false_keeps_only_final_image(self, patched_sampling_deps):
        """Test that quick_sample returns only the final image when return_steps is False."""
        s = Sampler(n_samples=4, samples_per_device=4, n_devices=1, timesteps=3, image_size=8,
                    return_steps=False)
        imgs = s.quick_sample("mymodel")
        assert imgs.shape == (4, 1, 8, 8)

    def test_multiple_batches_are_concatenated(self, patched_sampling_deps):
        """Test that quick_sample correctly handles multiple batches and concatenates the results."""
        s = Sampler(n_samples=8, samples_per_device=4, n_devices=1, timesteps=2, image_size=8)
        imgs = s.quick_sample("mymodel")
        assert imgs.shape == (8, 3, 1, 8, 8)
        assert len(_fake_edm_sampling.calls) == 2

    def test_context_shape_infers_n_samples(self, patched_sampling_deps):
        """Test that quick_sample infers n_samples from the shape of the provided context array."""
        s = Sampler(samples_per_device=1000, n_devices=1, timesteps=2, image_size=8)
        context = np.zeros((6, 3))
        s.quick_sample("mymodel", context=context)
        assert s.settings["n_samples"] == 6

    def test_context_and_labels_shape_mismatch_raises_assertion_error(self, patched_sampling_deps):
        """Test that quick_sample raises an AssertionError when context and labels have mismatched first dimensions."""
        # Regression test for the `self.context` typo (should be the local `context` parameter) that made this
        # branch raise AttributeError instead of validating shapes whenever both were passed together.
        s = Sampler(samples_per_device=1000, n_devices=1, timesteps=2, image_size=8)
        context = np.zeros((6, 3))
        labels = np.zeros((5,))
        with pytest.raises(AssertionError):
            s.quick_sample("mymodel", context=context, labels=labels)

    def test_unrecognized_setting_raises_value_error(self, patched_sampling_deps):
        """Test that quick_sample raises a ValueError when an unrecognized setting keyword argument is provided."""
        s = Sampler()
        with pytest.raises(ValueError):
            s.quick_sample("mymodel", not_a_real_setting=123)

    def test_uses_provided_model_without_calling_load_model(self, monkeypatch, patched_sampling_deps):
        """Test that quick_sample uses the provided model instance and does not call model_utils.load_model."""
        load_calls = []
        monkeypatch.setattr(model_utils, "load_model", lambda *a, **k: load_calls.append(1) or _FakeLoadedModel())
        s = Sampler(n_samples=2, samples_per_device=2, n_devices=1, timesteps=2, image_size=8)
        s.quick_sample("mymodel", model=_FakeLoadedModel())
        assert load_calls == []

    def test_context_fn_generates_context_when_no_context_given(self, patched_sampling_deps):
        """Test that quick_sample calls context_fn to generate context when no context is provided."""
        s = Sampler(n_samples=4, samples_per_device=4, n_devices=1, timesteps=2, image_size=8)
        calls = []

        def context_fn(n):
            calls.append(n)
            return np.zeros((n, 2))

        s.quick_sample("mymodel", context_fn=context_fn)
        assert calls == [4]

    def test_context_and_context_fn_together_raises_assertion_error(self, patched_sampling_deps):
        """Test that quick_sample raises an AssertionError when both context and context_fn are provided."""
        s = Sampler(n_samples=2, samples_per_device=2, n_devices=1, timesteps=2, image_size=8)
        with pytest.raises(AssertionError):
            s.quick_sample("mymodel", context=np.zeros((2, 1)), context_fn=lambda n: np.zeros((n, 1)))

    def test_recognized_setting_kwarg_updates_settings(self, patched_sampling_deps):
        """Test that recognized setting keyword arguments update the Sampler's settings correctly."""
        s = Sampler(n_samples=2, samples_per_device=2, n_devices=1, image_size=8)
        imgs = s.quick_sample("mymodel", timesteps=4)
        assert s.settings["timesteps"] == 4
        assert imgs.shape[1] == 5  # timesteps+1 steps, confirming the setting actually reached edm_sampling

    def test_labels_are_reshaped_and_forwarded(self, patched_sampling_deps):
        """Test that quick_sample reshapes the labels array and forwards it to edm_sampling correctly."""
        s = Sampler(n_samples=4, samples_per_device=4, n_devices=1, timesteps=2, image_size=8)
        labels = np.array([0, 1, 2, 3])
        s.quick_sample("mymodel", labels=labels)
        np.testing.assert_array_equal(_fake_edm_sampling.calls[0]["label_batch"], labels)

    def test_latents_are_reshaped_and_forwarded(self, patched_sampling_deps):
        """Test that quick_sample reshapes the latents array and forwards it to edm_sampling correctly."""
        s = Sampler(n_samples=4, samples_per_device=4, n_devices=1, timesteps=2, image_size=8)
        latents = np.zeros(4 * 1 * 8 * 8, dtype=np.float32)
        s.quick_sample("mymodel", latents=latents)
        assert _fake_edm_sampling.calls[0]["latents"] is not None

    def test_flux_transform_inverse_is_applied(self, patched_sampling_deps, monkeypatch):
        """
        Test that quick_sample applies the inverse of the flux_transform to the sampled images before returning them.
        """
        from diffracc.data import flux_transforms

        class _FakeTransform:
            """Fake flux transform that multiplies images by 1000 in its inverse method."""
            def to_dict(self):
                return {"name": "fake"}

            def inverse(self, imgs):
                return imgs * 1000.0

        monkeypatch.setattr(flux_transforms, "load", lambda spec: _FakeTransform())
        s = Sampler(n_samples=2, samples_per_device=2, n_devices=1, timesteps=2, image_size=8,
                    flux_transform="fake_spec")
        imgs = s.quick_sample("mymodel")
        # fake edm_sampling fills every step with float(i); the flux_transform should multiply everything by 1000
        assert imgs.max() == pytest.approx((imgs.shape[1] - 1) * 1000.0)


class TestSample:
    """Tests for Sampler.sample(), which orchestrates sampling and saves the output to an HDF5 file."""

    def test_saves_output_to_h5_file(self, patched_sampling_deps, tmp_path):
        """Test that sample() saves the output to an HDF5 file with the expected structure and attributes."""
        s = Sampler(out_root=tmp_path, n_samples=2, samples_per_device=2, n_devices=1, timesteps=2, image_size=8)
        s.sample("mymodel")
        out_file = tmp_path / "mymodel" / "mymodel_samples.h5"
        assert out_file.exists()
        with h5py.File(out_file, "r") as f:
            assert f["samples"].shape[0] == 2

    def test_custom_file_name(self, patched_sampling_deps, tmp_path):
        """Test that sample() saves the output to a custom HDF5 file name when provided."""
        s = Sampler(out_root=tmp_path, n_samples=2, samples_per_device=2, n_devices=1, timesteps=2, image_size=8)
        s.sample("mymodel", file_name="custom.h5")
        assert (tmp_path / "mymodel" / "custom.h5").exists()

    def test_recognized_setting_kwarg_updates_settings(self, patched_sampling_deps, tmp_path):
        """
        Test that recognized setting keyword arguments update the Sampler's settings correctly when calling sample().
        """
        s = Sampler(out_root=tmp_path, n_samples=2, samples_per_device=2, n_devices=1, image_size=8)
        s.sample("mymodel", timesteps=4)
        assert s.settings["timesteps"] == 4
        with h5py.File(tmp_path / "mymodel" / "mymodel_samples.h5", "r") as f:
            assert f["samples"].shape[1] == 5  # timesteps+1 steps

    def test_unrecognized_setting_raises_value_error(self, patched_sampling_deps, tmp_path):
        """Test that providing an unrecognized setting keyword argument raises a ValueError."""
        s = Sampler(out_root=tmp_path)
        with pytest.raises(ValueError):
            s.sample("mymodel", not_a_real_setting=123)

    def test_comment_appends_to_dataset_name(self, patched_sampling_deps, tmp_path):
        """Test that providing a comment appends it to the dataset name in the HDF5 file."""
        s = Sampler(out_root=tmp_path, n_samples=2, samples_per_device=2, n_devices=1, timesteps=2, image_size=8,
                    comment="run1")
        s.sample("mymodel")
        with h5py.File(tmp_path / "mymodel" / "mymodel_samples.h5", "r") as f:
            assert "samples_run1" in f
