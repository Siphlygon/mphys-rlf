"""
Unit tests for diffracc/utils/img_data_arrays.py.

ImageDataArrays.__init__ orchestrates a heavy pipeline (PyBDSF logs, residual/model FITS, dataset HDF5/FITS,
catalogs) that's out of scope for a unit test - these tests exercise its well-isolated helper methods directly,
via a bare instance built with object.__new__ (bypassing __init__) so only the attributes each method actually
touches need to be set up.
"""
import numpy as np
import pytest

from diffracc.utils import paths
from diffracc.utils.img_data_arrays import ImageDataArrays, NamedArrays, SubdirData
from diffracc.utils.logger import get_logger


def _bare_instance(**attrs):
    """Build an ImageDataArrays instance without running __init__, with only the given attributes set."""
    instance = object.__new__(ImageDataArrays)
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


class TestSubdirDataGetArrayNames:
    """Tests for the get_array_names method of the SubdirData dataclass."""

    def test_returns_all_ndarray_field_names(self):
        """Test that get_array_names returns the names of all fields that are numpy ndarrays."""
        names = SubdirData().get_array_names()
        assert set(names) == {
            "images", "residual_images", "model_images", "model_fluxes", "peak_fluxes",
            "sigma_clipped_means", "sigma_clipped_rmsds", "image_scale_factors", "las_values",
        }

    def test_defaults_are_independent_empty_arrays(self):
        """Test that the default values for the ndarray fields are independent empty arrays."""
        # dataclass mutable-default footgun check - two instances must not share the same underlying array
        a, b = SubdirData(), SubdirData()
        a.images = np.array([1.0])
        assert b.images.shape == (0,)


class TestAlignArrays:
    """Tests for the _align_arrays method of the ImageDataArrays class."""

    def test_restricts_to_intersection_of_indices(self):
        """Test that _align_arrays restricts the arrays to the intersection of their indices."""
        source1 = NamedArrays({"a": np.array([10, 20, 30, 50])}, np.array([1, 2, 3, 5]))
        source2 = NamedArrays({"b": np.array([200, 300, 400, 500])}, np.array([2, 3, 4, 5]))
        instance = _bare_instance()

        aligned = instance._align_arrays([source1, source2])

        # intersection of {1,2,3,5} and {2,3,4,5} is {2,3,5}
        np.testing.assert_array_equal(aligned["a"], [20, 30, 50])
        np.testing.assert_array_equal(aligned["b"], [200, 300, 500])

    def test_reorders_source_not_sorted_by_index(self):
        """Test that if a source's indices are not sorted, the arrays are reordered to match the sorted index order."""
        # source1's indices arrive out of order; alignment must still match by index value, not position
        source1 = NamedArrays({"a": np.array([30, 10, 20])}, np.array([3, 1, 2]))
        source2 = NamedArrays({"b": np.array([100, 200, 300])}, np.array([1, 2, 3]))
        instance = _bare_instance()

        aligned = instance._align_arrays([source1, source2])

        np.testing.assert_array_equal(aligned["a"], [10, 20, 30])
        np.testing.assert_array_equal(aligned["b"], [100, 200, 300])

    def test_single_source_returns_all_its_arrays_unfiltered(self):
        """Test that if only a single source is provided, all its arrays are returned unfiltered."""
        source = NamedArrays({"a": np.array([1, 2, 3])}, np.array([0, 1, 2]))
        instance = _bare_instance()
        aligned = instance._align_arrays([source])
        np.testing.assert_array_equal(aligned["a"], [1, 2, 3])


class TestComputePeakFluxes:
    """Tests for the _compute_peak_fluxes method of the ImageDataArrays class."""

    def test_h5_dataset_path_uses_image_max_directly(self):#
        """Test that when use_dataset_h5=True, the peak fluxes are computed directly from the image max values."""
        images = np.stack([np.full((4, 4), 0.001), np.full((4, 4), 0.002)])  # Jy/beam
        aligned = {"images": images}
        instance = _bare_instance()

        peak_fluxes_mjy, image_max = instance._compute_peak_fluxes(aligned, use_dataset_h5=True, subdir="x")

        np.testing.assert_allclose(image_max, [0.001, 0.002])
        np.testing.assert_allclose(peak_fluxes_mjy, [1.0, 2.0])  # Jy -> mJy

    def test_fits_path_inverts_the_power_transform(self, np_array_parent):
        """Test that when use_dataset_h5=False, the peak fluxes are computed by inverting the power transform."""
        from diffracc.utils.power_transform import PeakFluxPowerTransformer

        (np_array_parent / "gen").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        maxvals = np.abs(rng.normal(5, 1, 30)) + 1.0
        images = np.stack([np.full((4, 4), v) for v in maxvals])
        pt = PeakFluxPowerTransformer("gen", maxvals=maxvals)
        true_peak_mjy = maxvals[:5]
        transformed = pt.transform(true_peak_mjy / 1000)
        aligned = {"images": images[:5], "peak_fluxes_transformed": transformed}
        instance = _bare_instance()

        peak_fluxes_mjy, image_max = instance._compute_peak_fluxes(aligned, use_dataset_h5=False, subdir="gen")

        np.testing.assert_allclose(peak_fluxes_mjy, true_peak_mjy, rtol=1e-4)
        np.testing.assert_allclose(image_max, maxvals[:5])


class TestBuildSubdirData:
    """Tests for the _build_subdir_data method of the ImageDataArrays class."""

    def _aligned(self, n=3):
        """Helper to create a dict of aligned arrays for testing _build_subdir_data."""
        images = np.stack([np.full((4, 4), float(i + 1)) for i in range(n)])  # image max = i+1
        return {
            "images": images,
            "residual_images": np.ones((n, 4, 4)),
            "model_images": np.full((n, 4, 4), 2.0),
            "normalized_model_fluxes": np.full(n, 3.0),
            "las_values": np.arange(n),
            "sigma_clipped_means": np.full(n, 0.5),
            "sigma_clipped_rmsds": np.full(n, 0.1),
        }

    def test_do_unscaling_true_scales_by_recovered_peak_flux_ratio(self):
        """
        Test that when do_unscaling=True, the images and model fluxes are scaled by the ratio of recovered peak flux to
        image max.
        """
        instance = _bare_instance(config={"do_unscaling": "True"})
        aligned = self._aligned(n=2)
        # use_dataset_h5=True so peak_fluxes_mjy = image_max * 1000 directly, giving scale factor = 1000 always
        data = instance._build_subdir_data("gen", aligned, use_dataset_h5=True)

        np.testing.assert_allclose(data.image_scale_factors, [1000.0, 1000.0])
        np.testing.assert_allclose(data.images, aligned["images"] * 1000.0)
        np.testing.assert_allclose(data.model_fluxes, aligned["normalized_model_fluxes"] * 1000.0)
        np.testing.assert_allclose(data.peak_fluxes, [1000.0, 2000.0])

    def test_do_unscaling_false_leaves_scale_factors_as_ones(self):
        """Test that when do_unscaling=False, the image scale factors are left as ones."""
        instance = _bare_instance(config={"do_unscaling": "False"})
        aligned = self._aligned(n=2)

        data = instance._build_subdir_data("gen", aligned, use_dataset_h5=True)

        np.testing.assert_allclose(data.image_scale_factors, [1.0, 1.0])
        np.testing.assert_allclose(data.images, aligned["images"])
        np.testing.assert_allclose(data.model_fluxes, aligned["normalized_model_fluxes"])

    def test_las_values_pass_through_unscaled(self):
        """Test that the LAS values are passed through unscaled."""
        instance = _bare_instance(config={"do_unscaling": "True"})
        aligned = self._aligned(n=3)
        data = instance._build_subdir_data("gen", aligned, use_dataset_h5=True)
        np.testing.assert_array_equal(data.las_values, aligned["las_values"])


class TestLoadFromCacheAndSaveArrays:
    """Tests for the load_from_cache and save_arrays methods of the ImageDataArrays class."""

    def _instance(self):
        """
        Helper to create a bare ImageDataArrays instance with a logger for testing load_from_cache and save_arrays.
        """
        return _bare_instance(logger=get_logger("test_img_data_arrays"))

    def test_round_trips_all_arrays_through_save_and_load(self, np_array_parent):
        """Test that saving arrays to disk and then loading them back returns the same data."""
        instance = self._instance()
        (np_array_parent / "gen").mkdir(parents=True, exist_ok=True)
        arrays = {name: np.arange(5).astype(np.float64) for name in SubdirData().get_array_names()}

        instance.save_arrays("gen", **arrays)
        loaded = instance.load_from_cache("gen")

        for name, expected in arrays.items():
            np.testing.assert_array_equal(getattr(loaded, name), expected)

    def test_missing_array_file_raises_file_not_found(self, np_array_parent):
        """Test that loading a non-existent array file raises a FileNotFoundError."""
        instance = self._instance()
        (np_array_parent / "gen").mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            instance.load_from_cache("gen")

    def test_save_arrays_ignores_non_ndarray_kwargs(self, np_array_parent):
        """Test that save_arrays ignores any keyword arguments that are not numpy ndarrays."""
        instance = self._instance()
        (np_array_parent / "gen").mkdir(parents=True, exist_ok=True)
        instance.save_arrays("gen", images=np.array([1.0, 2.0]), not_an_array="hello")
        assert (np_array_parent / "gen" / "images.npy").exists()
        assert not (np_array_parent / "gen" / "not_an_array.npy").exists()


class TestSaveAllArrays:
    """Tests for the save_all_arrays method of the ImageDataArrays class."""

    def test_saves_arrays_for_both_subdirs(self, np_array_parent):
        """Test that save_all_arrays saves the arrays for both the dataset and generated subdirectories."""
        instance = _bare_instance(
            logger=get_logger("test_img_data_arrays"),
            config={"dataset_subdir": "dset", "generated_subdir": "gen"},
            dataset_data=SubdirData(images=np.array([1.0])),
            generated_data=SubdirData(images=np.array([2.0])),
        )
        (np_array_parent / "dset").mkdir(parents=True, exist_ok=True)
        (np_array_parent / "gen").mkdir(parents=True, exist_ok=True)

        instance.save_all_arrays()

        np.testing.assert_array_equal(np.load(np_array_parent / "dset" / "images.npy"), [1.0])
        np.testing.assert_array_equal(np.load(np_array_parent / "gen" / "images.npy"), [2.0])

    def test_only_subdirs_restricts_which_subdir_is_saved(self, np_array_parent):
        """Test that specifying only_subdirs restricts the saving to only those subdirectories."""
        instance = _bare_instance(
            logger=get_logger("test_img_data_arrays"),
            config={"dataset_subdir": "dset", "generated_subdir": "gen"},
            dataset_data=SubdirData(images=np.array([1.0])),
            generated_data=SubdirData(images=np.array([2.0])),
        )
        (np_array_parent / "dset").mkdir(parents=True, exist_ok=True)
        (np_array_parent / "gen").mkdir(parents=True, exist_ok=True)

        instance.save_all_arrays(only_subdirs={"gen"})

        assert not (np_array_parent / "dset" / "images.npy").exists()
        np.testing.assert_array_equal(np.load(np_array_parent / "gen" / "images.npy"), [2.0])
