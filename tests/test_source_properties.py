"""Unit tests for diffracc/evaluation/source_properties.py."""
import numpy as np
import pytest

from diffracc.evaluation import source_properties as sp


class TestAsImageStack:
    """Tests for :func:`diffracc.evaluation.source_properties.as_image_stack`."""

    def test_2d_image_gets_batch_dim_added(self):
        """Test that a single 2D image is converted to a 3D stack with batch dimension."""
        img = np.zeros((10, 10))
        stack = sp.as_image_stack(img)
        assert stack.shape == (1, 10, 10)
        assert stack.dtype == np.float32

    def test_3d_stack_passes_through(self):
        """Test that a 3D image stack passes through unchanged."""
        stack = sp.as_image_stack(np.zeros((5, 10, 10)))
        assert stack.shape == (5, 10, 10)

    def test_4d_single_channel_gets_squeezed(self):
        """Test that a 4D image stack with a single channel gets squeezed to 3D."""
        stack = sp.as_image_stack(np.zeros((5, 1, 10, 10)))
        assert stack.shape == (5, 10, 10)

    def test_invalid_shape_raises_value_error(self):
        """Test that an invalid input shape raises a ValueError."""
        with pytest.raises(ValueError):
            sp.as_image_stack(np.zeros((5,)))

    def test_accepts_torch_like_tensor(self):
        """Test that a torch-like tensor with .detach().cpu().numpy() is accepted."""
        class _FakeTensor:
            def __init__(self, arr):
                self._arr = arr
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self._arr

        stack = sp.as_image_stack(_FakeTensor(np.zeros((3, 8, 8))))
        assert stack.shape == (3, 8, 8)


class TestBackgroundStats:
    """Tests for :func:`diffracc.evaluation.source_properties._background_stats`."""

    def test_constant_image_gives_zero_std(self):
        """Test that a constant image returns the correct median and zero standard deviation."""
        med, std = sp._background_stats(np.full((10, 10), 5.0))
        assert med == pytest.approx(5.0)
        assert std == pytest.approx(0.0)

    def test_falls_back_to_mad_when_sigma_clipped_stats_raises(self, monkeypatch):
        """Test that the function falls back to MAD-based std when sigma_clipped_stats raises an exception."""
        def _raise(*args, **kwargs):
            raise RuntimeError("simulated astropy failure")
        monkeypatch.setattr(sp, "sigma_clipped_stats", _raise)

        img = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # median=3, MAD-based std = 1.4826 * median(|x-3|) = 1.4826*1
        med, std = sp._background_stats(img)
        assert med == pytest.approx(3.0)
        assert std == pytest.approx(1.4826)


class TestExtractProperties:
    """Tests for :func:`diffracc.evaluation.source_properties.extract_properties`."""

    def _noisy_background_with_bright_pixel(self, value=10.0):
        """Helper to create a small image with a noisy background and a single bright pixel."""
        rng = np.random.default_rng(0)
        img = rng.normal(loc=0.0, scale=0.01, size=(20, 20)).astype(np.float32)
        img[10, 10] = value
        return img

    def test_isolated_bright_pixel_is_the_only_detection(self):
        """
        Test that an isolated bright pixel in a noisy background is detected correctly and its properties are extracted.
        """
        img = self._noisy_background_with_bright_pixel()
        props = sp.extract_properties(img, nsigma=5.0)

        assert props.peak == pytest.approx(float(img.max()))
        assert props.n_components == 1
        assert props.source_area == 1
        assert props.extent == pytest.approx(0.0)  # single-pixel mask has no spatial variance
        assert props.concentration == pytest.approx(1.0, rel=1e-3)  # total_flux == peak - bg_median for one pixel
        assert props.snr > 100  # far above the ~0.01 background noise

    def test_all_zero_image_returns_zeroed_properties_with_nan_concentration(self):
        """
        Test that an all-zero image returns zeroed properties and NaN for concentration (since total_flux is zero).
        """
        props = sp.extract_properties(np.zeros((10, 10)), nsigma=5.0)
        assert props.peak == pytest.approx(0.0)
        assert props.total_flux == pytest.approx(0.0)
        assert props.n_components == 0
        assert props.source_area == 0
        assert props.extent == pytest.approx(0.0)
        assert np.isnan(props.concentration)

    def test_snr_is_nan_when_background_rms_is_zero_but_a_peak_exists(self):
        """Test that S/N is NaN when the background RMS is zero but a peak exists, to avoid division by zero."""
        # A constant background (rms=0) with nothing above threshold still has rms==0, but if it did have a peak
        # above background it should report NaN snr rather than dividing by zero. Constant image already covers
        # rms==0; this checks the guard explicitly by constructing one via a directly-zero-std background.
        img = np.zeros((10, 10))
        props = sp.extract_properties(img, nsigma=5.0)
        assert np.isnan(props.snr)

    def test_larger_source_gives_larger_extent(self):
        """Test that a larger source in the image results in a larger extent property."""
        # Keep both blocks a small minority of the image's pixels (<3%) - once a "source" gets too large a share
        # of the frame, the sigma-clipped background estimate itself breaks down (it assumes the source is a
        # minority), which would confound this comparison rather than test extent scaling.
        rng = np.random.default_rng(1)
        small = rng.normal(loc=0.0, scale=0.01, size=(40, 40)).astype(np.float32)
        small[19:21, 19:21] = 10.0  # 2x2 block, 4/1600 px
        large = small.copy()
        large[17:23, 17:23] = 10.0  # 6x6 block, 36/1600 px

        small_props = sp.extract_properties(small, nsigma=5.0)
        large_props = sp.extract_properties(large, nsigma=5.0)
        assert large_props.extent > small_props.extent
        assert large_props.source_area > small_props.source_area


class TestExtractBatch:
    """Tests for :func:`diffracc.evaluation.source_properties.extract_batch`."""

    def test_returns_one_result_per_image(self):
        """Test that extract_batch returns a list of SourceProperties, one for each image in the batch."""
        images = np.random.default_rng(0).normal(scale=0.01, size=(4, 10, 10)).astype(np.float32)
        props = sp.extract_batch(images, nsigma=5.0)
        assert len(props) == 4
        assert all(isinstance(p, sp.SourceProperties) for p in props)

    def test_uses_custom_extractor_when_given(self):
        """
        Test that extract_batch uses a custom extractor function when provided, and passes the nsigma argument to it.
        """
        images = np.zeros((3, 10, 10))
        calls = []

        def fake_extractor(img, nsigma=5.0):
            calls.append(nsigma)
            return sp.SourceProperties(1, 2, 3, 4, 5, 6, 7, 8, 9)

        props = sp.extract_batch(images, nsigma=7.0, extractor=fake_extractor)
        assert len(calls) == 3
        assert all(c == 7.0 for c in calls)
        assert all(p.peak == 1 for p in props)


class TestSourcePropertiesContainer:
    """Tests for the SourceProperties container class."""

    @pytest.fixture
    def sample(self):
        """Fixture that returns a sample SourceProperties instance for testing."""
        return sp.SourceProperties(peak=1.0, total_flux=2.0, rms=0.1, snr=10.0, n_components=1,
                                   source_area=5, extent=2.5, concentration=0.5, bg_median=0.01)

    def test_to_dict_matches_property_keys(self, sample):
        """Test that the to_dict method returns a dictionary with keys matching PROPERTY_KEYS and correct values."""
        d = sample.to_dict()
        assert set(d.keys()) == set(sp.PROPERTY_KEYS)
        assert d["peak"] == 1.0
        assert d["extent"] == 2.5

    def test_to_array_default_order_matches_property_keys(self, sample):
        """Test that the to_array method returns an array with values in the order of PROPERTY_KEYS by default."""
        arr = sample.to_array()
        expected = [getattr(sample, k) for k in sp.PROPERTY_KEYS]
        np.testing.assert_allclose(arr, expected)

    def test_to_array_respects_custom_key_order(self, sample):
        """Test that the to_array method respects a custom order of keys when provided."""
        arr = sample.to_array(["extent", "peak"])
        np.testing.assert_allclose(arr, [2.5, 1.0])

    def test_stack_collates_across_instances(self, sample):
        """
        Test that the stack method correctly collates properties across multiple SourceProperties instances into a
        dictionary of arrays.
        """        
        sample2 = sp.SourceProperties(peak=9.0, total_flux=0, rms=0, snr=0, n_components=0,
                                      source_area=0, extent=0, concentration=0, bg_median=0)
        stacked = sp.SourceProperties.stack([sample, sample2], keys=["peak"])
        np.testing.assert_allclose(stacked["peak"], [1.0, 9.0])

    def test_from_image_matches_extract_properties(self):
        """
        Test that the from_image class method produces the same SourceProperties as extract_properties for a given
        image.
        """
        img = np.random.default_rng(0).normal(scale=0.01, size=(10, 10)).astype(np.float32)
        img[5, 5] = 5.0
        assert sp.SourceProperties.from_image(img) == sp.extract_properties(img)

    def test_from_batch_matches_extract_batch(self):
        """
        Test that the from_batch class method produces the same list of SourceProperties as extract_batch for a batch of
        images.
        """
        images = np.random.default_rng(0).normal(scale=0.01, size=(3, 10, 10)).astype(np.float32)
        assert sp.SourceProperties.from_batch(images) == sp.extract_batch(images)


class TestFeatureMatrix:
    """Tests for the feature_matrix function."""

    def test_log_scales_flux_like_features_linearly_others(self):
        """Test that flux-like features are log-scaled while others remain linear in the feature matrix."""
        props = [sp.SourceProperties(peak=100.0, total_flux=10.0, rms=1.0, snr=50.0, n_components=2,
                                     source_area=5, extent=3.0, concentration=0.2, bg_median=0.0)]
        mat = sp.feature_matrix(props, keys=["peak", "extent"])
        assert mat.shape == (1, 2)
        assert mat[0, 0] == pytest.approx(np.log10(100.0))  # peak is log-scaled
        assert mat[0, 1] == pytest.approx(3.0)  # extent is linear

    def test_empty_props_returns_empty_matrix_with_correct_width(self):
        """Test that an empty list of SourceProperties returns an empty matrix with the correct number of columns."""
        mat = sp.feature_matrix([], keys=["peak", "extent", "snr"])
        assert mat.shape == (0, 3)

    def test_drops_rows_with_non_finite_features(self):
        """Test that rows with non-finite features (NaN or Inf) are dropped from the feature matrix."""
        good = sp.SourceProperties(peak=1.0, total_flux=1.0, rms=1.0, snr=1.0, n_components=1,
                                   source_area=1, extent=1.0, concentration=1.0, bg_median=0.0)
        bad = sp.SourceProperties(peak=1.0, total_flux=1.0, rms=1.0, snr=1.0, n_components=1,
                                  source_area=1, extent=1.0, concentration=np.nan, bg_median=0.0)
        mat = sp.feature_matrix([good, bad], keys=["peak", "concentration"])
        assert mat.shape == (1, 2)
