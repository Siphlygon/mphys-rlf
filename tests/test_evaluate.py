"""Unit tests for diffracc/evaluation/evaluate.py's Tier-1 evaluation reports."""
import numpy as np
import pytest

from diffracc.evaluation import evaluate
from diffracc.evaluation.source_properties import PROPERTY_KEYS


def _make_batch(peaks, size=30, noise_scale=0.01, seed_offset=0):
    """
    Build a batch of synthetic images with the given peak flux values, well above a small independent noise floor
    so the source finder reliably recovers peak == the requested value exactly (every blob pixel is set to the
    same peak value, so the max - and hence the recovered peak - is unaffected by blob shape).

    Source morphology (blob count and size) is varied across the batch, not just peak flux. A single fixed-shape
    source at a fixed location leaves n_components/source_area/extent/concentration exactly constant across the
    whole batch, which makes the feature covariance matrix singular regardless of sample count - that's numerically
    borderline for frechet_distance's scipy.linalg.sqrtm (succeeds on some BLAS/LAPACK builds, raises SqrtmError on
    others), not something more samples alone fixes.
    """
    imgs = []
    for i, peak in enumerate(peaks):
        rng = np.random.default_rng(seed_offset + i)
        img = rng.normal(loc=0.0, scale=noise_scale, size=(size, size)).astype(np.float32)
        n_blobs = 1 + (i % 3)
        blob_size = 1 + (i % 2)
        for j in range(n_blobs):
            r = c = 5 + 8 * j
            img[r:r + blob_size, c:c + blob_size] = peak
        imgs.append(img)
    return np.stack(imgs)


class TestPhysicalDistributionReport:
    """
    Tests for the physical_distribution_report function, which compares the distributions of source properties in
    generated and real images, using the source finder to extract properties from each image.
    """

    def test_reports_expected_top_level_keys_and_counts(self):
        """Test that the report includes the expected top-level keys and counts of generated and real images."""
        generated = _make_batch([1.0, 2.0, 3.0])
        real = _make_batch([1.0, 2.0, 3.0])
        report = evaluate.physical_distribution_report(generated, real)

        assert report["n_generated"] == 3
        assert report["n_real"] == 3
        assert set(report["per_property"].keys()) == set(PROPERTY_KEYS)
        assert "physical_fid" in report
        assert "physical_kid" in report

    def test_per_property_entries_have_expected_fields(self):
        """
        Test that each property in the report includes the expected fields, such as KS statistic and p-value, and the
        median values for generated and real images.
        """
        rng = np.random.default_rng(0)
        peaks = 10 ** rng.uniform(-0.5, 0.5, 15)
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        report = evaluate.physical_distribution_report(generated, real)
        peak_stats = report["per_property"]["peak"]
        assert set(peak_stats.keys()) == {"w1", "ks_stat", "ks_pvalue", "gen_median", "real_median"}

    def test_fid_and_kid_are_small_for_matched_distributions(self):
        """
        Test that the Frechet Inception Distance (FID) and Kernel Inception Distance (KID) are small when the generated
        and real distributions are closely matched.
        """
        rng = np.random.default_rng(0)
        peaks = 10 ** rng.uniform(-0.5, 0.5, 40)
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        report = evaluate.physical_distribution_report(generated, real)
        assert report["physical_fid"] < 0.5
        assert abs(report["physical_kid"]) < 0.5

    def test_fid_is_larger_for_clearly_different_distributions(self):
        """Test that the Frechet Inception Distance (FID) is larger for clearly different distributions."""
        rng = np.random.default_rng(0)
        real_peaks = 10 ** rng.uniform(-0.5, 0.5, 40)
        gen_peaks = 10 ** rng.uniform(1.5, 2.5, 40)  # systematically much brighter
        matched_report = evaluate.physical_distribution_report(_make_batch(real_peaks), _make_batch(real_peaks))
        different_report = evaluate.physical_distribution_report(_make_batch(gen_peaks), _make_batch(real_peaks))
        assert different_report["physical_fid"] > matched_report["physical_fid"]

    def test_respects_custom_feature_keys(self):
        """Test that the report respects custom feature keys when provided."""
        generated = _make_batch([1.0, 2.0])
        real = _make_batch([1.0, 2.0])
        report = evaluate.physical_distribution_report(generated, real, feature_keys=["peak", "extent"])
        assert report["feature_keys"] == ["peak", "extent"]


class TestCalibrationReport:
    """
    Tests for the calibration_report function, which compares the prompted peak values to the recovered peak values
    in generated images, assessing the linearity and bias of the recovery process.
    """

    def test_recovers_exact_slope_one_intercept_zero_for_perfect_conditioning(self):
        """
        Test that the calibration report recovers a slope of 1 and an intercept of 0 when the generated peak values
        perfectly match the prompted peak values.
        """
        rng = np.random.default_rng(0)
        peaks = 10 ** rng.uniform(-1.0, 1.0, 30)
        generated = _make_batch(peaks)

        report = evaluate.calibration_report(generated, peaks)

        assert report["n_used"] == 30
        assert report["slope"] == pytest.approx(1.0, abs=1e-6)
        assert report["intercept"] == pytest.approx(0.0, abs=1e-6)
        assert report["r2"] == pytest.approx(1.0, abs=1e-6)
        assert report["bias"] == pytest.approx(1.0, abs=1e-6)
        assert report["scatter_dex"] == pytest.approx(0.0, abs=1e-6)

    def test_mismatched_length_raises_value_error(self):
        """Test that a ValueError is raised when the lengths of the generated and prompted peak arrays do not match."""
        generated = _make_batch([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            evaluate.calibration_report(generated, prompted_peak=np.array([1.0, 2.0]))

    def test_binned_curve_has_n_bins_entries(self):
        """Test that the binned curve has the correct number of entries."""
        rng = np.random.default_rng(0)
        peaks = 10 ** rng.uniform(-1.0, 1.0, 50)
        report = evaluate.calibration_report(_make_batch(peaks), peaks, n_bins=5)
        assert report["bin_centers"].shape == (5,)
        assert report["bin_median_ratio"].shape == (5,)
        assert report["bin_scatter_dex"].shape == (5,)
        assert report["bin_counts"].shape == (5,)
        assert report["bin_counts"].sum() == 50

    def test_systematic_bias_is_recovered(self):
        """Test that a systematic bias in the generated peak values is correctly recovered in the calibration report."""
        # Recovered peak is always exactly 2x the prompted value -> a clean multiplicative bias of 2.
        rng = np.random.default_rng(0)
        prompted = 10 ** rng.uniform(-1.0, 1.0, 30)
        generated = _make_batch(prompted * 2.0)
        report = evaluate.calibration_report(generated, prompted)
        assert report["bias"] == pytest.approx(2.0, rel=1e-3)
        assert report["slope"] == pytest.approx(1.0, abs=1e-6)  # still linear, just offset


class TestMemorisationVectors:
    """
    Tests for the _memorisation_vectors function, which computes a vector representation of each image for
    nearest-neighbour distance calculations, using block-averaging and peak-normalisation.
    """

    def test_hand_computed_block_average_and_peak_normalisation(self):
        """Test that the function correctly computes block-averaged and peak-normalised vectors for a simple image."""
        img = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.float32)
        vec = evaluate._memorisation_vectors(img[None], downsample=2)
        np.testing.assert_allclose(vec[0], [0.25, 0.5, 0.75, 1.0])

    def test_no_downsampling_when_downsample_not_smaller_than_image(self):
        """Test that no downsampling is applied when the downsample parameter is not smaller than the image size."""
        img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        vec = evaluate._memorisation_vectors(img[None], downsample=100)
        np.testing.assert_allclose(vec[0], [0.25, 0.5, 0.75, 1.0])  # peak-normalised, flattened, unblocked

    def test_all_zero_image_does_not_divide_by_zero(self):
        """Test that an image with all zero values does not cause a division by zero error."""
        img = np.zeros((4, 4), dtype=np.float32)
        vec = evaluate._memorisation_vectors(img[None], downsample=2)
        assert np.all(np.isfinite(vec))
        np.testing.assert_allclose(vec, 0.0)


class TestMemorisationReport:
    """Tests for the memorisation_report function, which computes nearest-neighbour distances between generated and
    training images to assess potential memorisation of training data."""

    def test_exact_duplicate_has_near_zero_nn_distance(self):
        """Test that an exact duplicate of a training image has a near-zero nearest-neighbour distance in the report."""
        train = _make_batch([1.0, 2.0, 3.0])
        generated = train[:1].copy()  # an exact copy of the first training image
        report = evaluate.memorisation_report(generated, train, downsample=10)
        assert report["gen_nn_median"] == pytest.approx(0.0, abs=1e-5)

    def test_far_from_training_set_has_larger_nn_distance(self):
        """
        Test that an image structurally different from the training set has a larger nearest-neighbour distance than an
        exact duplicate.
        """
        train = _make_batch([1.0, 2.0, 3.0], size=20)
        # A structurally different image (bright pixel in a different location) should sit further from every
        # training image than an exact duplicate does.
        far_image = np.zeros((1, 20, 20), dtype=np.float32)
        far_image[0, 0, 0] = 1.0
        duplicate_report = evaluate.memorisation_report(train[:1].copy(), train, downsample=10)
        far_report = evaluate.memorisation_report(far_image, train, downsample=10)
        assert far_report["gen_nn_median"] > duplicate_report["gen_nn_median"]

    def test_includes_validation_baseline_when_val_given(self):
        """Test that the validation baseline is included in the report when a validation set is provided."""
        train = _make_batch([1.0, 2.0, 3.0])
        generated = _make_batch([1.0, 2.0])
        val = _make_batch([4.0, 5.0])
        report = evaluate.memorisation_report(generated, train, val=val, downsample=10)
        assert "val_nn_median" in report
        assert "median_ratio_gen_over_val" in report
        assert "w1_gen_vs_val" in report

    def test_omits_validation_keys_when_val_not_given(self):
        """Test that validation-related keys are omitted from the report when no validation set is provided."""
        train = _make_batch([1.0, 2.0, 3.0])
        generated = _make_batch([1.0, 2.0])
        report = evaluate.memorisation_report(generated, train, downsample=10)
        assert "val_nn_median" not in report
        assert "median_ratio_gen_over_val" not in report


def _varied_peaks(n=15, seed=0):
    """Generate a set of peak values that vary over several orders of magnitude, for testing."""
    rng = np.random.default_rng(seed)
    return 10 ** rng.uniform(-0.5, 0.5, n)


class TestFullReport:
    """
    Tests for the full_report function, which combines physical distribution, calibration, and memorisation reports into
    a single comprehensive report.
    """

    def test_includes_only_physical_distribution_by_default(self):
        """
        Test that the full report includes only the physical distribution section by default, without calibration or
        memorisation sections.
        """
        peaks = _varied_peaks()
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        report = evaluate.full_report(generated, real)
        assert "physical_distribution" in report
        assert "calibration" not in report
        assert "memorisation" not in report

    def test_includes_calibration_when_prompted_peak_given(self):
        """
        Test that the full report includes the calibration section when a prompted peak is provided.
        """
        peaks = _varied_peaks()
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        report = evaluate.full_report(generated, real, prompted_peak=peaks)
        assert "calibration" in report

    def test_includes_memorisation_when_train_given(self):
        """Test that the full report includes the memorisation section when a training set is provided."""
        peaks = _varied_peaks()
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        train = _make_batch(_varied_peaks(seed=1))
        report = evaluate.full_report(generated, real, train=train)
        assert "memorisation" in report


class TestSummarise:
    """Tests for the summarise function, which generates a human-readable summary of the evaluation report."""

    def test_includes_all_present_sections(self):
        """Test that the summary includes all sections present in the report."""
        peaks = _varied_peaks()
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        train = _make_batch(_varied_peaks(seed=1))
        report = evaluate.full_report(generated, real, prompted_peak=peaks, train=train)

        text = evaluate.summarise(report)

        assert "Physical distribution" in text
        assert "Calibration" in text
        assert "Memorisation" in text

    def test_omits_absent_sections(self):
        """Test that the summary omits sections that are not present in the report."""
        peaks = _varied_peaks()
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        report = evaluate.full_report(generated, real)

        text = evaluate.summarise(report)

        assert "Physical distribution" in text
        assert "Calibration" not in text
        assert "Memorisation" not in text

    def test_includes_val_baseline_ratio_when_val_given(self):
        """Test that the summary includes the validation baseline ratio when a validation set is provided."""
        peaks = _varied_peaks()
        generated = _make_batch(peaks)
        real = _make_batch(peaks)
        train = _make_batch(_varied_peaks(seed=1))
        val = _make_batch(_varied_peaks(seed=2))
        report = evaluate.full_report(generated, real, train=train, val=val)

        text = evaluate.summarise(report)

        assert "val NN median" in text
        assert "ratio =" in text
