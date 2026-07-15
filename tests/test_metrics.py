"""Unit tests for diffracc/evaluation/metrics.py's embedding-agnostic statistical distances."""
import numpy as np
import pytest

from diffracc.evaluation import metrics


class TestCleanSamples:
    """Tests for the _clean_samples function, which removes NaN and Inf values and converts to float64."""

    def test_drops_nan_and_inf(self):
        """Test that _clean_samples correctly removes NaN and Inf values from the input arrays."""
        cleaned, = metrics._clean_samples(np.array([1.0, np.nan, 2.0, np.inf, -np.inf, 3.0]))
        np.testing.assert_allclose(np.sort(cleaned), [1.0, 2.0, 3.0])

    def test_converts_to_float64(self):
        """Test that _clean_samples converts integer arrays to float64."""
        cleaned, = metrics._clean_samples(np.array([1, 2, 3], dtype=np.int32))
        assert cleaned.dtype == np.float64

    def test_cleans_multiple_samples_independently(self):
        """Test that _clean_samples cleans multiple input arrays independently."""
        a, b = metrics._clean_samples(np.array([1.0, np.nan]), np.array([np.inf, 5.0, 6.0]))
        np.testing.assert_allclose(a, [1.0])
        np.testing.assert_allclose(b, [5.0, 6.0])


class TestWasserstein1d:
    """
    Tests for the wasserstein_1d function, which computes the 1-D Wasserstein distance between two empirical
    distributions.
    """

    def test_zero_for_identical_samples(self):
        """Test that the Wasserstein distance is zero for two identical samples."""
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.wasserstein_1d(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_matches_exact_shift_between_equal_shaped_samples(self):
        """Test that the Wasserstein distance matches the exact shift between two samples of equal size."""
        # For two empirical distributions of equal size related by a constant shift, W1 is exactly that shift -
        # the optimal transport plan is just "move point i to point i".
        sample1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample2 = sample1 + 3.0
        assert metrics.wasserstein_1d(sample1, sample2) == pytest.approx(3.0)

    def test_ignores_nan_and_inf(self):
        """Test that the Wasserstein distance computation ignores NaN and Inf values in the input samples."""
        sample1 = np.array([1.0, 2.0, 3.0])
        sample2 = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
        assert metrics.wasserstein_1d(sample1, sample2) == pytest.approx(0.0, abs=1e-9)


class TestKS2samp:
    """Tests for the ks_2samp function, which computes the two-sample Kolmogorov-Smirnov test statistic and p-value."""

    def test_identical_samples_give_zero_statistic_and_pvalue_one(self):
        """Test that the KS statistic is zero and the p-value is one for two identical samples."""
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stat, pvalue = metrics.ks_2samp(sample, sample)
        assert stat == pytest.approx(0.0)
        assert pvalue == pytest.approx(1.0)

    def test_returns_plain_floats(self):
        """Test that the KS statistic and p-value are returned as plain Python floats."""
        stat, pvalue = metrics.ks_2samp(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
        assert isinstance(stat, float)
        assert isinstance(pvalue, float)

    def test_well_separated_samples_give_large_statistic_and_small_pvalue(self):
        """Test that well-separated samples yield a large KS statistic and a small p-value."""
        rng = np.random.default_rng(0)
        sample1 = rng.normal(loc=0.0, scale=1.0, size=200)
        sample2 = rng.normal(loc=100.0, scale=1.0, size=200)
        stat, pvalue = metrics.ks_2samp(sample1, sample2)
        assert stat == pytest.approx(1.0)
        assert pvalue < 0.01


class TestFrechetDistance:
    """
    Tests for the frechet_distance function, which computes the Fréchet distance between two multivariate distributions.
    """

    def test_zero_for_identical_feature_matrices(self):
        """Test that the Fréchet distance is zero for two identical feature matrices."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 3))
        assert metrics.frechet_distance(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_matches_squared_mean_shift_for_equal_covariance_1d(self):
        """Test that the Fréchet distance matches the squared mean shift for 1-D distributions with equal covariance."""
        # x and y=x+delta share identical covariance exactly (same samples, shifted), so the trace term vanishes
        # and the Frechet distance reduces to the squared mean shift, delta**2.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]).reshape(-1, 1)
        delta = 2.5
        y = x + delta
        assert metrics.frechet_distance(x, y) == pytest.approx(delta**2, rel=1e-6)

    def test_symmetric(self):
        """Test that the Fréchet distance is symmetric with respect to its inputs."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=(30, 2))
        y = rng.normal(loc=1.0, size=(30, 2))
        assert metrics.frechet_distance(x, y) == pytest.approx(metrics.frechet_distance(y, x), rel=1e-6)

    def test_larger_for_more_separated_distributions(self):
        """Test that the Fréchet distance is larger for more separated distributions."""
        rng = np.random.default_rng(2)
        x = rng.normal(loc=0.0, scale=1.0, size=(100, 2))
        y_close = rng.normal(loc=0.1, scale=1.0, size=(100, 2))
        y_far = rng.normal(loc=10.0, scale=1.0, size=(100, 2))
        assert metrics.frechet_distance(x, y_far) > metrics.frechet_distance(x, y_close)


class TestKernelDistance:
    """
    Tests for the kernel_distance function, which computes the polynomial-kernel MMD (KID) between two distributions.
    """

    def test_symmetric(self):
        """Test that the kernel distance is symmetric with respect to its inputs."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 2))
        y = rng.normal(loc=1.0, size=(20, 2))
        assert metrics.kernel_distance(x, y) == pytest.approx(metrics.kernel_distance(y, x), rel=1e-6)

    def test_near_zero_for_samples_from_the_same_distribution(self):
        """Test that the kernel distance is near zero for samples drawn from the same distribution."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 2))
        y = rng.normal(size=(300, 2))
        assert abs(metrics.kernel_distance(x, y)) < 0.1

    def test_larger_for_well_separated_distributions(self):
        """Test that the kernel distance is larger for well-separated distributions."""
        rng = np.random.default_rng(3)
        x = rng.normal(loc=0.0, size=(100, 2))
        y_close = rng.normal(loc=0.0, size=(100, 2))
        y_far = rng.normal(loc=20.0, size=(100, 2))
        assert metrics.kernel_distance(x, y_far) > metrics.kernel_distance(x, y_close)


class TestStandardise:
    """Tests for the standardise function, which standardises samples to zero mean and unit variance."""

    def test_reference_is_standardised_to_zero_mean_unit_std(self):
        """Test that the reference sample is standardised to have zero mean and unit standard deviation."""
        rng = np.random.default_rng(0)
        reference = rng.normal(loc=5.0, scale=2.0, size=(500, 3))
        std_ref, = metrics.standardise(reference)
        np.testing.assert_allclose(std_ref.mean(0), 0.0, atol=1e-9)
        np.testing.assert_allclose(std_ref.std(0), 1.0, atol=1e-9)

    def test_others_use_the_reference_scaler_not_their_own(self):
        """Test that other samples are standardised using the reference sample's mean and std, not their own."""
        reference = np.array([[0.0], [10.0]])  # mean=5, std=5
        other = np.array([[5.0], [15.0]])
        std_ref, std_other = metrics.standardise(reference, other)
        np.testing.assert_allclose(std_ref, [[-1.0], [1.0]])
        # other standardised with reference's mu=5, sigma=5 -> (5-5)/5=0, (15-5)/5=2
        np.testing.assert_allclose(std_other, [[0.0], [2.0]])

    def test_constant_reference_column_does_not_divide_by_zero(self):
        """
        Test that a constant column in the reference sample does not lead to division by zero during standardisation.
        """
        reference = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])  # first column constant
        std_ref, = metrics.standardise(reference)
        assert np.all(np.isfinite(std_ref))
        np.testing.assert_allclose(std_ref[:, 0], 0.0)  # (3-3)/1.0 (sigma guarded to 1.0), not (3-3)/0
