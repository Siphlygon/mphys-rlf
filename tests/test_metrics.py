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


class TestDensityCoverage:
    """
    Tests for density/coverage (Naeem et al. 2020): density counts real k-NN balls each generated point falls into
    (normalised by k), coverage is the fraction of real samples with a generated neighbour in their own ball.
    """

    def test_identical_sets_give_full_coverage(self):
        """With generated == real, every real ball contains its twin, so coverage = 1 and density > 0."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((300, 4))
        d, c = metrics.density_coverage(x, x.copy(), k=5)
        assert c == pytest.approx(1.0)
        assert d > 0.0

    def test_matched_distributions_density_near_one(self):
        """Two independent draws from the same distribution give density ~ 1 (generated sit where real is dense)."""
        rng = np.random.default_rng(1)
        real = rng.standard_normal((1000, 3))
        gen = rng.standard_normal((1000, 3))
        d, c = metrics.density_coverage(real, gen, k=5)
        assert 0.7 < d < 1.3
        assert c > 0.6

    def test_far_apart_sets_give_zero_density_and_coverage(self):
        """Disjoint, far-separated clouds: no generated point lands in any real ball."""
        rng = np.random.default_rng(2)
        real = rng.standard_normal((300, 4))
        gen = rng.standard_normal((300, 4)) + 100.0
        d, c = metrics.density_coverage(real, gen, k=5)
        assert d == pytest.approx(0.0)
        assert c == pytest.approx(0.0)

    def test_coverage_in_unit_interval_density_nonnegative(self):
        """Coverage is a fraction in [0, 1]; density is non-negative."""
        rng = np.random.default_rng(3)
        real = rng.standard_normal((200, 5))
        gen = rng.standard_normal((200, 5)) * 1.2
        d, c = metrics.density_coverage(real, gen, k=5)
        assert d >= 0.0
        assert 0.0 <= c <= 1.0

    def test_too_few_samples_raises(self):
        """Fewer than k+1 real samples cannot define the k-NN balls."""
        real = np.zeros((4, 2))
        gen = np.zeros((10, 2))
        with pytest.raises(ValueError, match="need at least"):
            metrics.density_coverage(real, gen, k=5)


class TestKnnRadii:
    """Tests for the _knn_radii helper underpinning the manifold metrics."""

    def test_excludes_self_and_returns_kth_distance(self):
        """On collinear points spaced by 1, the 1st-NN distance is 1 for interior points (self is excluded)."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        radii = metrics._knn_radii(x, k=1)
        # nearest non-self neighbour is at distance 1 for all four points
        np.testing.assert_allclose(radii, [1.0, 1.0, 1.0, 1.0])

    def test_second_neighbour(self):
        """k=2 returns the distance to the second-nearest non-self neighbour."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        radii = metrics._knn_radii(x, k=2)
        # for point 0: neighbours at 1, 2, 3 -> 2nd nearest is 2; for point 1: neighbours at 1,1,2 -> 2nd nearest is 1
        assert radii[0] == pytest.approx(2.0)
        assert radii[1] == pytest.approx(1.0)
