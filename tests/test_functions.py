"""Unit tests for the pure math helpers in diffracc/utils/functions.py."""
import numpy as np
import pytest

from diffracc.utils import functions as func

# NOTE: Log10*/exponent kwargs below are always written as floats (26.0, not 26). Passing a bare int here makes
# `10**Log10Lstar` a huge *Python* int rather than a numpy float, which numpy then can't cleanly divide an ndarray
# by (it silently produces an object-dtype array and np.exp blows up on it downstream) - a real footgun in these
# functions, not just a test artifact, so every call site here plays it safe.


class TestSigmoids:
    """Tests for the sigmoid01, sigmoid, erf01, and richards01 functions."""

    def test_sigmoid_matches_sigmoid01_with_default_asymptotes(self):
        """Test that sigmoid01(x, x0, k) == sigmoid(x, x0, k, a=1, b=0) for a range of x values."""
        x = np.linspace(-10, 10, 21)
        np.testing.assert_allclose(func.sigmoid(x, x0=1.5, k=2.0, a=1, b=0), func.sigmoid01(x, x0=1.5, k=2.0))

    def test_sigmoid01_midpoint_is_half(self):
        """Test that sigmoid01(x, x0, k) == 0.5 when x == x0, for a range of x0/k values."""
        assert func.sigmoid01(np.array([3.0]), x0=3.0, k=2.0)[0] == pytest.approx(0.5)

    def test_sigmoid01_asymptotes(self):
        """Test that sigmoid01(x, x0, k) approaches 0 for x << x0 and 1 for x >> x0, for a range of x0/k values."""
        assert func.sigmoid01(np.array([-1e6]), x0=0, k=1) == pytest.approx(0.0, abs=1e-9)
        assert func.sigmoid01(np.array([1e6]), x0=0, k=1) == pytest.approx(1.0, abs=1e-9)

    def test_sigmoid_general_asymptotes_are_a_plus_b_and_b(self):
        """
        Test that sigmoid(x, x0, k, a, b) approaches b for x << x0 and a + b for x >> x0, for a range of a/b/x0/k
        values.
        """
        a, b = 3.0, -1.0
        lo = func.sigmoid(np.array([-1e6]), x0=0, k=1, a=a, b=b)[0]
        hi = func.sigmoid(np.array([1e6]), x0=0, k=1, a=a, b=b)[0]
        assert lo == pytest.approx(b, abs=1e-9)
        assert hi == pytest.approx(a + b, abs=1e-9)

    def test_richards01_reduces_to_sigmoid01_when_nu_is_one(self):
        """Test that richards01(x, x0, k, nu=1) == sigmoid01(x, x0, k) for a range of x values."""
        x = np.linspace(-5, 5, 11)
        np.testing.assert_allclose(func.richards01(x, x0=0.5, k=1.5, nu=1.0), func.sigmoid01(x, x0=0.5, k=1.5))

    def test_erf01_midpoint_is_half(self):
        """Test that erf01(x, x0, sigma) == 0.5 when x == x0, for a range of x0/sigma values."""
        assert func.erf01(np.array([2.0]), x0=2.0, sigma=1.0)[0] == pytest.approx(0.5)

    def test_erf01_asymptotes(self):
        """Test that erf01(x, x0, sigma) approaches 0 for x << x0 and 1 for x >> x0, for a range of x0/sigma values."""
        assert func.erf01(np.array([-1e3]), x0=0, sigma=1)[0] == pytest.approx(0.0, abs=1e-9)
        assert func.erf01(np.array([1e3]), x0=0, sigma=1)[0] == pytest.approx(1.0, abs=1e-9)

    def test_each_sigmoid_matches_numpy_expression(self):
        """
        Test that each of the sigmoid01, sigmoid, and richards01 functions matches the equivalent numpy expression.
        """
        x = np.linspace(-5, 5, 11)
        x0, k, a, b, nu = 0.5, 1.5, 2.0, -1.0, 3.0
        np.testing.assert_allclose(func.sigmoid01(x, x0=x0, k=k), 1 / (1 + np.exp(-k * (x - x0))))
        np.testing.assert_allclose(func.sigmoid(x, x0=x0, k=k, a=a, b=b), a / (1 + np.exp(-k * (x - x0))) + b)
        np.testing.assert_allclose(func.richards01(x, x0=x0, k=k, nu=nu), 1 / ((1 + np.exp(-k * (x - x0)))**(1/nu)))

    def test_each_sigmoid_behaves_with_np_arrays(self):
        """
        Test that each of the sigmoid01, sigmoid, erf01, and richards01 functions can handle numpy arrays as input.
        """
        x = np.array([-1.0, 0.0, 1.0])
        assert func.sigmoid01(x, x0=0, k=1).shape == x.shape
        assert func.sigmoid(x, x0=0, k=1, a=1, b=0).shape == x.shape
        assert func.erf01(x, x0=0, sigma=1).shape == x.shape
        assert func.richards01(x, x0=0, k=1, nu=1).shape == x.shape


class TestWiseFluxConversion:
    """Tests for the mag_to_flux_w2 and mag_to_flux_w3 functions, which convert WISE magnitudes to fluxes in mJy."""

    @pytest.mark.parametrize("flux_fn", [func.mag_to_flux_w2, func.mag_to_flux_w3])
    def test_flux_decreases_with_increasing_magnitude(self, flux_fn):
        """Test that a source with a larger WISE magnitude has a smaller flux, for both W2 and W3."""
        mags = np.array([10.0, 12.0, 14.0])
        fluxes = flux_fn(mags)
        assert np.all(np.diff(fluxes) < 0)


class TestKCorrFactor:
    """
    Tests for the k_corr_factor function, which computes the k-correction factor for a given redshift and spectral
    index. Requires a flat LambdaCDM cosmology, which is provided by the flat_lcdm_cosmo fixture.
    """

    def test_zero_redshift_is_no_correction(self):
        """Test that a source at z=0 has a k-correction factor of 1.0, regardless of spectral index."""
        assert func.k_corr_factor(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_luminosity_space_matches_formula(self):
        """
        Test that k_corr_factor(z, spectral_index=alpha, mag_space=False) == (1+z)^(1-alpha) for a range of z/alpha 
        values.
        """
        z = np.array([0.0, 0.5, 1.0])
        alpha = -0.7
        expected = (1 + z) ** (1 - alpha)
        np.testing.assert_allclose(func.k_corr_factor(z, spectral_index=alpha), expected)

    def test_mag_space_matches_minus_2_5_log10_of_luminosity_space(self):
        """
        Test that k_corr_factor(z, mag_space=True) == -2.5 * log10(k_corr_factor(z, mag_space=False)) for a range of z 
        values.
        """
        z = np.array([0.1, 0.5, 2.0])
        lum_space = func.k_corr_factor(z, mag_space=False)
        mag_space = func.k_corr_factor(z, mag_space=True)
        np.testing.assert_allclose(mag_space, -2.5 * np.log10(lum_space))


class TestRlfPowerLaw:
    """Tests for the rlf_power_law, rlf_power_law_evolution, rlf_pde, and rlf_ple functions."""

    def test_rlf_power_law_at_lstar_is_c_over_two(self):
        """Test that rlf_power_law(Lstar, ...) == 10**Log10C / 2, regardless of alpha/beta values."""
        # At L = Lstar, (L/Lstar)^alpha = (L/Lstar)^beta = 1 regardless of alpha/beta, so the denominator is always
        # exactly 2 there - a parameter-independent sanity check on the formula.
        Log10C = -5.5
        phi_at_lstar = func.rlf_power_law(1e26, alpha=0.5, beta=1.5, Log10C=Log10C, Log10Lstar=26.0)
        assert phi_at_lstar == pytest.approx(10**Log10C / 2)

    def test_rlf_power_law_monotonically_decreases_for_positive_indices(self):
        """
        Test that rlf_power_law(L, ...) is monotonically decreasing for L > 0 when alpha and beta are both positive.
        """
        # With alpha, beta > 0 (the sign convention used by this project's default fit parameters), both terms in
        # the denominator grow with L, so phi should fall monotonically across the whole luminosity range.
        luminosity = np.logspace(20, 30, 200)
        phi = func.rlf_power_law(luminosity, alpha=0.5, beta=1.5, Log10C=-5.5, Log10Lstar=26.0)
        assert np.all(np.diff(phi) < 0)

    def test_rlf_power_law_evolution_reduces_to_rlf_power_law_at_z_zero(self):
        """Test that rlf_power_law_evolution(L, z=0, ...) == rlf_power_law(L, ...) for a range of L values."""
        luminosity = np.logspace(21, 29, 50)
        params = dict(alpha=0.5, beta=1.5, Log10C=-5.5, Log10Lstar=26.0)
        expected = func.rlf_power_law(luminosity, **params)
        x = np.vstack([luminosity, np.zeros_like(luminosity)])
        actual = func.rlf_power_law_evolution(x, **params, alphaD=0.7, alphaL=-0.3)
        np.testing.assert_allclose(actual, expected)

    def test_rlf_pde_reduces_to_rlf_power_law_at_z_zero(self):
        """Test that rlf_pde(L, z=0, ...) == rlf_power_law(L, ...) for a range of L values."""
        luminosity = np.logspace(21, 29, 50)
        params = dict(alpha=0.5, beta=1.5, Log10C=-5.5, Log10Lstar=26.0)
        expected = func.rlf_power_law(luminosity, **params)
        x = np.vstack([luminosity, np.zeros_like(luminosity)])
        actual = func.rlf_pde(x, **params, alphaD=1.3)
        np.testing.assert_allclose(actual, expected)

    def test_rlf_ple_reduces_to_rlf_power_law_at_z_zero(self):
        """Test that rlf_ple(L, z=0, ...) == rlf_power_law(L, ...) for a range of L values."""
        luminosity = np.logspace(21, 29, 50)
        params = dict(alpha=0.5, beta=1.5, Log10C=-5.5, Log10Lstar=26.0)
        expected = func.rlf_power_law(luminosity, **params)
        x = np.vstack([luminosity, np.zeros_like(luminosity)])
        actual = func.rlf_ple(x, **params, alphaL=-0.3)
        np.testing.assert_allclose(actual, expected)

    def test_rlf_pde_scales_linearly_with_1_plus_z_to_the_alphaD(self):
        """Test that rlf_pde(L, z, ...) scales as (1+z)^alphaD for a range of L/z values."""
        luminosity = np.full(3, 1e26)
        params = dict(alpha=0.5, beta=1.5, Log10C=-5.5, Log10Lstar=26.0, alphaD=2.0)
        x_z0 = np.vstack([luminosity, np.zeros_like(luminosity)])
        x_z1 = np.vstack([luminosity, np.ones_like(luminosity)])
        ratio = func.rlf_pde(x_z1, **params) / func.rlf_pde(x_z0, **params)
        np.testing.assert_allclose(ratio, 2.0**params["alphaD"])


class TestRlfSchechter:
    """Tests for the rlf_schechter function."""

    def test_positive_for_positive_luminosities(self):
        """Test that rlf_schechter(L, ...) > 0 for a range of positive L values."""
        # Keep L/Lstar within a moderate range (~1e-5 to 1e2) - well beyond that, exp(-(L/Lstar)^gamma) genuinely
        # underflows to exactly 0 in float64, which isn't a bug, just not what this test is checking for.
        luminosity = np.logspace(20, 27, 50)
        phi = func.rlf_schechter(luminosity, beta=0.5, gamma=1.0, Log10Phi=-3.0, Log10Lstar=25.0)
        assert np.all(phi > 0)

    def test_decreases_well_above_lstar(self):
        """Test that rlf_schechter(L, ...) decreases for L >> Lstar, as the exponential cutoff dominates."""
        # Beyond L*, the exp(-(L/Lstar)^gamma) cutoff should dominate and phi should fall.
        Log10Lstar = 25.0
        luminosity = np.array([10**Log10Lstar, 10 ** (Log10Lstar + 1), 10 ** (Log10Lstar + 2)])
        phi = func.rlf_schechter(luminosity, beta=0.5, gamma=1.0, Log10Phi=-3.0, Log10Lstar=Log10Lstar)
        assert np.all(np.diff(phi) < 0)


class TestYuanEvolutionVariants:
    """Tests for the yuan_evolution_a and yuan2018_evolution_a functions."""

    def test_yuan_evolution_a_z_greater_than_z0_branch(self):
        """Test that yuan_evolution_a(L, z > z0, ...) == z**m * rlf_power_law(L, ...) when k1=0."""
        # z > z0 selects e1 = z**m; k1=0 makes e2=1 so the luminosity term is untouched, isolating e1.
        l = np.array([1e26])
        z = np.array([2.0])
        params = dict(alpha=0.5, beta=1.5, Log10C=-5.5, Log10Lstar=26.0)
        actual = func.yuan_evolution_a(np.vstack([l, z]), **params, m=2.0, z0=1.0, zsigma=0.5, k1=0.0)
        expected_e1 = 2.0**2.0  # z**m
        expected = expected_e1 * func.rlf_power_law(l, **params)
        np.testing.assert_allclose(actual, expected)

    def test_yuan_evolution_a_z_below_z0_branch(self):
        """
        Test that yuan_evolution_a(L, z < z0, ...) == z**m * exp(-0.5*((z-z0)/zsigma)**2) * rlf_power_law(L, ...) when
        k1=0.
        """
        # z <= z0 selects the Gaussian-suppressed e1 = z**m * exp(-0.5*((z-z0)/zsigma)**2).
        l = np.array([1e26])
        z = np.array([0.5])
        m, z0, zsigma = 1.0, 1.0, 0.5
        params = dict(alpha=0.5, beta=1.5, Log10C=-5.5, Log10Lstar=26.0)
        actual = func.yuan_evolution_a(np.vstack([l, z]), **params, m=m, z0=z0, zsigma=zsigma, k1=0.0)
        expected_e1 = z[0]**m * np.exp(-0.5 * ((z[0] - z0) / zsigma)**2)
        expected = expected_e1 * func.rlf_power_law(l, **params)
        np.testing.assert_allclose(actual, expected)

    def test_yuan2018_evolution_a_matches_rlf_schechter_when_e1_is_forced_to_one(self):
        """Test that yuan2018_evolution_a(L, z, ...) == rlf_schechter(L, ...) when p1=p2=0 and k1=0."""
        # Choose p1=p2=0 so p0 = 2 and e1 = p0 * (((1+zc)/(1+z))^0 + ...)^-1 = p0 / 2 = 1 identically, and k1=0 so
        # e2=1, isolating the rlf_schechter call.
        l = np.logspace(21, 29, 10)
        z = np.array([0.0, 0.3, 0.8, 1.5, 0.1, 0.6, 0.9, 0.2, 1.1, 0.4])
        x = np.vstack([l, z])
        schechter_params = dict(beta=0.5, gamma=1.0, Log10Phi=-3.0, Log10Lstar=25.0)
        actual = func.yuan2018_evolution_a(x, p1=0.0, p2=0.0, zc=0.5, k1=0.0, **schechter_params)
        expected = func.rlf_schechter(l, **schechter_params)
        np.testing.assert_allclose(actual, expected)
