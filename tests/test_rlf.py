"""
Unit tests for diffracc/rlf/rlf.py's RLF class.

These use the rlf_factory fixture (tests/conftest.py), which builds RLF instances against a small temp config.ini and
completeness-args file rather than the real ones - real N_MC_PTS (100000) and LUM_BINS (16) would make this suite far
too slow, per rlf_config_path's small N_MC_PTS/LUM_BINS.
"""
import numpy as np
import pytest

from diffracc.rlf.rlf_constants import shimwell_data
from diffracc.utils import functions as func

EMPTY = np.array([])
EMPTY_BOOL = np.array([], dtype=bool)


class TestConstructorBinning:
    """
    Tests that the RLF constructor correctly derives z/l bins from config.ini and that it raises when the completeness
    file is missing.
    """

    def test_bins_derived_from_config(self, rlf_factory):
        """
        Test that the z/l bins are derived from the config.ini fixture, and that the phi array has the expected shape.
        """
        rlf = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL)
        # rlf_config_path fixture: Z_MIN=0.01, Z_MAX=1.0, dz=0.5, HARDCASTLE_Z_BINS=False, DEJONG_Z_BINS=False
        np.testing.assert_allclose(rlf.z_bins, np.arange(0.01, 1.0, 0.5))
        assert rlf.n_z_bins == rlf.z_bins.shape[0] - 1
        assert rlf.l_bins.shape[0] == rlf.lum_bins_count
        assert rlf.n_lum_bins == rlf.lum_bins_count - 1
        assert rlf.phi.shape == (rlf.n_z_bins, rlf.n_lum_bins)

    def test_missing_completeness_file_raises(self, rlf_factory, tmp_path):
        """Test that the constructor raises a FileNotFoundError when the completeness file is missing."""
        with pytest.raises(FileNotFoundError):
            rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, completeness_path=tmp_path / "does_not_exist.txt")


class TestGetCompleteness:
    """
    Tests that the _get_completeness method returns the expected values for resolved and unresolved sources, and that
    the bias parameter shifts the resolved completeness curve as expected.
    """
    
    def test_resolved_source_below_flux_cut_is_zero(self, rlf_factory):
        """Test that a resolved source below the flux cut returns a completeness of 0.0."""
        rlf = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, flux_cut_jy=1.1e-3)
        completeness = rlf._get_completeness(np.array([1e-4]), resolved=np.array([True]))
        assert completeness[0] == 0.0

    def test_resolved_source_above_flux_cut_matches_sigmoid(self, rlf_factory):
        """Test that a resolved source above the flux cut returns the completeness fit evaluated at that flux."""
        rlf = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, flux_cut_jy=1.1e-3, bias=0)
        flux = 5e-3  # 5 mJy, above the cut
        completeness = rlf._get_completeness(np.array([flux]), resolved=np.array([True]))
        # The fixture fit is a `sigmoid` fitted against log10(mJy); _get_completeness hands it flux in mJy and the fit
        # applies the log10 itself, so the expected value is the raw function at log10(flux / mJy).
        expected = func.sigmoid(np.log10(flux * 1000), *rlf.completeness_fit.popt)
        assert completeness[0] == pytest.approx(expected)

    def test_bias_shifts_the_resolved_sigmoid_completeness(self, rlf_factory):
        """Test that the bias parameter shifts the resolved completeness curve as expected."""
        flux = 5e-3
        rlf_no_bias = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, bias=0)
        rlf_biased = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, bias=5)
        c0 = rlf_no_bias._get_completeness(np.array([flux]), resolved=np.array([True]))[0]
        c1 = rlf_biased._get_completeness(np.array([flux]), resolved=np.array([True]))[0]
        assert c0 != pytest.approx(c1)

    def test_unresolved_uses_shimwell_completeness_by_default(self, rlf_factory):
        """
        Test that an unresolved source above the flux cut returns a completeness matching the Shimwell et al. (2017)
        completeness curve by default.
        """
        rlf = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, flux_cut_jy=1.1e-3, use_shimwell=True)
        flux = shimwell_data[0, 18] / 1000  # 1.25 mJy, above the 1.1 mJy default flux cut
        completeness = rlf._get_completeness(np.array([flux]), resolved=np.array([False]))
        assert completeness[0] == pytest.approx(shimwell_data[1, 18])

    def test_unresolved_uses_step_function_when_shimwell_disabled(self, rlf_factory):
        """
        Test that an unresolved source above the flux cut returns a completeness of 1.0 when the Shimwell completeness
        curve is disabled.
        """
        rlf = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, flux_cut_jy=1.1e-3, use_shimwell=False)
        below = rlf._get_completeness(np.array([1e-4]), resolved=np.array([False]))[0]
        above = rlf._get_completeness(np.array([5e-3]), resolved=np.array([False]))[0]
        assert below == 0.0
        assert above == 1.0

    def test_biased_call_does_not_mutate_the_completeness_fit(self, rlf_factory):
        """Test that evaluating _get_completeness with a nonzero bias leaves the stored fit parameters untouched."""
        # The bias shifts the curve's midpoint only for the duration of the evaluation (CompletenessFit.evaluate works
        # on a copy of popt), so the fit reused across the millions of Monte Carlo calls must never drift. Guard that
        # the stored popt is byte-for-byte unchanged after a biased call.
        rlf = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL, bias=3)
        original = rlf.completeness_fit.popt.copy()
        rlf._get_completeness(np.array([5e-3]), resolved=np.array([True]))
        np.testing.assert_array_equal(rlf.completeness_fit.popt, original)


class TestCalculateRlfBinning:
    """Tests that the RLF binning methods correctly count sources and produce phi arrays of the expected shape."""
    
    @staticmethod
    def _synthetic_catalogue(rng, n=200):
        """Test helper: generate a synthetic catalogue of fluxes, redshifts, and resolved flags."""
        redshifts = rng.uniform(0.02, 0.49, size=n)
        fluxes = 10 ** rng.uniform(-3, -1, size=n)  # 1-100 mJy, straddling the 1.1 mJy default flux cut
        resolved = rng.random(n) > 0.5
        return fluxes, redshifts, resolved

    def test_vmax_binning_counts_match_source_count(self, rlf_factory):
        """Test that the Vmax binning method counts every source that falls inside the z/l bin ranges exactly once."""
        rng = np.random.default_rng(0)
        fluxes, redshifts, resolved = self._synthetic_catalogue(rng)
        rlf = rlf_factory(fluxes, redshifts, np.zeros_like(fluxes), resolved, vmax_method=True)

        z, lum, res = rlf._preprocess_sources()
        rlf._calculate_rlf_vmax(z, lum, res)

        assert rlf.phi.shape == (rlf.n_z_bins, rlf.n_lum_bins)
        # phi can legitimately be +inf: 1/Vmax blows up when a source's Monte-Carlo-estimated Vmax rounds to
        # exactly 0 (it sits right at the edge of detectability across the whole bin's volume range) - this is a
        # real, RLF.py-acknowledged limitation of the estimator (see the "Monte Carlo failure" log message in
        # _warn_on_zero_bin_integrals), not something a unit test should paper over by only using "safe" data.
        assert not np.any(np.isnan(rlf.phi))
        assert np.all(rlf.phi >= 0)
        assert np.all(rlf.counts >= 0)
        # every source that fell inside the z/l bin ranges should be counted exactly once
        in_range = (z >= rlf.z_bins[0]) & (z < rlf.z_bins[-1]) & (lum >= rlf.l_bins[0]) & (lum < rlf.l_bins[-1])
        assert rlf.counts.sum() == in_range.sum()

    def test_page_carrera_binning_counts_match_source_count(self, rlf_factory):
        """
        Test that the Page-Carrera binning method counts every source that falls inside the z/l bin ranges exactly once.
        """
        rng = np.random.default_rng(1)
        fluxes, redshifts, resolved = self._synthetic_catalogue(rng)
        rlf = rlf_factory(fluxes, redshifts, np.zeros_like(fluxes), resolved, vmax_method=False)

        z, lum, res = rlf._preprocess_sources()
        rlf._calculate_rlf_page_carrera(z, lum, res)

        assert rlf.phi.shape == (rlf.n_z_bins, rlf.n_lum_bins)
        # see the comment in test_vmax_binning_counts_match_source_count - the same edge case applies here.
        assert not np.any(np.isnan(rlf.phi))
        assert np.all(rlf.phi >= 0)
        in_range = (z >= rlf.z_bins[0]) & (z < rlf.z_bins[-1]) & (lum >= rlf.l_bins[0]) & (lum < rlf.l_bins[-1])
        assert rlf.counts.sum() == in_range.sum()

    def test_zero_flux_sources_are_dropped_before_binning(self, rlf_factory):
        """
        Test that sources with zero flux are dropped before binning, since they cannot be detected and would otherwise
        cause a divide-by-zero in the Vmax calculation.
        """
        rng = np.random.default_rng(2)
        fluxes, redshifts, resolved = self._synthetic_catalogue(rng, n=20)
        fluxes[0] = 0.0
        rlf = rlf_factory(fluxes, redshifts, np.zeros_like(fluxes), resolved)
        z, lum, res = rlf._preprocess_sources()
        assert z.shape[0] == 19


class TestFitRlfIndividually:
    """Tests that the RLF fitting method correctly recovers known power-law parameters from synthetic data."""

    def test_recovers_known_power_law_parameters(self, rlf_factory):
        """Test that fit_rlf_individually recovers known power-law parameters from synthetic data."""
        # Bypass calculate_rlf's binning/Monte Carlo entirely: set phi to an exact rlf_power_law curve (with a
        # small fixed relative error) and check curve_fit recovers parameters close to the ones used to generate
        # it. rlf_config_path gives 8 luminosity bins, comfortably more than the model's 4 free parameters.
        rlf = rlf_factory(EMPTY, EMPTY, EMPTY, EMPTY_BOOL)
        true_params = [0.5, 1.5, -5.5, 26.0]
        bin_centres = (rlf.l_bins[:-1] + rlf.l_bins[1:]) / 2
        phi = func.rlf_power_law(bin_centres, *true_params)

        rlf.phi = np.tile(phi, (rlf.n_z_bins, 1))
        rlf.phi_err = rlf.phi * 0.05
        rlf.counts = np.ones_like(rlf.phi)

        rlf.fit_rlf_individually()

        for i_z in range(rlf.n_z_bins):
            fitted_values = rlf.rlf_fit_params[i_z, :, 0]
            np.testing.assert_allclose(fitted_values, true_params, rtol=0.05)
