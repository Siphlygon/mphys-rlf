"""
Unit tests for diffracc/completeness/completeness_estimator.py's CompletenessEstimator.

Built against the completeness_estimator_factory fixture (tests/conftest.py), which constructs instances with
override_data=True (skipping ImageDataArrays) and a fake RMSDistribution (skipping the real Hardcastle catalogue),
so nothing here touches real datasets, FITS files, or catalogues - only tiny synthetic arrays/DataFrames.
"""
import matplotlib

matplotlib.use("Agg")  # non-interactive backend before pyplot is used - see test_agn_selection.py for why.

import numpy as np
import pandas as pd
import pytest
import scipy.signal

from diffracc.utils.functions import erf01, richards01, sigmoid, sigmoid01


class TestCreateBeamCorrNoise:
    """
    Tests that the _create_beam_corr_noise method produces an output of the requested shape and matches a manual
    implementation.
    """

    def test_output_shape_matches_requested_shape(self, completeness_estimator_factory):
        """Test that the output shape of _create_beam_corr_noise matches the requested shape."""
        ce = completeness_estimator_factory()
        kernel = np.ones((1, 3, 3)) / 9
        out = ce._create_beam_corr_noise(kernel, rms=0.05, shape=(2, 10, 10))
        assert out.shape == (2, 10, 10)

    def test_matches_manual_composition_with_same_random_seed(self, completeness_estimator_factory):
        """
        Test that _create_beam_corr_noise matches a manual composition of np.random.normal and scipy.signal.fftconvolve
        with the same random seed.
        """
        # _create_beam_corr_noise is just np.random.normal + scipy.signal.fftconvolve - reproduce both calls
        # independently with the same seed and compare, which tests the composition/argument-passing rather than
        # re-deriving the maths of either building block.
        ce = completeness_estimator_factory()
        filter_kernel = np.array([[[1.0]]])  # a true 1x1 identity kernel
        shape = (3, 4, 4)
        rms = 0.1

        np.random.seed(42)
        actual = ce._create_beam_corr_noise(filter_kernel, rms=rms, shape=shape)

        np.random.seed(42)
        expected_raw = np.random.normal(loc=0.0, scale=rms, size=shape)
        expected = scipy.signal.fftconvolve(expected_raw, filter_kernel, mode='same')

        np.testing.assert_allclose(actual, expected)


class TestComputeCompletenessPerBin:
    """
    Tests that the _compute_completeness_per_bin method correctly computes completeness and error bars per flux bin.
    """

    def test_completeness_matches_detected_fraction_per_bin(self, completeness_estimator_factory):
        """Test that the completeness per bin matches the fraction of detectable sources in that bin."""
        ce = completeness_estimator_factory()
        int_flux_bins = np.array([0.0, 1.0, 2.0, 3.0])
        mock_sources = pd.DataFrame({
            'mock_flux':  [0.5, 0.5, 0.5, 1.5, 1.5, 2.5],
            'detectable': [True, True, False, True, False, True],
        })

        completeness, yerr = ce._compute_completeness_per_bin(int_flux_bins, mock_sources, show_progress=False)

        np.testing.assert_allclose(completeness, [2 / 3, 1 / 2, 1.0])
        assert yerr.shape == (3,)
        assert np.all(np.isfinite(yerr))
        assert np.all(yerr >= 0)

    def test_empty_bin_gives_zero_completeness_and_zero_error(self, completeness_estimator_factory):
        """Test that an empty flux bin (no mock sources) returns a completeness of 0.0 and an error of 0.0."""
        ce = completeness_estimator_factory()
        int_flux_bins = np.array([0.0, 1.0, 2.0])
        mock_sources = pd.DataFrame({'mock_flux': [1.5, 1.5], 'detectable': [True, False]})

        completeness, yerr = ce._compute_completeness_per_bin(int_flux_bins, mock_sources, show_progress=False)

        assert completeness[0] == 0.0
        assert yerr[0] == 0.0

    def test_fully_detected_bin_gives_completeness_one(self, completeness_estimator_factory):
        """Test that a bin with all detectable sources returns a completeness of 1.0."""
        ce = completeness_estimator_factory()
        int_flux_bins = np.array([0.0, 1.0])
        mock_sources = pd.DataFrame({'mock_flux': [0.5, 0.5, 0.5], 'detectable': [True, True, True]})

        completeness, yerr = ce._compute_completeness_per_bin(int_flux_bins, mock_sources, show_progress=False)

        assert completeness[0] == 1.0
        assert np.isfinite(yerr[0])


class TestFitFunction:
    """
    Tests that the _fit_function method correctly fits a sigmoid function to completeness data and handles edge cases.
    """
    TRUE_PARAMS = [0.0, 5.0, 1.0, 0.0]  # x0, k, a, b for utils.functions.sigmoid

    def _synthetic_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A helper method to generate a synthetic completeness curve using the sigmoid function with known parameters.
        
        Returns
        -------
        bin_centers : np.ndarray
            The centers of the flux bins.
        completeness : np.ndarray
            The synthetic completeness values generated from the sigmoid function.
        yerr : np.ndarray
            The synthetic error values for the completeness data.
        """
        bin_centers = np.linspace(-1, 1, 15)
        completeness = sigmoid(bin_centers, *self.TRUE_PARAMS)
        yerr = np.full_like(completeness, 0.02)
        return bin_centers, completeness, yerr

    def test_recovers_known_sigmoid_parameters_without_explicit_initial_guess(self, completeness_estimator_factory):
        """Test that _fit_function recovers known sigmoid parameters without providing an explicit initial guess."""
        ce = completeness_estimator_factory()
        bin_centers, completeness, yerr = self._synthetic_curve()

        popt, pcov = ce._fit_function(bin_centers, completeness, yerr)

        assert popt.shape == (4,)
        np.testing.assert_allclose(popt, self.TRUE_PARAMS, atol=0.1)

    def test_drops_non_finite_and_zero_sigma_points_before_fitting(self, completeness_estimator_factory):
        """Test that _fit_function drops non-finite completeness points and zero-sigma points before fitting."""
        ce = completeness_estimator_factory()
        bin_centers, completeness, yerr = self._synthetic_curve()
        completeness = completeness.copy()
        yerr = yerr.copy()
        completeness[3] = np.nan  # non-finite point
        yerr[7] = 0.0  # zero-sigma point

        popt, pcov = ce._fit_function(bin_centers, completeness, yerr)

        assert popt.shape == (4,)
        np.testing.assert_allclose(popt, self.TRUE_PARAMS, atol=0.2)

    def test_shape_mismatch_without_explicit_initial_guess_raises_indexerror(self, completeness_estimator_factory):
        """
        Test that a shape mismatch between bin_centers and completeness without an explicit initial guess raises
        IndexError.
        """
        # Another rough edge in the error handling: the "bin_centers/completeness shape mismatch" ValueError check
        # lives inside the try/except, but the default-initial-guess computation runs *before* that try block and
        # itself indexes bin_centers using an index derived from completeness's shape - so a shape mismatch never
        # reaches the intended graceful handling, it raises IndexError first. Documenting actual behavior.
        ce = completeness_estimator_factory()
        with pytest.raises(IndexError):
            ce._fit_function(np.array([1.0, 2.0]), np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.1, 0.1]))

    def test_shape_mismatch_with_explicit_initial_guess_is_caught_and_returns_empty_arrays(
        self,
        completeness_estimator_factory):
        """
        Test that a shape mismatch between bin_centers and completeness with an explicit initial guess is caught and
        returns empty arrays.
        """
        # Providing initial_guess explicitly skips the crashing auto-guess block above, so the shape-mismatch
        # ValueError inside the try/except is reached and handled as intended.
        ce = completeness_estimator_factory()
        popt, pcov = ce._fit_function(np.array([1.0, 2.0]), np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.1, 0.1]),
                                      initial_guess=[0.0, 1.0, 1.0, 0.0])
        assert popt.size == 0
        assert pcov.size == 0

    def test_non_callable_function_argument_raises_in_the_error_handler(self, completeness_estimator_factory):
        """Test that providing a non-callable function argument raises an AttributeError in the error handler."""
        # Not a clean failure mode: the try/except wraps a `raise TypeError` when `function` isn't callable, but
        # the except block itself does `function.__name__` to log the error - which raises AttributeError for a
        # non-callable `function` (e.g. a plain string) instead of the intended graceful "return empty arrays"
        # fallback. Documenting the actual current behavior here rather than silently assuming it's handled.
        ce = completeness_estimator_factory()
        with pytest.raises(AttributeError):
            ce._fit_function(np.array([0.0]), np.array([0.5]), np.array([0.1]), function="not callable")

    def test_saves_popt_and_pcov_to_output_file(self, completeness_estimator_factory, tmp_path):
        """Test that _fit_function saves the fitted parameters and covariance matrix to an output file if specified."""
        ce = completeness_estimator_factory()
        bin_centers, completeness, yerr = self._synthetic_curve()
        out_path = tmp_path / "fit_params.txt"

        ce._fit_function(bin_centers, completeness, yerr, output_file=out_path)

        assert out_path.exists()
        content = out_path.read_text()
        assert "Fitted parameters" in content
        assert "Covariance matrix" in content

    def test_empty_data_falls_back_to_median_x0_guess_and_is_caught(self, completeness_estimator_factory):
        """
        Test that providing empty data arrays triggers the np.median(bin_centers) x0_guess fallback and is caught,
        returning empty arrays rather than raising an exception.
        """
        # completeness.size == 0 triggers the np.median(bin_centers) x0_guess fallback (rather than indexing into
        # an empty completeness array), and then curve_fit itself fails on zero data points - caught by the outer
        # except, returning empty arrays rather than raising.
        ce = completeness_estimator_factory()
        popt, pcov = ce._fit_function(np.array([]), np.array([]), np.array([]))
        assert popt.size == 0
        assert pcov.size == 0

    def test_uses_scipy_default_p0_when_arity_is_not_2_3_or_4(self, completeness_estimator_factory):
        """Test that _fit_function uses scipy's default p0 when the function's arity is not 2, 3, or 4."""
        # A 1-parameter model isn't handled by the n_params in {2,3,4} initial-guess heuristics, so initial_guess
        # stays None through that block and curve_fit falls back to its own default p0.
        ce = completeness_estimator_factory()

        def constant(x, a):
            return np.full_like(x, a)

        bin_centers = np.linspace(-1, 1, 10)
        true_a = 0.7
        completeness = constant(bin_centers, true_a)
        yerr = np.full_like(completeness, 0.01)

        popt, pcov = ce._fit_function(bin_centers, completeness, yerr, function=constant)

        assert popt.shape == (1,)
        assert popt[0] == pytest.approx(true_a, abs=0.05)

    def test_yerr_shape_mismatch_is_caught_and_returns_empty_arrays(self, completeness_estimator_factory):
        """Test that a shape mismatch between yerr and completeness is caught and returns empty arrays."""
        # Unlike the bin_centers/completeness mismatch, this one doesn't touch the auto-guess block (which never
        # indexes yerr), so it's reachable without an explicit initial_guess.
        ce = completeness_estimator_factory()
        bin_centers, completeness, _ = self._synthetic_curve()
        popt, pcov = ce._fit_function(bin_centers, completeness, np.array([0.1, 0.1]))
        assert popt.size == 0
        assert pcov.size == 0

    @pytest.mark.parametrize("function,true_params", [
        (richards01, [0.0, 5.0, 1.0]),          # 3-parameter function
        (erf01, [0.0, 0.2]),                    # 2-parameter, second param name "sigma" -> width_guess branch
        (sigmoid01, [0.0, 5.0]),                # 2-parameter, second param name "k" -> k_guess branch
    ])
    def test_recovers_known_parameters_for_other_arities(self, completeness_estimator_factory, function, true_params):
        """Test that _fit_function recovers known parameters for other function arities (2 or 3)."""
        ce = completeness_estimator_factory()
        bin_centers = np.linspace(-1, 1, 15)
        completeness = function(bin_centers, *true_params)
        yerr = np.full_like(completeness, 0.02)

        popt, pcov = ce._fit_function(bin_centers, completeness, yerr, function=function)

        assert popt.shape == (len(true_params),)
        np.testing.assert_allclose(popt, true_params, atol=0.15)


class TestPlotCompleteness:
    """Smoke test for plot_completeness - forced onto the Agg backend (see module-level matplotlib.use above)."""

    def test_runs_without_error_and_saves_a_figure(self, completeness_estimator_factory, monkeypatch, tmp_path):
        """Test that plot_completeness runs without error and saves a figure to the specified path."""
        import matplotlib.pyplot as plt

        monkeypatch.chdir(tmp_path)
        ce = completeness_estimator_factory()
        true_params = [0.0, 5.0, 1.0, 0.0]
        bin_centers = np.linspace(-1, 1, 10)
        completeness = sigmoid(bin_centers, *true_params)
        yerr = np.full_like(completeness, 0.05)

        try:
            ce.plot_completeness(bin_centers, completeness, yerr, popt=true_params, save_name="completeness.png")
            assert (tmp_path / "completeness.png").exists()
        finally:
            plt.close("all")

    def test_raises_assertion_error_without_a_fitted_popt(self, completeness_estimator_factory):
        """Test that plot_completeness raises an AssertionError if no fitted popt is provided."""
        ce = completeness_estimator_factory()
        bin_centers = np.linspace(-1, 1, 10)
        completeness = np.linspace(0, 1, 10)
        yerr = np.full_like(completeness, 0.05)

        with pytest.raises(AssertionError):
            ce.plot_completeness(bin_centers, completeness, yerr, popt=None)


class TestDetectMockSources:
    """Tests that the _detect_mock_sources method correctly identifies detectable sources and writes the output file."""

    def test_bright_source_is_always_detected_and_faint_source_never_is(self, completeness_estimator_factory,
                                                                         monkeypatch, tmp_path):
        """
        Test that a bright source is always detected and a faint source is never detected, regardless of the random
        noise draw, and that the output shapes are correct.
        """
        # A background of +/-1000 dominates any plausible noise realization (rms ~ 0.095, threshold ~ 0.475), so
        # detectability is deterministic here regardless of the random noise draw - no seeding needed.
        monkeypatch.chdir(tmp_path)
        ce = completeness_estimator_factory()  # N_NOISE_PATCHES=3, DETECTION_SIGMA_THRESHOLD=5, fixed rms=95e-3
        images = np.stack([
            np.full((80, 80), 1000.0),
            np.full((80, 80), -1000.0),
        ])
        model_fluxes = np.array([5.0, 2.0])

        mock_fluxes, detectable = ce._detect_mock_sources(images, model_fluxes, show_progress=False)

        n = ce.num_noise_patches
        assert mock_fluxes.shape == (2 * n,)
        assert detectable.shape == (2 * n,)
        np.testing.assert_allclose(mock_fluxes[:n], 5.0)
        np.testing.assert_allclose(mock_fluxes[n:], 2.0)
        assert np.all(detectable[:n])
        assert not np.any(detectable[n:])

    def test_writes_mock_fluxes_detectability_file_to_cwd(self, completeness_estimator_factory, monkeypatch, tmp_path):
        """
        Test that _detect_mock_sources writes the mock fluxes and detectability to a file in the current working
        directory.
        """
        monkeypatch.chdir(tmp_path)
        ce = completeness_estimator_factory()
        images = np.full((1, 80, 80), 1000.0)
        model_fluxes = np.array([3.0])

        ce._detect_mock_sources(images, model_fluxes, show_progress=False)

        out_file = tmp_path / "mock_fluxes_detectability.txt"
        assert out_file.exists()
        lines = out_file.read_text().splitlines()
        assert lines[0] == "Mock_Flux(mJy/beam)\tDetectable"
        assert len(lines) == 1 + ce.num_noise_patches

    def test_squeezes_leading_singleton_dimension_from_images(self, completeness_estimator_factory,
                                                               monkeypatch, tmp_path):
        """Test that _detect_mock_sources squeezes a leading singleton dimension from images before processing."""
        # images[i] can come through as shape (1, 80, 80) from some FITS readers - _detect_mock_sources should
        # squeeze that down to 2D rather than raising ValueError.
        monkeypatch.chdir(tmp_path)
        ce = completeness_estimator_factory()
        images = np.full((1, 1, 80, 80), 1000.0)
        model_fluxes = np.array([3.0])

        mock_fluxes, detectable = ce._detect_mock_sources(images, model_fluxes, show_progress=False)

        assert np.all(detectable)
