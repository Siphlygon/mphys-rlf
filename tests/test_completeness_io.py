"""Unit tests for diffracc/completeness/completeness_io.py."""
import json

import numpy as np
import pytest

from diffracc.completeness import completeness_io as cio
from diffracc.utils import functions as func


class TestCompletenessFitValidation:
    """
    Unit tests for the validation logic in CompletenessFit.__post_init__, which is the only place the function name and
    x_space are checked, and where popt and pcov are converted to float arrays.
    """

    def test_unknown_function_name_raises_value_error(self):
        """Test that an unknown function name is rejected with a ValueError."""
        with pytest.raises(ValueError):
            cio.CompletenessFit(function_name="not_a_real_function", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])

    def test_unknown_x_space_raises_value_error(self):
        """Test that an unknown x_space is rejected with a ValueError."""
        with pytest.raises(ValueError):
            cio.CompletenessFit(function_name="sigmoid01", x_space="furlongs", popt=[1.0, 2.0])

    def test_popt_is_converted_to_a_float_array(self):
        """Test that popt is converted to a float array."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1, 2])
        assert isinstance(fit.popt, np.ndarray)
        assert fit.popt.dtype == np.float64

    def test_pcov_is_converted_to_a_float_array_when_given(self):
        """Test that pcov is converted to a float array when given."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0],
                                  pcov=[[1, 0], [0, 1]])
        assert isinstance(fit.pcov, np.ndarray)
        assert fit.pcov.dtype == np.float64

    def test_pcov_defaults_to_none(self):
        """Test that pcov defaults to None when not given."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])
        assert fit.pcov is None

    def test_wrong_parameter_count_raises_value_error(self):
        """
        Test that a mismatch between the number of parameters in popt and the function's signature raises a ValueError.
        """
        # sigmoid01 takes 2 params (x0, k); giving 3 is a mismatch
        with pytest.raises(ValueError):
            cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0, 3.0])

    def test_non_1d_popt_raises_value_error(self):
        """Test that a non-1D popt raises a ValueError."""
        with pytest.raises(ValueError):
            cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[[1.0, 2.0]])

    @pytest.mark.parametrize("function_name,n_params", [
        ("sigmoid", 4), ("sigmoid01", 2), ("richards01", 3), ("erf01", 2),
    ])
    def test_correct_parameter_count_is_accepted_for_every_registered_function(self, function_name, n_params):
        """Test that the correct number of parameters is accepted for every registered function."""
        fit = cio.CompletenessFit(function_name=function_name, x_space=cio.X_SPACE_MJY,
                                  popt=list(range(1, n_params + 1)))
        assert fit.popt.shape == (n_params,)


class TestCompletenessFitFunctionProperty:
    """
    Unit tests for the CompletenessFit.function property, which returns the callable corresponding to the function name.
    """

    def test_returns_the_registered_callable(self):
        """Test that the function property returns the registered callable for the given function name."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])
        assert fit.function is func.sigmoid01

    def test_erf01_returns_erf01(self):
        """Test that the function property returns the registered callable for the given function name."""
        fit = cio.CompletenessFit(function_name="erf01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])
        assert fit.function is func.erf01


class TestCompletenessFitEvaluate:
    """Unit tests for the CompletenessFit.evaluate method, which evaluates the completeness curve at given fluxes."""

    def test_mjy_space_matches_direct_function_call(self):
        """
        Test that evaluating in mJy space matches a direct call to the registered function with the same parameters.
        """
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[5.0, 1.5])
        fluxes = np.array([1.0, 3.0, 5.0, 8.0])
        expected = func.sigmoid01(fluxes, 5.0, 1.5)
        np.testing.assert_allclose(fit.evaluate(fluxes), expected)

    def test_log10_space_evaluates_the_function_on_log10_of_flux(self):
        """Test that evaluating in log10(mJy) space evaluates the function on the log10 of the fluxes."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_LOG10_MJY, popt=[0.7, 1.5])
        fluxes = np.array([1.0, 3.0, 5.0, 8.0])
        expected = func.sigmoid01(np.log10(fluxes), 0.7, 1.5)
        np.testing.assert_allclose(fit.evaluate(fluxes), expected)

    def test_at_the_midpoint_sigmoid01_is_exactly_one_half(self):
        """Test that the sigmoid01 function evaluates to 0.5 at its midpoint, in both x_spaces."""
        # sigmoid01(x0, x0, k) == 0.5 for any k - true in both x_spaces
        fit_linear = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[5.0, 2.0])
        assert fit_linear.evaluate(5.0) == pytest.approx(0.5)

        fit_log = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_LOG10_MJY,
                                      popt=[np.log10(5.0), 2.0])
        assert fit_log.evaluate(5.0) == pytest.approx(0.5)

    def test_zero_flux_in_log10_space_gives_zero_completeness_without_raising(self):
        """Test that evaluating at zero flux in log10 space gives 0 completeness without raising an error."""
        # log10(0) = -inf (the honest limit), which drives every registered curve to 0 completeness.
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_LOG10_MJY, popt=[0.7, 1.5])
        result = fit.evaluate(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0])

    def test_negative_flux_in_log10_space_gives_nan_without_raising(self):
        """Test that evaluating at a negative flux in log10 space gives NaN without raising an error."""
        # log10(negative) has no real value at all (unlike log10(0)'s well-defined -inf limit), so this is NaN
        # rather than 0 - still doesn't raise, and NaN safely fails every downstream comparison.
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_LOG10_MJY, popt=[0.7, 1.5])
        result = fit.evaluate(np.array([-1.0]))
        assert np.isnan(result[0])

    def test_zero_shift_leaves_curve_unchanged(self):
        """Test that a zero s0_shift_mjy leaves the curve unchanged."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[5.0, 2.0])
        fluxes = np.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(fit.evaluate(fluxes, s0_shift_mjy=0.0), fit.evaluate(fluxes))

    def test_shift_moves_midpoint_in_linear_mjy_space(self):
        """Test that a non-zero s0_shift_mjy moves the curve's midpoint in linear mJy space."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[5.0, 2.0])
        # midpoint shifted from 5.0 to 7.0 mJy -> evaluating at 7.0 should now give 0.5
        assert fit.evaluate(7.0, s0_shift_mjy=2.0) == pytest.approx(0.5)

    def test_shift_in_log_space_is_applied_in_linear_flux_not_to_log_x0_directly(self):
        """Test that a non-zero s0_shift_mjy is applied in linear flux space, not added to log10(x0) directly."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_LOG10_MJY, popt=[np.log10(5.0), 2.0])
        # midpoint (linear) shifted from 5.0 to 7.0 mJy -> evaluating at 7.0 should now give 0.5
        assert fit.evaluate(7.0, s0_shift_mjy=2.0) == pytest.approx(0.5)

    def test_shift_that_makes_midpoint_non_positive_raises_value_error(self):
        """Test that a shift that makes the curve's midpoint non-positive raises a ValueError."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])
        with pytest.raises(ValueError):
            fit.evaluate(1.0, s0_shift_mjy=-2.0)

    def test_evaluate_does_not_mutate_popt(self):
        """Test that calling evaluate does not mutate the fit's popt."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[5.0, 2.0])
        original = fit.popt.copy()
        fit.evaluate(7.0, s0_shift_mjy=2.0)
        np.testing.assert_array_equal(fit.popt, original)

    def test_accepts_scalar_flux(self):
        """Test that evaluate accepts a scalar flux and returns a scalar completeness."""
        fit = cio.CompletenessFit(function_name="erf01", x_space=cio.X_SPACE_MJY, popt=[5.0, 1.0])
        result = fit.evaluate(5.0)
        assert np.isscalar(result) or result.shape == ()


class TestWriteCompletenessFit:
    """Unit tests for the write_completeness_fit function, which writes a CompletenessFit to a JSON file."""

    def test_rejects_non_json_suffix(self, tmp_path):
        """Test that write_completeness_fit rejects a path without a .json suffix."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])
        with pytest.raises(ValueError):
            cio.write_completeness_fit(tmp_path / "fit.txt", fit)

    def test_creates_parent_directories(self, tmp_path):
        """Test that write_completeness_fit creates parent directories if they do not exist."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])
        out_path = tmp_path / "nested" / "dir" / "fit.json"
        result = cio.write_completeness_fit(out_path, fit)
        assert result == out_path
        assert out_path.exists()

    def test_writes_valid_json_with_expected_fields(self, tmp_path):
        """Test that write_completeness_fit writes a valid JSON file with the expected fields."""
        fit = cio.CompletenessFit(function_name="erf01", x_space=cio.X_SPACE_LOG10_MJY, popt=[0.5, 1.2],
                                  param_names=["x0", "sigma"], provenance="unit test")
        out_path = tmp_path / "fit.json"
        cio.write_completeness_fit(out_path, fit)

        payload = json.loads(out_path.read_text())
        assert payload["function"] == "erf01"
        assert payload["x_space"] == cio.X_SPACE_LOG10_MJY
        assert payload["popt"] == [0.5, 1.2]
        assert payload["pcov"] is None
        assert payload["param_names"] == ["x0", "sigma"]
        assert payload["provenance"] == "unit test"


class TestReadCompletenessFit:
    """Unit tests for the read_completeness_fit function, which reads a CompletenessFit from a JSON file."""

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        """Test that read_completeness_fit raises FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            cio.read_completeness_fit(tmp_path / "does_not_exist.json")

    def test_non_json_file_raises_value_error(self, tmp_path):
        """Test that read_completeness_fit raises ValueError for a non-JSON file."""
        legacy = tmp_path / "legacy.txt"
        legacy.write_text("1.0 2.0 3.0")
        with pytest.raises(ValueError):
            cio.read_completeness_fit(legacy)

    def test_json_missing_required_fields_raises_value_error(self, tmp_path):
        """Test that read_completeness_fit raises ValueError for a JSON file missing required fields."""
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps({"function": "sigmoid01"}))  # missing x_space, popt
        with pytest.raises(ValueError):
            cio.read_completeness_fit(path)

    def test_round_trips_full_fit(self, tmp_path):
        """Test that a full CompletenessFit can be written and read back, preserving all fields."""
        fit = cio.CompletenessFit(
            function_name="richards01", x_space=cio.X_SPACE_MJY, popt=[3.0, 1.0, 2.0],
            pcov=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            param_names=["x0", "k", "nu"], provenance="round-trip test",
        )
        path = tmp_path / "fit.json"
        cio.write_completeness_fit(path, fit)

        loaded = cio.read_completeness_fit(path)

        assert loaded.function_name == fit.function_name
        assert loaded.x_space == fit.x_space
        np.testing.assert_allclose(loaded.popt, fit.popt)
        np.testing.assert_allclose(loaded.pcov, fit.pcov)
        assert loaded.param_names == fit.param_names
        assert loaded.provenance == fit.provenance

    def test_round_trips_fit_with_no_pcov(self, tmp_path):
        """Test that a CompletenessFit with no pcov can be written and read back, preserving all fields."""
        fit = cio.CompletenessFit(function_name="sigmoid01", x_space=cio.X_SPACE_MJY, popt=[1.0, 2.0])
        path = tmp_path / "fit.json"
        cio.write_completeness_fit(path, fit)

        loaded = cio.read_completeness_fit(path)

        assert loaded.pcov is None

    def test_missing_optional_fields_default_sensibly(self, tmp_path):
        """Test that a JSON file missing optional fields defaults them sensibly when read back."""
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps({"function": "sigmoid01", "x_space": cio.X_SPACE_MJY, "popt": [1.0, 2.0]}))

        loaded = cio.read_completeness_fit(path)

        assert loaded.pcov is None
        assert loaded.param_names == []
        assert loaded.provenance == ""

    def test_loaded_fit_evaluates_the_same_as_the_original(self, tmp_path):
        """Test that a CompletenessFit loaded from JSON evaluates the same as the original fit."""
        fit = cio.CompletenessFit(function_name="erf01", x_space=cio.X_SPACE_LOG10_MJY, popt=[0.5, 1.2])
        path = tmp_path / "fit.json"
        cio.write_completeness_fit(path, fit)

        loaded = cio.read_completeness_fit(path)

        fluxes = np.array([1.0, 3.0, 10.0])
        np.testing.assert_allclose(loaded.evaluate(fluxes), fit.evaluate(fluxes))
