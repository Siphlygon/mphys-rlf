"""
Unit tests for diffracc/analysis/log_analyzer.py's PyBDSF log-parsing functions.

Each function searches for one fixed regex pattern in a text file. Synthetic log text is written to tmp_path,
reusing the literal filler characters ("....." / "..............") straight out of the source regex, since a
literal "." trivially satisfies the regex "." metacharacter - no need to reverse-engineer the real PyBDSF log
format, just to satisfy the pattern.
"""
from pathlib import Path

import pytest

from diffracc.analysis import log_analyzer as la


def write_log(tmp_path, content: str, name: str = "source.pybdsf.log") -> Path:
    """
    Write a temporary PyBDSF log file with the given content to tmp_path, returning the path to it.

    Parameters
    ----------
    tmp_path : pytest.TempPathFactory
        A pytest fixture providing a temporary directory unique to the test invocation.
    content : str
        The text content to write to the log file.
    name : str, optional
        The name of the log file to create, by default "source.pybdsf.log"

    Returns
    -------
    Path
        The path to the created log file.
    """
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestGetTotalFlux:
    """Tests for the get_total_flux() function in log_analyzer.py."""

    def test_extracts_total_flux_value(self, tmp_path):
        """Test that get_total_flux() correctly extracts the total flux value from a log file."""
        path = write_log(tmp_path, "Flux from sum of (non-blank) pixels ..... : 1.234 Jy\n")
        assert la.get_total_flux(path) == pytest.approx(1.234)

    def test_raises_when_pattern_not_found(self, tmp_path):
        """
        Test that get_total_flux() raises an AttributeError when the expected pattern is not found in the log file.
        """
        path = write_log(tmp_path, "nothing relevant here\n")
        with pytest.raises(AttributeError):
            la.get_total_flux(path)


class TestGetModelFlux:
    """Tests for the get_model_flux() function in log_analyzer.py."""

    def test_extracts_model_flux_value(self, tmp_path):
        """Test that get_model_flux() correctly extracts the model flux value from a log file."""
        path = write_log(tmp_path, "Total flux density in model ............. : 5.678 Jy\n")
        assert la.get_model_flux(path) == pytest.approx(5.678)

    def test_defaults_to_zero_when_no_model_flux_line(self, tmp_path):
        """Test that get_model_flux() returns 0 when the expected model flux line is not found in the log file."""
        # Unlike get_flux, this one is documented to default to 0 rather than raise, since PyBDSF omits this line
        # entirely when no model flux is found.
        path = write_log(tmp_path, "no model flux line in this log\n")
        assert la.get_model_flux(path) == 0


class TestGetMean:
    """Tests for the get_mean() function in log_analyzer.py."""

    def test_extracts_mean_value(self, tmp_path):
        """Test that get_mean() correctly extracts the mean value from a log file."""
        path = write_log(tmp_path, "Raw mean (Stokes I) =  0.512 mJy\n")
        assert la.get_mean(path) == pytest.approx(0.512)

    def test_raises_when_pattern_not_found(self, tmp_path):
        """Test that get_mean() raises an AttributeError when the expected pattern is not found in the log file."""
        path = write_log(tmp_path, "nothing relevant here\n")
        with pytest.raises(AttributeError):
            la.get_mean(path)


class TestGetSigmaClippedMean:
    """Tests for the get_sigma_clipped_mean() function in log_analyzer.py."""

    def test_extracts_positive_value(self, tmp_path):
        """Test that get_sigma_clipped_mean() correctly extracts a positive mean value from a log file."""
        path = write_log(tmp_path, "sigma clipped mean (Stokes I) =  0.123 mJy\n")
        assert la.get_sigma_clipped_mean(path) == pytest.approx(0.123)

    def test_extracts_negative_value_with_sign_preserved(self, tmp_path):
        """Test that get_sigma_clipped_mean() correctly extracts a negative mean value from a log file."""
        path = write_log(tmp_path, "sigma clipped mean (Stokes I) =  -0.123 mJy\n")
        assert la.get_sigma_clipped_mean(path) == pytest.approx(-0.123)

    def test_raises_when_pattern_not_found(self, tmp_path):
        """
        Test that get_sigma_clipped_mean() raises an AttributeError when the expected pattern is not found in the log
        file.
        """
        path = write_log(tmp_path, "nothing relevant here\n")
        with pytest.raises(AttributeError):
            la.get_sigma_clipped_mean(path)


class TestGetRms:
    """Tests for the get_rms() function in log_analyzer.py."""

    def test_extracts_rms_value(self, tmp_path):
        """Test that get_rms() correctly extracts the rms value from a log file."""
        path = write_log(tmp_path, "raw rms =  0.045 mJy\n")
        assert la.get_rms(path) == pytest.approx(0.045)

    def test_raises_when_pattern_not_found(self, tmp_path):
        """Test that get_rms() raises an AttributeError when the expected pattern is not found in the log file."""
        path = write_log(tmp_path, "nothing relevant here\n")
        with pytest.raises(AttributeError):
            la.get_rms(path)


class TestGetSigmaClippedRms:
    """Tests for the get_sigma_clipped_rms() function in log_analyzer.py."""

    def test_extracts_sigma_clipped_rms_value(self, tmp_path):
        """Test that get_sigma_clipped_rms() correctly extracts the sigma clipped rms value from a log file."""
        path = write_log(tmp_path, "sigma clipped rms =  0.038 mJy\n")
        assert la.get_sigma_clipped_rms(path) == pytest.approx(0.038)

    def test_raises_when_pattern_not_found(self, tmp_path):
        """
        Test that get_sigma_clipped_rms() raises an AttributeError when the expected pattern is not found in the log
        file.
        """
        path = write_log(tmp_path, "nothing relevant here\n")
        with pytest.raises(AttributeError):
            la.get_sigma_clipped_rms(path)


class TestGetFluxMeanRms:
    """Tests for the get_flux_mean_rms() function in log_analyzer.py."""

    def test_extracts_all_three_values(self, tmp_path):
        """Test that get_flux_mean_rms() correctly extracts flux, mean, and rms values from a log file."""
        # The combined regex requires "raw mean ... and raw rms ..." together, then (with re.DOTALL letting
        # anything - including newlines - fall in between) the flux line later in the file.
        content = (
            "Raw mean (Stokes I) =  0.512 mJy and raw rms =  0.045 mJy\n"
            "... other log content in between ...\n"
            "Flux from sum of (non-blank) pixels ..... : 1.234 Jy\n"
        )
        path = write_log(tmp_path, content)
        flux, mean, rms = la.get_flux_mean_rms(path)
        assert flux == pytest.approx(1.234)
        assert mean == pytest.approx(0.512)
        assert rms == pytest.approx(0.045)

    def test_raises_when_mean_and_rms_not_on_the_same_line(self, tmp_path):
        """
        Test that get_flux_mean_rms() raises an AttributeError when the mean and rms values are not on the same line.
        """
        # get_mean/get_rms search independently and would each succeed here, but get_flux_mean_rms requires them
        # joined by " and " on one line - splitting them across lines should not match.
        content = (
            "Raw mean (Stokes I) =  0.512 mJy\n"
            "raw rms =  0.045 mJy\n"
            "Flux from sum of (non-blank) pixels ..... : 1.234 Jy\n"
        )
        path = write_log(tmp_path, content)
        with pytest.raises(AttributeError):
            la.get_flux_mean_rms(path)

    def test_raises_when_pattern_not_found(self, tmp_path):
        """
        Test that get_flux_mean_rms() raises an AttributeError when the expected pattern is not found in the log file.
        """
        path = write_log(tmp_path, "nothing relevant here\n")
        with pytest.raises(AttributeError):
            la.get_flux_mean_rms(path)
