"""
Unit tests for diffracc/analysis/run_analysis.py's analyze() orchestration function.

run_analysis.py imports ImageAnalyzer (from image_analyzer.py) at module level, which itself imports the real
`bdsf` package - not installable on Windows, hence the same importorskip guard as test_image_analyzer.py. See that
file's docstring for how to run this one under WSL/conda.
"""
import pytest

pytest.importorskip("bdsf", reason="bdsf (PyBDSF) is not installable on Windows; run this file under WSL/conda.")

from diffracc.analysis import run_analysis as ra
from diffracc.utils import paths


class _FakeAnalyzer:
    """Records the kwargs it was constructed with, and whether/with-what kwargs analyze_all_fits_in_input() ran."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.ran = False
        self.run_kwargs = None
        _FakeAnalyzer.instances.append(self)

    def analyze_all_fits_in_input(self, **kwargs):
        self.ran = True
        self.run_kwargs = kwargs


@pytest.fixture(autouse=True)
def _clear_fake_analyzer_instances():
    """Clear the _FakeAnalyzer.instances list before and after each test."""
    _FakeAnalyzer.instances = []
    yield
    _FakeAnalyzer.instances = []


class TestAnalyze:
    """Tests for the analyze() function in run_analysis.py."""

    def test_constructs_and_runs_one_analyzer_per_subdir(self, monkeypatch):
        """Test that analyze() constructs and runs one ImageAnalyzer per subdir."""
        monkeypatch.setattr(ra, "ImageAnalyzer", _FakeAnalyzer)

        ra.analyze(["sub1", "sub2"], fits_input_dir="/some/dir")

        assert len(_FakeAnalyzer.instances) == 2
        first, second = _FakeAnalyzer.instances
        assert first.kwargs["subdir"] == "sub1"
        assert second.kwargs["subdir"] == "sub2"
        assert first.ran and second.ran

    def test_passes_fits_input_dir_export_images_and_catalog_format(self, monkeypatch):
        """Test that analyze() passes fits_input_dir, export_images, and catalog_format to ImageAnalyzer."""
        monkeypatch.setattr(ra, "ImageAnalyzer", _FakeAnalyzer)

        ra.analyze(["sub1"], fits_input_dir="/some/dir")

        kwargs = _FakeAnalyzer.instances[0].kwargs
        assert kwargs["fits_input_dir"] == "/some/dir"
        assert kwargs["export_images"] == ["gaus_model", "gaus_resid"]
        assert kwargs["catalog_format"] == "fits"

    def test_defaults_fits_input_dir_to_paths_fits_parent_for_every_subdir(self, monkeypatch):
        """Test that analyze() defaults fits_input_dir to paths.FITS_PARENT for every subdir."""
        monkeypatch.setattr(ra, "ImageAnalyzer", _FakeAnalyzer)

        ra.analyze(["sub1", "sub2"])

        assert _FakeAnalyzer.instances[0].kwargs["fits_input_dir"] == paths.FITS_PARENT
        assert _FakeAnalyzer.instances[1].kwargs["fits_input_dir"] == paths.FITS_PARENT

    def test_empty_subdir_list_constructs_nothing(self, monkeypatch):
        """Test that analyze() constructs nothing if the subdir list is empty."""
        monkeypatch.setattr(ra, "ImageAnalyzer", _FakeAnalyzer)

        ra.analyze([])

        assert _FakeAnalyzer.instances == []
