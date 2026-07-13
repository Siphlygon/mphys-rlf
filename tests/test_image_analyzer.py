"""
Unit tests for diffracc/analysis/image_analyzer.py.

Requires the real `bdsf` package to be importable (it's a top-level import in the module under test), which is not
installable on Windows - run this file via the WSL conda env `mphysrlf`, e.g.:
    wsl.exe -e bash -lc "cd '<repo>' && ~/miniconda3/envs/mphysrlf/bin/python -m pytest tests/test_image_analyzer.py"
bdsf.process_image itself is always monkeypatched out below - no real PyBDSF processing runs in these tests.

The importorskip below means this whole module is cleanly SKIPPED (not errored) wherever bdsf isn't installed -
without it, a bare `import bdsf` failure here would abort the entire pytest session's collection, not just this
file, since pytest treats a collection ImportError as fatal unless --continue-on-collection-errors is passed.
"""
from pathlib import PurePath

import numpy as np
import pytest
from astropy.io import fits

pytest.importorskip("bdsf", reason="bdsf (PyBDSF) is not installable on Windows; run this file under WSL/conda.")

from diffracc.analysis import image_analyzer as ia
from diffracc.utils import paths


class TestProcessArgs:
    """
    Tests for the ProcessArgs dataclass in image_analyzer.py, which reads PyBDSF config from a TOML file and
    allows overriding individual keys via kwargs.
    """

    def test_from_toml_reads_all_sections(self):
        """
        Test that ProcessArgs.from_toml() reads the TOML file and populates all expected fields with the correct values.
        """
        args = ia.ProcessArgs.from_toml(paths.PYBDSF_CONFIG)
        assert args.beam == (0.00166667, 0.00166667, 0.0) or list(args.beam) == [0.00166667, 0.00166667, 0.0]
        assert args.frequency == pytest.approx(144e6)
        assert args.mean_map == "zero"
        assert args.adaptive_thresh == 150
        assert args.atrous_jmax == 4
        assert args.ini_method == "intensity"

    def test_from_toml_raises_on_missing_file(self, tmp_path):
        """Test that ProcessArgs.from_toml() raises a FileNotFoundError when the specified TOML file does not exist."""
        with pytest.raises(FileNotFoundError):
            ia.ProcessArgs.from_toml(tmp_path / "does_not_exist.toml")

    def test_to_dict_matches_dataclass_fields(self):
        """Test that ProcessArgs.to_dict() returns a dictionary with keys matching the dataclass fields."""
        args = ia.ProcessArgs(thresh_pix=3.0)
        d = args.to_dict()
        assert d["thresh_pix"] == 3.0
        assert d["thresh_isl"] == 4.0  # untouched default


@pytest.fixture
def make_analyzer(tmp_path):
    """
    A factory fixture that creates an ImageAnalyzer instance with default paths set to tmp_path and allows overriding
    any kwargs. The subdir is fixed to "mysubdir" for testing purposes.

    Returns
    -------
    _make : Callable[..., ia.ImageAnalyzer]
        A factory function that creates an ImageAnalyzer instance with default paths set to tmp_path and allows
        overriding any kwargs. The subdir is fixed to "mysubdir" for testing purposes
    """
    def _make(**kwargs) -> ia.ImageAnalyzer:
        kwargs.setdefault("fits_input_dir", tmp_path / "fits")
        kwargs.setdefault("log_dir", tmp_path / "logs")
        kwargs.setdefault("catalog_dir", tmp_path / "catalogs")
        kwargs.setdefault("img_dir", tmp_path / "images")
        return ia.ImageAnalyzer(subdir="mysubdir", **kwargs)
    return _make


class TestImageAnalyzerInitKwargRouting:
    """
    Tests that ImageAnalyzer correctly routes kwargs to the appropriate sub-objects (ProcessArgs, ExportImgArgs, and
    CatalogArgs).
    """

    def test_catalog_type_kwarg_expands_to_catalog_type(self, make_analyzer):
        """Test that a catalog_type kwarg passed to ImageAnalyzer is correctly routed to the CatalogArgs sub-object."""
        analyzer = make_analyzer(catalog_type="srl")
        assert analyzer.catalog_args["catalog_type"] == "srl"

    def test_catalog_catalog_type_kwarg_is_skipped(self, make_analyzer):
        """
        Test that a catalog_catalog_type kwarg passed to ImageAnalyzer is ignored and does not override catalog_type.
        """
        analyzer = make_analyzer(catalog_catalog_type="srl")
        assert "catalog_type" not in analyzer.catalog_args or analyzer.catalog_args.get("catalog_type") != "srl"

    def test_catalog_kwarg_ignored_when_write_catalog_false(self, make_analyzer):
        """Test that a catalog kwarg passed to ImageAnalyzer is ignored when write_catalog is False."""
        analyzer = make_analyzer(write_catalog=False, catalog_type="srl")
        assert analyzer.catalog_args == {}

    def test_write_catalog_defaults_clobber_true(self, make_analyzer):
        """Test that when write_catalog is True, the clobber kwarg defaults to True in CatalogArgs."""
        analyzer = make_analyzer()
        assert analyzer.catalog_args["clobber"] is True

    def test_export_img_kwarg_routes_to_correct_image_type(self, make_analyzer):
        """Test that an export_img kwarg passed to ImageAnalyzer is correctly routed to the ExportImgArgs sub-object."""
        analyzer = make_analyzer(export_images=["gaus_model", "gaus_resid"], gaus_model_clobber=False)
        assert analyzer.export_img_args["gaus_model"] == {"img_type": "gaus_model", "clobber": False}
        assert analyzer.export_img_args["gaus_resid"] == {"img_type": "gaus_resid", "clobber": True}

    def test_process_kwarg_overrides_toml_default(self, make_analyzer):
        """Test that a process kwarg passed to ImageAnalyzer overrides the default value from the TOML config."""
        analyzer = make_analyzer(process_thresh_pix=3.0)
        assert analyzer.process_args["thresh_pix"] == 3.0
        assert analyzer.process_args["thresh_isl"] == 4.0  # untouched default from pybdsf_config.toml

    def test_process_args_default_to_toml_when_not_overridden(self, make_analyzer):
        """
        Test that when no process kwarg is passed to ImageAnalyzer, the default values from the TOML config are used.
        """
        analyzer = make_analyzer()
        assert analyzer.process_args["mean_map"] == "zero"
        assert analyzer.process_args["frequency"] == pytest.approx(144e6)

    def test_unused_kwarg_does_not_raise(self, make_analyzer):
        """Test that an unused kwarg passed to ImageAnalyzer does not raise an error."""
        # Just needs to not error - the mismatch is only logged as a warning.
        make_analyzer(totally_unrelated_kwarg=1)


class TestGetPostfix:
    """
    Tests for the get_postfix() method of ImageAnalyzer, which returns the path parts after the subdir in a given path.
    """

    def test_returns_path_parts_after_subdir(self, make_analyzer, tmp_path):
        """Test that get_postfix() correctly returns the path parts after the subdir in a given path."""
        analyzer = make_analyzer(fits_input_dir=tmp_path / "fits")
        path = tmp_path / "fits" / "mysubdir" / "bin1" / "image5.fits"
        assert analyzer.get_postfix(path) == PurePath("bin1", "image5.fits")

    def test_raises_when_subdir_not_in_path(self, make_analyzer, tmp_path):
        """Test that get_postfix() raises a ValueError when the subdir is not present in the given path."""
        analyzer = make_analyzer(fits_input_dir=tmp_path / "fits")
        with pytest.raises(ValueError):
            analyzer.get_postfix(tmp_path / "somewhere" / "else.fits")


class TestSaveImageToFits:
    """
    Tests for the save_image_to_fits() method of ImageAnalyzer, which saves a numpy array to a FITS file with a
    specified header.
    """

    def test_writes_image_with_expected_header_and_data(self, make_analyzer, tmp_path):
        """Test that save_image_to_fits() writes the image data and expected header to a FITS file."""
        analyzer = make_analyzer()
        image = np.arange(16, dtype=np.float32).reshape(4, 4)

        analyzer.save_image_to_fits(image, "example.fits", FXSCLD=0.42)

        out_path = tmp_path / "fits" / "mysubdir" / "example.fits"
        assert out_path.exists()
        with fits.open(out_path) as hdul:
            np.testing.assert_allclose(hdul[0].data, image)
            assert hdul[0].header["FXSCLD"] == pytest.approx(0.42)
            assert hdul[0].header["CTYPE1"] == "RA---SIN"
            assert hdul[0].header["CUNIT1"] == "deg"

    def test_creates_nested_postfix_directories(self, make_analyzer, tmp_path):
        """Test that save_image_to_fits() creates nested directories for the postfix if they do not exist."""
        analyzer = make_analyzer()
        image = np.zeros((4, 4), dtype=np.float32)

        analyzer.save_image_to_fits(image, "bin1/example.fits")

        assert (tmp_path / "fits" / "mysubdir" / "bin1" / "example.fits").exists()


class _FakeBdsfImage:
    """A fake image object for testing purposes."""

    def __init__(self):
        self.export_image_calls = []
        self.write_catalog_calls = []

    def export_image(self, outfile, **kwargs):
        """Record the call to export_image and create a flag file for testing purposes."""
        self.export_image_calls.append((outfile, kwargs))
        # PyBDSF actually writes the file; a fake stand-in needs to too, since the caller only touches the flag
        # file afterward regardless of whether the real output file exists.
        open(outfile, "w", encoding="utf-8").close()

    def write_catalog(self, outfile, **kwargs):
        """Record the call to write_catalog and create a flag file for testing purposes."""
        self.write_catalog_calls.append((outfile, kwargs))
        open(outfile, "w", encoding="utf-8").close()


class TestAnalyzeFitsAtPath:
    """Tests for the analyze_fits_at_path() method of ImageAnalyzer, which processes a FITS file and writes outputs."""

    def _make_fits_file(self, path):
        """Helper method to create a dummy FITS file at the specified path for testing purposes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32)).writeto(path)

    def test_raises_for_missing_path(self, make_analyzer, tmp_path):
        """Test that analyze_fits_at_path() raises an AssertionError when the specified FITS file does not exist."""
        analyzer = make_analyzer()
        with pytest.raises(AssertionError):
            analyzer.analyze_fits_at_path(tmp_path / "does_not_exist.fits")

    def test_raises_for_directory_path(self, make_analyzer, tmp_path):
        """Test that analyze_fits_at_path() raises a ValueError when the specified path is a directory."""
        analyzer = make_analyzer()
        directory = tmp_path / "a_directory.fits"
        directory.mkdir()
        with pytest.raises(ValueError):
            analyzer.analyze_fits_at_path(directory)

    def test_raises_for_non_fits_suffix(self, make_analyzer, tmp_path):
        """Test that analyze_fits_at_path() raises a ValueError when the specified path does not have a .fits suffix."""
        analyzer = make_analyzer()
        bad_file = tmp_path / "not_fits.txt"
        bad_file.write_text("hello")
        with pytest.raises(ValueError):
            analyzer.analyze_fits_at_path(bad_file)

    def test_skips_processing_when_nothing_to_do(self, make_analyzer, tmp_path, monkeypatch):
        """
        Test that analyze_fits_at_path() skips processing when the log file, image export flag, and catalog flag already
        exist, and does not call bdsf.process_image.
        """
        analyzer = make_analyzer(export_images=["gaus_model"])
        fits_path = tmp_path / "fits" / "mysubdir" / "image0.fits"
        self._make_fits_file(fits_path)

        # Pre-create everything analyze_fits_at_path checks for: the log file, the image export flag, and (since
        # write_catalog defaults True) the catalog flag.
        (tmp_path / "logs" / "mysubdir" / "image0.fits.pybdsf.log").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs" / "mysubdir" / "image0.fits.pybdsf.log").touch()
        (tmp_path / "images" / "mysubdir" / "gaus_model" / "image0.fits.flag").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "mysubdir" / "gaus_model" / "image0.fits.flag").touch()
        (tmp_path / "catalogs" / "mysubdir" / "image0.fits.flag").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "catalogs" / "mysubdir" / "image0.fits.flag").touch()

        def _unexpected_process_image(*args, **kwargs):
            raise AssertionError("bdsf.process_image should not be called when there's no work to do")
        monkeypatch.setattr(ia.bdsf, "process_image", _unexpected_process_image)

        analyzer.analyze_fits_at_path(fits_path)  # must not raise

    def test_processes_and_writes_outputs_when_work_is_needed(self, make_analyzer, tmp_path, monkeypatch):
        """
        Test that analyze_fits_at_path() calls bdsf.process_image and writes the expected outputs when work is needed.
        """
        analyzer = make_analyzer(export_images=["gaus_model"])
        fits_path = tmp_path / "fits" / "mysubdir" / "image0.fits"
        self._make_fits_file(fits_path)

        fake_image = _FakeBdsfImage()
        process_calls = []
        monkeypatch.setattr(ia.bdsf, "process_image",
                            lambda path, **kwargs: (process_calls.append((path, kwargs)), fake_image)[1])

        analyzer.analyze_fits_at_path(fits_path)

        assert len(process_calls) == 1
        assert len(fake_image.export_image_calls) == 1
        assert len(fake_image.write_catalog_calls) == 1
        assert (tmp_path / "images" / "mysubdir" / "gaus_model" / "image0.fits.flag").exists()
        assert (tmp_path / "catalogs" / "mysubdir" / "image0.fits.flag").exists()

    def test_still_writes_flag_files_when_processing_fails(self, make_analyzer, tmp_path, monkeypatch):
        """Test that analyze_fits_at_path() still writes flag files when bdsf.process_image raises an exception."""
        # A ValueError from bdsf.process_image is caught internally; flag files should still be written so a
        # permanently-unprocessable image isn't retried forever, but the (nonexistent) image/catalog aren't
        # written.
        analyzer = make_analyzer(export_images=["gaus_model"])
        fits_path = tmp_path / "fits" / "mysubdir" / "image0.fits"
        self._make_fits_file(fits_path)

        def _fail(*args, **kwargs):
            raise ValueError("PyBDSF could not process this image")
        monkeypatch.setattr(ia.bdsf, "process_image", _fail)

        analyzer.analyze_fits_at_path(fits_path)  # must not raise

        assert (tmp_path / "images" / "mysubdir" / "gaus_model" / "image0.fits.flag").exists()
        assert (tmp_path / "catalogs" / "mysubdir" / "image0.fits.flag").exists()
        assert not (tmp_path / "images" / "mysubdir" / "gaus_model" / "image0.fits").exists()

    def test_runtime_error_from_unphysical_rms_is_caught(self, make_analyzer, tmp_path, monkeypatch):
        """
        Test that analyze_fits_at_path() catches a RuntimeError from bdsf.process_image when the RMS is unphysical.
        """
        analyzer = make_analyzer(export_images=["gaus_model"])
        fits_path = tmp_path / "fits" / "mysubdir" / "image0.fits"
        self._make_fits_file(fits_path)

        def _fail(*args, **kwargs):
            raise RuntimeError("unphysical RMS")
        monkeypatch.setattr(ia.bdsf, "process_image", _fail)

        analyzer.analyze_fits_at_path(fits_path)  # must not raise

        assert (tmp_path / "images" / "mysubdir" / "gaus_model" / "image0.fits.flag").exists()

    def test_accepts_a_string_path(self, make_analyzer, tmp_path, monkeypatch):
        """Test that analyze_fits_at_path() accepts a string path as well as a Path object."""
        analyzer = make_analyzer()
        fits_path = tmp_path / "fits" / "mysubdir" / "image0.fits"
        self._make_fits_file(fits_path)
        monkeypatch.setattr(ia.bdsf, "process_image", lambda path, **kwargs: _FakeBdsfImage())

        analyzer.analyze_fits_at_path(str(fits_path))  # must not raise despite being a str, not a Path


class TestAnalyzeImage:
    """
    Tests for the analyze_image() method of ImageAnalyzer, which saves a numpy array to a FITS file and analyzes it.
    """

    def test_saves_and_analyzes_in_one_call(self, make_analyzer, tmp_path, monkeypatch):
        """Test that analyze_image() saves the image to a FITS file and then analyzes it in one call."""
        analyzer = make_analyzer()
        calls = []
        monkeypatch.setattr(analyzer, "analyze_fits_at_path", lambda path: calls.append(path))
        image = np.zeros((4, 4), dtype=np.float32)

        analyzer.analyze_image(image, fscaled=0.7, postfix="example.fits")

        saved_path = tmp_path / "fits" / "mysubdir" / "example.fits"
        assert saved_path.exists()
        with fits.open(saved_path) as hdul:
            assert hdul[0].header["FXSCLD"] == pytest.approx(0.7)
        assert calls == [saved_path]
