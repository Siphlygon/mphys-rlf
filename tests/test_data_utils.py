"""Unit tests for diffracc/utils/data_utils.py."""
import h5py
import numpy as np
import pytest
from astropy.io import fits

from diffracc.utils import data_utils as du
from diffracc.utils.logger import get_logger


@pytest.fixture
def logger():
    """Fixture to provide a logger instance for tests."""
    return get_logger("test_data_utils")


class _FakeColumn:
    """A fake object to simulate a FITS column with a name and format."""
    def __init__(self, name, format):
        self.name = name
        self.format = format


class TestPadTo80x80:
    """Tests for the _pad_to_80x80 function, which pads smaller arrays to 80x80 with NaN values."""

    def test_pads_smaller_array_with_nan(self):
        """Test that a smaller array is padded to 80x80 with NaN values."""
        arr = np.ones((10, 10), dtype=np.float32)
        padded = du._pad_to_80x80(arr)
        assert padded.shape == (80, 80)
        np.testing.assert_array_equal(padded[:10, :10], arr)
        assert np.all(np.isnan(padded[10:, :]))
        assert np.all(np.isnan(padded[:, 10:]))

    def test_full_size_array_is_unchanged_aside_from_dtype(self):
        """Test that an 80x80 array is returned unchanged, except for dtype conversion to float64."""
        arr = np.arange(80 * 80, dtype=np.float64).reshape(80, 80)
        padded = du._pad_to_80x80(arr)
        np.testing.assert_allclose(padded, arr)


class TestBuildCustomDtype:
    """Tests for the _build_custom_dtype function, which maps FITS column formats to NumPy dtypes."""

    @pytest.mark.parametrize("fmt,expected", [
        ("E", np.float32),
        ("D", np.float64),
        ("I", np.int16),
        ("J", np.int32),
        ("K", np.int64),
        ("L", np.bool_),
    ])
    def test_maps_known_fits_formats(self, fmt, expected):
        """Test that known FITS formats are correctly mapped to NumPy dtypes."""
        dtype = du._build_custom_dtype([_FakeColumn("col", fmt)])
        assert dtype["col"] == np.dtype(expected)

    def test_maps_character_string_format_to_fixed_length_string(self):
        """Test that a character string format is mapped to a fixed-length string dtype."""
        dtype = du._build_custom_dtype([_FakeColumn("name", "10A")])
        assert dtype["name"] == np.dtype("S10")

    def test_unsupported_format_raises_value_error(self):
        """Test that an unsupported FITS format raises a ValueError."""
        with pytest.raises(ValueError):
            du._build_custom_dtype([_FakeColumn("bad", "Z")])

    def test_preserves_column_order(self):
        """Test that the order of columns is preserved in the resulting dtype."""
        dtype = du._build_custom_dtype([_FakeColumn("a", "E"), _FakeColumn("b", "J")])
        assert dtype.names == ("a", "b")


class TestLoadSingleCutout:
    """Tests for the load_single_cutout function, which loads a single FITS cutout image and pads it to 80x80."""

    def test_loads_correctly_shaped_image_unchanged(self, tmp_path, logger):
        """Test that a correctly shaped 80x80 image is loaded unchanged."""
        data = np.random.default_rng(0).normal(size=(80, 80)).astype(np.float32)
        path = tmp_path / "cutout.fits"
        fits.PrimaryHDU(data=data).writeto(path)

        loaded = du.load_single_cutout(path, logger)

        np.testing.assert_allclose(loaded, data)
        assert loaded.dtype == np.float32

    def test_pads_undersized_image(self, tmp_path, logger):
        """Test that a smaller image is padded to 80x80 with NaN values."""
        data = np.ones((40, 40), dtype=np.float32)
        path = tmp_path / "small_cutout.fits"
        fits.PrimaryHDU(data=data).writeto(path)

        loaded = du.load_single_cutout(path, logger)

        assert loaded.shape == (80, 80)
        np.testing.assert_array_equal(loaded[:40, :40], data)
        assert np.all(np.isnan(loaded[40:, :]))

    def test_missing_file_returns_all_nan(self, tmp_path, logger):
        """Test that if the FITS file does not exist, an 80x80 array of NaN values is returned."""
        loaded = du.load_single_cutout(tmp_path / "does_not_exist.fits", logger)
        assert loaded.shape == (80, 80)
        assert np.all(np.isnan(loaded))


class TestLoadFitsCatalogue:
    """
    Tests for the load_fits_catalogue function, which loads a FITS catalogue and returns data, header, and columns.
    """

    def test_reads_back_data_header_and_columns(self, tmp_path):
        """Test that a FITS catalogue is read back correctly, returning the data, header, and columns."""
        col = fits.Column(name="flux", format="E", array=np.array([1.0, 2.0], dtype=np.float32))
        hdu = fits.BinTableHDU.from_columns([col])
        hdu.header["TESTKEY"] = "hello"
        path = tmp_path / "catalogue.fits"
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

        data, header, columns = du.load_fits_catalogue(path)

        np.testing.assert_allclose(data["flux"], [1.0, 2.0])
        assert header["TESTKEY"] == "hello"
        assert columns.names == ["flux"]


class TestSaveToFits:
    """Tests for the save_to_fits function, which saves catalogue information and cutout images to a FITS file."""

    def _make_cat_info(self):
        """Helper method to create a simple catalogue info array for testing."""
        col = fits.Column(name="flux", format="E", array=np.array([1.0, 2.0], dtype=np.float32))
        return fits.BinTableHDU.from_columns([col]).data

    def test_round_trips_catalogue_indices_and_images(self, tmp_path, logger):
        """
        Test that catalogue info, indices, and cutout images are saved to a FITS file and can be read back correctly.
        """
        cat_info = self._make_cat_info()
        pixel_values = [np.full((5, 5), 1.0, dtype=np.float32), np.full((5, 5), 2.0, dtype=np.float32)]
        indices = np.array([10, 20])
        save_path = tmp_path / "out.fits"

        du.save_to_fits(cat_info, pixel_values, indices, logger, save_path=save_path)

        with fits.open(save_path) as hdul:
            names = [hdu.name for hdu in hdul]
            assert "CATALOGUE_INFO" in names
            assert "CATALOGUE_INDEX" in names
            assert "CUTOUT_IMAGE0" in names
            assert "CUTOUT_IMAGE1" in names

            np.testing.assert_allclose(hdul["CATALOGUE_INFO"].data["flux"], [1.0, 2.0])
            np.testing.assert_array_equal(hdul["CATALOGUE_INDEX"].data["INDEX"], indices)
            np.testing.assert_allclose(hdul["CUTOUT_IMAGE0"].data, pixel_values[0])
            np.testing.assert_allclose(hdul["CUTOUT_IMAGE1"].data, pixel_values[1])

    def test_writes_wcs_header_keys_and_catalogue_index(self, tmp_path, logger):
        """Test that the WCS header keys and catalogue index are correctly written to the FITS file."""
        cat_info = self._make_cat_info()
        pixel_values = [np.zeros((5, 5), dtype=np.float32)]
        indices = np.array([0])
        save_path = tmp_path / "out.fits"

        du.save_to_fits(cat_info, pixel_values, indices, logger, save_path=save_path)

        with fits.open(save_path) as hdul:
            header = hdul["CUTOUT_IMAGE0"].header
            assert header["CTYPE1"] == "RA---SIN"
            assert header["CTYPE2"] == "DEC--SIN"
            assert header["CATIDX"] == 0


class TestSaveToHdf5:
    """Tests for the save_to_hdf5 function, which saves catalogue information and cutout images to an HDF5 file."""

    def _make_cat_info_and_columns(self):
        """Helper method to create a simple catalogue info array and its corresponding columns for testing."""
        col = fits.Column(name="flux", format="E", array=np.array([1.0, 2.0], dtype=np.float32))
        hdu = fits.BinTableHDU.from_columns([col])
        return hdu.data, hdu.columns

    def test_saves_without_custom_dtype(self, tmp_path, logger):
        """Test that catalogue info, pixel values, and indices are saved to an HDF5 file without a custom dtype."""
        cat_info, _ = self._make_cat_info_and_columns()
        pixel_values = np.stack([np.zeros((5, 5)), np.ones((5, 5))])
        indices = np.array([0, 1])
        save_path = tmp_path / "out.h5"

        du.save_to_hdf5(cat_info, pixel_values, indices, logger, cat_columns=None, save_path=save_path)

        with h5py.File(save_path, "r") as f:
            np.testing.assert_allclose(f["images"][:], pixel_values)
            np.testing.assert_array_equal(f["indices"][:], indices)
            np.testing.assert_allclose(f["cat_info"]["flux"][:], [1.0, 2.0])

    def test_saves_with_custom_dtype(self, tmp_path, logger):
        """Test that catalogue info is saved to an HDF5 file with a custom dtype derived from FITS columns."""
        cat_info, columns = self._make_cat_info_and_columns()
        pixel_values = np.stack([np.zeros((5, 5)), np.ones((5, 5))])
        indices = np.array([0, 1])
        save_path = tmp_path / "out_custom.h5"

        du.save_to_hdf5(cat_info, pixel_values, indices, logger, cat_columns=columns, save_path=save_path)

        with h5py.File(save_path, "r") as f:
            np.testing.assert_allclose(f["cat_info"]["flux"][:], [1.0, 2.0])
            assert f["cat_info"].dtype["flux"] == np.dtype(np.float32)
