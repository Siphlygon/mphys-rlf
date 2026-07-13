"""
Unit tests for diffracc/data/catalogue_downloader.py's CatalogueDownloader.

_create_stripped_catalogue/_get_positions_from_hardcastle/_write_positions_to_file are exercised against small
synthetic FITS/text files in tmp_path (no real catalogue). download_catalogue's network call is mocked via a fake
requests.get - no real HTTP request is ever made.
"""
import numpy as np
import pytest
from astropy.io import fits

from diffracc.data import catalogue_downloader as cd


class TestCataloguesSchema:
    """Test that each catalogue entry in CATALOGUES has the required keys and valid values."""

    def test_every_catalogue_has_file_name_and_url(self):
        """
        Test that every catalogue entry in CATALOGUES has a 'file_name' and a 'url', and that the URL starts with
        'https://'.
        """
        for name, entry in cd.CATALOGUES.items():
            assert "file_name" in entry, name
            assert "url" in entry, name
            assert entry["url"].startswith("https://"), name


@pytest.fixture
def downloader():
    """Fixture that returns a CatalogueDownloader instance for use in tests."""
    return cd.CatalogueDownloader()


def _write_catalogue_fits(path, columns: dict):
    """Helper function to write a FITS file with the specified columns for testing purposes."""
    cols = fits.ColDefs([
        fits.Column(name=name, format='E' if arr.dtype.kind == 'f' else 'L', array=arr)
        for name, arr in columns.items()
    ])
    hdu = fits.BinTableHDU.from_columns(cols)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


class TestCreateStrippedCatalogue:
    """
    Tests that CatalogueDownloader._create_stripped_catalogue correctly creates a stripped FITS file with only the
    desired columns.
    """

    def test_keeps_only_desired_columns(self, downloader, tmp_path):
        """Test that _create_stripped_catalogue keeps only the desired columns in the output FITS file."""
        raw_path = tmp_path / "raw.fits"
        _write_catalogue_fits(raw_path, {
            "RA": np.array([1.0, 2.0], dtype=np.float32),
            "DEC": np.array([3.0, 4.0], dtype=np.float32),
            "Total_flux": np.array([5.0, 6.0], dtype=np.float32),
            "not_a_desired_column": np.array([7.0, 8.0], dtype=np.float32),
        })
        stripped_path = tmp_path / "stripped.fits"

        downloader._create_stripped_catalogue(file_path=stripped_path, catalogue_path=raw_path)

        with fits.open(stripped_path) as hdul:
            names = set(hdul[1].columns.names)
        assert names == {"RA", "DEC", "Total_flux"}

    def test_skips_creation_if_stripped_file_already_exists(self, downloader, tmp_path):
        """Test that _create_stripped_catalogue skips creation if the stripped file already exists."""
        stripped_path = tmp_path / "stripped.fits"
        stripped_path.write_bytes(b"already here")

        # catalogue_path deliberately doesn't exist - if the method tried to actually process it, this would raise.
        downloader._create_stripped_catalogue(file_path=stripped_path, catalogue_path=tmp_path / "missing.fits")

        assert stripped_path.read_bytes() == b"already here"

    def test_raises_on_missing_raw_catalogue(self, downloader, tmp_path):
        """Test that _create_stripped_catalogue raises an exception when the raw catalogue file is missing."""
        with pytest.raises(Exception):
            downloader._create_stripped_catalogue(
                file_path=tmp_path / "stripped.fits", catalogue_path=tmp_path / "missing.fits")


class TestGetPositionsFromHardcastle:
    """
    Tests that CatalogueDownloader._get_positions_from_hardcastle correctly reads RA/DEC positions from a stripped FITS
    file and returns only the resolved sources.
    """

    def test_raises_on_missing_catalogue_file(self, downloader, tmp_path):
        """Test that _get_positions_from_hardcastle raises an exception when the catalogue file is missing."""
        with pytest.raises(Exception):
            downloader._get_positions_from_hardcastle(catalogue_path=tmp_path / "missing.fits")

    def test_returns_only_resolved_source_positions(self, downloader, tmp_path):
        """Test that _get_positions_from_hardcastle returns only the positions of resolved sources."""
        catalogue_path = tmp_path / "catalogue.fits"
        _write_catalogue_fits(catalogue_path, {
            "RA": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "DEC": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "Resolved": np.array([True, False, True]),
        })

        positions = downloader._get_positions_from_hardcastle(catalogue_path=catalogue_path)

        assert positions == [(10.0, 1.0), (30.0, 3.0)]


class TestWritePositionsToFile:
    """Tests that CatalogueDownloader._write_positions_to_file correctly writes RA/DEC pairs to a text file."""

    def test_writes_one_ra_dec_pair_per_line(self, downloader, tmp_path):
        """Test that _write_positions_to_file writes one RA/DEC pair per line in the output text file."""
        out_path = tmp_path / "positions.txt"
        downloader._write_positions_to_file([(1.5, 2.5), (3.0, 4.0)], positions_path=out_path)

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert lines == ["1.5 2.5", "3.0 4.0"]

    def test_logs_error_instead_of_raising_for_an_unwritable_path(self, downloader, tmp_path):
        """Test that _write_positions_to_file logs an error instead of raising an exception for an unwritable path."""
        # A directory as the "file" path can never be opened for writing - the method should catch this rather
        # than propagate the exception.
        bad_path = tmp_path  # a directory, not a file
        downloader._write_positions_to_file([(1.0, 2.0)], positions_path=bad_path)  # must not raise


class _FakeResponse:
    """A fake response object to simulate requests.get responses for testing purposes."""
    def __init__(self, status_code=200, content=b"fake fits bytes"):
        self.status_code = status_code
        self._content = content

    def iter_content(self, chunk_size=8192):
        yield self._content


class TestDownloadCatalogue:
    """Tests that CatalogueDownloader.download_catalogue correctly downloads a catalogue, saves it, and strips it."""

    def test_skips_download_and_strips_if_raw_file_already_exists(self, downloader, tmp_path, monkeypatch):
        """Test that download_catalogue skips downloading and strips the catalogue if the raw file already exists."""
        calls = []
        monkeypatch.setattr(downloader, "_create_stripped_catalogue",
                            lambda file_path, catalogue_path: calls.append((file_path, catalogue_path)))
        raw_path = tmp_path / "raw.fits"
        raw_path.write_bytes(b"already downloaded")
        stripped_path = tmp_path / "stripped.fits"

        def _unexpected_get(*args, **kwargs):
            raise AssertionError("requests.get should not be called when the raw file already exists")
        monkeypatch.setattr(cd.requests, "get", _unexpected_get)

        downloader.download_catalogue("hardcastle2023", raw_catalogue_path=raw_path,
                                      stripped_catalogue_path=stripped_path)

        assert calls == [(stripped_path, raw_path)]

    def test_invalid_catalogue_name_returns_without_downloading(self, downloader, tmp_path, monkeypatch):
        """Test that download_catalogue returns without downloading when given an invalid catalogue name."""
        def _unexpected_get(*args, **kwargs):
            raise AssertionError("requests.get should not be called for an unknown catalogue name")
        monkeypatch.setattr(cd.requests, "get", _unexpected_get)

        downloader.download_catalogue("not_a_real_catalogue",
                                      raw_catalogue_path=tmp_path / "raw.fits",
                                      stripped_catalogue_path=tmp_path / "stripped.fits")

        assert not (tmp_path / "raw.fits").exists()

    def test_successful_download_writes_file_and_strips_it(self, downloader, tmp_path, monkeypatch):
        """Test that download_catalogue successfully downloads a catalogue, writes it to a file, and strips it."""
        monkeypatch.setattr(cd.requests, "get", lambda *a, **k: _FakeResponse(200, b"fake fits bytes"))
        calls = []
        monkeypatch.setattr(downloader, "_create_stripped_catalogue",
                            lambda file_path, catalogue_path: calls.append((file_path, catalogue_path)))
        raw_path = tmp_path / "raw.fits"
        stripped_path = tmp_path / "stripped.fits"

        downloader.download_catalogue("hardcastle2023", raw_catalogue_path=raw_path,
                                      stripped_catalogue_path=stripped_path)

        assert raw_path.read_bytes() == b"fake fits bytes"
        assert calls == [(stripped_path, raw_path)]

    def test_failed_download_raises_runtime_error(self, downloader, tmp_path, monkeypatch):
        """Test that download_catalogue raises a RuntimeError when the download fails (non-200 status code)."""
        monkeypatch.setattr(cd.requests, "get", lambda *a, **k: _FakeResponse(404))

        with pytest.raises(RuntimeError):
            downloader.download_catalogue("hardcastle2023",
                                          raw_catalogue_path=tmp_path / "raw.fits",
                                          stripped_catalogue_path=tmp_path / "stripped.fits")
        assert not (tmp_path / "raw.fits").exists()
