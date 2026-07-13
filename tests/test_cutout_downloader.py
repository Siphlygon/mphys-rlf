"""
Unit tests for diffracc/data/cutout_downloader.py's CutoutDownloader.

Built against the cutout_downloader_factory fixture (tests/conftest.py), which constructs instances against a temp
FOLDER_SIZE-only config. No real network requests are made anywhere here - _get_cutout/download_all_cutouts are
always mocked or monkeypatched out.
"""
import time

import numpy as np
import pytest
from astropy.io import fits

from diffracc.data import cutout_downloader as cdl


class TestReadPositions:
    """Tests that CutoutDownloader._read_positions correctly reads RA/Dec pairs from a text file."""

    def test_parses_one_ra_dec_pair_per_line(self, cutout_downloader_factory, tmp_path):
        """Test that _read_positions correctly parses a text file with one RA/Dec pair per line."""
        downloader = cutout_downloader_factory()
        positions_path = tmp_path / "positions.txt"
        positions_path.write_text("1.5 2.5\n3.0 4.0\n", encoding="utf-8")

        positions = downloader._read_positions(file_path=positions_path)

        assert positions == [(1.5, 2.5), (3.0, 4.0)]

    def test_raises_on_missing_file(self, cutout_downloader_factory, tmp_path):
        """Test that _read_positions raises an exception when the specified file does not exist."""
        downloader = cutout_downloader_factory()
        with pytest.raises(Exception):
            downloader._read_positions(file_path=tmp_path / "does_not_exist.txt")


class TestMakeFolder:
    """Tests that CutoutDownloader._make_folder correctly creates a folder for a given index range."""

    def test_folder_name_matches_index_range(self, cutout_downloader_factory, tmp_path):
        """Test that _make_folder creates a folder with the correct name based on the index range."""
        downloader = cutout_downloader_factory()  # FOLDER_SIZE=100
        folder_path = downloader._make_folder(2, directory_path=tmp_path)
        assert folder_path.name == "200-299"
        assert folder_path.exists()

    def test_first_folder_starts_at_zero(self, cutout_downloader_factory, tmp_path):
        """Test that the first folder created by _make_folder starts at index 0."""
        downloader = cutout_downloader_factory()
        folder_path = downloader._make_folder(0, directory_path=tmp_path)
        assert folder_path.name == "0-99"

    def test_does_not_error_if_folder_already_exists(self, cutout_downloader_factory, tmp_path):
        """Test that _make_folder does not raise an exception if the folder already exists."""
        downloader = cutout_downloader_factory()
        downloader._make_folder(0, directory_path=tmp_path)
        # Calling again for the same folder_num must not raise.
        folder_path = downloader._make_folder(0, directory_path=tmp_path)
        assert folder_path.exists()


class TestRateLimit:
    """Tests that CutoutDownloader._rate_limit correctly implements rate limiting based on recent errors."""

    def test_no_sleep_when_no_recent_errors(self, cutout_downloader_factory, monkeypatch):
        """Test that _rate_limit does not sleep when there are no recent errors."""
        downloader = cutout_downloader_factory()
        downloader.recent_errors = 0
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        downloader._rate_limit()

        assert sleeps == []

    def test_short_sleep_for_a_few_recent_errors(self, cutout_downloader_factory, monkeypatch):
        """Test that _rate_limit sleeps for a short duration when there are a few recent errors."""
        downloader = cutout_downloader_factory()
        downloader.recent_errors = 2
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        downloader._rate_limit()

        assert sleeps == [0.15]

    def test_longer_sleep_when_error_count_exceeds_three(self, cutout_downloader_factory, monkeypatch):
        """Test that _rate_limit sleeps for a longer duration when there are more than three recent errors."""
        downloader = cutout_downloader_factory()
        downloader.recent_errors = 5
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        downloader._rate_limit()

        assert sleeps == [0.5]


class TestDownloadOne:
    """
    Tests that CutoutDownloader._download_one correctly handles downloading a single cutout, including skipping existing
    files and handling errors.
    """

    def test_skips_download_if_file_already_exists(self, cutout_downloader_factory, tmp_path, monkeypatch):
        """Test that _download_one skips downloading if the cutout file already exists."""
        downloader = cutout_downloader_factory()
        path = tmp_path / "cutout0.fits"
        path.write_bytes(b"already downloaded")

        def _unexpected_get_cutout(*args, **kwargs):
            raise AssertionError("_get_cutout should not be called when the file already exists")
        monkeypatch.setattr(downloader, "_get_cutout", _unexpected_get_cutout)

        i, err = downloader._download_one((0, 1.0, 2.0, path))

        assert (i, err) == (0, "exists")

    def test_success_returns_none_error_and_decrements_recent_errors(self, cutout_downloader_factory,
                                                                     tmp_path, monkeypatch):
        """Test that _download_one returns None for the error and decrements recent_errors on a successful download."""
        downloader = cutout_downloader_factory()
        downloader.recent_errors = 2
        monkeypatch.setattr(downloader, "_get_cutout", lambda *a, **k: None)
        path = tmp_path / "cutout0.fits"

        i, err = downloader._download_one((0, 1.0, 2.0, path))

        assert (i, err) == (0, None)
        assert downloader.recent_errors == 1

    def test_failure_returns_error_message_and_increments_recent_errors(self, cutout_downloader_factory,
                                                                        tmp_path,monkeypatch):
        """Test that _download_one returns an error message and increments recent_errors on a failed download."""
        downloader = cutout_downloader_factory()
        downloader.recent_errors = 0

        def _fail(*args, **kwargs):
            raise RuntimeError("Status 500")
        monkeypatch.setattr(downloader, "_get_cutout", _fail)
        path = tmp_path / "cutout0.fits"

        i, err = downloader._download_one((0, 1.0, 2.0, path))

        assert i == 0
        assert err == "Status 500"
        assert downloader.recent_errors == 1


class TestLoadSingleCutout:
    """Tests that CutoutDownloader._test_load_single_cutout correctly identifies valid and corrupted FITS files."""

    def test_valid_fits_file_returns_true(self, cutout_downloader_factory, tmp_path):
        """Test that _test_load_single_cutout returns True for a valid FITS file and does not delete it."""
        downloader = cutout_downloader_factory()
        path = tmp_path / "cutout0.fits"
        fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(path)

        assert downloader._test_load_single_cutout(path) is True
        assert path.exists()

    def test_corrupted_file_returns_false_and_deletes_it(self, cutout_downloader_factory, tmp_path):
        """Test that _test_load_single_cutout returns False for a corrupted FITS file and deletes it."""
        downloader = cutout_downloader_factory()
        path = tmp_path / "cutout1.fits"
        path.write_bytes(b"this is not a valid fits file")

        assert downloader._test_load_single_cutout(path) is False
        assert not path.exists()


class TestVerifyDownloads:
    """
    Tests that CutoutDownloader.verify_downloads correctly identifies missing and corrupted cutouts and requests their
    redownload.
    """

    def test_identifies_missing_and_corrupted_cutouts_and_requests_their_redownload(
        self, cutout_downloader_factory, tmp_path, monkeypatch
    ):
        """Test that verify_downloads identifies missing and corrupted cutouts and requests their redownload."""
        downloader = cutout_downloader_factory()
        positions = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]  # indices 0, 1, 2
        monkeypatch.setattr(downloader, "_read_positions", lambda: positions)

        download_dir = tmp_path / "cutouts"
        download_dir.mkdir()
        fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(download_dir / "cutout0.fits")  # valid, present
        # index 1 is entirely missing
        (download_dir / "cutout2.fits").write_bytes(b"not a real fits file")  # present but corrupted

        redownload_calls = []
        monkeypatch.setattr(downloader, "download_all_cutouts",
                            lambda custom_positions=None, **kwargs: redownload_calls.append(custom_positions))

        downloader.verify_downloads(download_path=download_dir)

        # the corrupted file should have been deleted by _test_load_single_cutout
        assert not (download_dir / "cutout2.fits").exists()
        assert len(redownload_calls) == 1
        requested = {tuple(p) for p in redownload_calls[0]}
        assert requested == {(2.0, 20.0), (3.0, 30.0)}  # positions for indices 1 (missing) and 2 (corrupted)

    def test_no_redownload_requested_when_everything_is_present_and_valid(
        self, cutout_downloader_factory, tmp_path, monkeypatch
    ):
        """Test that verify_downloads does not request any redownload when all cutouts are present and valid."""
        downloader = cutout_downloader_factory()
        positions = [(1.0, 10.0), (2.0, 20.0)]
        monkeypatch.setattr(downloader, "_read_positions", lambda: positions)

        download_dir = tmp_path / "cutouts"
        download_dir.mkdir()
        fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(download_dir / "cutout0.fits")
        fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(download_dir / "cutout1.fits")

        def _unexpected_redownload(*args, **kwargs):
            raise AssertionError("download_all_cutouts should not be called when nothing is missing/corrupted")
        monkeypatch.setattr(downloader, "download_all_cutouts", _unexpected_redownload)

        downloader.verify_downloads(download_path=download_dir)  # must not raise
