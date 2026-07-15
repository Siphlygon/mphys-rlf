"""Unit tests for diffracc/scripts/size_comp.py's main()."""
import argparse

import numpy as np
import pytest
from astropy.io import fits

from diffracc.scripts import size_comp
from diffracc.utils import paths


@pytest.fixture
def fits_subdir(tmp_path, monkeypatch):
    """Monkeypatch paths.FITS_PARENT to a temp root and return the (empty) subdir to fill with FITS files."""
    fits_root = tmp_path / "fits_root"
    monkeypatch.setattr(paths, "FITS_PARENT", fits_root)
    subdir = fits_root / "generated"
    subdir.mkdir(parents=True)
    return subdir


def _write_fits_file(subdir, name, data, lasize):
    """Helper function to write a FITS file with given data and LASIZE header."""
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.header["LASIZE"] = float(lasize)
    hdu.writeto(subdir / name)


def _expected_las_from_img(data):
    """Reproduce main()'s own proxy formula directly, for hand-verifiable test data."""
    return np.sqrt((data > np.median(data) + 3 * np.std(data)).sum())


class TestMain:
    """Unit tests for the main function of size_comp.py."""

    def _args(self, subdir="generated"):
        """Helper function to create an argparse.Namespace with default values, overridden by any provided arguments."""
        return argparse.Namespace(subdir=subdir)

    def test_runs_without_error_and_saves_a_figure(self, fits_subdir, tmp_path, monkeypatch):
        """Test that main() runs without error and saves a figure to the current directory."""
        data = np.zeros((5, 5))
        data[0, 0] = 1000.0
        _write_fits_file(fits_subdir, "img0.fits", data, lasize=3.0)

        monkeypatch.chdir(tmp_path)
        size_comp.main(self._args())

        assert (tmp_path / "size_comp.png").exists()

    def test_scatter_matches_lasize_header_and_proxy_formula(self, fits_subdir, tmp_path, monkeypatch):
        """Test that the scatter plot points match the LASIZE header and the proxy formula from the image data."""
        data1 = np.zeros((6, 6))
        data1[0, 0] = 1000.0  # single outlier pixel
        data2 = np.zeros((6, 6))
        data2[0, :4] = 1000.0  # four outlier pixels
        _write_fits_file(fits_subdir, "img0.fits", data1, lasize=2.0)
        _write_fits_file(fits_subdir, "img1.fits", data2, lasize=5.0)

        monkeypatch.chdir(tmp_path)
        size_comp.main(self._args())

        import matplotlib.pyplot as plt
        offsets = plt.gca().collections[0].get_offsets()
        by_lasize = {round(x): y for x, y in offsets}
        assert by_lasize[2] == pytest.approx(_expected_las_from_img(data1))
        assert by_lasize[5] == pytest.approx(_expected_las_from_img(data2))

    def test_reads_files_recursively_from_nested_subdirectories(self, fits_subdir, tmp_path, monkeypatch):
        """Test that main() reads FITS files recursively from nested subdirectories."""
        nested = fits_subdir / "nested"
        nested.mkdir()
        data = np.zeros((5, 5))
        data[0, 0] = 1000.0
        _write_fits_file(nested, "deep.fits", data, lasize=1.0)

        monkeypatch.chdir(tmp_path)
        size_comp.main(self._args())

        import matplotlib.pyplot as plt
        offsets = plt.gca().collections[0].get_offsets()
        assert len(offsets) == 1

    def test_empty_subdir_raises_a_clear_runtime_error(self, fits_subdir, tmp_path, monkeypatch):
        """Test that main() raises a RuntimeError with a clear message when the subdir contains no FITS files."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="No FITS files found"):
            size_comp.main(self._args())
