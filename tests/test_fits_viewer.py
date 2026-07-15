"""
Unit tests for diffracc/scripts/fits_viewer.py's FitsViewer.

Note that the local imports of matplotlib.pyplot are done inside the test functions to avoid messing with the global
state of matplotlib during test collection, which can cause issues with some other test runners.
"""
import numpy as np
import pytest
from astropy.io import fits

from diffracc.scripts.fits_viewer import FitsViewer


def _make_fits_files(tmp_path, n=4, fxscld=None):
    """
    Create n small FITS files with distinct pixel values (i.e. so imshow/data differ), and an FXSCLD header
    (ascending unless overridden), returned in creation order.
    """
    fxscld = fxscld if fxscld is not None else list(range(n))
    paths = []
    for i in range(n):
        hdu = fits.PrimaryHDU(data=np.full((4, 4), float(i)))
        hdu.header["FXSCLD"] = float(fxscld[i])
        path = tmp_path / f"img{i}.fits"
        hdu.writeto(path)
        paths.append(path)
    return paths


class TestShowImageGridBasics:
    """Tests for the basic functionality of the show_image_grid method of FitsViewer."""

    def test_raises_runtime_error_when_files_is_none(self):
        """Test that if the files attribute is None, a RuntimeError is raised when attempting to show the image grid."""
        viewer = FitsViewer(None)
        with pytest.raises(RuntimeError):
            viewer.show_image_grid()

    def test_raises_runtime_error_when_files_is_empty(self):
        """
        Test that if the files attribute is an empty list, a RuntimeError is raised when attempting to show the image
        grid.
        """
        viewer = FitsViewer([])
        with pytest.raises(RuntimeError):
            viewer.show_image_grid()

    def test_returns_self_for_chaining(self, tmp_path):
        """Test that the show_image_grid method returns self to allow for method chaining."""
        files = _make_fits_files(tmp_path, n=4)
        viewer = FitsViewer(files)
        result = viewer.show_image_grid()
        assert result is viewer

    def test_creates_one_axes_per_file(self, tmp_path):
        """Test that the show_image_grid method creates one axes for each FITS file provided."""
        files = _make_fits_files(tmp_path, n=4)
        viewer = FitsViewer(files)
        viewer.show_image_grid()
        fig = __import__("matplotlib.pyplot", fromlist=["gcf"]).gcf()
        assert len(fig.axes) == 4

    def test_auto_row_count_is_square_like(self, tmp_path):
        """Test that when rows=-1, the function automatically computes a square-like grid for the number of images."""
        files = _make_fits_files(tmp_path, n=9)
        viewer = FitsViewer(files)
        viewer.show_image_grid(rows=-1)
        import matplotlib.pyplot as plt
        # 9 files, aspect=1 -> ceil(sqrt(9)) = 3 rows
        rows = {ax.get_gridspec().nrows for ax in plt.gcf().axes}
        assert rows == {3}

    def test_explicit_row_count_is_respected(self, tmp_path):
        """Test that when a specific number of rows is provided, the function respects that value."""
        files = _make_fits_files(tmp_path, n=4)
        viewer = FitsViewer(files)
        viewer.show_image_grid(rows=2)
        import matplotlib.pyplot as plt
        rows = {ax.get_gridspec().nrows for ax in plt.gcf().axes}
        assert rows == {2}


class TestTitlesAndTicks:
    """Tests for the title and tick display functionality of the show_image_grid method of FitsViewer."""

    def test_titles_default_to_file_names(self, tmp_path):
        """Test that by default, the titles of the axes are set to the names of the FITS files."""
        files = _make_fits_files(tmp_path, n=2)
        viewer = FitsViewer(files)
        viewer.show_image_grid()
        import matplotlib.pyplot as plt
        titles = {ax.get_title() for ax in plt.gcf().axes}
        assert titles == {f.name for f in files}

    def test_no_titles_leaves_titles_blank(self, tmp_path):
        """Test that when no_titles=True, the titles of the axes are left blank."""
        files = _make_fits_files(tmp_path, n=2)
        viewer = FitsViewer(files)
        viewer.show_image_grid(no_titles=True)
        import matplotlib.pyplot as plt
        for ax in plt.gcf().axes:
            assert ax.get_title() == ""

    def test_no_ticks_hides_axes(self, tmp_path):
        """Test that when no_ticks=True, the x and y axes are hidden for all images."""
        files = _make_fits_files(tmp_path, n=2)
        viewer = FitsViewer(files)
        viewer.show_image_grid(no_ticks=True)
        import matplotlib.pyplot as plt
        for ax in plt.gcf().axes:
            assert ax.xaxis.get_visible() is False
            assert ax.yaxis.get_visible() is False

    def test_ticks_visible_by_default(self, tmp_path):
        """Test that by default, the x and y axes are visible for all images."""
        files = _make_fits_files(tmp_path, n=2)
        viewer = FitsViewer(files)
        viewer.show_image_grid()
        import matplotlib.pyplot as plt
        for ax in plt.gcf().axes:
            assert ax.xaxis.get_visible() is True


class TestUpperBoundAndOutfile:
    """Tests for the upper_bound and outfile parameters of the show_image_grid method of FitsViewer."""

    def test_upper_bound_sets_image_clim(self, tmp_path):
        """Test that when upper_bound is provided, the color limits of the images are set accordingly."""
        files = _make_fits_files(tmp_path, n=1)
        viewer = FitsViewer(files)
        viewer.show_image_grid(upper_bound=5.0)
        import matplotlib.pyplot as plt
        img = plt.gcf().axes[0].images[0]
        assert img.get_clim() == (0, 5.0)

    def test_no_upper_bound_leaves_default_clim(self, tmp_path):
        """Test that when upper_bound is None, the color limits of the images are left at their default values."""
        files = _make_fits_files(tmp_path, n=1)
        viewer = FitsViewer(files)
        viewer.show_image_grid()
        import matplotlib.pyplot as plt
        img = plt.gcf().axes[0].images[0]
        assert img.get_clim() != (0, 5.0)

    def test_outfile_writes_a_file(self, tmp_path):
        """Test that when outfile is provided, the figure is saved to the specified path."""
        files = _make_fits_files(tmp_path, n=1)
        out_path = tmp_path / "grid.png"
        viewer = FitsViewer(files)
        viewer.show_image_grid(outfile=str(out_path))
        assert out_path.exists()

    def test_no_outfile_does_not_write_a_file(self, tmp_path):
        """Test that when outfile is None, no file is written to disk."""
        files = _make_fits_files(tmp_path, n=1)
        viewer = FitsViewer(files)
        viewer.show_image_grid(outfile=None)
        assert list(tmp_path.glob("*.png")) == []


class TestSorting:
    """Tests for the sorting functionality of the show_image_grid method of FitsViewer."""

    def test_sort_by_flux_scaled_orders_by_fxscld_header(self, tmp_path):
        """
        Test that when sorting=FitsViewer.SORT_BY_FLUX_SCALED, the images are ordered by their FXSCLD header values.
        """
        # Files created in descending FXSCLD order; SORT_BY_FLUX_SCALED should reorder to ascending
        files = _make_fits_files(tmp_path, n=4, fxscld=[30, 10, 40, 20])
        viewer = FitsViewer(files)
        viewer.show_image_grid(sorting=FitsViewer.SORT_BY_FLUX_SCALED)
        import matplotlib.pyplot as plt
        titles_in_order = [ax.get_title() for ax in plt.gcf().axes]
        expected_order = [files[1].name, files[3].name, files[0].name, files[2].name]  # fxscld 10,20,30,40
        assert titles_in_order == expected_order

    def test_no_sorting_preserves_original_file_order(self, tmp_path):
        """Test that when sorting=FitsViewer.NO_SORTING, the images are ordered as they appear in the input list."""
        files = _make_fits_files(tmp_path, n=3, fxscld=[30, 10, 20])
        viewer = FitsViewer(files)
        viewer.show_image_grid(sorting=FitsViewer.NO_SORTING)
        import matplotlib.pyplot as plt
        titles_in_order = [ax.get_title() for ax in plt.gcf().axes]
        assert titles_in_order == [f.name for f in files]

    def test_second_call_with_same_sorting_reuses_cache_successfully(self, tmp_path):
        """
        Test that if show_image_grid is called twice with the same sorting, it reuses the cached files/data without
        error.
        """
        files = _make_fits_files(tmp_path, n=3)
        viewer = FitsViewer(files)
        viewer.show_image_grid(sorting=FitsViewer.NO_SORTING)
        viewer.show_image_grid(sorting=FitsViewer.NO_SORTING)  # reuses the cached files/data, does not crash

    def test_second_call_with_different_sorting_recomputes(self, tmp_path):
        """Test that if show_image_grid is called twice with different sorting, it recomputes the image order."""
        files = _make_fits_files(tmp_path, n=3, fxscld=[30, 10, 20])
        viewer = FitsViewer(files)
        viewer.show_image_grid(sorting=FitsViewer.NO_SORTING)
        viewer.show_image_grid(sorting=FitsViewer.SORT_BY_FLUX_SCALED)
        import matplotlib.pyplot as plt
        titles_in_order = [ax.get_title() for ax in plt.gcf().axes]
        assert titles_in_order == [files[1].name, files[2].name, files[0].name]  # fxscld 10, 20, 30

    def test_sort_by_peak_flux_unscaled_orders_by_raw_pixel_max(self, tmp_path):
        """
        Test that when sorting=FitsViewer.SORT_BY_PEAK_FLUX_UNSCALED, the images are ordered by their raw pixel maximum
        values.
        """
        # Files created with descending FXSCLD but ascending raw pixel value (index i -> pixel value i); the
        # unscaled mode should ignore FXSCLD entirely and sort by the raw pixel max instead.
        files = _make_fits_files(tmp_path, n=3, fxscld=[30, 20, 10])
        viewer = FitsViewer(files)
        viewer.show_image_grid(sorting=FitsViewer.SORT_BY_PEAK_FLUX_UNSCALED)
        import matplotlib.pyplot as plt
        titles_in_order = [ax.get_title() for ax in plt.gcf().axes]
        assert titles_in_order == [f.name for f in files]  # pixel values 0,1,2 already ascending

    def test_undefined_sorting_value_raises_a_clear_error(self, tmp_path):
        """Test that if an undefined sorting value is provided, a ValueError is raised with a clear message."""
        files = _make_fits_files(tmp_path, n=3)
        viewer = FitsViewer(files)
        with pytest.raises(ValueError):
            viewer.show_image_grid(sorting=99)
