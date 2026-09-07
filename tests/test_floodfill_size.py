"""
Unit tests for diffracc/completeness/floodfill_size.py, the faithful port of the LoTSS-DR2 flood-fill sizing.

The geometry helpers (`_mask_ellipse`, `largest_pixel_separation`, `_beam_area_pixels`, `_bad_image`) are tested against
hand-computable values, and `flood_mask` is tested on synthetic blobs (connectivity, keeping the source island,
excluding a neighbour). Everything runs on small in-memory arrays - no PyBDSF, no catalogue files. The size-selection
rule that combines this with the component-based size lives in `AngularSizeFinder.select_angular_size` and is tested in
`test_angular_size_finder.py`.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from diffracc.completeness import floodfill_size as ff


def _header(ny=80, nx=80, pix_arcsec=1.5, beam_arcsec=6.0):
    """A minimal 2D FITS header with a SIN WCS, matching the dr2 cutout geometry (1.5"/pixel, 6" circular beam)."""
    pix = pix_arcsec / 3600.0
    h = fits.Header()
    for k, v in {'NAXIS': 2, 'NAXIS1': nx, 'NAXIS2': ny, 'CRPIX1': nx // 2, 'CRPIX2': ny // 2,
                 'CRVAL1': 180.0, 'CRVAL2': 30.0, 'CDELT1': -pix, 'CDELT2': pix,
                 'BMAJ': beam_arcsec / 3600.0, 'BMIN': beam_arcsec / 3600.0, 'BPA': 0.0}.items():
        h[k] = v
    h['CTYPE1'] = 'RA---SIN'
    h['CTYPE2'] = 'DEC--SIN'
    return h


def _comp_at(header, px, py, maj_arcsec, min_arcsec=None, pa=0.0):
    """Build a one-row component array `(ra, dec, maj_deg, min_deg, pa)` for the ellipse centred on pixel (px, py)."""
    if min_arcsec is None:
        min_arcsec = maj_arcsec
    ra, dec = WCS(header).celestial.wcs_pix2world([[px, py]], 1)[0]
    return np.array([[ra, dec, maj_arcsec / 3600.0, min_arcsec / 3600.0, pa]])


class TestMaskEllipse:
    """The ellipse indicator, pinned to the (deliberately non-standard) LoTSS normalisation."""

    def test_circle_area_matches_pi_r_squared(self):
        """Test that the LoTSS normalisation of the ellipse mask gives the expected area for a circle."""
        d = 40.0  # full axis in pixels -> radius 20
        mask = ff._mask_ellipse((120, 120), 60.0, 60.0, d, d, 0.0)
        assert mask.sum() == pytest.approx(np.pi * (d / 2.0) ** 2, rel=2e-2)

    def test_minor_axis_extent_is_half_minpix(self):
        """
        Test that the LoTSS normalisation of the ellipse mask gives the expected minor-axis extent for a
        highly-elongated ellipse.
        """
        mask = ff._mask_ellipse((120, 120), 60.0, 60.0, 60.0, 20.0, 0.0)  # pa=0 -> +90 internally
        # column extent through the centre row
        col = np.nonzero(mask[60, :])[0]
        row = np.nonzero(mask[:, 60])[0]
        # one direction spans minpix (=20 -> ~20px), the other the geometric-mean major (sqrt(60*20)=~34.6 -> ~34px)
        short = min(np.ptp(col), np.ptp(row))
        long = max(np.ptp(col), np.ptp(row))
        assert short == pytest.approx(20, abs=2)
        assert long == pytest.approx(np.sqrt(60.0 * 20.0), abs=2)


class TestLargestPixelSeparation:
    """Unit tests for the function that computes the max pairwise pixel distance of the non-NaN entries."""

    def test_two_points(self):
        """Test that the largest separation of two points is the Euclidean distance between them."""
        a = np.full((10, 10), np.nan)
        a[2, 1] = 1.0
        a[6, 4] = 1.0  # dx=3, dy=4 -> 5
        assert ff.largest_pixel_separation(a) == pytest.approx(5.0)

    def test_fewer_than_two_returns_zero(self):
        """Test that the largest separation of fewer than two points is zero."""
        a = np.full((5, 5), np.nan)
        assert ff.largest_pixel_separation(a) == 0.0
        a[2, 2] = 1.0
        assert ff.largest_pixel_separation(a) == 0.0


class TestBeamArea:
    """Unit tests for the beam area in pixels, 2*pi*bmaj*bmin/(2 sqrt(2 ln2))^2 with axes in pixels."""

    def test_matches_formula(self):
        """
        Test that the beam area in pixels matches the expected formula for a circular 6" beam on a 1.5"/pixel image.
        """
        h = _header(beam_arcsec=6.0)  # 6"/1.5" = 4 px per axis
        gfac = 2.0 * np.sqrt(2.0 * np.log(2.0))
        assert ff._beam_area_pixels(h) == pytest.approx(2.0 * np.pi * 4.0 * 4.0 / gfac ** 2)


class TestBadImage:
    """Unit tests for the Bad_image flag: fewer than five fully-neighboured positive pixels."""

    def test_tiny_region_is_bad(self):
        """Test that a single positive pixel is flagged as a bad image."""
        a = np.full((20, 20), np.nan)
        a[10, 10] = 1.0
        assert ff._bad_image(a) is True

    def test_solid_block_is_good(self):
        """Test that a 6x6 block of positive pixels is not flagged as a bad image."""
        a = np.full((20, 20), np.nan)
        a[8:14, 8:14] = 1.0  # 6x6 block -> a 4x4 interior of fully-neighboured pixels = 16 >= 5
        assert ff._bad_image(a) is False


class TestFloodMask:
    """Unit tests for the flood mask function, which performs flood-fill on a thresholded image."""

    def _blob(self, shape, cx, cy, amp, s):
        """Create a 2D Gaussian blob for testing, with the given centre, amplitude, and sigma."""
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        return amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2)))

    def test_keeps_source_island_only(self):
        """Test that a neighbouring island is excluded if it is not included in the component list (not overlapping)."""
        h = _header()
        img = self._blob((80, 80), 30, 40, 0.05, 3.0) + self._blob((80, 80), 60, 40, 0.05, 3.0)
        include = _comp_at(h, 30, 40, 10.0)
        _, flooded = ff.flood_mask(img, h, include, np.empty((0, 5)), threshold=1e-3)
        xs = np.nonzero(~np.isnan(flooded))[1]
        assert xs.max() < 45  # only the left (source) island survives; the right neighbour is gone

    def test_exclusion_shrinks_a_merged_neighbour(self):
        """
        Test that a neighbouring island is shrunk if it is included in the component list but also excluded by the
        ellipse.
        """
        h = _header()
        img = self._blob((80, 80), 36, 40, 0.05, 4.0) + self._blob((80, 80), 52, 40, 0.05, 4.0)
        include = _comp_at(h, 36, 40, 10.0)
        # exclusion ellipse large enough to cover the neighbour's above-threshold emission, not just its FWHM core
        exclude = _comp_at(h, 52, 40, 40.0)
        _, merged = ff.flood_mask(img, h, include, np.empty((0, 5)), threshold=1e-3)
        _, split = ff.flood_mask(img, h, include, exclude, threshold=1e-3)
        assert ff.largest_pixel_separation(split) < ff.largest_pixel_separation(merged)


class TestMeasureSource:
    """
    Unit tests for the measure_source function, which performs end-to-end per-source measurement."""

    def test_returns_finite_size_and_flags(self):
        """Test that a simple Gaussian source returns a finite size and flux, and no bad flags."""
        h = _header()
        yy, xx = np.mgrid[0:80, 0:80]
        img = 0.05 * np.exp(-(((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 4.0 ** 2)))
        include = _comp_at(h, 40, 40, 10.0)
        out = ff.measure_source(img, h, include, np.empty((0, 5)), rms=1e-4, peak=0.05, badflux=1e-6)
        assert out["LM_size"] > 0
        assert out["LM_flux"] > 0
        assert out["Bad_flux"] is False
        assert out["Bad_image"] is False
