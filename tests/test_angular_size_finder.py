"""
Unit tests for diffracc/completeness/angular_size_finder.py.

MakeShape owns the shape geometry, so its helpers (MakeShape._ellipse_polygon, MakeShape._furthest_pair) and its
stateless size entry point (MakeShape.estimate_size) are tested directly with hand-computable expected values (ellipse
area = pi*a*b, a single-component shape's length = 2*semi-major-axis). AngularSizeFinder's logic is tested on ordinary
instances - __init__ is cheap (it only stores a path and builds an unused RecursiveFileAnalyzer), so there is no need
to bypass it. _extract_component_data and estimate_angular_sizes's cache branch are tested against small real files in
tmp_path rather than the real catalogue.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from diffracc.completeness.angular_size_finder import AngularSizeFinder, MakeShape


class TestEllipse:
    """
    Tests that MakeShape._ellipse_polygon produces a polygon with the expected area, centroid, and independence of
    position angle.
    """

    def test_area_matches_pi_a_b(self):
        """Test that the area of the polygon returned by _ellipse_polygon matches the analytic area formula pi*a*b."""
        a, b = 10.0, 4.0
        poly = MakeShape._ellipse_polygon(0.0, 0.0, a, b, pa=0.0, n=400)
        assert poly.area == pytest.approx(np.pi * a * b, rel=1e-3)

    def test_centered_at_x0_y0_regardless_of_position_angle(self):
        """Test that the centroid of the polygon returned by _ellipse_polygon is at (x0, y0) regardless of PA."""
        poly = MakeShape._ellipse_polygon(5.0, -3.0, 10.0, 4.0, pa=30.0, n=400)
        centroid = poly.centroid
        assert centroid.x == pytest.approx(5.0, abs=1e-6)
        assert centroid.y == pytest.approx(-3.0, abs=1e-6)

    def test_area_independent_of_position_angle(self):
        """
        Test that the area of the polygon returned by _ellipse_polygon is independent of position angle. If the area
        changes with PA, the ellipse is being distorted by the rotation.
        """
        a, b = 8.0, 3.0
        areas = [MakeShape._ellipse_polygon(0, 0, a, b, pa=pa, n=400).area for pa in (0, 45, 90, 137)]
        for area in areas:
            assert area == pytest.approx(np.pi * a * b, rel=1e-3)


class TestFindFurthestPoints:
    """
    Tests that MakeShape._furthest_pair correctly identifies the two points in a set that are furthest apart.
    """

    def test_max_distance_pair_in_unit_square(self):
        """
        Test that the two furthest points in a unit square are the two diagonal corners, with a squared distance of 2.
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        best_coords, mdist2 = MakeShape._furthest_pair(points)
        assert mdist2 == pytest.approx(2.0)  # the two diagonal corners

    def test_empty_points_returns_zero_without_raising(self):
        """
        Test that _furthest_pair returns ((0,0),(0,0)), 0 for an empty input array instead of raising an error.
        """
        best_coords, mdist2 = MakeShape._furthest_pair(np.empty((0, 2)))
        assert best_coords == ((0, 0), (0, 0))
        assert mdist2 == 0


class TestMakeShapeLength:
    """
    Tests that MakeShape.length() returns the expected length for a single-component shape and for two widely-separated
    components.
    """

    def test_single_component_length_equals_twice_semi_major_axis(self):
        """
        Test that a single-component shape's length is exactly twice the semi-major axis (plus the 0.1 arcsec buffer).
        """
        # With PA=0, _ellipse's internal +90 degree offset aligns the major axis (DC_Maj) along y, so for a single
        # component (whose centre is the mean of itself, i.e. offset (0,0)) the two furthest hull points are
        # exactly the major-axis endpoints - length should be exactly 2 * (DC_Maj_arcsec + 0.1 buffer).
        dc_maj_deg, dc_min_deg = 0.01, 0.005  # 36, 18 arcsec
        clist = pd.DataFrame([{'RA': 10.0, 'DEC': 20.0, 'DC_Maj': dc_maj_deg, 'DC_Min': dc_min_deg, 'PA': 0.0}])
        shape = MakeShape(clist)

        expected_semi_major_arcsec = dc_maj_deg * 3600 + 0.1
        assert shape.length() == pytest.approx(2 * expected_semi_major_arcsec, rel=1e-3)

    def test_two_widely_separated_components_length_reflects_separation(self):
        """
        Test that a shape built from two widely-separated components has a length dominated by the separation, not the
        components' own sizes.
        """
        # Two small, far-apart components: the estimated size should be dominated by the ~3600 arcsec (1 degree)
        # separation between them, not by either component's own small size.
        clist = pd.DataFrame([
            {'RA': 0.0, 'DEC': 0.0, 'DC_Maj': 0.0001, 'DC_Min': 0.00005, 'PA': 0.0},
            {'RA': 1.0, 'DEC': 0.0, 'DC_Maj': 0.0001, 'DC_Min': 0.00005, 'PA': 0.0},
        ])
        shape = MakeShape(clist)
        assert shape.length() == pytest.approx(3600.0, rel=0.01)


class TestFilterComponents:
    """Tests that AngularSizeFinder._filter_components correctly filters components to reach the flux threshold."""

    @pytest.fixture
    def finder(self):
        """Fixture that returns an AngularSizeFinder instance with a flux threshold set for testing."""
        return AngularSizeFinder(root_dir=Path("unused"), flux_threshold=0.95)

    def test_keeps_components_until_flux_threshold_reached(self, finder):
        """Test that _filter_components keeps the brightest components until the flux threshold is reached."""
        # total flux = 16.5; 0.95 * 16.5 = 15.675 -> top 2 (10+5=15) undershoots, top 3 (10+5+1=16) reaches it.
        components = [(10.0, 0, 0, 0, 0, 0), (5.0, 0, 0, 0, 0, 0), (1.0, 0, 0, 0, 0, 0), (0.5, 0, 0, 0, 0, 0)]

        filtered = finder._filter_components(list(components))

        assert [c[0] for c in filtered] == [10.0, 5.0, 1.0]

    def test_sorts_components_by_flux_descending_regardless_of_input_order(self, finder):
        """Test that _filter_components sorts the components by flux in descending order, regardless of input order."""
        components = [(1.0, 0, 0, 0, 0, 0), (10.0, 0, 0, 0, 0, 0), (5.0, 0, 0, 0, 0, 0)]
        filtered = finder._filter_components(list(components))
        assert [c[0] for c in filtered] == [10.0, 5.0, 1.0]

    def test_raises_on_empty_components(self, finder):
        """Test that _filter_components raises an AssertionError when given an empty list of components."""
        with pytest.raises(AssertionError):
            finder._filter_components([])

    def test_raises_on_zero_total_flux(self, finder):
        """Test that _filter_components raises a ValueError when the total flux of the components is zero."""
        with pytest.raises(ValueError):
            finder._filter_components([(0.0, 0, 0, 0, 0, 0), (0.0, 0, 0, 0, 0, 0)])


class TestEstimateSize:
    """
    Tests that MakeShape.estimate_size (the stateless size entry point) estimates the angular size from components.
    """

    def test_matches_makeshape_length_for_given_components(self):
        """
        Test that estimate_size returns the same buffered length as MakeShape(...).length() for the given components.
        """
        dc_maj_deg, dc_min_deg = 0.01, 0.005
        components = [(10.0, 10.0, 20.0, dc_maj_deg, dc_min_deg, 0.0)]

        size = MakeShape.estimate_size(components)

        expected = 2 * (dc_maj_deg * 3600 + 0.1)
        assert size == pytest.approx(expected, rel=1e-3)

    def test_raises_on_empty_components(self):
        """Test that estimate_size raises an AssertionError when given an empty list of components."""
        with pytest.raises(AssertionError):
            MakeShape.estimate_size([])


class TestExtractComponentData:
    """Tests that AngularSizeFinder._extract_component_data correctly reads and filters components from a FITS file."""

    def _write_component_fits(self, path, fluxes, ra, dec, dc_maj, dc_min, pa):
        """Helper method to write a FITS file with the specified component data for testing."""
        cols = fits.ColDefs([
            fits.Column(name='Total_flux', format='E', array=np.asarray(fluxes, dtype=np.float32)),
            fits.Column(name='RA', format='E', array=np.asarray(ra, dtype=np.float32)),
            fits.Column(name='DEC', format='E', array=np.asarray(dec, dtype=np.float32)),
            fits.Column(name='DC_Maj', format='E', array=np.asarray(dc_maj, dtype=np.float32)),
            fits.Column(name='DC_Min', format='E', array=np.asarray(dc_min, dtype=np.float32)),
            fits.Column(name='PA', format='E', array=np.asarray(pa, dtype=np.float32)),
        ])
        hdu = fits.BinTableHDU.from_columns(cols)
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    def test_reads_and_filters_components_from_fits_file(self, tmp_path):
        """Test that _extract_component_data reads a FITS file and filters components to reach the flux threshold."""
        finder = AngularSizeFinder(root_dir=tmp_path, flux_threshold=0.95)
        fits_path = tmp_path / "source_1.fits"
        self._write_component_fits(
            fits_path,
            fluxes=[10.0, 5.0, 1.0, 0.5],
            ra=[10.0, 10.001, 10.002, 10.003],
            dec=[20.0, 20.001, 20.002, 20.003],
            dc_maj=[0.001] * 4,
            dc_min=[0.0005] * 4,
            pa=[0.0] * 4,
        )

        components = finder._extract_component_data(fits_path)

        # matches TestFilterComponents' threshold arithmetic: top 3 of 4 components reach 0.95 of total flux.
        assert len(components) == 3
        assert components[0][0] == pytest.approx(10.0, rel=1e-5)


class TestEstimateAngularSizesCache:
    """
    Tests that AngularSizeFinder.estimate_angular_sizes correctly reads from an existing output file instead of running
    the full pipeline when the output file exists.
    """

    def test_reads_from_existing_output_file_instead_of_running_the_fits_pipeline(self, tmp_path):
        """Test that estimate_angular_sizes reads from an existing output file instead of running the FITS pipeline."""
        fits_dir = tmp_path / "cats"
        fits_dir.mkdir()
        # Only the filenames matter for the cache-read branch - it never opens these files.
        (fits_dir / "source_1.fits").write_bytes(b"")
        (fits_dir / "source_2.fits").write_bytes(b"")

        output_file = tmp_path / "sizes.csv"
        output_file.write_text("Estimated Angular Size (arcseconds)\n12.5\n30.0\n")

        finder = AngularSizeFinder(root_dir=fits_dir)
        indices, sizes = finder.estimate_angular_sizes(fits_dir=fits_dir, output_file=output_file)

        np.testing.assert_allclose(sorted(sizes), [12.5, 30.0])
        assert set(indices) == {1, 2}


class TestEstimateAngularSizesFullPipeline:
    """Covers the non-cache branch: scanning FITS files, extracting/filtering components, and estimating sizes."""

    def _write_component_fits(self, path, fluxes, ra, dec, dc_maj, dc_min, pa):
        """Helper method to write a FITS file with the specified component data for testing."""
        cols = fits.ColDefs([
            fits.Column(name='Total_flux', format='E', array=np.asarray(fluxes, dtype=np.float32)),
            fits.Column(name='RA', format='E', array=np.asarray(ra, dtype=np.float32)),
            fits.Column(name='DEC', format='E', array=np.asarray(dec, dtype=np.float32)),
            fits.Column(name='DC_Maj', format='E', array=np.asarray(dc_maj, dtype=np.float32)),
            fits.Column(name='DC_Min', format='E', array=np.asarray(dc_min, dtype=np.float32)),
            fits.Column(name='PA', format='E', array=np.asarray(pa, dtype=np.float32)),
        ])
        hdu = fits.BinTableHDU.from_columns(cols)
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    def test_computes_sizes_from_scratch_and_saves_output_file(self, tmp_path):
        """Test that estimate_angular_sizes computes sizes from scratch and saves the output file."""
        fits_dir = tmp_path / "cats"
        fits_dir.mkdir()

        # source_1: a single component survives filtering -> hits the len(components)==1 special case, which uses
        # 2*DC_Maj_deg*3600 directly (no MakeShape/_ellipse +0.1 arcsec buffer).
        dc_maj_deg_1 = 0.002  # 7.2 arcsec
        self._write_component_fits(fits_dir / "source_1.fits",
                                   fluxes=[1.0], ra=[10.0], dec=[20.0],
                                   dc_maj=[dc_maj_deg_1], dc_min=[0.001], pa=[0.0])

        # source_2: two equal-flux, widely-separated (1 degree = 3600 arcsec) components both survive filtering ->
        # goes through MakeShape, and the ~3600 arcsec separation should dominate the estimated size.
        self._write_component_fits(fits_dir / "source_2.fits",
                                   fluxes=[1.0, 1.0], ra=[0.0, 1.0], dec=[0.0, 0.0],
                                   dc_maj=[0.0001, 0.0001], dc_min=[0.00005, 0.00005], pa=[0.0, 0.0])

        output_file = tmp_path / "sizes.csv"
        finder = AngularSizeFinder(root_dir=fits_dir)

        indices, sizes = finder.estimate_angular_sizes(fits_dir=fits_dir, output_file=output_file)

        assert set(indices) == {1, 2}
        by_index = dict(zip(indices, sizes))
        assert by_index[1] == pytest.approx(2 * dc_maj_deg_1 * 3600, rel=1e-3)
        assert by_index[2] == pytest.approx(3600.0, rel=0.01)
        assert output_file.exists()


class TestMakeShapePlot:
    """Smoke test for MakeShape.plot() - forced onto the Agg backend so it never opens a real window."""

    def test_runs_without_error(self, monkeypatch):
        """Test that MakeShape.plot() runs without error on the Agg backend."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clist = pd.DataFrame([{'RA': 0.0, 'DEC': 0.0, 'DC_Maj': 0.001, 'DC_Min': 0.0005, 'PA': 0.0}])
        try:
            MakeShape(clist).plot()
        finally:
            plt.close("all")
