"""
Unit tests for diffracc.las.make_shape, which is used by diffracc.las.angular_size_finder to estimate the angular size
of a source from its pyBDSF components.
"""

import numpy as np
import pandas as pd
import pytest

from diffracc.las.make_shape import MakeShape


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