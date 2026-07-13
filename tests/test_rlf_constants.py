"""Unit tests for diffracc/rlf/rlf_constants.py."""
import numpy as np
import pytest

from diffracc.rlf.rlf_constants import colors, z_from_v


class TestZFromV:
    """Tests for the z_from_v function, which interpolates redshift from comoving volume."""

    def test_matches_grid_points_exactly(self):
        """Test that z_from_v returns the exact redshift values at the grid points."""
        volume_grid = np.array([0.0, 10.0, 100.0, 1000.0])
        redshift_grid = np.array([0.01, 0.05, 0.2, 0.5])
        np.testing.assert_allclose(z_from_v(volume_grid, volume_grid, redshift_grid), redshift_grid)

    def test_interpolates_linearly_between_grid_points(self):
        """Test that z_from_v interpolates linearly between the grid points."""
        volume_grid = np.array([0.0, 100.0])
        redshift_grid = np.array([0.0, 1.0])
        assert z_from_v(50.0, volume_grid, redshift_grid) == pytest.approx(0.5)

    def test_clamps_outside_grid_range(self):
        """Test that z_from_v clamps to the nearest grid point when the volume is outside the grid range."""
        # np.interp clamps to the edge values rather than extrapolating - RLF relies on this to avoid producing
        # redshifts outside [z_min, z_max] for volumes slightly outside the interpolation grid's rounding error.
        volume_grid = np.array([10.0, 20.0])
        redshift_grid = np.array([0.1, 0.2])
        assert z_from_v(0.0, volume_grid, redshift_grid) == pytest.approx(0.1)
        assert z_from_v(1000.0, volume_grid, redshift_grid) == pytest.approx(0.2)


class TestColors:
    """Tests for the colors constant, which is a list of RGB triples."""
    def test_colors_are_normalised_rgb_triples(self):
        """Test that colors is a list of RGB triples with values in the range [0, 1]."""
        assert len(colors) > 0
        for color in colors:
            assert len(color) == 3
            assert all(0.0 <= channel <= 1.0 for channel in color)
