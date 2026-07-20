"""
Unit tests for diffracc/utils/plotting.py.

These exercise the helper's contracts (return values, subsampling, style isolation, required limits) rather than pixel
output - the actual rendering is smoke-tested by drawing onto an Agg figure and checking no exception is raised and the
expected artists appear.
"""
import matplotlib

matplotlib.use("Agg")  # headless backend before pyplot is imported, so tests never open a window

import matplotlib.pyplot as plt
import numpy as np
import pytest

from diffracc.utils import plotting
from diffracc.utils.plotting import Boundary, Population, density_scatter, paper_style


@pytest.fixture
def rng():
    """Fixture providing a reproducible random number generator for tests that need to sample points."""
    return np.random.default_rng(0)


class TestPopulation:
    """Unit tests for the Population class, which is a simple data container with some derived properties."""

    def test_count_ignores_non_finite_points(self):
        """Test that the count property only counts points where both x and y are finite."""
        pop = Population("x", np.array([1.0, np.nan, 3.0, np.inf]), np.array([1.0, 2.0, np.nan, 4.0]), color="red")
        assert pop.count == 1  # only index 0 is finite in both x and y

    def test_legend_label_appends_count(self):
        """Test that the legend_label property appends the count of finite points in parentheses."""
        pop = Population("SFG", np.array([1.0, 2.0]), np.array([1.0, 2.0]), color="red")
        assert pop.legend_label == "SFG (2)"

    def test_legend_label_without_count(self):
        """Test that the legend_label property does not append the count if show_count is False."""
        pop = Population("SFG", np.array([1.0]), np.array([1.0]), color="red", show_count=False)
        assert pop.legend_label == "SFG"

    def test_scatter_xy_subsamples_to_max_scatter(self, rng):
        """Test that scatter_xy returns a subsample of points if the population exceeds max_scatter."""
        x = np.arange(1000.0)
        pop = Population("x", x, x, color="red", max_scatter=100)
        sx, sy = pop.scatter_xy(rng)
        assert sx.size == 100 and sy.size == 100
        assert set(sx).issubset(set(x))  # a genuine subset, no invented points

    def test_scatter_xy_keeps_all_when_below_max(self, rng):
        """Test that scatter_xy returns all points if the population is below max_scatter."""
        x = np.array([1.0, 2.0, 3.0])
        pop = Population("x", x, x, color="red", max_scatter=100)
        sx, _ = pop.scatter_xy(rng)
        assert sx.size == 3


class TestPaperStyle:
    """Unit tests for the paper_style context manager, which temporarily sets matplotlib rcParams."""

    def test_restores_rcparams_after_context(self):
        """
        Test that the paper_style context manager restores rcParams to their original values after exiting the context.
        """
        key = "xtick.direction"
        before = plt.rcParams[key]
        with paper_style():
            assert plt.rcParams[key] == "in"
        assert plt.rcParams[key] == before

    def test_overrides_are_applied(self):
        """Test that the paper_style context manager applies overrides to rcParams within the context."""
        with paper_style({"font.size": 42}):
            assert plt.rcParams["font.size"] == 42


class TestDensityScatter:
    """
    Unit tests for the density_scatter function, which draws a scatter plot with optional populations and boundaries.
    """

    def test_requires_explicit_limits(self):
        """
        Test that density_scatter raises a ValueError if xlim or ylim are not provided, since it cannot infer limits
        from the data.
        """
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError):
                density_scatter(ax, np.array([1.0, 2.0]), np.array([1.0, 2.0]), xlabel="x", ylabel="y")
        finally:
            plt.close(fig)

    def test_draws_background_populations_and_boundaries(self, rng):
        """
        Test that density_scatter draws the background points, populations, and boundaries without error, and returns
        the axis.
        """
        x_all = rng.normal(0, 1, 5000)
        y_all = 10 ** rng.normal(24, 1, 5000)
        pop = Population("blob", rng.normal(0, 0.5, 200), 10 ** rng.normal(24, 0.5, 200), color="#2c7fb8")
        line_x = np.linspace(-3, 3, 50)
        boundary = Boundary(line_x, 10 ** (24 + line_x / 3), label="cut")

        fig, ax = plt.subplots()
        try:
            out = density_scatter(
                ax, x_all, y_all, populations=[pop], boundaries=[boundary],
                xlabel="x", ylabel="y", xlim=(-3, 3), ylim=(1e22, 1e26), ylog=True)
            assert out is ax
            assert ax.get_yscale() == "log"
            assert ax.get_xlim() == (-3, 3)
            # one scatter collection for the population, one line for the boundary, and a legend
            assert len(ax.collections) >= 1
            assert any("cut" == t.get_text() or "blob" in t.get_text() for t in ax.get_legend().get_texts())
        finally:
            plt.close(fig)

    def test_inverted_xlim_is_respected(self, rng):
        """Test that density_scatter respects an inverted xlim, which is common in astronomy plots."""
        fig, ax = plt.subplots()
        try:
            density_scatter(ax, rng.normal(-25, 2, 1000), 10 ** rng.normal(24, 1, 1000),
                            xlabel="x", ylabel="y", xlim=(-18, -34), ylim=(1e22, 1e26), ylog=True)
            assert ax.get_xlim() == (-18, -34)  # inverted axis preserved
        finally:
            plt.close(fig)

    def test_hexbin_density_mode_runs_and_draws(self, rng):
        """Test that density_scatter with density="hexbin" runs without error and draws a hexbin collection."""
        fig, ax = plt.subplots()
        try:
            density_scatter(ax, rng.normal(-25, 2, 5000), 10 ** rng.normal(24, 1, 5000),
                            xlabel="x", ylabel="y", xlim=(-18, -34), ylim=(1e22, 1e26),
                            ylog=True, density="hexbin")
            assert len(ax.collections) >= 1  # hexbin adds a PolyCollection
            assert ax.get_xlim() == (-18, -34)
        finally:
            plt.close(fig)

    def test_shaded_population_draws_filled_contours(self, rng):
        """
        Test that a Population with shade=True adds filled contours to the plot, which is a more visually appealing
        representation of the density.
        """
        fig, ax = plt.subplots()
        try:
            pop = Population("blob", rng.normal(0, 0.5, 2000), 10 ** rng.normal(24, 0.5, 2000),
                             color="#2c7fb8", shade=True, max_scatter=100)
            n_before = len(ax.collections)
            density_scatter(ax, rng.normal(0, 1, 3000), 10 ** rng.normal(24, 1, 3000),
                            populations=[pop], xlabel="x", ylabel="y",
                            xlim=(-3, 3), ylim=(1e22, 1e26), ylog=True)
            # a shaded population adds its own filled-contour artist on top of the background + scatter
            assert len(ax.collections) > n_before + 1
        finally:
            plt.close(fig)

    def test_handles_all_nan_background_without_error(self):
        """
        Test that density_scatter does not raise an error when the background data is all NaN, which can happen in some
        cases.
        """
        fig, ax = plt.subplots()
        try:
            density_scatter(ax, np.full(10, np.nan), np.full(10, np.nan),
                            xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1))
        finally:
            plt.close(fig)
