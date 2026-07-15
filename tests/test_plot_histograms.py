"""Unit tests for diffracc/plotting/plot_histograms.py."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from diffracc.plotting.plot_histograms import HistogramErrorDrawer, HistogramPlotter


@pytest.fixture
def hist_data():
    """3 samples in [0,1), 5 in [1,2), 2 in [2,3) - deliberately hand-countable bin populations."""
    return np.array([0.5] * 3 + [1.5] * 5 + [2.5] * 2, dtype=float)


class TestHistogramErrorDrawer:
    """
    Unit tests for the HistogramErrorDrawer class, which draws histograms with error bars on a given matplotlib Axes.
    """

    def test_raises_when_density_and_relative_both_true(self, hist_data):
        """Test that a RuntimeError is raised when both density and relative are set to True, as this is not allowed."""
        fig, ax = plt.subplots()
        drawer = HistogramErrorDrawer()
        with pytest.raises(RuntimeError):
            drawer.draw(hist_data, ax, bins=3, range=(0, 3), label="x", color="b", density=True, relative=True)

    def test_raw_counts_plotted_when_neither_density_nor_relative(self, hist_data):
        """Test that raw counts are plotted when both density and relative are set to False."""
        fig, ax = plt.subplots()
        HistogramErrorDrawer().draw(hist_data, ax, bins=3, range=(0, 3), label="x", color="b",
                                    density=False, relative=False)
        step_line = ax.lines[0]
        # np.histogram bin edges plus a trailing zero appended for the step plot's final "post" segment
        np.testing.assert_allclose(step_line.get_xdata(), [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(step_line.get_ydata(), [3.0, 5.0, 2.0, 0.0])

        errorbar_line = ax.containers[0][0]
        np.testing.assert_allclose(errorbar_line.get_xdata(), [0.5, 1.5, 2.5])
        np.testing.assert_allclose(errorbar_line.get_ydata(), [3.0, 5.0, 2.0])

    def test_relative_frequency_divides_counts_by_sample_count(self, hist_data):
        """Test that relative frequency is computed by dividing counts by the total number of samples."""
        fig, ax = plt.subplots()
        HistogramErrorDrawer().draw(hist_data, ax, bins=3, range=(0, 3), label="x", color="b",
                                    density=False, relative=True)
        step_line = ax.lines[0]
        np.testing.assert_allclose(step_line.get_ydata(), [0.3, 0.5, 0.2, 0.0])

        errorbar_line = ax.containers[0][0]
        np.testing.assert_allclose(errorbar_line.get_ydata(), [0.3, 0.5, 0.2])

    def test_density_matches_numpy_histogram_density(self, hist_data):
        """Test that the density option produces a histogram matching numpy's density=True output."""
        fig, ax = plt.subplots()
        HistogramErrorDrawer().draw(hist_data, ax, bins=3, range=(0, 3), label="x", color="b",
                                    density=True, relative=False)
        step_line = ax.lines[0]
        expected_density = np.histogram(hist_data, bins=3, range=(0, 3), density=True)[0]
        np.testing.assert_allclose(step_line.get_ydata()[:-1], expected_density)

    def test_yerr_bars_are_centred_on_the_plotted_value(self, hist_data):
        """Test that the error bars are centred on the plotted histogram values."""
        fig, ax = plt.subplots()
        HistogramErrorDrawer().draw(hist_data, ax, bins=3, range=(0, 3), label="x", color="b",
                                    density=False, relative=False)
        errorbar_line = ax.containers[0][0]
        yerr_segments = ax.containers[0][2][0].get_segments()
        plotted_y = errorbar_line.get_ydata()
        for (seg, y) in zip(yerr_segments, plotted_y):
            lower, upper = seg[0][1], seg[1][1]
            assert lower < y < upper


class TestHistogramPlotterSetUpFigure:
    """Tests for the set_up_figure method of the HistogramPlotter class."""

    def test_returns_four_axes_matching_titles_and_labels(self):
        """Test that set_up_figure returns four axes with titles and labels matching the provided arguments."""
        plotter = HistogramPlotter(generated_subdir="gen", dataset_subdir="dset")
        titles = ["A", "B", "C", "D"]
        ranges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        xlabels = ["xa", "xb", "xc", "xd"]
        ylabels = ["ya", "yb", "yc", "yd"]

        fig, axes = plotter.set_up_figure(titles, ranges, xlabels, ylabels)

        assert len(axes) == 4
        for ax, title, xlabel, ylabel in zip(axes, titles, xlabels, ylabels):
            assert ax.get_title() == title
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel
            assert ax.get_yscale() == "log"

    def test_init_stores_config(self):
        """Test that the HistogramPlotter constructor correctly stores the provided configuration parameters."""
        plotter = HistogramPlotter(generated_subdir="gen", dataset_subdir="dset", config_name="cfg", bin_count=10)
        assert plotter.generated_subdir == "gen"
        assert plotter.dataset_subdir == "dset"
        assert plotter.config_name == "cfg"
        assert plotter.bin_count == 10


class TestHistogramPlotterPlotHistograms:
    """Tests for the plot_histograms method of the HistogramPlotter class."""

    def test_runs_without_error_and_saves_a_figure(self, monkeypatch, tmp_path, np_array_parent):
        """Test that the plot_histograms method runs without error and saves a figure to the current directory."""
        rng = np.random.default_rng(0)
        for subdir in ("generated", "dataset"):
            (np_array_parent / subdir).mkdir(parents=True, exist_ok=True)
            np.save(np_array_parent / subdir / "integrated_fluxes_normalized.npy", rng.uniform(1, 10, 20))
            np.save(np_array_parent / subdir / "histogram_data.npy", rng.normal(0, 0.05, size=(20, 4, 4)))

        monkeypatch.chdir(tmp_path)

        plotter = HistogramPlotter(generated_subdir="generated", dataset_subdir="dataset", config_name="cfg")
        plotter.plot_histograms()

        assert (tmp_path / "hist_cfg.png").exists()

    def test_uses_plain_hist_filename_when_no_config_name(self, monkeypatch, tmp_path, np_array_parent):
        """Test that the plot_histograms method uses 'hist.png' as the filename when no config_name is provided."""
        rng = np.random.default_rng(0)
        for subdir in ("generated", "dataset"):
            (np_array_parent / subdir).mkdir(parents=True, exist_ok=True)
            np.save(np_array_parent / subdir / "integrated_fluxes_normalized.npy", rng.uniform(1, 10, 20))
            np.save(np_array_parent / subdir / "histogram_data.npy", rng.normal(0, 0.05, size=(20, 4, 4)))

        monkeypatch.chdir(tmp_path)

        plotter = HistogramPlotter(generated_subdir="generated", dataset_subdir="dataset")
        plotter.plot_histograms()

        assert (tmp_path / "hist.png").exists()
