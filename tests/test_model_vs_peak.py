"""Unit tests for diffracc/plotting/model_vs_peak.py."""
import numpy as np

from diffracc.plotting import model_vs_peak as mvp
from diffracc.utils import paths


class _FakeSubdirData:
    """A fake object to simulate the data arrays for a subdirectory (dataset or generated)."""
    def __init__(self, peak_fluxes: np.ndarray, model_fluxes: np.ndarray):
        self.peak_fluxes = peak_fluxes
        self.model_fluxes = model_fluxes


class _FakeImageDataArrays:
    """A fake object to simulate the ImageDataArrays class."""
    def __init__(self, config_name: str):
        self.generated_data = _FakeSubdirData(peak_fluxes=np.array([1.0, 2.0]), model_fluxes=np.array([1.1, 2.1]))
        self.dataset_data = _FakeSubdirData(peak_fluxes=np.array([3.0, 4.0]), model_fluxes=np.array([3.1, 4.1]))


class TestPlotPeakVsModelFlux:
    """Unit tests for the plot_peak_vs_model_flux function."""

    def test_runs_without_error_and_saves_a_figure(self, monkeypatch, tmp_path):
        """Test that the function runs without error and saves a figure."""
        monkeypatch.setattr(paths, "config", {"my_config": {"dataset_subdir": "dset", "generated_subdir": "gen"}})
        monkeypatch.setattr(mvp, "ImageDataArrays", _FakeImageDataArrays)
        monkeypatch.chdir(tmp_path)

        mvp.plot_peak_vs_model_flux("my_config")

        assert (tmp_path / "peak_vs_model_flux_my_config.png").exists()

    def test_legend_labels_match_their_data_source(self, monkeypatch, tmp_path):
        """Test that the legend labels in the plot match their corresponding data source."""
        monkeypatch.setattr(paths, "config", {"my_config": {"dataset_subdir": "dset", "generated_subdir": "gen"}})
        monkeypatch.setattr(mvp, "ImageDataArrays", _FakeImageDataArrays)
        monkeypatch.chdir(tmp_path)

        mvp.plot_peak_vs_model_flux("my_config")

        import matplotlib.pyplot as plt
        ax = plt.gca()
        scatter_by_label = {c.get_label(): c for c in ax.collections}
        np.testing.assert_allclose(scatter_by_label["gen"].get_offsets(), [[1.0, 1.1], [2.0, 2.1]])
        np.testing.assert_allclose(scatter_by_label["dset"].get_offsets(), [[3.0, 3.1], [4.0, 4.1]])
