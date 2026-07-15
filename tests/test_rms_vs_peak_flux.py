"""Unit tests for diffracc/plotting/rms_vs_peak_flux.py."""
import numpy as np

from diffracc.plotting import rms_vs_peak_flux as rvp
from diffracc.utils import paths


class _FakeSubdirData:
    """A fake object to simulate the data arrays for a subdirectory (dataset or generated)."""
    def __init__(self, peak_fluxes: np.ndarray, residual_images: np.ndarray):
        self.peak_fluxes = peak_fluxes
        self.residual_images = residual_images


def _residual_images_with_std(stds: list[float]) -> np.ndarray:
    """Helper function to create a 3D array of residual images with specified standard deviations."""
    rng = np.random.default_rng(0)
    return np.stack([rng.normal(0.0, std, size=(4, 4)) for std in stds])


class _FakeImageDataArrays:
    """A fake object to simulate the ImageDataArrays class."""
    def __init__(self, config_name: str):
        self.generated_data = _FakeSubdirData(peak_fluxes=np.array([1.0, 2.0]),
                                              residual_images=_residual_images_with_std([0.1, 0.2]))
        self.dataset_data = _FakeSubdirData(peak_fluxes=np.array([3.0, 4.0]),
                                            residual_images=_residual_images_with_std([0.3, 0.4]))


class TestPlotRmsVsPeakFlux:
    """Unit tests for the plot_rms_vs_peak_flux function."""

    def test_runs_without_error_and_saves_a_figure(self, monkeypatch, tmp_path):
        """Test that the function runs without error and saves a figure."""
        monkeypatch.setattr(paths, "config", {"my_config": {"dataset_subdir": "dset", "generated_subdir": "gen"}})
        monkeypatch.setattr(rvp, "ImageDataArrays", _FakeImageDataArrays)
        monkeypatch.chdir(tmp_path)

        rvp.plot_rms_vs_peak_flux("my_config")

        assert (tmp_path / "peak_vs_rms.png").exists()

    def test_plots_std_of_residual_images_against_peak_flux(self, monkeypatch, tmp_path):
        """Test that the plot correctly shows the standard deviation of residual images against peak flux."""
        monkeypatch.setattr(paths, "config", {"my_config": {"dataset_subdir": "dset", "generated_subdir": "gen"}})
        monkeypatch.setattr(rvp, "ImageDataArrays", _FakeImageDataArrays)
        monkeypatch.chdir(tmp_path)

        rvp.plot_rms_vs_peak_flux("my_config")

        import matplotlib.pyplot as plt
        ax = plt.gca()
        scatter_by_label = {c.get_label(): c for c in ax.collections}
        offsets = scatter_by_label["gen"].get_offsets()
        np.testing.assert_allclose(offsets[:, 0], [1.0, 2.0])
        expected_stds = np.std(_residual_images_with_std([0.1, 0.2]), axis=(1, 2))
        np.testing.assert_allclose(offsets[:, 1], expected_stds)
