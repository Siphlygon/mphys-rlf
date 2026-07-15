"""Unit tests for diffracc/plotting/flux_vs_residual.py."""
import numpy as np

from diffracc.plotting import flux_vs_residual as fvr
from diffracc.utils import paths


class _FakeSubdirData:
    """A simple container for the data returned by ImageDataArrays._load_subdir_data."""

    def __init__(self, images, peak_fluxes, image_scale_factors, residual_images):
        self.images = images
        self.peak_fluxes = peak_fluxes
        self.image_scale_factors = image_scale_factors
        self.residual_images = residual_images


def _make_subdir_data(n=5, seed=0):
    """Helper function to create a _FakeSubdirData instance with synthetic data."""
    rng = np.random.default_rng(seed)
    images = rng.uniform(1.0, 10.0, size=(n, 4, 4)).astype(np.float64)
    peak_fluxes = np.array([0.1, 0.6, 1.0, 5.0, 10.0])[:n]  # mJy - one below the 0.5 mJy cut
    image_scale_factors = np.full(n, 1.0)
    residual_images = rng.normal(0.5, 0.2, size=(n, 4, 4))  # mostly positive, so delta isn't degenerately zero
    return _FakeSubdirData(images, peak_fluxes, image_scale_factors, residual_images)


class _FakeImageDataArrays:
    """A fake object to simulate the ImageDataArrays class."""
    def __init__(self, config_name: str):
        self.generated_data = _make_subdir_data(seed=0)
        self.dataset_data = _make_subdir_data(seed=1)


class TestPlotFluxVsResiduals:
    """Tests for the plot_flux_vs_residuals function."""

    def test_runs_without_error_and_saves_a_figure(self, monkeypatch, tmp_path, np_array_parent):
        """Test that the function runs without error and saves a figure."""
        monkeypatch.setattr(paths, "config", {"my_config": {"generated_subdir": "generated",
                                                            "dataset_subdir": "dataset"}})
        monkeypatch.setattr(fvr, "ImageDataArrays", _FakeImageDataArrays)
        (np_array_parent / "generated").mkdir(parents=True, exist_ok=True)
        (np_array_parent / "dataset").mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)

        fvr.plot_flux_vs_residuals("my_config")

        assert (tmp_path / "scatter.png").exists()

    def test_filters_out_peak_fluxes_at_or_below_half_mjy(self, monkeypatch, tmp_path, np_array_parent):
        """Test that the function filters out peak fluxes at or below 0.5 mJy."""
        monkeypatch.setattr(paths, "config", {"my_config": {"generated_subdir": "generated",
                                                            "dataset_subdir": "dataset"}})
        monkeypatch.setattr(fvr, "ImageDataArrays", _FakeImageDataArrays)
        (np_array_parent / "generated").mkdir(parents=True, exist_ok=True)
        (np_array_parent / "dataset").mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)

        fvr.plot_flux_vs_residuals("my_config")

        import matplotlib.pyplot as plt
        ax = plt.gca()
        scatter_by_label = {c.get_label(): c for c in ax.collections}
        # 5 fake images per subdir, one with peak_flux=0.1 mJy (<=0.5) filtered out -> 4 points plotted
        assert len(scatter_by_label["generated"].get_offsets()) == 4
        assert len(scatter_by_label["dataset"].get_offsets()) == 4
