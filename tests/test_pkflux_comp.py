"""Unit tests for diffracc/scripts/pkflux_comp.py's main()."""
import argparse

import numpy as np
import pytest
from astropy.io import fits

from diffracc.scripts import pkflux_comp
from diffracc.utils import paths
from diffracc.utils.power_transform import PeakFluxPowerTransformer


@pytest.fixture
def fits_subdir(tmp_path, monkeypatch):
    """Monkeypatch paths.FITS_PARENT to a temp root and return the (empty) subdir to fill with FITS files."""
    fits_root = tmp_path / "fits_root"
    monkeypatch.setattr(paths, "FITS_PARENT", fits_root)
    subdir = fits_root / "generated"
    subdir.mkdir(parents=True)
    return subdir


@pytest.fixture
def fitted_transformer(np_array_parent):
    """
    Fit a real PeakFluxPowerTransformer for 'generated' against known maxvals, so main()'s own (no-maxvals-
    argument) construction of PeakFluxPowerTransformer('generated') finds a valid cached maxvals.npy on disk.
    """
    (np_array_parent / "generated").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    maxvals = np.abs(rng.normal(5, 1, 30)) + 1.0
    return PeakFluxPowerTransformer("generated", maxvals=maxvals)


def _write_fits_file(subdir, name, pixel_max, fxscld):
    data = np.full((4, 4), pixel_max, dtype=np.float32)
    data[0, 0] = pixel_max  # ensure this is genuinely the max
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["FXSCLD"] = float(fxscld)
    hdu.writeto(subdir / name)


class TestMain:
    """Unit tests for the main function of pkflux_comp.py."""

    def _args(self, subdir="generated"):
        """Helper function to create an argparse.Namespace with default values, overridden by any provided arguments."""
        return argparse.Namespace(subdir=subdir)

    def test_runs_without_error_and_saves_a_figure(self, fits_subdir, fitted_transformer, tmp_path, monkeypatch):
        """Test that main() runs without error and saves a figure to the current directory."""
        true_peaks = np.array([1.0, 2.0, 3.0])
        transformed = fitted_transformer.transform(true_peaks)
        for i, (peak, fxscld) in enumerate(zip(true_peaks, transformed)):
            _write_fits_file(fits_subdir, f"img{i}.fits", pixel_max=peak, fxscld=fxscld)

        monkeypatch.chdir(tmp_path)
        pkflux_comp.main(self._args())

        assert (tmp_path / "peak_flux_conditioning.png").exists()

    def test_recovers_conditioned_peak_flux_close_to_pixel_peak(self, fits_subdir, fitted_transformer, tmp_path,
                                                                 monkeypatch):
        """Test that the conditioned peak fluxes plotted by main() are close to the true pixel peak values."""
        true_peaks = np.array([1.0, 2.0, 3.0])
        transformed = fitted_transformer.transform(true_peaks)
        for i, (peak, fxscld) in enumerate(zip(true_peaks, transformed)):
            _write_fits_file(fits_subdir, f"img{i}.fits", pixel_max=peak, fxscld=fxscld)

        monkeypatch.chdir(tmp_path)
        pkflux_comp.main(self._args())

        import matplotlib.pyplot as plt
        offsets = plt.gca().collections[0].get_offsets()
        conditioned = np.sort(offsets[:, 0])
        from_imgs = np.sort(offsets[:, 1])
        np.testing.assert_allclose(conditioned, true_peaks, rtol=1e-3)
        np.testing.assert_allclose(from_imgs, true_peaks, rtol=1e-6)

    def test_reads_files_recursively_from_nested_subdirectories(self, fits_subdir, fitted_transformer, tmp_path,
                                                                  monkeypatch):
        """Test that main() reads FITS files recursively from nested subdirectories."""
        nested = fits_subdir / "nested"
        nested.mkdir()
        true_peak = 2.0
        fxscld = fitted_transformer.transform(np.array([true_peak]))[0]
        _write_fits_file(nested, "deep.fits", pixel_max=true_peak, fxscld=fxscld)

        monkeypatch.chdir(tmp_path)
        pkflux_comp.main(self._args())

        import matplotlib.pyplot as plt
        offsets = plt.gca().collections[0].get_offsets()
        assert len(offsets) == 1

    def test_empty_subdir_raises_a_clear_runtime_error(self, fits_subdir, fitted_transformer, tmp_path, monkeypatch):
        """Test that main() raises a RuntimeError with a clear message when the subdir contains no FITS files."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="No FITS files found"):
            pkflux_comp.main(self._args())
