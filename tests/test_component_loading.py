"""
Unit tests for diffracc/las/component_loading.py.
"""
import numpy as np
import pytest
from astropy.io import fits

from diffracc.las.component_loading import ComponentLoader


class TestFilterComponents:
    """Tests that ComponentLoader._filter_by_flux correctly filters components to reach the flux threshold."""

    def test_keeps_components_until_flux_threshold_reached(self):
        """Test that _filter_by_flux keeps the brightest components until the flux threshold is reached."""
        # total flux = 16.5; 0.95 * 16.5 = 15.675 -> top 2 (10+5=15) undershoots, top 3 (10+5+1=16) reaches it.
        components = [(10.0, 0, 0, 0, 0, 0), (5.0, 0, 0, 0, 0, 0), (1.0, 0, 0, 0, 0, 0), (0.5, 0, 0, 0, 0, 0)]

        filtered = ComponentLoader._filter_by_flux(list(components), 0.95)

        assert [c[0] for c in filtered] == [10.0, 5.0, 1.0]

    def test_sorts_components_by_flux_descending_regardless_of_input_order(self):
        """Test that _filter_by_flux sorts the components by flux in descending order, regardless of input order."""
        components = [(1.0, 0, 0, 0, 0, 0), (10.0, 0, 0, 0, 0, 0), (5.0, 0, 0, 0, 0, 0)]
        filtered = ComponentLoader._filter_by_flux(list(components), 0.95)
        assert [c[0] for c in filtered] == [10.0, 5.0, 1.0]

    def test_raises_on_empty_components(self):
        """Test that _filter_by_flux raises an AssertionError when given an empty list of components."""
        with pytest.raises(AssertionError):
            ComponentLoader._filter_by_flux([], 0.95)

    def test_raises_on_zero_total_flux(self):
        """Test that _filter_by_flux raises a ValueError when the total flux of the components is zero."""
        with pytest.raises(ValueError):
            ComponentLoader._filter_by_flux([(0.0, 0, 0, 0, 0, 0), (0.0, 0, 0, 0, 0, 0)], 0.95)


class TestReadAndFilter:
    """Tests that ComponentLoader._read_and_filter correctly reads and filters components from a FITS file."""

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
        """Test that _read_and_filter reads a FITS file and filters components to reach the flux threshold."""
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

        components = ComponentLoader._read_and_filter(fits_path, 0.95)

        # matches TestFilterComponents' threshold arithmetic: top 3 of 4 components reach 0.95 of total flux.
        assert len(components) == 3
        assert components[0][0] == pytest.approx(10.0, rel=1e-5)
