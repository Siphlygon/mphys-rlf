"""
Unit tests for diffracc/las/angular_size_estimator.py.
"""
import numpy as np
import pytest
from astropy.io import fits

from diffracc.las.angular_size_estimator import AngularSizeEstimator


class TestEstimateAngularSizesCache:
    """
    Tests that AngularSizeEstimator.estimate_angular_sizes correctly reads from an existing output file instead of running
    the full pipeline when the output file exists.
    """

    def test_reads_from_existing_output_file_instead_of_running_the_fits_pipeline(self, tmp_path):
        """Test that estimate_angular_sizes reads from an existing output file instead of running the FITS pipeline."""
        fits_dir = tmp_path / "cats"
        fits_dir.mkdir()

        # The read-from-file branch takes both the indices and the sizes straight from the CSV columns, never touching
        # fits_dir.
        output_file = tmp_path / "sizes.csv"
        output_file.write_text("fits_index,estimated_las_arcsec\n1,12.5\n2,30.0\n")

        estimator = AngularSizeEstimator(root_dir=fits_dir)
        indices, sizes = estimator.estimate_angular_sizes(fits_dir=fits_dir, output_file=output_file, read_from_file=True)

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
        estimator = AngularSizeEstimator(root_dir=fits_dir)

        # load_from_catalogue=False keeps the pipeline on the FITS-extraction path (these tmp files), rather than the
        # real DR2 component catalogue.
        indices, sizes = estimator.estimate_angular_sizes(fits_dir=fits_dir, output_file=output_file,
                                                         load_from_catalogue=False)

        assert set(indices) == {1, 2}
        by_index = dict(zip(indices, sizes))
        assert by_index[1] == pytest.approx(2 * dc_maj_deg_1 * 3600, rel=1e-3)
        assert by_index[2] == pytest.approx(3600.0, rel=0.01)
        assert output_file.exists()


class TestSelectAngularSize:
    """
    AngularSizeEstimator.select_angular_size chooses between the component-based and flood-fill sizes: it takes the
    flood-fill size only when unflagged, the flux matches within 20%, and the component size is in 30-600 arcsec.
    """

    def test_selects_floodfill_when_all_gates_pass(self):
        """Test that select_angular_size selects the flood-fill size when all gates pass."""
        las, src = AngularSizeEstimator.select_angular_size(40.0, 55.0, 1.0, 1.0, False, False)
        assert las == 55.0 and src == "Flood-fill"

    def test_keeps_component_when_size_below_30(self):
        """Test that select_angular_size keeps the component size when it is below 30 arcsec."""
        las, src = AngularSizeEstimator.select_angular_size(20.0, 55.0, 1.0, 1.0, False, False)
        assert las == 20.0 and src == "Catalogue"

    def test_keeps_component_when_size_above_600(self):
        """Test that select_angular_size keeps the component size when it is above 600 arcsec."""
        las, src = AngularSizeEstimator.select_angular_size(700.0, 55.0, 1.0, 1.0, False, False)
        assert las == 700.0 and src == "Catalogue"

    def test_keeps_component_on_flux_mismatch(self):
        """Test that select_angular_size keeps the component size when the fluxes mismatch by more than 20%."""
        las, src = AngularSizeEstimator.select_angular_size(40.0, 55.0, 0.5, 1.0, False, False)  # ratio 0.5 < 0.8
        assert las == 40.0 and src == "Catalogue"

    def test_keeps_component_on_bad_flags(self):
        """Test that select_angular_size keeps the component size when either bad_flux or bad_image is True."""
        assert AngularSizeEstimator.select_angular_size(40.0, 55.0, 1.0, 1.0, True, False)[0] == 40.0
        assert AngularSizeEstimator.select_angular_size(40.0, 55.0, 1.0, 1.0, False, True)[0] == 40.0

    def test_vectorised(self):
        """Test that select_angular_size works with vectorised inputs."""
        las, src = AngularSizeEstimator.select_angular_size([40, 20, 40], [55, 55, 55], [1.0, 1.0, 0.5],
                                                         [1.0, 1.0, 1.0], [False, False, False], [False, False, False])
        np.testing.assert_array_equal(las, [55, 20, 40])
        np.testing.assert_array_equal(src, ["Flood-fill", "Catalogue", "Catalogue"])
