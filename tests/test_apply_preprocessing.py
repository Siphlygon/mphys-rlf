"""
Unit tests for diffracc/data/apply_preprocessing.py's CutoutPreprocessor.

Built against the cutout_preprocessor_factory fixture (tests/conftest.py), which constructs instances against a
temp cosmology-only config, so nothing here touches the real Hardcastle catalogue or cutout files - only tiny
synthetic images and hand-built catalogue records.
"""
import numpy as np
import pandas as pd
import pytest

from diffracc.rlf.agn_selection import select_rlagn


class TestCalculateSNR:
    """
    Tests that CutoutPreprocessor._calculate_snr_vectorised and _calculate_snr_single produce the same results
    for the same inputs, and that they handle zero-noise cases correctly.
    """
    
    def test_vectorised_matches_single_for_nonzero_noise(self, cutout_preprocessor_factory):
        """Test that the vectorised S/N calculation matches the single-value calculation for non-zero noise values."""
        cp = cutout_preprocessor_factory()
        noise = np.array([2.0, 4.0])
        peak = np.array([10.0, 12.0])
        vectorised = cp._calculate_snr_vectorised(noise, peak)
        singles = np.array([cp._calculate_snr_single(n, p) for n, p in zip(noise, peak)])
        np.testing.assert_allclose(vectorised, singles)
        np.testing.assert_allclose(vectorised, [5.0, 3.0])

    def test_vectorised_returns_minus_one_for_zero_noise(self, cutout_preprocessor_factory):
        """Test that the vectorised S/N calculation returns -1 for zero noise values."""
        cp = cutout_preprocessor_factory()
        result = cp._calculate_snr_vectorised(np.array([0.0, 2.0]), np.array([10.0, 10.0]))
        np.testing.assert_allclose(result, [-1.0, 5.0])

    def test_single_returns_minus_one_for_zero_noise(self, cutout_preprocessor_factory):
        """Test that the single-value S/N calculation returns -1 for zero noise values."""
        cp = cutout_preprocessor_factory()
        assert cp._calculate_snr_single(0.0, 10.0) == -1


class TestIdentifyImageStatus:
    """Tests that CutoutPreprocessor correctly identifies broken and incomplete images based on NaN values."""

    def test_all_nan_image_is_broken_not_incomplete(self, cutout_preprocessor_factory):
        """Test that an image with all NaN values is identified as broken and not incomplete."""
        cp = cutout_preprocessor_factory()
        image = np.full((80, 80), np.nan)
        assert cp._identify_broken_source_single(image) == True
        assert cp._identify_incomplete_image_single(image) == False

    def test_some_nan_image_is_incomplete_not_broken(self, cutout_preprocessor_factory):
        """Test that an image with some NaN values is identified as incomplete and not broken."""
        cp = cutout_preprocessor_factory()
        image = np.ones((80, 80))
        image[0, 0] = np.nan
        assert cp._identify_incomplete_image_single(image) == True
        assert cp._identify_broken_source_single(image) == False

    def test_no_nan_image_is_neither(self, cutout_preprocessor_factory):
        """Test that an image with no NaN values is identified as neither broken nor incomplete."""
        cp = cutout_preprocessor_factory()
        image = np.ones((80, 80))
        assert cp._identify_broken_source_single(image) == False
        assert cp._identify_incomplete_image_single(image) == False


class TestCalculateEdgeMaxSingle:
    """
    Tests that CutoutPreprocessor._calculate_edge_max_single correctly computes the edge_max ratio for a single image.
    """

    def test_matches_hand_computed_ratio(self, cutout_preprocessor_factory):
        """Test that the edge_max ratio is computed correctly for a hand-built image."""
        cp = cutout_preprocessor_factory()
        image = np.ones((80, 80))  # all edges = 1.0
        image[5, 5] = 10.0  # interior max
        assert cp._calculate_edge_max_single(image) == pytest.approx(1.0 / 10.0)

    def test_edge_dominated_image_gives_ratio_near_one(self, cutout_preprocessor_factory):
        """Test that an image dominated by edge pixels gives an edge_max ratio near 1."""
        cp = cutout_preprocessor_factory()
        image = np.zeros((80, 80))
        image[0, 0] = 5.0  # a corner (edge) pixel is the global max
        assert cp._calculate_edge_max_single(image) == pytest.approx(1.0)


class _SyntheticCatalogue:
    """Two hand-built synthetic sources with known edge_max/S-N/RLAGN-relevant quantities, shared by the vectorised
    and iterative flag-computation tests below so their outputs can be cross-checked against each other."""

    def __init__(self):
        """Builds a synthetic catalogue with two sources, each with a synthetic image and associated catalogue info."""
        # image A: border pixels = 1.0, interior peak = 10.0 -> edge_max ratio = 0.1
        self.image_a = np.ones((80, 80))
        self.image_a[40, 40] = 10.0
        # image B: border pixels = 3.0, interior peak = 6.0 -> edge_max ratio = 0.5
        self.image_b = np.full((80, 80), 3.0)
        self.image_b[40, 40] = 6.0

        self.dataset = pd.DataFrame([
            {'index': 0, 'pixel_values': self.image_a, 'broken': False, 'incomplete': False,
             'size': 0.0, 'S/N': 0.0, 'edge_max': 0.0, 'peak_flux': 0.0, 'rlagn': False},
            {'index': 1, 'pixel_values': self.image_b, 'broken': False, 'incomplete': False,
             'size': 0.0, 'S/N': 0.0, 'edge_max': 0.0, 'peak_flux': 0.0, 'rlagn': False},
        ])
        self.cat_info = [
            {'LAS': 10.0, 'Isl_rms': 0.5, 'mag_w1': 17.0, 'mag_w2': 15.0, 'mag_w3': 13.0, 'magerr_w3': 0.1,
             'L_144': 1e26, 'z_best': 0.3, 'Total_flux': 1000.0},  # mJy
            {'LAS': 20.0, 'Isl_rms': 1.0, 'mag_w1': 16.0, 'mag_w2': 14.0, 'mag_w3': 12.0, 'magerr_w3': 0.1,
             'L_144': 1e23, 'z_best': 0.5, 'Total_flux': 2000.0},  # mJy
        ]


class TestComputeFlags:
    """
    Cross-checks _compute_vectorised_flags and _compute_iterative_flags against each other and against
    hand-computed expected values, since both should implement the same per-image logic.
    """

    def _expected_rlagn(self, cp, cat):
        """
        Compute the expected RLAGN flags for the synthetic catalogue, using the same logic as the CutoutPreprocessor.
        """
        # Recompute independently via select_rlagn with the same per-source arrays/units the flag-computation
        # methods themselves pass to it, cross-checking argument order/units rather than re-deriving the astro.
        wise_1_mag = np.array([r['mag_w1'] for r in cat.cat_info])
        wise_2_mag = np.array([r['mag_w2'] for r in cat.cat_info])
        wise_3_mag = np.array([r['mag_w3'] for r in cat.cat_info])
        wise_3_magerr = np.array([r['magerr_w3'] for r in cat.cat_info])
        luminosities = np.array([r['L_144'] for r in cat.cat_info])
        redshifts = np.array([r['z_best'] for r in cat.cat_info])
        total_fluxes = np.array([r['Total_flux'] / 1000 for r in cat.cat_info])  # convert from mJy to Jy
        return select_rlagn(wise_1_mag, wise_2_mag, wise_3_mag, wise_3_magerr, luminosities, redshifts, total_fluxes,
                            cosmo=cp.cosmo, exclusive=cp.exclusive)[0]

    def test_vectorised_flags_match_hand_computed_values(self, cutout_preprocessor_factory):
        """Test that _compute_vectorised_flags produces the expected values for the synthetic catalogue."""
        cp = cutout_preprocessor_factory()
        cat = _SyntheticCatalogue()

        cp._compute_vectorised_flags(cat.dataset, cat.cat_info)

        np.testing.assert_allclose(cat.dataset['edge_max'].to_numpy(dtype=float), [0.1, 0.5])
        np.testing.assert_allclose(cat.dataset['size'].to_numpy(dtype=float), [10.0, 20.0])
        np.testing.assert_allclose(cat.dataset['peak_flux'].to_numpy(dtype=float), [10000.0, 6000.0])
        np.testing.assert_allclose(cat.dataset['S/N'].to_numpy(dtype=float), [20000.0, 6000.0])
        np.testing.assert_array_equal(cat.dataset['rlagn'].to_numpy(), self._expected_rlagn(cp, cat))

    def test_iterative_flags_match_hand_computed_values(self, cutout_preprocessor_factory):
        """Test that _compute_iterative_flags produces the expected values for the synthetic catalogue."""
        cp = cutout_preprocessor_factory()
        cat = _SyntheticCatalogue()

        cp._compute_iterative_flags(cat.dataset, cat.cat_info)

        np.testing.assert_allclose(cat.dataset['edge_max'].to_numpy(dtype=float), [0.1, 0.5])
        np.testing.assert_allclose(cat.dataset['size'].to_numpy(dtype=float), [10.0, 20.0])
        np.testing.assert_allclose(cat.dataset['peak_flux'].to_numpy(dtype=float), [10000.0, 6000.0])
        np.testing.assert_allclose(cat.dataset['S/N'].to_numpy(dtype=float), [20000.0, 6000.0])
        np.testing.assert_array_equal(cat.dataset['rlagn'].to_numpy(), self._expected_rlagn(cp, cat))

    def test_vectorised_and_iterative_agree(self, cutout_preprocessor_factory):
        """
        Test that _compute_vectorised_flags and _compute_iterative_flags produce the same results for the same inputs.
        """
        cp = cutout_preprocessor_factory()
        cat_v = _SyntheticCatalogue()
        cat_i = _SyntheticCatalogue()

        cp._compute_vectorised_flags(cat_v.dataset, cat_v.cat_info)
        cp._compute_iterative_flags(cat_i.dataset, cat_i.cat_info)

        pd.testing.assert_series_equal(cat_v.dataset['edge_max'], cat_i.dataset['edge_max'], check_dtype=False)
        pd.testing.assert_series_equal(cat_v.dataset['S/N'], cat_i.dataset['S/N'], check_dtype=False)
        pd.testing.assert_series_equal(cat_v.dataset['rlagn'], cat_i.dataset['rlagn'], check_dtype=False)

    def test_broken_and_incomplete_images_are_skipped(self, cutout_preprocessor_factory):
        """
        Test that broken and incomplete images are skipped during flag computation, leaving their default values intact.
        """
        cp = cutout_preprocessor_factory()
        cat = _SyntheticCatalogue()
        cat.dataset.loc[1, 'broken'] = True

        cp._compute_iterative_flags(cat.dataset, cat.cat_info)

        # untouched default values for the broken row
        assert cat.dataset.loc[1, 'edge_max'] == 0.0
        assert cat.dataset.loc[1, 'S/N'] == 0.0
        # the valid row was still processed
        assert cat.dataset.loc[0, 'edge_max'] == pytest.approx(0.1)
