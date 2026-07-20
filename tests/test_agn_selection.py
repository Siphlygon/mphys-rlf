"""
Unit tests for the RLAGN/SFG/RQQ selection logic in diffracc/rlf/agn_selection.py.

select_rlagn's masks depend on wise3_absmag, which _get_wise3_absmag derives from magnitudes + cosmology through a
k-correction. Rather than reverse-engineer magnitudes that land exactly on a classification boundary, these tests
monkeypatch _get_wise3_absmag to return chosen values directly, isolating the mask combination logic (the AND/OR/NOT
combinatorics, the exclusive-mode data-sufficiency cut, and the peak-flux/redshift override) from the magnitude/
cosmology conversion.
"""
import matplotlib

matplotlib.use("Agg")  # force a non-interactive backend before agn_selection imports pyplot - avoids opening a
                       # real GUI window (the environment's default backend is TkAgg) when tests exercise plotting.

import astropy.units as u
import numpy as np
import pytest

from diffracc.rlf import agn_selection
from diffracc.rlf.agn_selection import (
    WISE_2_FREQ,
    WISE_3_FREQ,
    _get_wise3_absmag,
    select_non_contaminants,
    select_rlagn,
)
from diffracc.utils.functions import k_corr_factor, mag_to_flux_w2, mag_to_flux_w3

# The luminosity below which a source is never classified RQQ - Hardcastle's rqq &= logl>=24.8 lower bound.
RQQ_LOWER_LUM = 10 ** 24.8


def _sfg_cutoff_lum(absmag: float) -> float:
    return 10 ** (14 - absmag / 2.5)


def _rqq_cutoff_lum(absmag: float) -> float:
    return 10 ** (25.3 + (-absmag - 27) * 2.0/7)


@pytest.fixture
def patch_wise3_absmag(monkeypatch):
    """Force _get_wise3_absmag to return a fixed array regardless of its (mag/redshift/cosmo) inputs."""
    def _patch(absmag: np.ndarray):
        monkeypatch.setattr(agn_selection, "_get_wise3_absmag", lambda *args, **kwargs: absmag)
    return _patch


def _default_rlagn_selection(flat_lcdm_cosmo,
                             wise1_mag=np.array([17.0]),
                             wise2_mag=np.array([15.0]),
                             wise3_mag=np.array([13.0]),
                             wise3_magerr=np.array([0.1]),
                             luminosities=np.array([1.0]),
                             redshifts=np.array([0.5]),
                             tot_fluxes=np.array([1.0]),
                             exclusive=False,
                             ):
    return select_rlagn(
        wise1_mag=wise1_mag, wise2_mag=wise2_mag, wise3_mag=wise3_mag,
        wise3_magerr=wise3_magerr, luminosities=luminosities, redshifts=redshifts,
        tot_fluxes=tot_fluxes, cosmo=flat_lcdm_cosmo, exclusive=exclusive)


class TestSelectRlagnMaskLogic:
    """Test the AND/OR/NOT logic of the RLAGN/SFG/RQQ masks, independent of the magnitude/cosmology conversion."""

    def test_faint_source_below_both_sfg_cutoffs_is_classified_sfg(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test a source with a luminosity below the SFG cutoff (and below 10**24.8) is classified as SFG, even if its
        absolute magnitude is bright enough to fail the RQQ cutoff. This ensures the SFG mask's second AND-condition
        (luminosity < 10**24.8) is applied correctly, and that the SFG mask is applied before the RQQ mask.
        """
        absmag = np.array([-25.0])
        cutoff = min(_sfg_cutoff_lum(absmag[0]), 10**24.8)
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(flat_lcdm_cosmo, luminosities=np.array([cutoff * 1e-3]))
        assert sfg == [True]
        assert rqq == [False]
        assert rlagn == [False]

    def test_source_above_sfg_cutoff_and_absmag_too_bright_for_rqq_is_rlagn(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test a source with a luminosity above the SFG cutoff and an absolute magnitude too bright to be RQQ is
        classified as RLAGN.
        """
        # absmag = -20 fails the RQQ absmag < -27 requirement outright, and a luminosity well above the SFG cutoff
        # fails that too, so the source should fall through to RLAGN.
        absmag = np.array([-20.0])
        cutoff = min(_sfg_cutoff_lum(absmag[0]), 10**24.8)
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(flat_lcdm_cosmo, luminosities=np.array([cutoff * 1e3]))
        assert sfg == [False]
        assert rqq == [False]
        assert rlagn == [True]

    def test_faint_source_below_rqq_cutoff_with_absmag_below_minus_27_is_classified_rqq(self,
                                                                                        patch_wise3_absmag,
                                                                                        flat_lcdm_cosmo):
        """
        Test a source with a luminosity below the RQQ cutoff and an absolute magnitude below -27 is classified as RQQ,
        even if its luminosity is above the SFG cutoff.
        """
        absmag = np.array([-30.0])
        rqq_cutoff = _rqq_cutoff_lum(absmag[0])
        # Keep well above 10**24.8 so both the SFG mask's second AND-condition and the RQQ mask's lower bound are
        # unambiguous, isolating RQQ. (1e25, not 10**25: the latter is a Python int too large for int64, which numpy
        # would make an object array that np.isfinite in the sample mask rejects.)
        luminosity = max(rqq_cutoff * 1e-3, 1e25)
        assert RQQ_LOWER_LUM < luminosity < rqq_cutoff, "test luminosity must sit in the RQQ band for this scenario"
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(flat_lcdm_cosmo, luminosities=np.array([luminosity]))
        assert rqq == [True]
        assert sfg == [False]
        assert rlagn == [False]

    def test_source_below_rqq_lower_luminosity_bound_is_not_rqq(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test the RQQ mask's lower luminosity bound (Hardcastle's rqq &= logl>=24.8): a source with abs_w3 < -27 and a
        luminosity below the RQQ cutoff line, but *below* 10**24.8, must NOT be classified RQQ. With abs_w3 = -30 such
        a source falls below the SFG divide line too, so it is classified SFG instead - which is exactly why the lower
        bound matters (without it, the same source is double-counted into RQQ, inflating the RQQ total).
        """
        absmag = np.array([-30.0])
        luminosity = 10**24.0  # below 10**24.8, and below both the SFG divide and the RQQ cutoff for absmag=-30
        assert luminosity < RQQ_LOWER_LUM and luminosity < _rqq_cutoff_lum(absmag[0])
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(flat_lcdm_cosmo, luminosities=np.array([luminosity]))
        assert rqq == [False]  # excluded by the lower bound despite being below the RQQ cutoff line
        assert sfg == [True]
        assert rlagn == [False]

    def test_nan_wise3_magerr_forces_sfg_and_rqq_false_in_inclusive_mode(self,
                                                                         patch_wise3_absmag,
                                                                         flat_lcdm_cosmo):
        """
        Test that in the default (inclusive, exclusive=False) mode a source with np.nan wise3_magerr is never
        classified as SFG or RQQ, even if its luminosity is below the SFG/RQQ cutoffs - it falls through to RLAGN. In
        inclusive mode the magerr-sufficiency cut is applied to the SFG/RQQ masks, so a source with no trustworthy W3
        measurement cannot be positively excluded. (The exclusive/narrow mode instead classifies such an in-zone
        source - see TestExclusiveMode.)
        """
        absmag = np.array([-30.0])  # in the RQQ zone by position, but has no W3 detection to confirm it
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(flat_lcdm_cosmo,
                                                   luminosities=np.array([1.0]), wise3_magerr=np.array([np.nan]))
        assert sfg == [False]
        assert rqq == [False]
        assert rlagn == [True]

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    @pytest.mark.parametrize("band", ["wise1_mag", "wise2_mag", "wise3_mag"])
    def test_non_finite_wise_magnitude_excludes_source_from_all_masks(self, patch_wise3_absmag, flat_lcdm_cosmo,
                                                                      band, bad):
        """
        Test the sample-completeness cut (Hardcastle's selection.py requires mag_w1/w2/w3 all present): a source with a
        non-finite magnitude in any of the three WISE bands is dropped from all three class masks. np.isfinite (not
        np.isnan) is used deliberately so an inf sentinel is caught as well as a NaN.
        """
        absmag = np.array([-20.0])  # would otherwise classify as RLAGN
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(
            flat_lcdm_cosmo, luminosities=np.array([_sfg_cutoff_lum(-20.0) * 1e3]), **{band: np.array([bad])})
        assert rlagn == [False]
        assert sfg == [False]
        assert rqq == [False]


class TestExclusiveMode:
    """
    Test the `exclusive` argument's effect on sources with insufficient data to classify as SFG/RQQ, independent of the
    magnitude/cosmology conversion.
    """

    def _insufficient_data_source(self, flat_lcdm_cosmo, patch_wise3_absmag, exclusive: bool):
        """
        Return a source with np.nan wise3_magerr and a luminosity below the SFG/RQQ cutoffs, which is insufficient data
        to classify as SFG/RQQ. The `exclusive` argument controls whether the source is dropped from RLAGN or not.
        """
        absmag = np.array([-30.0])
        patch_wise3_absmag(absmag)
        return _default_rlagn_selection(flat_lcdm_cosmo,
                                        luminosities=np.array([1.0]), wise3_magerr=np.array([np.nan]),
                                        exclusive=exclusive)

    def test_non_exclusive_keeps_sources_with_insufficient_data_as_rlagn(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test that a source with insufficient data to classify as SFG/RQQ is still classified as RLAGN when
        exclusive=False.
        """
        rlagn, sfg, rqq = self._insufficient_data_source(flat_lcdm_cosmo, patch_wise3_absmag, exclusive=False)
        assert rlagn == [True]

    def test_exclusive_drops_in_zone_source_with_insufficient_data_from_rlagn(self, patch_wise3_absmag,
                                                                              flat_lcdm_cosmo):
        """
        Test that in exclusive (narrow) mode an in-zone source with insufficient data (NaN wise3_magerr) is dropped
        from RLAGN. Unlike inclusive mode, exclusive mode does NOT require a W3 detection to classify: the source is
        positively classified SFG on the strength of its (upper-limit) position, which is what removes it from RLAGN.
        """
        rlagn, sfg, rqq = self._insufficient_data_source(flat_lcdm_cosmo, patch_wise3_absmag, exclusive=True)
        assert rlagn == [False]
        # In narrow mode the source is classified by position despite the missing magerr - here as SFG (absmag=-30,
        # luminosity=1.0 sits in the SFG zone), which is precisely the mechanism that excludes it from RLAGN.
        assert sfg == [True]
        assert rqq == [False]

    def test_exclusive_keeps_out_of_zone_source_with_insufficient_data_as_rlagn(self, patch_wise3_absmag,
                                                                                flat_lcdm_cosmo):
        """
        Test that even in exclusive (narrow) mode an insufficient-data (NaN wise3_magerr) source lying OUTSIDE the
        SFG/RQQ zones is kept as RLAGN. Exclusive mode does not wholesale-drop magerr-less sources; it only excludes
        the ones that positively classify as SFG/RQQ by position, so a source confidently RLAGN by position survives.
        """
        absmag = np.array([-20.0])  # too faint in W3 for RQQ, paired with a high luminosity so it is not SFG either
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(
            flat_lcdm_cosmo, luminosities=np.array([_sfg_cutoff_lum(-20.0) * 1e3]),
            wise3_magerr=np.array([np.nan]), exclusive=True)
        assert rlagn == [True]
        assert sfg == [False]
        assert rqq == [False]

    def test_exclusive_does_not_drop_sources_with_complete_data(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test that a source with sufficient data to classify as SFG/RQQ is not dropped from RLAGN when
        exclusive=True.
        """
        absmag = np.array([-20.0])  # deliberately not SFG/RQQ so it falls through to RLAGN
        cutoff = min(_sfg_cutoff_lum(absmag[0]), 10**24.8)
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(flat_lcdm_cosmo,
                                                    luminosities=np.array([cutoff * 1e3]),
                                                    exclusive=True)
        assert rlagn == [True]


class TestLowFluxAndRedshiftOverride:
    """
    Test the survey cut that excludes sources at or below the peak-flux threshold, or at or below z=0.01, from all
    three class masks (they are dropped from the sample entirely, independent of the WISE-based classification).
    """

    def test_low_tot_fluxes_excludes_source_even_when_it_would_classify(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test a source with a total flux at or below the threshold is dropped from all three masks, even though its
        luminosity/magnitude would otherwise place it in one of the classes.
        """
        absmag = np.array([-25.0])
        cutoff = min(_sfg_cutoff_lum(absmag[0]), 10**24.8)
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(
            flat_lcdm_cosmo,
            luminosities=np.array([cutoff * 1e-3]),
            tot_fluxes=np.array([1.1e-3]),  # exactly at the flux threshold
        )
        assert rlagn == [False]
        assert sfg == [False]
        assert rqq == [False]

    def test_single_source_below_threshold_does_not_crash_the_real_absmag_path(self, flat_lcdm_cosmo):
        """
        Regression test for apply_preprocessing's per-source iterative flag path: select_rlagn called with scalar
        inputs for a single source that fails the flux/redshift gate (the routine case there, not a rare one) must
        not crash.
        """
        # Deliberately does NOT use patch_wise3_absmag - every other test in this file bypasses the real
        # _get_wise3_absmag, which is exactly why this crash went unnoticed: the empty-array bug only fires inside
        # the real cosmo.luminosity_distance call, when `sample` (here, a single element) is all-False because this
        # one source is below the flux threshold. See test_all_invalid_input_returns_nan_without_crashing in
        # TestGetWise3Absmag for the direct reproduction of the underlying astropy/numpy interaction.
        rlagn, sfg, rqq = select_rlagn(
            wise1_mag=14.0, wise2_mag=15.0, wise3_mag=13.0, wise3_magerr=0.1,
            luminosities=1e26, redshifts=0.3,
            tot_fluxes=5e-4,  # below the 1.1e-3 Jy threshold -> sample is all-False for this single source
            cosmo=flat_lcdm_cosmo, exclusive=True,
        )
        assert rlagn == [False]
        assert sfg == [False]
        assert rqq == [False]

    def test_low_redshift_excludes_source_even_when_it_would_classify(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test a source with a redshift at or below 0.01 is dropped from all three masks, even though its
        luminosity/magnitude would otherwise place it in one of the classes.
        """
        absmag = np.array([-30.0])
        rqq_cutoff = _rqq_cutoff_lum(absmag[0])
        luminosity = max(rqq_cutoff * 1e-3, 1e25)  # 1e25 float, not 10**25 int (would become an object array)
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(
            flat_lcdm_cosmo,
            luminosities=np.array([luminosity]),
            redshifts=np.array([0.01]),  # exactly at the redshift threshold
        )
        assert rlagn == [False]
        assert sfg == [False]
        assert rqq == [False]

    def test_override_does_not_affect_sources_above_both_thresholds(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test a source with a total flux and redshift above the default thresholds is classified according to the SFG/RQQ
        mask logic, even if its luminosity is below the SFG/RQQ cutoffs.
        """
        absmag = np.array([-25.0])
        cutoff = min(_sfg_cutoff_lum(absmag[0]), 10**24.8)
        patch_wise3_absmag(absmag)

        rlagn, sfg, rqq = _default_rlagn_selection(
            flat_lcdm_cosmo,
            luminosities=np.array([cutoff * 1e-3]),
            redshifts=np.array([0.5]),
            tot_fluxes=np.array([1.0]),
        )
        assert sfg == [True]
        assert rqq == [False]
        assert rlagn == [False]


class _FakeCatalogue:
    """
    Minimal stand-in for HardcastleCatalogue, backed by an in-memory dict of columns keyed by Source.value. Lets
    get_catalogue_info be tested without hitting the real FITS catalogue file (or triggering CatalogueDownloader's
    network download).
    """
    def __init__(self, columns: dict):
        self._columns = columns

    def get_value_column(self, source):
        key = source.value if hasattr(source, "value") else source
        return self._columns[key]


@pytest.fixture
def fake_catalogue(monkeypatch):
    """
    Returns a factory that makes agn_selection.HardcastleCatalogue(...) return a _FakeCatalogue with given columns.
    """
    def _patch(columns: dict):
        monkeypatch.setattr(agn_selection, "HardcastleCatalogue", lambda resolved_only=False: _FakeCatalogue(columns))
    return _patch


class TestGetCatalogueInfo:
    """
    get_catalogue_info returns a 4-tuple (redshifts, fluxes, luminosities, resolved) - not a dict - restricted to
    rlagn_mask. _get_wise3_absmag is patched as elsewhere in this file, isolating these tests from the magnitude/
    cosmology conversion; HardcastleCatalogue is patched via fake_catalogue to avoid any real I/O.
    """

    def test_converts_catalogue_flux_from_mjy_to_jy(self, fake_catalogue, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test that get_catalogue_info converts the catalogue's Total_flux from mJy to Jy in its output fluxes array.
        """
        # A single source that classifies cleanly as RLAGN (bright patched absmag, high luminosity) and sits well clear
        # of the survey cut (z > 0.01, peak flux > threshold), so it is kept and the flux unit conversion is what's
        # under test, isolated from the classification logic.
        patch_wise3_absmag(np.array([-20.0]))
        fake_catalogue({
            "z_best": np.array([0.3]),
            "Total_flux": np.array([50.0]),  # mJy, as stored in the real catalogue
            "L_144": np.array([1e26]),
            "mag_w1": np.array([17.0]),
            "mag_w2": np.array([15.0]),
            "mag_w3": np.array([13.0]),
            "magerr_w3": np.array([0.1]),
            "Resolved": np.array([True]),
        })

        redshifts, fluxes, luminosities, resolved = agn_selection.get_catalogue_info(cosmo=flat_lcdm_cosmo)

        np.testing.assert_allclose(fluxes, [0.05])  # 50 mJy -> 0.05 Jy

    def test_returns_only_rlagn_sources_across_all_four_columns_in_order(self,
                                                                         fake_catalogue,
                                                                         patch_wise3_absmag,
                                                                         flat_lcdm_cosmo):
        """
        Test that get_catalogue_info returns only the sources classified as RLAGN, across all four output arrays, in the
        same order as the input catalogue.
        """
        # Two sources: index 0 is a clear RLAGN (bright absmag, high luminosity), index 1 is a clear SFG (faint
        # absmag, luminosity below its own SFG cutoff) - both with fluxes/redshifts well clear of the override, so
        # only the SFG/RLAGN classification decides the outcome.
        rlagn_absmag, sfg_absmag = -20.0, -25.0
        rlagn_luminosity = _sfg_cutoff_lum(rlagn_absmag) * 1e3
        sfg_luminosity = _sfg_cutoff_lum(sfg_absmag) * 1e-3
        patch_wise3_absmag(np.array([rlagn_absmag, sfg_absmag]))
        fake_catalogue({
            "z_best": np.array([0.3, 0.4]),
            "Total_flux": np.array([50.0, 60.0]),  # mJy
            "L_144": np.array([rlagn_luminosity, sfg_luminosity]),
            "mag_w1": np.array([17.0, 17.0]),
            "mag_w2": np.array([15.0, 15.0]),
            "mag_w3": np.array([13.0, 13.0]),
            "magerr_w3": np.array([0.1, 0.1]),
            "Resolved": np.array([True, False]),
        })

        redshifts, fluxes, luminosities, resolved = agn_selection.get_catalogue_info(cosmo=flat_lcdm_cosmo)

        np.testing.assert_allclose(redshifts, [0.3])
        np.testing.assert_allclose(fluxes, [0.05])
        np.testing.assert_allclose(luminosities, [rlagn_luminosity])
        np.testing.assert_array_equal(resolved, [True])

    def test_keeps_nan_magerr_source_as_rlagn_in_inclusive_mode(self,
                                                                fake_catalogue,
                                                                patch_wise3_absmag,
                                                                flat_lcdm_cosmo):
        """
        get_catalogue_info runs select_rlagn in inclusive (exclusive=False) mode, so a source with a NaN wise3_magerr
        that is not otherwise SFG/RQQ is kept as RLAGN rather than dropped. Source 0 (NaN magerr, but RLAGN by
        position) survives; source 1 (a detected SFG) does not.
        """
        rlagn_absmag, sfg_absmag = -20.0, -25.0
        rlagn_luminosity = _sfg_cutoff_lum(rlagn_absmag) * 1e3  # high -> not SFG; absmag -20 -> not RQQ
        sfg_luminosity = _sfg_cutoff_lum(sfg_absmag) * 1e-3     # low -> SFG
        patch_wise3_absmag(np.array([rlagn_absmag, sfg_absmag]))
        fake_catalogue({
            "z_best": np.array([0.3, 0.4]),
            "Total_flux": np.array([50.0, 60.0]),  # mJy
            "L_144": np.array([rlagn_luminosity, sfg_luminosity]),
            "mag_w1": np.array([17.0, 17.0]),
            "mag_w2": np.array([15.0, 15.0]),
            "mag_w3": np.array([13.0, 13.0]),
            "magerr_w3": np.array([np.nan, 1.0]),  # source 0 has no W3 detection, but is RLAGN by position
            "Resolved": np.array([True, False]),
        })

        redshifts, fluxes, luminosities, resolved = agn_selection.get_catalogue_info(cosmo=flat_lcdm_cosmo)

        np.testing.assert_allclose(redshifts, [0.3])  # only the RLAGN source (index 0) is returned
        np.testing.assert_allclose(luminosities, [rlagn_luminosity])
        np.testing.assert_array_equal(resolved, [True])

    def test_plot_rlagn_selection_contour_false_does_not_call_plotting(self,
                                                                       fake_catalogue,
                                                                       patch_wise3_absmag,
                                                                       flat_lcdm_cosmo,
                                                                       monkeypatch):
        """
        Test that get_catalogue_info does not call _plot_rlagn_selection_contour when
        plot_rlagn_selection_contour=False.
        """
        called = []
        monkeypatch.setattr(agn_selection, "_plot_rlagn_selection_contour", lambda *a, **k: called.append(True))
        patch_wise3_absmag(np.array([-20.0]))
        fake_catalogue({
            "z_best": np.array([0.3]),
            "Total_flux": np.array([50.0]),
            "L_144": np.array([1e26]),
            "mag_w1": np.array([17.0]),
            "mag_w2": np.array([15.0]),
            "mag_w3": np.array([13.0]),
            "magerr_w3": np.array([0.1]),
            "Resolved": np.array([True]),
        })

        agn_selection.get_catalogue_info(cosmo=flat_lcdm_cosmo, plot_rlagn_selection_contour=False)

        assert called == []

    def test_plot_rlagn_selection_contour_true_calls_plotting(self,
                                                              fake_catalogue,
                                                              patch_wise3_absmag,
                                                              flat_lcdm_cosmo,
                                                              monkeypatch):
        """Test that get_catalogue_info calls _plot_rlagn_selection_contour when plot_rlagn_selection_contour=True."""
        called = []
        monkeypatch.setattr(agn_selection, "_plot_rlagn_selection_contour", lambda *a, **k: called.append(True))
        patch_wise3_absmag(np.array([-20.0]))
        fake_catalogue({
            "z_best": np.array([0.3]),
            "Total_flux": np.array([50.0]),
            "L_144": np.array([1e26]),
            "mag_w1": np.array([17.0]),
            "mag_w2": np.array([15.0]),
            "mag_w3": np.array([13.0]),
            "magerr_w3": np.array([0.1]),
            "Resolved": np.array([True]),
        })

        agn_selection.get_catalogue_info(cosmo=flat_lcdm_cosmo, plot_rlagn_selection_contour=True)

        assert called == [True]


class TestGetWise3Absmag:
    """
    Direct tests of _get_wise3_absmag's own formula, since every other test in this file bypasses it via
    patch_wise3_absmag. Expected values are hand-assembled from the already-independently-tested mag_to_flux_w2/w3
    and k_corr_factor building blocks, so this only tests _get_wise3_absmag's own combination logic (the distance
    modulus and spectral-index derivation), not those building blocks themselves.
    """

    def test_matches_hand_computed_value(self, flat_lcdm_cosmo):
        """Test that _get_wise3_absmag returns the expected value for a known input, matching a hand-computed result."""
        wise3_mag = np.array([12.0])
        wise2_mag = np.array([13.5])
        redshifts = np.array([0.3])

        actual = _get_wise3_absmag(wise3_mag, wise2_mag, redshifts, flat_lcdm_cosmo)

        wise3_flux = mag_to_flux_w3(wise3_mag)
        wise2_flux = mag_to_flux_w2(wise2_mag)
        # Use the module's own band frequencies so this expected value can't drift from the implementation (the code
        # uses 12.1um for W3, not 12um).
        spectral_inds = -np.log(wise3_flux / wise2_flux) / np.log(WISE_3_FREQ / WISE_2_FREQ)
        d_pc = flat_lcdm_cosmo.luminosity_distance(redshifts).to(u.parsec).value
        expected = wise3_mag - 5 * (np.log10(d_pc) - 1) \
            - k_corr_factor(redshifts, mag_space=True, spectral_index=spectral_inds)

        np.testing.assert_allclose(actual, expected)

    def test_brighter_apparent_magnitude_gives_brighter_absolute_magnitude(self, flat_lcdm_cosmo):
        """
        Test that a brighter apparent magnitude (lower numeric value) gives a brighter absolute magnitude (also lower
        numeric value) at fixed redshift and color.
        """
        # Absolute magnitude should track apparent magnitude in the same direction at fixed redshift/color (lower
        # numeric value = brighter, so a smaller wise3_mag should give a smaller wise3_absmag).
        wise2_mag = np.array([13.5])
        redshifts = np.array([0.3])

        dim = _get_wise3_absmag(np.array([14.0]), wise2_mag, redshifts, flat_lcdm_cosmo)[0]
        bright = _get_wise3_absmag(np.array([12.0]), wise2_mag, redshifts, flat_lcdm_cosmo)[0]
        assert bright < dim

    def test_all_invalid_input_returns_nan_without_crashing(self, flat_lcdm_cosmo):
        """
        Test that _get_wise3_absmag returns NaN (rather than crashing) when every source fails the redshift/magnitude
        validity check, so `valid` is all-False and cosmo.luminosity_distance would otherwise be called with a
        zero-length array.
        """
        # astropy's FLRW distance methods fall back to a numerically-integrated path wrapped in np.vectorize whenever
        # there's no closed-form comoving distance (e.g. non-zero Tcmb0, as flat_lcdm_cosmo has, adds a radiation
        # term) - and np.vectorize raises ValueError on zero-length input rather than returning an empty array. This
        # is exactly what apply_preprocessing's per-source iterative flag path hits every time a single source fails
        # the validity check, which is routine, not rare.
        result = _get_wise3_absmag(np.array([np.nan]), np.array([15.0]), np.array([0.3]), flat_lcdm_cosmo)
        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_scalar_all_invalid_input_returns_nan_without_crashing(self, flat_lcdm_cosmo):
        """Test the scalar-input form of the same all-invalid case (the shape apply_preprocessing calls with)."""
        result = _get_wise3_absmag(np.nan, 15.0, 0.3, flat_lcdm_cosmo)
        assert np.isscalar(result) or result.ndim == 0
        assert np.isnan(result)


class TestPlotRlagnSelectionContour:
    """
    Smoke test for the plotting function itself - forced onto the Agg backend (see module-level matplotlib.use at
    the top of this file) so it never opens a real window, and run inside a tmp_path so it doesn't litter the repo
    with lum_vs_w3.png.
    """

    @pytest.mark.parametrize("with_masks", [False, True])
    def test_runs_without_error_and_saves_a_figure(self, monkeypatch, tmp_path, flat_lcdm_cosmo, with_masks):
        """
        Test that _plot_rlagn_selection_contour runs without error and saves a figure to the current working
        directory, both with and without the SFG/RQQ population overlays.
        """
        import matplotlib.pyplot as plt

        monkeypatch.chdir(tmp_path)
        rng = np.random.default_rng(0)
        n = 20
        wise2_mag = rng.uniform(12, 16, n)
        wise3_mag = rng.uniform(10, 14, n)
        redshifts = rng.uniform(0.05, 0.5, n)
        luminosities = 10 ** rng.uniform(22, 27, n)
        masks = dict(sfg_mask=rng.random(n) < 0.3, rqq_mask=rng.random(n) < 0.2) if with_masks else {}

        try:
            agn_selection._plot_rlagn_selection_contour(
                wise2_mag, wise3_mag, redshifts, luminosities, flat_lcdm_cosmo, **masks)
            assert (tmp_path / "lum_vs_w3.png").exists()
        finally:
            plt.close("all")


class TestSelectNonContaminants:
    """
    Tests for select_non_contaminants, which answers "is this source NOT a known contaminant?" rather than
    select_rlagn's "is this source in the H25 RLAGN sample?". The distinction only shows up for sources that cannot
    be classified at all - select_rlagn drops them (they can't be confirmed RLAGN), this keeps them (they can't be
    shown to be contaminants) - so these tests deliberately focus on those, and on confirming that a source which
    *is* positively classified is still removed.
    """

    def test_positively_classified_sfg_is_removed(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """Test that a source the WISE criteria positively place in the SFG zone is dropped."""
        absmag = np.array([-25.0])
        cutoff = min(_sfg_cutoff_lum(absmag[0]), RQQ_LOWER_LUM)
        patch_wise3_absmag(absmag)

        keep = select_non_contaminants(
            wise1_mag=np.array([17.0]), wise2_mag=np.array([15.0]), wise3_mag=np.array([13.0]),
            wise3_magerr=np.array([0.1]), luminosities=np.array([cutoff * 1e-3]),
            redshifts=np.array([0.5]), tot_fluxes=np.array([1.0]), cosmo=flat_lcdm_cosmo)

        assert keep == [False]

    def test_unclassifiable_source_is_kept_unlike_select_rlagn(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test that a source failing the H25 sample gate is kept, where select_rlagn would drop it.

        This is the entire point of the mode: a source below the 1.1 mJy total-flux cut cannot be classified, so it
        is not a known contaminant and survives - even though select_rlagn (correctly, for the RLF) excludes it.
        """
        absmag = np.array([-20.0])
        patch_wise3_absmag(absmag)
        common = dict(
            wise1_mag=np.array([17.0]), wise2_mag=np.array([15.0]), wise3_mag=np.array([13.0]),
            wise3_magerr=np.array([0.1]), luminosities=np.array([1e26]),
            redshifts=np.array([0.5]), tot_fluxes=np.array([5e-4]),  # below the 1.1e-3 Jy sample cut
        )

        rlagn, _, _ = select_rlagn(**common, cosmo=flat_lcdm_cosmo, exclusive=False)
        keep = select_non_contaminants(**common, cosmo=flat_lcdm_cosmo)

        assert rlagn == [False]  # select_rlagn drops it: can't confirm it's RLAGN
        assert keep == [True]    # select_non_contaminants keeps it: can't show it's a contaminant

    def test_source_without_wise_data_is_kept(self, flat_lcdm_cosmo):
        """
        Test that a source with no WISE magnitudes at all is kept.

        Deliberately does not patch _get_wise3_absmag, so this also covers the real all-invalid path: with NaN
        magnitudes the sample gate is all-False and no classification is possible.
        """
        keep = select_non_contaminants(
            wise1_mag=np.array([np.nan]), wise2_mag=np.array([np.nan]), wise3_mag=np.array([np.nan]),
            wise3_magerr=np.array([np.nan]), luminosities=np.array([1e26]),
            redshifts=np.array([0.5]), tot_fluxes=np.array([5e-3]), cosmo=flat_lcdm_cosmo)

        assert keep == [True]

    def test_in_zone_w3_upper_limit_source_is_kept(self, patch_wise3_absmag, flat_lcdm_cosmo):
        """
        Test that a source inside the SFG zone but with only a W3 upper limit (NaN wise3_magerr) is kept.

        Without a real W3 detection the zone classification rests on a limiting magnitude, so the source is not
        *definitely* a contaminant. This is fixed behaviour, not configurable - contrast select_rlagn(exclusive=True),
        which does drop such a source.
        """
        absmag = np.array([-25.0])
        cutoff = min(_sfg_cutoff_lum(absmag[0]), RQQ_LOWER_LUM)
        patch_wise3_absmag(absmag)
        common = dict(
            wise1_mag=np.array([17.0]), wise2_mag=np.array([15.0]), wise3_mag=np.array([13.0]),
            wise3_magerr=np.array([np.nan]),  # W3 upper limit, not a detection
            luminosities=np.array([cutoff * 1e-3]), redshifts=np.array([0.5]), tot_fluxes=np.array([1.0]),
        )

        keep = select_non_contaminants(**common, cosmo=flat_lcdm_cosmo)
        rlagn_exclusive, _, _ = select_rlagn(**common, cosmo=flat_lcdm_cosmo, exclusive=True)

        assert keep == [True]
        assert rlagn_exclusive == [False]

    def test_keeps_a_superset_of_select_rlagn_across_a_mixed_population(self, flat_lcdm_cosmo):
        """
        Test that over a mixed population the kept set is a strict superset of select_rlagn's RLAGN sample.

        Every RLAGN is by definition not a contaminant, so anything select_rlagn keeps this must keep too; the
        converse does not hold, since this also keeps the unclassifiable sources.
        """
        rng = np.random.default_rng(0)
        n = 200
        common = dict(
            wise1_mag=rng.uniform(12, 18, n), wise2_mag=rng.uniform(12, 18, n), wise3_mag=rng.uniform(10, 17, n),
            wise3_magerr=np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0.01, 0.2, n)),
            luminosities=10 ** rng.uniform(21, 28, n), redshifts=rng.uniform(0.005, 0.6, n),
            tot_fluxes=10 ** rng.uniform(-4, -1, n),
        )

        rlagn, _, _ = select_rlagn(**common, cosmo=flat_lcdm_cosmo, exclusive=False)
        keep = select_non_contaminants(**common, cosmo=flat_lcdm_cosmo)

        assert np.all(keep[rlagn])       # everything select_rlagn keeps is also kept here
        assert keep.sum() > rlagn.sum()  # and strictly more, from the unclassifiable sources
