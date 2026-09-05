"""
Unit tests for diffracc/data/contamination.py.

The two flag functions accept a pre-loaded `components` dict, so every test here runs against tiny hand-built component
catalogues - nothing touches the real 1 GB component FITS. Geometry is set up in arcsec offsets from a source position
so the expected cropped/foreign outcomes can be reasoned about by hand.
"""
import numpy as np
import pytest

from diffracc.data import cutout_quality as con

SRC_RA, SRC_DEC = 180.0, 40.0
HALF = con.CUTOUT_SIZE_ARCSEC / 2.0  # 60"


def _components(specs):
    """
    Build a components dict from a list of specs, each a dict with keys:
    parent, dra, ddec (arcsec offsets from the source), peak (mJy), and optional maj/min (arcsec FWHM), pa (deg).
    """
    cos_dec = np.cos(np.radians(SRC_DEC))
    return {
        "ra": np.array([SRC_RA + s["dra"] / 3600.0 / cos_dec for s in specs], dtype=float),
        "dec": np.array([SRC_DEC + s["ddec"] / 3600.0 for s in specs], dtype=float),
        "peak": np.array([s["peak"] for s in specs], dtype=float),
        "maj": np.array([s.get("maj", 6.0) for s in specs], dtype=float),
        "min": np.array([s.get("min", 6.0) for s in specs], dtype=float),
        "pa": np.array([s.get("pa", 0.0) for s in specs], dtype=float),
        "parent": np.array([s["parent"] for s in specs], dtype="<U32"),
    }


def _one_source(name="SRC", isl_rms=0.1):
    """A single-source argument bundle (ra, dec, name, isl_rms arrays) centred on the test position."""
    return (np.array([SRC_RA]), np.array([SRC_DEC]), np.array([name]), np.array([isl_rms]))


# ------------------------- helpers -------------------------
class TestHelpers:
    """Helpers are private, but we can still test them in isolation."""
    def test_as_str_array_decodes_bytes_and_strips(self):
        """Test that _as_str_array() decodes bytes to str and strips whitespace."""
        out = con._as_str_array(np.array([b"ILT J1 ", b" ILT J2"], dtype="S8"))
        np.testing.assert_array_equal(out, ["ILT J1", "ILT J2"])

    def test_as_str_array_passes_unicode_through(self):
        """Test that _as_str_array() passes unicode strings through unchanged."""
        out = con._as_str_array(np.array([" A ", "B"]))
        np.testing.assert_array_equal(out, ["A", "B"])

    def test_unit_vectors_are_normalised(self):
        """Test that _unit_vectors() returns unit vectors for given spherical coordinates."""
        v = con._unit_vectors(np.array([0.0, 90.0, 180.0]), np.array([0.0, 45.0, -30.0]))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)

    def test_group_by_parent_recovers_each_sources_components(self):
        """Test that _group_by_parent() correctly groups components by their parent source."""
        parent = np.array(["B", "A", "B", "C", "A"])
        names, starts, order = con._group_by_parent(parent)
        np.testing.assert_array_equal(names, ["A", "B", "C"])
        # the block for "A" should point at the two "A" rows (indices 1 and 4)
        a_block = order[starts[0]:starts[1]]
        assert sorted(a_block.tolist()) == [1, 4]


# ------------------------- foreign contamination -------------------------
class TestFlagForeignComponents:
    """Unit tests for the `flag_foreign_components()` function, which flags sources with bright foreign neighbours."""

    def test_bright_foreign_neighbour_is_flagged(self):
        """Test that a bright foreign neighbour is correctly flagged as a contaminant."""
        comp = _components([
            {"parent": "SRC", "dra": 0, "ddec": 0, "peak": 1.0},     # the target itself
            {"parent": "OTHER", "dra": 0, "ddec": 30, "peak": 1.0},  # neighbour, 30" away, 10 sigma
        ])
        res = con.flag_foreign_components(*_one_source(), components=comp)
        assert bool(res["foreign_contaminant"][0]) is True
        assert int(res["n_foreign_detected"][0]) == 1
        assert res["brightest_foreign_flux"][0] == pytest.approx(1.0)
        assert res["brightest_foreign_snr"][0] == pytest.approx(10.0)

    def test_faint_foreign_neighbour_below_threshold_not_flagged(self):
        """Test that a faint foreign neighbour below the detection threshold is not flagged as a contaminant."""
        comp = _components([
            {"parent": "SRC", "dra": 0, "ddec": 0, "peak": 1.0},
            {"parent": "OTHER", "dra": 0, "ddec": 30, "peak": 0.3},  # 3 sigma < 5 sigma
        ])
        res = con.flag_foreign_components(*_one_source(), components=comp)
        assert bool(res["foreign_contaminant"][0]) is False
        assert int(res["n_foreign_components"][0]) == 1   # present in-frame...
        assert int(res["n_foreign_detected"][0]) == 0     # ...but not detected

    def test_foreign_neighbour_outside_cutout_is_ignored(self):
        """Test that a bright foreign neighbour outside the cutout is not flagged as a contaminant."""
        comp = _components([
            {"parent": "SRC", "dra": 0, "ddec": 0, "peak": 1.0},
            {"parent": "OTHER", "dra": 0, "ddec": 80, "peak": 5.0},  # bright but 80" > 60" half-width
        ])
        res = con.flag_foreign_components(*_one_source(), components=comp)
        assert bool(res["foreign_contaminant"][0]) is False
        assert int(res["n_foreign_components"][0]) == 0

    def test_own_components_never_count_as_foreign(self):
        """Test that a bright component belonging to the source itself is not flagged as a foreign contaminant."""
        comp = _components([
            {"parent": "SRC", "dra": 0, "ddec": 0, "peak": 1.0},
            {"parent": "SRC", "dra": 0, "ddec": 40, "peak": 5.0},   # a bright *own* lobe
        ])
        res = con.flag_foreign_components(*_one_source(), components=comp)
        assert bool(res["foreign_contaminant"][0]) is False
        assert int(res["n_foreign_components"][0]) == 0

    def test_threshold_is_configurable(self):
        """Test that the sigma threshold for flagging foreign components can be configured."""
        comp = _components([
            {"parent": "SRC", "dra": 0, "ddec": 0, "peak": 1.0},
            {"parent": "OTHER", "dra": 0, "ddec": 30, "peak": 0.3},  # 3 sigma
        ])
        res = con.flag_foreign_components(*_one_source(), components=comp, sigma_threshold=2.0)
        assert bool(res["foreign_contaminant"][0]) is True   # 3 sigma clears a 2-sigma bar


# ------------------------- cropping -------------------------
class TestFlagCroppedSources:
    """
    Unit tests for the `flag_cropped_sources()` function, which flags sources with components cropped by the cutout.
    """

    def test_compact_central_source_is_not_cropped(self):
        """Test that a compact source with its only component well within the cutout is not flagged as cropped."""
        comp = _components([{"parent": "SRC", "dra": 0, "ddec": 0, "peak": 1.0, "maj": 6.0, "min": 6.0}])
        res = con.flag_cropped_sources(*_one_source(), components=comp)
        assert bool(res["cropped"][0]) is False
        assert int(res["n_own_components"][0]) == 1

    def test_component_reaching_past_the_edge_is_cropped(self):
        """Test that a component whose extent reaches past the cutout edge is flagged as cropped."""
        # centre at 58" + FWHM half-axis 3" -> 61" > 60"
        comp = _components([{"parent": "SRC", "dra": 0, "ddec": 58, "peak": 1.0, "maj": 6.0, "min": 6.0}])
        res = con.flag_cropped_sources(*_one_source(), components=comp)
        assert bool(res["cropped"][0]) is True

    def test_component_centre_outside_frame_is_cropped(self):
        """Test that a component whose centre is outside the cutout is flagged as cropped."""
        comp = _components([{"parent": "SRC", "dra": 0, "ddec": 70, "peak": 1.0, "maj": 6.0, "min": 6.0}])
        res = con.flag_cropped_sources(*_one_source(), components=comp)
        assert bool(res["cropped"][0]) is True

    def test_unknown_source_name_has_no_components(self):
        """Test that a source name with no components is not flagged as cropped."""
        comp = _components([{"parent": "SOMETHING_ELSE", "dra": 0, "ddec": 0, "peak": 1.0}])
        res = con.flag_cropped_sources(*_one_source(name="SRC"), components=comp)
        assert int(res["n_own_components"][0]) == 0
        assert bool(res["cropped"][0]) is False

    def test_multi_component_source_spanning_the_frame_is_cropped(self):
        """Test that a source with multiple components, some of which are cropped, is flagged as cropped."""
        # two own lobes at +40" and +80" Dec -> the far one is well outside the 60" half-width
        comp = _components([
            {"parent": "SRC", "dra": 0, "ddec": 40, "peak": 1.0, "maj": 6.0, "min": 6.0},
            {"parent": "SRC", "dra": 0, "ddec": 80, "peak": 1.0, "maj": 6.0, "min": 6.0},
        ])
        res = con.flag_cropped_sources(*_one_source(), components=comp)
        assert int(res["n_own_components"][0]) == 2
        assert bool(res["cropped"][0]) is True
        # own_extent is the larger bbox side: Dec spans 37..83 -> ~46"
        assert res["own_extent"][0] == pytest.approx(46.0, abs=1e-6)

    def test_boundary_sigma_extends_bright_component_wings(self):
        """Test that the boundary_sigma parameter extends the effective extent of bright components."""
        # centre 55" + FWHM half-axis 3" = 58" (fits), but a bright peak's 3-sigma contour reaches past 60"
        comp = _components([{"parent": "SRC", "dra": 0, "ddec": 55, "peak": 100.0, "maj": 6.0, "min": 6.0}])
        src = _one_source(isl_rms=0.1)
        assert bool(con.flag_cropped_sources(*src, components=comp, boundary_sigma=None)["cropped"][0]) is False
        assert bool(con.flag_cropped_sources(*src, components=comp, boundary_sigma=3.0)["cropped"][0]) is True
