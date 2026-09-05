"""
Smoke tests for diffracc/plotting/cutout_overlays.py's pure drawing function `draw_cutout`.

These build a synthetic image and a minimal WCS, so nothing touches the real cutout FITS or the catalogues; they check
that components are drawn (as ellipses), coloured own-vs-foreign correctly, and the target centre is marked.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

from diffracc.plotting import cutout_overlays as co


def _synthetic_wcs(ra=180.0, dec=40.0):
    """A minimal 1.5\"/pixel SIN WCS centred on (ra, dec) at pixel (10.5, 10.5)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [10.5, 10.5]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-1.5 / 3600.0, 1.5 / 3600.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return w


def _components(ra=180.0, dec=40.0):
    """One own component at the centre and one foreign component 20\" east."""
    cos_dec = np.cos(np.radians(dec))
    return {
        "ra": np.array([ra, ra + 20.0 / 3600.0 / cos_dec]),
        "dec": np.array([dec, dec]),
        "peak": np.array([2.0, 1.0]),
        "maj": np.array([6.0, 6.0]),
        "min": np.array([6.0, 6.0]),
        "pa": np.array([0.0, 0.0]),
        "parent": np.array(["SRC", "NBR"], dtype="<U8"),
    }


def test_draw_cutout_draws_an_ellipse_per_component_and_marks_centre():
    """Test that the drawing function draws one ellipse per component, marks the centre, and returns a colour dict."""
    image = np.random.default_rng(0).normal(size=(20, 20))
    fig, ax = plt.subplots()
    colours = co.draw_cutout(ax, image, _synthetic_wcs(), 180.0, 40.0, "SRC", _components())
    assert len(ax.patches) == 2                 # one ellipse per component
    assert len(ax.lines) == 1                   # the white centre cross
    assert ax.images                            # the imshow background
    # only foreign parents get a colour entry; the own source ("SRC") does not
    assert "NBR" in colours and "SRC" not in colours
    plt.close(fig)


def test_draw_cutout_handles_no_components():
    """Test that the drawing function does not crash when there are no components."""
    image = np.zeros((20, 20))
    empty = {k: np.array([], dtype=("<U8" if k == "parent" else float)) for k in co._COMP_KEYS}
    fig, ax = plt.subplots()
    colours = co.draw_cutout(ax, image, _synthetic_wcs(), 180.0, 40.0, "SRC", empty)
    assert len(ax.patches) == 0
    assert colours == {}
    plt.close(fig)
