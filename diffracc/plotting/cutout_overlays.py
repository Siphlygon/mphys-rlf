"""
WCS-aware cutout visualisation, specifically considering multiple components in one DR2 cutout image.

This module allows you to draw a LoTSS-DR2 cutout with the component catalogue overlaid, colouring each component by the
source it belongs to (`Parent_Source`) - white for the cutout's own target, a distinct colour per foreign neighbour,
each labelled with its catalogued peak flux.

The flag logic lives in `diffracc.data.cutout_quality`; this module only draws. It reuses that module's catalogue loader
and geometry helpers, so the ellipses shown are exactly the ellipses the flags are computed from.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree

from ..data import cutout_quality as cq
from ..utils import paths
from ..utils.plotting import paper_style

# The target's own components are drawn white; foreign sources cycle through these.
FOREIGN_COLOURS = ["#00e5ff", "#ff5252", "#ffea00", "#69f0ae", "#ff80ab", "#b388ff", "#ffab40"]
_COMP_KEYS = ("ra", "dec", "peak", "maj", "min", "pa", "parent")



def _sky_ellipse(wcs: WCS,
                 ra: float,
                 dec: float,
                 maj: float,
                 minr: float,
                 pa: float,
                 pixscale: float,
                 **kwargs) -> Ellipse:
    """
    A matplotlib `Ellipse` for a component's fitted ellipse (`Maj`/`Min` in arcsec FWHM, `PA` North-through-East),
    positioned and oriented through the cutout WCS.
    
    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The cutout's WCS, used to place the ellipse.
    ra, dec : float
        The component's position in degrees.
    maj, minr : float
        The component's fitted ellipse major and minor axes in arcsec FWHM.
    pa : float
        The component's fitted ellipse position angle in degrees, North-through-East.
    pixscale : float
        The cutout's pixel scale in arcsec/pixel.
    kwargs : dict
        Extra keyword arguments passed to `Ellipse` (e.g. `edgecolor`, `lw`, `linestyle`).
    
    Returns
    -------
    matplotlib.patches.Ellipse
        The component's fitted ellipse, positioned and oriented through the cutout WCS.
    """
    x0, y0 = wcs.all_world2pix(ra, dec, 0)
    d = (maj / 2.0) / 3600.0
    d_ra = d * np.sin(np.radians(pa)) / np.cos(np.radians(dec))   # a step along the major axis, on the sky
    d_dec = d * np.cos(np.radians(pa))
    xt, yt = wcs.all_world2pix(ra + d_ra, dec + d_dec, 0)
    angle = np.degrees(np.arctan2(float(yt - y0), float(xt - x0)))
    return Ellipse((float(x0), float(y0)), maj / pixscale, minr / pixscale, angle=angle, fill=False, **kwargs)


def draw_cutout(ax, image, wcs: WCS, source_ra: float, source_dec: float, source_name: str, components: dict) -> dict:
    """
    Draw one cutout on `ax`: the image plus every component in `components` as its fitted ellipse - white if it belongs
    to `source_name` (the target), a distinct colour per foreign parent - labelled with its peak flux, and the target
    position marked with a white cross. Pure (no file I/O), so it is straightforward to test with synthetic inputs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    image : np.ndarray
        2D cutout pixel values (display is percentile-normalised, so absolute units do not matter).
    wcs : astropy.wcs.WCS
        The cutout's WCS, used to place catalogue components.
    source_ra, source_dec : float
        The target's position in degrees (the cutout centre).
    source_name : str
        The target's `Source_Name`; components with this `parent` are drawn as "own".
    components : dict
        In-frame components to draw, with keys `ra`, `dec` (deg), `peak` (mJy/beam), `maj`, `min` (arcsec FWHM),
        `pa` (deg) and `parent` (`Source_Name`).

    Returns
    -------
    dict
        Mapping of foreign `Parent_Source` -> the colour used, for building a legend if desired.
    """
    try:
        pixscale = float(proj_plane_pixel_scales(wcs)[0]) * 3600.0    # arcsec / pixel
    except Exception:
        pixscale = cq.PIXEL_SCALE_ARCSEC
    ax.imshow(image, origin="lower", cmap="inferno",
              norm=ImageNormalize(image, interval=PercentileInterval(99.5), stretch=AsinhStretch()))

    foreign_colour = {}
    for k in range(len(components["ra"])):
        parent = components["parent"][k]
        is_own = parent == source_name
        if is_own:
            colour = "white"
        else:
            foreign_colour.setdefault(parent, FOREIGN_COLOURS[len(foreign_colour) % len(FOREIGN_COLOURS)])
            colour = foreign_colour[parent]
        ax.add_patch(_sky_ellipse(wcs, components["ra"][k], components["dec"][k], components["maj"][k],
                                  components["min"][k], components["pa"][k], pixscale,
                                  edgecolor=colour, lw=1.8 if is_own else 1.5, linestyle="-" if is_own else "--"))
        x0, y0 = wcs.all_world2pix(components["ra"][k], components["dec"][k], 0)
        ax.text(float(x0), float(y0) + 3.5, f"{components['peak'][k]:.2f}", color=colour, fontsize=6.5,
                ha="center", va="bottom", fontweight="bold")

    # Find the target's pixel position and mark it with a white cross.
    cx, cy = wcs.all_world2pix(source_ra, source_dec, 0)
    ax.plot(float(cx), float(cy), "+", color="white", ms=9, mew=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    return foreign_colour


class CutoutOverlayPlotter:
    """
    Load the catalogues once, then draw dr2 cutouts with their catalogued components overlaid (see :func:`draw_cutout`).

    Parameters
    ----------
    component_catalogue_path : Path, optional
        The component catalogue, by default `paths.COMPONENT_CATALOGUE_PATH`.
    source_catalogue_path : Path, optional
        A source catalogue carrying `Source_Name`, by default `paths.STRIPPED_CATALOGUE_PATH`.
    cutouts_path : Path, optional
        Folder of `cutout{index}.fits` files, by default `paths.CUTOUTS_PATH`.
    flags : pd.DataFrame or None, optional
        Optional `cutout_quality` flags (row-aligned to the resolved source order) used only for panel titles.
    """

    def __init__(self,
                 component_catalogue_path: Path = paths.COMPONENT_CATALOGUE_PATH,
                 source_catalogue_path: Path = paths.STRIPPED_CATALOGUE_PATH,
                 cutouts_path: Path = paths.CUTOUTS_PATH,
                 flags: pd.DataFrame | None = None):
        self.components = cq.load_component_catalogue(component_catalogue_path)
        self._tree = cKDTree(cq._unit_vectors(self.components["ra"], self.components["dec"]))
        with fits.open(source_catalogue_path, memmap=True) as hdul:
            data = hdul[1].data
            res = np.asarray(data["Resolved"]).astype(bool)
            self.src_ra = np.asarray(data["RA"][res], float)
            self.src_dec = np.asarray(data["DEC"][res], float)
            self.src_name = cq._as_str_array(data["Source_Name"][res])
        self.cutouts_path = cutouts_path
        self.flags = flags
        self._half = cq.CUTOUT_SIZE_ARCSEC / 2.0
        self._qrad = np.radians((self._half * np.sqrt(2.0) + float(self.components["maj"].max()) / 2.0) / 3600.0)

    def cutout_path(self, index: int) -> Path:
        """
        Path to `cutout{index}.fits` (assuming files are bucketed into 10k-index subfolders).
        
        Parameters
        ----------
        index : int
            The resolved source index (row in the source catalogue).
        
        Returns
        -------
        Path
            The path to the cutout FITS file for the given index.
        """
        base = (index // 10000) * 10000
        return self.cutouts_path / f"{base}-{base + 9999}" / f"cutout{index}.fits"

    def _load_cutout(self, index: int) -> tuple[np.ndarray, WCS]:
        """
        Load the cutout image and WCS for resolved source `index`.

        Parameters
        ----------
        index : int
            The resolved source index (row in the source catalogue).

        Returns
        -------
        tuple[np.ndarray, WCS]
            A tuple containing the cutout image and its World Coordinate System (WCS).
        """
        with fits.open(self.cutout_path(index)) as hdul:
            return np.asarray(hdul[0].data, float), WCS(hdul[0].header)

    def _components_in_cutout(self, index: int) -> dict:
        """
        Get the catalogue components whose fitted ellipse overlaps source `index`'s cutout frame.
        
        Parameters
        ----------
        index : int
            The resolved source index (row in the source catalogue).
            
        Returns
        -------
        dict
            A dictionary of component properties (keys `ra`, `dec`, `peak`, `maj`, `min`, `pa`, `parent`) for components
            that overlap the cutout frame of the specified source index.
        """
        vec = cq._unit_vectors(np.array([self.src_ra[index]]), np.array([self.src_dec[index]]))[0]
        cand = np.asarray(self._tree.query_ball_point(vec, self._qrad), dtype=int)
        if cand.size:
            x0 = (self.components["ra"][cand] - self.src_ra[index]) * np.cos(np.radians(self.src_dec[index])) * 3600.0
            y0 = (self.components["dec"][cand] - self.src_dec[index]) * 3600.0
            d_ra, d_dec = cq._ellipse_halfwidths(self.components["maj"][cand], self.components["min"][cand],
                                                 self.components["pa"][cand])
            cand = cand[(x0 - d_ra <= self._half) & (x0 + d_ra >= -self._half)
                        & (y0 - d_dec <= self._half) & (y0 + d_dec >= -self._half)]
        return {key: self.components[key][cand] for key in _COMP_KEYS}

    def _default_title(self, index: int) -> str:
        """
        Default title for a cutout panel, showing the index and any foreign components detected in the cutout.

        Parameters
        ----------
        index : int
            The resolved source index (row in the source catalogue).

        Returns
        -------
        str
            The default title for the cutout panel.
        """
        if self.flags is None:
            return f"#{index}"
        row = self.flags.iloc[index]
        return (f"#{index}  {int(row.get('n_foreign_detected', 0))} foreign\n"
                f"brightest nbr {float(row.get('brightest_foreign_flux', 0.0)):.2f} mJy "
                f"({float(row.get('brightest_foreign_snr', 0.0)):.0f}$\\sigma$)")

    def plot_cutout(self, index: int, ax = None, title: str | None = None):
        """
        Draw the cutout for resolved-source `index` on `ax` (a new axis if None). Returns the axis.
        
        Parameters
        ----------
        index : int
            The resolved source index (row in the source catalogue).
        ax : matplotlib.axes.Axes, optional
            Axis to draw on; if None, a new axis is created.
        title : str, optional
            Title for the cutout panel; if None, a default title is generated based on the cutout's flags. 
        
        Returns
        -------
        matplotlib.axes.Axes
            The axis with the cutout drawn on it.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(3.5, 3.5))
        image, wcs = self._load_cutout(index)
        draw_cutout(ax, image, wcs, self.src_ra[index], self.src_dec[index], self.src_name[index],
                    self._components_in_cutout(index))
        ax.set_title(self._default_title(index) if title is None else title, fontsize=7.5)
        return ax

    def plot_grid(self, indices: list[int], ncols: int = 4, suptitle: str | None = None):
        """
        Tile `indices` as a titled grid under the project paper style. Returns the Figure.
        
        Parameters
        ----------
        indices : list[int]
            List of resolved source indices (rows in the source catalogue) to plot.
        ncols : int, optional
            Number of columns in the grid, by default 4.
        suptitle : str, optional
            Optional super-title for the entire figure; if None, no super-title is added.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the grid of cutouts.
        """
        indices = [int(i) for i in indices]
        nrows = int(np.ceil(len(indices) / ncols))
        with paper_style():
            fig, axs = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.7 * nrows))
            axs = np.atleast_1d(axs).ravel()
            for ax in axs:
                ax.axis("off")
            for ax, index in zip(axs, indices):
                ax.axis("on")
                self.plot_cutout(index, ax=ax)
            if suptitle:
                fig.suptitle(suptitle, fontsize=13, y=1.002)
            fig.tight_layout()
        return fig
