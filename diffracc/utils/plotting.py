"""
Reusable, paper-quality plotting helpers, currently only used in AGN Selection Plots.

The recurring figure style in this project (and in Hardcastle et al. 2025) is a greyscale density of *all* sources in
the background, one or more coloured sub-populations scattered on top, and a few analytic boundary lines - see
``density_scatter`` below. Everything is written against an explicit ``Axes`` so figures can be composed into
multi-panel layouts, and styling is applied through the ``paper_style`` context manager rather than by mutating global
rcParams, so importing this module never changes how unrelated plots look.

Example
-------
>>> with paper_style():
...     fig, ax = plt.subplots()
...     density_scatter(
...         ax, x_all, y_all,
...         populations=[Population("SFG", x_sfg, y_sfg, color="tab:blue")],
...         boundaries=[Boundary(line_x, line_y, label="SF exclusion")],
...         xlabel="Absolute $W3$ magnitude", ylabel="$L_{144}$ (W Hz$^{-1}$)",
...         xlim=(-18, -34), ylim=(4e20, 1e29), ylog=True,
...     )
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import gaussian_filter

# A clean, A&A-like house style: serif text, Computer-Modern maths, inward ticks on all four sides with minor ticks,
# and frameless legends. Applied via a context manager so it is opt-in and never leaks into other figures.
PAPER_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
}


@contextmanager
def paper_style(overrides: dict | None = None):
    """
    Context manager applying the project's paper style to any plotting done inside it.

    Parameters
    ----------
    overrides : dict | None
        rcParams to merge on top of ``PAPER_RC`` for this block only (e.g. ``{"font.size": 18}`` for a wide
        multi-panel figure).
    """
    rc = dict(PAPER_RC)
    if overrides:
        rc.update(overrides)
    with mpl.rc_context(rc):
        yield


@dataclass
class Population:
    """
    A coloured sub-population to scatter on top of the background density.

    Attributes
    ----------
    label : str
        Legend label. The source count is appended automatically unless ``show_count`` is False.
    x, y : np.ndarray
        The population's coordinates (same units/space as the background data).
    color : str
        Any matplotlib colour spec.
    size : float
        Marker area in points**2.
    marker : str
        Matplotlib marker.
    shade : bool
        If True, also draw a faint filled density contour of this population in its own colour (the shaded "blobs" in
        Hardcastle-style figures), which reads better than scatter alone for very large populations.
    max_scatter : int | None
        If set and the population is larger than this, a random subsample of this many points is scattered (to keep
        the figure legible and its file size sane). The legend count always reflects the full population, not the
        subsample. Pair a small value with ``shade=True`` for the Hardcastle look: a smooth blob plus a few points.
    show_count : bool
        Whether to append ``(N)`` to the legend label.
    alpha : float
        Opacity of the scattered points.
    """
    label: str
    x: np.ndarray
    y: np.ndarray
    color: str
    size: float = 20.0
    marker: str = "."
    shade: bool = False
    max_scatter: int | None = None
    show_count: bool = True
    alpha: float = 0.7

    @property
    def count(self) -> int:
        """Number of finite points in the population."""
        return int(np.count_nonzero(np.isfinite(self.x) & np.isfinite(self.y)))

    def scatter_xy(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """The (x, y) actually drawn: finite points, subsampled to ``max_scatter`` if the population is larger."""
        finite = np.isfinite(self.x) & np.isfinite(self.y)
        x, y = np.asarray(self.x)[finite], np.asarray(self.y)[finite]
        if self.max_scatter is not None and x.size > self.max_scatter:
            idx = rng.choice(x.size, self.max_scatter, replace=False)
            x, y = x[idx], y[idx]
        return x, y

    @property
    def legend_label(self) -> str:
        """Returns the legend label, optionally with the source count appended."""
        return f"{self.label} ({self.count})" if self.show_count else self.label


@dataclass
class Boundary:
    """An analytic boundary line (e.g. a selection cut) drawn over everything else."""
    x: np.ndarray
    y: np.ndarray
    label: str | None = None
    color: str = "0.15"
    linestyle: str = "-"
    linewidth: float = 1.4


def _mono_cmap(color: str) -> LinearSegmentedColormap:
    """A transparent-to-``color`` colormap, for shading a single population's density in its own hue."""
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list("mono", [(r, g, b, 0.0), (r, g, b, 1.0)])


def _smoothed_density(x: np.ndarray, y: np.ndarray, xlim, ylim, xlog, ylog, bins, smooth):
    """
    Gaussian-smoothed 2D histogram of (x, y) over the plot limits, returned as (x_centres, y_centres, H) ready for
    contourf. Working from a histogram (rather than a KDE) keeps this O(N) so it scales to the full ~10^6-source
    catalogue. Coordinates are histogrammed in log space on log axes so the smoothing kernel is uniform on screen.
    
    Parameters
    ----------
    x, y : np.ndarray
        The data to histogram.
    xlim, ylim : tuple
        The plot limits, used to define the histogram range.
    xlog, ylog : bool
        Whether each axis is logarithmic.
    bins : int
        Number of histogram bins per axis.
    smooth : float
        Gaussian smoothing sigma in histogram bins.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The x and y bin centres, and the smoothed histogram H[y, x].
    """
    xs = np.log10(x) if xlog else np.asarray(x, dtype=float)
    ys = np.log10(y) if ylog else np.asarray(y, dtype=float)
    xr = tuple(np.log10(xlim)) if xlog else tuple(xlim)
    yr = tuple(np.log10(ylim)) if ylog else tuple(ylim)

    finite = np.isfinite(xs) & np.isfinite(ys)
    hist, x_edges, y_edges = np.histogram2d(
        xs[finite], ys[finite], bins=bins, range=[sorted(xr), sorted(yr)])
    hist = gaussian_filter(hist, smooth)

    x_centres = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_centres = 0.5 * (y_edges[1:] + y_edges[:-1])
    if xlog:
        x_centres = 10 ** x_centres
    if ylog:
        y_centres = 10 ** y_centres
    return x_centres, y_centres, hist.T  # transpose: contourf expects H[y, x]


def density_scatter(ax,
                    x_all: np.ndarray,
                    y_all: np.ndarray,
                    *,
                    populations=(),
                    boundaries=(),
                    xlabel: str = "",
                    ylabel: str = "",
                    xlim=None,
                    ylim=None,
                    xlog: bool = False,
                    ylog: bool = False,
                    background_cmap: str = "Greys",
                    density: str = "contour",
                    n_levels: int = 16,
                    smooth: float = 2.0,
                    bins: int = 256,
                    gridsize: int = 60,
                    shade_smooth: float = 3.5,
                    shade_levels: int = 6,
                    shade_alpha: float = 0.55,
                    legend: bool = True,
                    legend_loc: str = "best",
                    title: str | None = None,
                    seed: int = 0):
    """
    Draw a greyscale background density of all sources, optional coloured populations, and optional boundary lines on
    ``ax``. Returns ``ax``. Nothing is saved or shown - the caller owns the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x_all, y_all : np.ndarray
        Coordinates of every source, for the background density.
    populations : Iterable[Population]
        Coloured sub-populations to overlay (scatter, optionally shaded).
    boundaries : Iterable[Boundary]
        Analytic lines drawn on top (e.g. selection cuts).
    xlabel, ylabel, title : str
        Axis labels and optional title.
    xlim, ylim : tuple | None
        Axis limits. Required if any axis is logarithmic (the density needs a finite range). May be given reversed
        (e.g. ``(-18, -34)``) to invert an axis.
    xlog, ylog : bool
        Whether each axis is logarithmic.
    background_cmap : str
        Colormap for the background density.
    density : str
        ``"contour"`` (smoothed filled contours - clean, emphasises the core) or ``"hexbin"`` (log-count hexagonal
        bins - shows the full extent down to single sources, matching the Hardcastle-style figures).
    n_levels : int
        Number of filled contour levels for the ``"contour"`` background.
    smooth : float
        Gaussian smoothing sigma (in histogram bins) for the ``"contour"`` density.
    bins : int
        Histogram resolution per axis for the ``"contour"`` density.
    gridsize : int
        Number of hexagons across the x-range for the ``"hexbin"`` density.
    legend, legend_loc : bool, str
        Legend control.
    """
    if xlim is None or ylim is None:
        # The density range is taken from the limits, so they must be explicit; falling back to data min/max would make
        # multi-panel figures inconsistent.
        raise ValueError("xlim and ylim are required so the background density has a well-defined range")

    if density == "hexbin":
        # Every populated hexagon down to a single source is shown (mincnt=1) with a log-count colour, so the density
        # spans the whole occupied plane
        xr, yr = sorted(xlim), sorted(ylim)
        finite = (np.isfinite(x_all) & np.isfinite(y_all)
                  & (x_all >= xr[0]) & (x_all <= xr[1]) & (y_all >= yr[0]) & (y_all <= yr[1]))
        ax.hexbin(np.asarray(x_all)[finite], np.asarray(y_all)[finite], gridsize=gridsize, bins="log",
                  cmap=background_cmap, mincnt=1, linewidths=0.0, zorder=1,
                  xscale="log" if xlog else "linear", yscale="log" if ylog else "linear")
    else:
        # sqrt compresses the density's dynamic range so low-count outskirts stay visible without the core saturating.
        xc, yc, hist = _smoothed_density(x_all, y_all, xlim, ylim, xlog, ylog, bins, smooth)
        hmax = float(hist.max())
        if hmax > 0:
            levels = np.sqrt(np.linspace(hmax * 0.015, hmax, n_levels))
            ax.contourf(xc, yc, np.sqrt(hist), levels=levels, cmap=background_cmap, extend="max", zorder=1)

    rng = np.random.default_rng(seed)
    for pop in populations:
        if pop.shade:
            # Filled contours of the population's own smoothed density, in its own hue (transparent -> colour), so the
            # bulk of a large population reads as a smooth "blob" while only a sparse scatter of individual points is
            # drawn on top. Heavier smoothing than the background gives clean, paper-style contours.
            pxc, pyc, phist = _smoothed_density(pop.x, pop.y, xlim, ylim, xlog, ylog, bins, shade_smooth)
            pmax = float(phist.max())
            if pmax > 0:
                plevels = np.sqrt(np.linspace(pmax * 0.08, pmax, shade_levels))
                ax.contourf(pxc, pyc, np.sqrt(phist), levels=plevels, cmap=_mono_cmap(pop.color),
                            alpha=shade_alpha, zorder=2)
        sx, sy = pop.scatter_xy(rng)
        ax.scatter(sx, sy, s=pop.size, c=pop.color, marker=pop.marker, linewidths=0.0, alpha=pop.alpha,
                   label=pop.legend_label, zorder=3, rasterized=True)

    for bound in boundaries:
        ax.plot(bound.x, bound.y, color=bound.color, linestyle=bound.linestyle, linewidth=bound.linewidth,
                label=bound.label, zorder=4)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.tick_params(which="both", top=True, right=True)

    if legend and (populations or any(b.label for b in boundaries)):
        leg = ax.legend(loc=legend_loc, markerscale=2.2, handletextpad=0.5, borderaxespad=0.8)
        leg.set_zorder(5)
    return ax
