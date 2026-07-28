"""
This module contains the `AngularSizeFinder` class, which is used to estimate the angular size of a set of radio galaxy
images on a 80x80 grid based on the component data extracted from PyBDSF catalogue FITS files. The class processes the
FITS files, filters the components based on total flux to remove any present noise islands, and estimates the angular
size of the sources by creating a shape from the components and calculating the maximum distance between points on the
convex hull of this shape.

The shape geometry is owned by `MakeShape` (adapted from LoTSS-Catalogue GitHub): it samples each component ellipse's
boundary, takes their convex hull, and measures its diameter. Since the convex hull of a union of shapes equals the
convex hull of all those shapes' boundary points, the GEOS polygon union (shapely `unary_union`) is unnecessary for the
size estimate (and expensive) and is only built inside `MakeShape.plot` for visualisation purposes.
"""
import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import transforms
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer


class MakeShape():
    """
    A radio source's shape, built from its component list. It samples the component ellipse boundaries, taking their
    convex hull, and measuring the hull's diameter (the angular-size estimate). An instance keeps the intermediate
    geometry so it can also be plotted; `estimate_size` is a stateless fast path for callers (e.g. the parallel
    pipeline) that only need the number.

    Code here is adapted from the LoTSS-Catalogue GitHub, which contains the code to create the optically-identified
    LoTSS catalogues (e.g., Hardcastle et al. 2023 for LoTSS-DR2). The exact file is found here:
    https://github.com/mhardcastle/lotss-catalogue/blob/master/dr2_catalogue/make_catalogue.py
    """

    # Number of points used to sample each component ellipse's boundary. The original code used 200; this is overkill,
    # and 50 points give sub-percent error on the length and cuts the point count down dramatically for the hull.
    # This number needs to be even so the two antipodal major-axis points (theta = 0 and pi) are always sampled exactly,
    # which makes a single-component source's size land exactly on twice its (buffered) major axis.
    DEFAULT_ELLIPSE_POINTS = 50

    # Buffer (arcsec) added to each ellipse's axes so neighbouring components overlap into a single connected shape.
    _ELLIPSE_BUFFER_ARCSEC = 0.1


    def __init__(self,
                 clist: pd.DataFrame,
                 n: int = DEFAULT_ELLIPSE_POINTS):
        """
        Build the shape for a source from the component information in `clist`.

        The angular size (`length`) is computed from the convex hull of the sampled ellipse boundaries; the filled
        `shapely` union is only built on demand by `plot`.

        Parameters
        ----------
        clist : pd.DataFrame
            A DataFrame containing the component information for the source, with columns 'RA', 'DEC', 'DC_Maj',
            'DC_Min', and 'PA' representing the right ascension, declination, major axis, minor axis, and position
            angle of each component, respectively.
        n : int, optional
            The number of points used to sample each component ellipse's boundary, by default `DEFAULT_ELLIPSE_POINTS`.
        """
        self.n = n

        # Component arrays (kept for lazy shapely reconstruction in plot())
        self._ra = np.asarray(clist['RA'], dtype=float)
        self._dec = np.asarray(clist['DEC'], dtype=float)
        self._dc_maj = np.asarray(clist['DC_Maj'], dtype=float)
        self._dc_min = np.asarray(clist['DC_Min'], dtype=float)
        self._pa = np.asarray(clist['PA'], dtype=float)

        # Set the RA and DEC of the source to the mean RA and DEC of its components
        self.ra = float(self._ra.mean())
        self.dec = float(self._dec.mean())

        # Sample every ellipse boundary, take the convex hull, and find the furthest pair of hull points, which gives
        # the angular-size estimate. No polygon union is built here.
        points = self._ellipse_points(self._ra, self._dec, self._dc_maj, self._dc_min, self._pa, n)
        self.hull_points = self._hull_vertices(points)
        self.best_coords, self.mdist2 = self._furthest_pair(self.hull_points)

        # Filled shapely union and its hull are only needed for plotting; built lazily to keep this path GEOS-free.
        self.combined_polygon: Polygon | MultiPolygon | None = None
        self.hull = None


    def length(self) -> float:
        """
        Calculate the angular size of the source as the maximum distance between points on the convex hull of the
        (union of) component ellipses.

        Returns
        -------
        float
            The estimated angular size of the source in arcseconds.
        """
        return np.sqrt(self.mdist2)


    @classmethod
    def estimate_size(cls, components, n: int = DEFAULT_ELLIPSE_POINTS) -> float:
        """
        Estimate a source's angular size directly from its components, without constructing an instance or any shapely
        geometry. This is the stateless fast path used by the pipeline. It is equivalent to `MakeShape(clist).length()`
        but skips the DataFrame and the stored plotting state.

        Parameters
        ----------
        components : array-like
            The source's components, as rows of `(Total_flux, RA, DEC, DC_Maj, DC_Min, PA)`.
        n : int, optional
            The number of points used to sample each component ellipse's boundary, by default `DEFAULT_ELLIPSE_POINTS`.

        Returns
        -------
        float
            The estimated angular size in arcseconds.
        """
        comp = np.asarray(components, dtype=float)
        assert comp.size, "No components to create shape from. Check the filtering step and the input data."

        points = cls._ellipse_points(comp[:, 1], comp[:, 2], comp[:, 3], comp[:, 4], comp[:, 5], n)
        _, mdist2 = cls._furthest_pair(cls._hull_vertices(points))
        return float(np.sqrt(mdist2))


    # ---------- GEOMETRY ----------
    @classmethod
    def _ellipse_points(cls,
                        ra: np.ndarray,
                        dec: np.ndarray,
                        dc_maj: np.ndarray,
                        dc_min: np.ndarray,
                        pa: np.ndarray,
                        n: int = DEFAULT_ELLIPSE_POINTS) -> np.ndarray:
        """
        Sample the boundaries of every component ellipse at once, returning all points as a single (k*n, 2) array of
        arcsecond offsets from the source centre (the mean RA/DEC of the components).

        Uses a tangent-plane projection (RA scaled by cos(dec), the +90 degree position-angle convention) and the
        `_ELLIPSE_BUFFER_ARCSEC` axis buffer, but builds no shapely objects and loops over no rows.

        Parameters
        ----------
        ra, dec : np.ndarray
            Component right ascensions and declinations, in degrees.
        dc_maj, dc_min : np.ndarray
            Component deconvolved major and minor axes, in degrees.
        pa : np.ndarray
            Component position angles, in degrees.
        n : int, optional
            Number of boundary points per ellipse, by default `DEFAULT_ELLIPSE_POINTS`.

        Returns
        -------
        np.ndarray
            A (k*n, 2) array of (x, y) boundary points in arcseconds.
        """
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        dc_maj = np.asarray(dc_maj, dtype=float)
        dc_min = np.asarray(dc_min, dtype=float)
        pa = np.asarray(pa, dtype=float)

        # Source centre is the mean of the component positions
        ra0 = ra.mean()
        dec0 = dec.mean()

        # Per-component centre offsets in arcseconds, accounting for the cosine of the declination on the RA component
        x0 = 3600 * np.cos(np.deg2rad(dec0)) * (ra0 - ra)
        y0 = 3600 * (dec - dec0)
        a = dc_maj * 3600 + cls._ELLIPSE_BUFFER_ARCSEC
        b = dc_min * 3600 + cls._ELLIPSE_BUFFER_ARCSEC

        # Convert the position angle from degrees to radians and adjust by 90 degrees to match the original convention
        ang = np.deg2rad(pa + 90)

        # Points evenly spaced around a unit circle, shared by every component
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ct = np.cos(theta)          # (n,)
        st = np.sin(theta)
        ca = np.cos(ang)[:, None]   # (k, 1)
        sa = np.sin(ang)[:, None]

        # Parametric ellipse, broadcast over all k components and n angles at once -> (k, n)
        px = x0[:, None] + a[:, None] * ca * ct - b[:, None] * sa * st
        py = y0[:, None] + a[:, None] * sa * ct + b[:, None] * ca * st
        return np.column_stack([px.ravel(), py.ravel()])


    @staticmethod
    def _hull_vertices(points: np.ndarray) -> np.ndarray:
        """
        Return the convex-hull vertices of a set of 2D points, falling back to the points themselves when a hull cannot
        be formed (fewer than three points, or a degenerate/collinear set that Qhull rejects). The maximum pairwise
        distance is unchanged by that fallback, so the size estimate stays correct.

        Parameters
        ----------
        points : np.ndarray
            An (m, 2) array of points.

        Returns
        -------
        np.ndarray
            The subset of `points` lying on the convex hull, or all of `points` if no hull could be formed.
        """
        if len(points) < 3:
            return points
        try:
            return points[ConvexHull(points).vertices]
        except QhullError:
            return points


    @staticmethod
    def _furthest_pair(points: np.ndarray) \
            -> tuple[tuple[tuple[float, float], tuple[float, float]], float]:
        """
        Find the pair of points that are furthest apart, returning both the pair and their squared distance.

        Intended to be called on convex-hull vertices only (a handful of points), so the O(m^2) all-pairs computation is
        cheap. `length` needs only the squared distance; `plot` also wants the actual pair to draw the max-distance
        line, so both are returned from one computation.

        Parameters
        ----------
        points : np.ndarray
            An (m, 2) array of points.

        Returns
        -------
        best_coords : tuple[tuple[float, float], tuple[float, float]]
            The pair of points that are furthest apart. `((0, 0), (0, 0))` when fewer than two points are given.
        mdist2 : float
            The maximum squared distance between any two points, or 0.0 when fewer than two points are given.
        """
        if len(points) < 2:
            return ((0.0, 0.0), (0.0, 0.0)), 0.0

        diff = points[:, None, :] - points[None, :, :]
        dist2 = (diff * diff).sum(axis=-1)
        i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
        return (points[i], points[j]), float(dist2[i, j])


    # ---------- VISUALISATION ----------
    @staticmethod
    def _ellipse_polygon(x0: float,
                         y0: float,
                         a: float,
                         b: float,
                         pa: float,
                         n: int = 200) -> Polygon:
        """
        Create a shapely Polygon approximating an ellipse centred at `(x0, y0)` with semi-axes `a`, `b` and position
        angle `pa`, using `n` points. Only used to build the filled shape for `plot`.

        Parameters
        ----------
        x0, y0 : float
            The centre of the ellipse.
        a, b : float
            The semi-major and semi-minor axes.
        pa : float
            The position angle in degrees.
        n : int, optional
            The number of points used to approximate the ellipse, by default 200.

        Returns
        -------
        Polygon
            A shapely Polygon representing the ellipse.
        """
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        st = np.sin(theta)
        ct = np.cos(theta)

        pa = np.deg2rad(pa + 90)
        sa = np.sin(pa)
        ca = np.cos(pa)

        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        return Polygon(p)


    @classmethod
    def _combined_polygon(cls,
                          ra: np.ndarray,
                          dec: np.ndarray,
                          dc_maj: np.ndarray,
                          dc_min: np.ndarray,
                          pa: np.ndarray,
                          n: int = 200) -> Polygon | MultiPolygon:
        """
        Build the filled shapely union of a source's component ellipses. Uses the same projection and axis buffer as
        `_ellipse_points`; only used for plotting.

        Parameters
        ----------
        ra, dec, dc_maj, dc_min, pa : np.ndarray
            Component positions (deg), axes (deg), and position angles (deg).
        n : int, optional
            The number of points used to approximate each ellipse, by default 200.

        Returns
        -------
        Polygon | MultiPolygon
            The union of the component ellipses.
        """
        ra0 = ra.mean()
        dec0 = dec.mean()

        x = 3600 * np.cos(np.deg2rad(dec0)) * (ra0 - ra)
        y = 3600 * (dec - dec0)
        a = dc_maj * 3600 + cls._ELLIPSE_BUFFER_ARCSEC
        b = dc_min * 3600 + cls._ELLIPSE_BUFFER_ARCSEC

        ellist = [cls._ellipse_polygon(x[i], y[i], a[i], b[i], pa[i], n) for i in range(len(ra))]
        return unary_union(ellist)


    def plot(self):
        """
        Plot the combined shape of the source and its convex hull, along with the points on the convex hull and the
        pair of points that are furthest apart, which are used to estimate the angular size of the source.
        """
        # Build the filled union + its hull lazily; these are only needed for the plot.
        if self.combined_polygon is None:
            self.combined_polygon = self._combined_polygon(self._ra, self._dec, self._dc_maj, self._dc_min, self._pa)
            self.hull = self.combined_polygon.convex_hull

        plt.figure(figsize=(8, 8))

        # Plot the combined shape of the source, which is formed by taking the union of ellipses representing each
        # component. Some sources are combined together, into a MultiPolygon.
        if isinstance(self.combined_polygon, MultiPolygon):
            for geom in self.combined_polygon.geoms:
                x, y = geom.exterior.xy
                plt.plot(x, y, label='Combined Shape', color='blue')
        else:
            x, y = self.combined_polygon.exterior.xy  # type: ignore
            plt.plot(x, y, label='Combined Shape', color='blue')

        xh, yh = self.hull.exterior.xy  # type: ignore
        plt.plot(xh, yh, label='Convex Hull', color='orange')

        xh_points, yh_points = self.hull_points[:, 0], self.hull_points[:, 1]
        plt.scatter(xh_points, yh_points, label='Hull Points', color='green', s=10)

        if self.best_coords is not None:
            bestcoords_x = [self.best_coords[0][0], self.best_coords[1][0]]
            bestcoords_y = [self.best_coords[0][1], self.best_coords[1][1]]
            plt.plot(bestcoords_x, bestcoords_y,
                     label='Max Distance Pair', color='red', linewidth=2)

        plt.xlabel('DEC Offset (arcseconds)')
        plt.ylabel('RA Offset (arcseconds)')

        # Rotate the plot by 90 degrees to align with the standard astronomical convention, where RA increases to the
        # left and DEC increases upwards. This is done by applying an affine transformation to the plot.
        tr = transforms.Affine2D().rotate_deg(90) + transforms.Affine2D().translate(0, 0) + plt.gca().transData
        for line in plt.gca().get_lines():
            line.set_transform(tr)

        # Ensure the axes are equal to avoid distortion of the shape
        max_x = max(abs(xh_points)+1)
        max_y = max(abs(yh_points)+1)
        plt.xlim(-max_x, max_x)
        plt.ylim(-max_y, max_y)

        plt.title('Combined Shape and Convex Hull of Source')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.axis('equal')
        plt.show()



class AngularSizeFinder:
    """
    A class to estimate the angular size of a set of radio galaxy images on a 80x80 grid based on the component data
    extracted from PyBDSF catalogue FITS files. It owns the pipeline (reading, flux-filtering, per-source conventions,
    and serial/parallel dispatch) and delegates the shape geometry to `MakeShape`.
    """
    def __init__(self,
                 root_dir: Path = paths.STORAGE_PARENT / "diffracc/completeness/retrained_loguniform_catalogs",
                 flux_threshold: float = 0.95,
                 n_points: int = MakeShape.DEFAULT_ELLIPSE_POINTS,
                 num_processes: int = 1):
        """
        This class processes PyBDSF catalogue FITS files containing Gaussian component data for radio sources, filters
        the components based on total flux, and estimates the angular size of the sources by creating a shape from the
        components and calculating the maximum distance between points on the convex hull of this shape.

        Parameters
        ----------
        root_dir : Path, optional
            The root directory containing the FITS files to be processed, by default
            paths.STORAGE_PARENT / "diffracc/completeness/retrained_loguniform_catalogs"
        flux_threshold : float, optional
            The fraction of total flux to keep when filtering components, by default 0.95. Components contributing to
            the dimmest flux are removed while keeping total flux above this threshold.
        n_points : int, optional
            The number of points used to sample each component ellipse's boundary when estimating sizes, by default
            `MakeShape.DEFAULT_ELLIPSE_POINTS`.
        num_processes : int, optional
            The number of worker processes to use for the (CPU-bound, per-source-independent) size-estimation step, by
            default 1 (serial). Values > 1 dispatch the estimation across a `ProcessPoolExecutor`.
        """
        self.logger = get_logger("AngularSizeFinder", LoggingLevels.INFO.value)
        self.root_dir = root_dir

        # Decide a flux threshold for filtering components. PyBDSF can sometimes fit islands to noise and so we sort and
        # then filter islands based on their fractional total flux. The threshold below represents the fraction of total
        # flux to keep, so the dimmest islands are removed while keeping total flux above this fractional threshold.
        self.flux_threshold = flux_threshold

        self.n_points = n_points
        self.num_processes = num_processes

        self.rfa = RecursiveFileAnalyzer(self.root_dir)


    # ---------- ASSEMBLING SIZE ESTIMATES ----------
    @staticmethod
    def _read_and_filter(file_path: Path, flux_threshold: float) -> list[tuple]:
        """
        Read one PyBDSF catalogue FITS file and return its flux-filtered components.

        A staticmethod taking `flux_threshold` explicitly (rather than reading `self`) so it is picklable (i.e.,
        shareable across processes) and can be dispatched to `RecursiveFileAnalyzer`'s process mode for parallel
        parsing.

        Parameters
        ----------
        file_path : Path
            The path to the FITS file containing the component data for a single source.
        flux_threshold : float
            The fraction of total flux to keep when filtering (see `_filter_by_flux`).

        Returns
        -------
        list[tuple]
            The filtered components, each a `(Total_flux, RA, DEC, DC_Maj, DC_Min, PA)` tuple.
        """
        # Fastest way to read certain columns from the table
        with fits.open(file_path, memmap=False) as hdul:
            data = hdul[1].data
            components = list(zip(data["Total_flux"], data["RA"], data["DEC"],
                                  data["DC_Maj"], data["DC_Min"], data["PA"]))

        return AngularSizeFinder._filter_by_flux(components, flux_threshold)


    @staticmethod
    def _filter_by_flux(components: list[tuple], flux_threshold: float) -> list[tuple]:
        """
        Keep the brightest components that together reach `flux_threshold` of the total flux, discarding the dimmest -
        which PyBDSF sometimes fits to noise islands.

        Parameters
        ----------
        components : list[tuple]
            The components, each a tuple whose first element is the total flux, followed by RA, DEC, major axis, minor
            axis, and position angle.
        flux_threshold : float
            The fraction of total flux to keep. The dimmest components are removed while the cumulative flux of those
            kept stays above this fraction.

        Returns
        -------
        list[tuple]
            The filtered components, brightest first.
        """
        assert components, "No components found in the data. Check the FITS file and the expected column names."

        # Sort components by total flux in descending order (a new list, leaving the caller's untouched)
        components = sorted(components, key=lambda c: c[0], reverse=True)

        sum_flux = sum(component[0] for component in components)
        if sum_flux == 0:
            raise ValueError("Total flux of the source is zero. Cannot filter components based on flux threshold.")

        # Keep the brightest components until their cumulative flux reaches the threshold fraction of the total
        filtered_components = []
        cumulative_flux = 0
        for component in components:
            cumulative_flux += component[0]
            filtered_components.append(component)
            if cumulative_flux / sum_flux >= flux_threshold:
                break

        return filtered_components


    @staticmethod
    def _size_worker(components: list[tuple] | None, n: int) -> float:
        """
        Estimate one source's angular size (arcseconds), applying the pipeline's per-source conventions and delegating
        the geometry to `MakeShape`. A staticmethod so it can be pickled and dispatched to a `ProcessPoolExecutor`.

        Parameters
        ----------
        components : list[tuple] | None
            The filtered components, each a `(Total_flux, RA, DEC, DC_Maj, DC_Min, PA)` tuple. `None` (a failed file
            read that `RecursiveFileAnalyzer` turned into `None`) yields `NaN` rather than crashing the whole run.
        n : int
            Number of boundary points per ellipse.

        Returns
        -------
        float
            The estimated angular size in arcseconds, or `NaN` if `components` is `None`.
        """
        if components is None:
            return float("nan")

        comp = np.asarray(components, dtype=float)

        # A single surviving component: return twice the (unbuffered) major axis directly. This is a pipeline
        # convention that deliberately skips the ellipse-buffer path MakeShape uses for multi-component shapes.
        if len(comp) == 1:
            return 2 * comp[0, 3] * 3600

        return MakeShape.estimate_size(comp, n)


    def _estimate_sizes(self, components_list: list[tuple] | np.ndarray) -> list[float]:
        """
        Estimate the angular size for every source's component list, serially or across a process pool.

        Parameters
        ----------
        components_list : list[tuple] | np.ndarray
            The per-source filtered component lists (as produced by `_read_and_filter` via the extraction pipeline).

        Returns
        -------
        list[float]
            The estimated angular sizes in arcseconds, one per source, in the same order as `components_list`.
        """
        worker = partial(self._size_worker, n=self.n_points)

        # Each source is independent and the work is CPU-bound pure numpy/scipy, so it parallelises cleanly. The
        # chunksize is tuned to keep the workers busy without overwhelming the main process with too many results.
        if self.num_processes and self.num_processes > 1:
            self.logger.info(f"Estimating angular sizes across {self.num_processes} processes")
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                return list(tqdm(executor.map(worker, components_list, chunksize=64),
                                 total=len(components_list),
                                 desc="Estimating angular sizes", mininterval=1.0))

        return [worker(components)
                for components in tqdm(components_list, desc="Estimating angular sizes", mininterval=1.0)]


    # ---------- EXTRACTION + CONSOLIDATION ----------
    def _extract_components(self, fits_dir: str | Path | None, pattern: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract and flux-filter the components from every catalogue FITS under `fits_dir`.

        Reading a PyBDSF binary table is dominated by astropy parsing which binds the GIL, so when `num_processes > 1`
        this runs across worker processes (`mode="process"`), which the GIL-bound parse actually benefits from;
        otherwise it stays on the default threaded file mode.

        Parameters
        ----------
        fits_dir : str | Path | None
            The root directory of catalogue FITS files. If `None`, the finder's `root_dir` is used.
        pattern : str
            The regex pattern matching the catalogue files, with a capture group for the source index.

        Returns
        -------
        PipelineResult
            The per-source filtered component lists and their extracted indices.
        """
        parallel = self.num_processes and self.num_processes > 1
        return self.rfa.run_pipeline(
            function=self._read_and_filter,
            flux_threshold=self.flux_threshold,
            root_dir=fits_dir if fits_dir else self.root_dir,
            pattern=pattern,
            return_nums=True,
            mode="process" if parallel else "file",
            num_workers=self.num_processes if parallel else None,
            progress_bar_desc="Extracting and filtering component data from FITS files",
        )


    def _load_or_extract_components(self,
                                    fits_dir: str | Path | None,
                                    pattern: str,
                                    components_cache: str | Path | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the per-source components and indices, using a one-time consolidated cache when available.

        Extracting the components re-parses every catalogue FITS (the dominant cost of the whole pipeline). When
        `components_cache` is given, the extracted result is consolidated into that single file so subsequent runs -
        e.g. re-estimating with a different `n_points`, or recovering after a downstream failure - skip the parse
        entirely. This is deliberately local to the finder for now; it may later move into a shared, program-wide
        consolidation step.

        Parameters
        ----------
        fits_dir : str | Path | None
            The root directory of catalogue FITS files. If `None`, the finder's `root_dir` is used.
        pattern : str
            The regex pattern matching the catalogue files.
        components_cache : str | Path | None
            Path to the consolidated components file. If it exists, it is loaded instead of re-parsing; if it does not
            exist, the freshly extracted components are written to it. If `None`, no consolidation is done.

        Returns
        -------
        components_list : np.ndarray
            The per-source filtered component lists.
        fits_indices : np.ndarray
            The source indices corresponding to `components_list`.
        """
        if components_cache is not None and os.path.exists(components_cache):
            self.logger.info(f"Loading consolidated components from {components_cache}")
            with open(components_cache, "rb") as f:
                cached = pickle.load(f)
            return cached["components"], cached["indices"]

        components_list, fits_indices = self._extract_components(fits_dir, pattern)

        if components_cache is not None:
            self.logger.info(f"Consolidating extracted components to {components_cache}")
            with open(components_cache, "wb") as f:
                pickle.dump({"components": components_list, "indices": fits_indices}, f)

        return components_list, fits_indices


    # ---------- RUNNING THE PIPELINE ----------
    def estimate_angular_sizes(self,
                               fits_dir: str | Path | None = None,
                               pattern: str = r'.*?\D+(\d+)\.fits$',
                               output_file: str | Path | None = None,
                               components_cache: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to estimate the angular sizes of sources from the FITS files in the root directory, and optionally save
        the results to a CSV file.

        Parameters
        ----------
        fits_dir : str | Path | None, optional
            The root directory containing the FITS files, by default `None`.
        pattern : str, optional
            The regex pattern to match FITS files, by default r'.*?\D+(\d+)\.fits$'.
        output_file : str | Path | None, optional
            The name of the CSV file to save the estimated angular sizes to, by default `None`.
        components_cache : str | Path | None, optional
            Path to a consolidated components file, by default `None`. When given, the extracted components are cached
            to (and reloaded from) this file, so re-runs skip the expensive re-parse of every catalogue FITS. See
            `_load_or_extract_components`.

        Returns
        -------
        indices : np.ndarray
            An array of indices corresponding to the FITS files processed.
        sizes : np.ndarray
            An array of estimated angular sizes for the sources, in arcseconds.
        """
        # If the output file already exists, read the sizes from the file and return them along with the corresponding
        # indices
        if output_file is not None and os.path.exists(output_file):
            try:
                self.logger.info(f"Reading estimated angular sizes from {output_file}")
                ang_sizes = np.genfromtxt(output_file, delimiter=',', skip_header=1)
            except Exception as e:
                raise Exception(f"Failed to read {output_file}. Please check the file and try again: {e}") from e

            fits_indices = self.rfa.get_unwrapped_list(path=fits_dir,
                                                       pattern=pattern,
                                                       return_nums=True).numbers
            return fits_indices, ang_sizes

        # Extract (or reload consolidated) component data for each FITS file
        components_list, fits_indices = self._load_or_extract_components(fits_dir, pattern, components_cache)

        # Estimate the angular size of each image based on the component data
        ang_sizes = self._estimate_sizes(components_list)

        # Save the estimated angular sizes to a CSV file if an output file name is provided
        if output_file:
            self.logger.info(f"Saving estimated angular sizes to {output_file}")
            df = pd.DataFrame(ang_sizes, columns=['Estimated Angular Size (arcseconds)'])
            df.to_csv(output_file, index=False)

        return fits_indices, np.array(ang_sizes)


def build_arg_parser():
    """
    Build the argument parser for the command line interface.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser for the command line interface.
    """
    parser = argparse.ArgumentParser(description="Estimate angular sizes of radio sources from FITS files.")
    parser.add_argument("--root-dir", type=str, default=None,
                        help="Root directory containing the FITS files. Default is "
                             "'diffracc/completeness/dr2_cutouts_download_catalogs'.")
    parser.add_argument("--output-file", type=str, default='estimated_angular_sizes.csv',
                        help="Output CSV file to save the estimated angular sizes. Default is "
                             "'estimated_angular_sizes.csv'.")
    parser.add_argument("--flux-threshold", type=float, default=0.95,
                        help="Fraction of total flux to keep when filtering components. Default is 0.95.")
    parser.add_argument("--pattern", type=str, default=r'.*?\D+(\d+)\.fits$',
                        help="Regex pattern to match FITS files. Default is r'.*?\D+(\d+)\.fits$'.")
    parser.add_argument("--outlier-threshold", type=float, default=200.0,
                        help="Threshold for identifying outliers in estimated angular sizes (in arcseconds). "
                             "Sources with estimated sizes above this threshold will be removed from the analysis. "
                             "Default is 200.0 arcseconds.")
    parser.add_argument("--num-points", type=int, default=MakeShape.DEFAULT_ELLIPSE_POINTS,
                        help="Number of points used to sample each component ellipse's boundary. Default is "
                             f"{MakeShape.DEFAULT_ELLIPSE_POINTS}.")
    parser.add_argument("--num-processes", type=int, default=8,
                        help="Number of worker processes for the extraction and size-estimation steps. Set to 1 for "
                        "serial/threaded execution. Default is 8.")
    parser.add_argument("--components-cache", type=str, default=None,
                        help="Optional path to a consolidated components file. When given, extracted components are "
                        "cached to (and reloaded from) it, so re-runs skip re-parsing every catalogue FITS. "
                        "Default is None (no consolidation).")

    return parser


if __name__ == "__main__":
    _default_root = paths.STORAGE_PARENT / "diffracc/completeness/dr2_cutouts_download_catalogs"

    parser = build_arg_parser()
    args = parser.parse_args()

    root = args.root_dir if args.root_dir else _default_root

    asf = AngularSizeFinder(root,
                            flux_threshold=args.flux_threshold,
                            n_points=args.num_points,
                            num_processes=args.num_processes)
    indices, sizes = asf.estimate_angular_sizes(output_file=args.output_file, components_cache=args.components_cache)

    # Check for estimated angular sizes that are above the outlier threshold - "outliers"
    outliers = np.where(sizes > args.outlier_threshold)[0]
    asf.logger.warning(f"Found {len(outliers)} outliers with estimated angular sizes above {args.outlier_threshold} "
                                   f"arcseconds. These will be removed from the analysis.")
    indices = np.delete(indices, outliers)
    sizes = np.delete(sizes, outliers)

    for i in range(0, round(max(sizes)), 5):
        print(f"Size bin: {i} - {i+5} arcseconds")
        print(f"Number of sources in this size bin: {len(sizes[(sizes >= i) & (sizes < i+5)])}")

    # Plot a histogram of the estimated angular sizes
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Estimated Angular Sizes of Radio Sources')
    plt.xlabel('Estimated Angular Size (arcseconds)')
    plt.ylabel('Number of Sources')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(args.output_file.replace('.csv', '_distribution.png'))
    plt.show()
