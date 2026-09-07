"""
The shape geometry of pyBDSF components is owned by the class `MakeShape` (adapted from LoTSS-Catalogue GitHub): it
samples each component ellipse's boundary, takes their convex hull, and measures its diameter.

Since the convex hull of a union of shapes equals the convex hull of all those shapes' boundary points, the GEOS polygon
union (shapely `unary_union`) is unnecessary for the size estimate (and expensive) and is only built inside
`MakeShape.plot` for visualisation purposes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


class MakeShape:
    """
    A radio source's shape, built from its component list. It samples the component ellipse boundaries, taking their
    convex hull, and measuring the hull's diameter (the angular-size estimate). An instance keeps the intermediate
    geometry so it can also be plotted; `estimate_size` is a stateless fast path for callers (e.g. the parallel
    pipeline) that only need the number.

    Code here is adapted from the LoTSS-Catalogue GitHub, which contains the code to create the optically-identified
    LoTSS catalogues (e.g., Hardcastle et al. 2023 for LoTSS-DR2). The exact file is found here:
    https://github.com/mhardcastle/lotss-catalogue/blob/master/dr2_catalogue/make_catalogue.py
    """

    # The number of points used to sample each component ellipse's boundary.
    # This number needs to be even so the two antipodal major-axis points (theta = 0 and pi) are always sampled exactly,
    # which makes a single-component source's size land exactly on twice its (buffered) major axis.
    DEFAULT_ELLIPSE_POINTS = 200

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
