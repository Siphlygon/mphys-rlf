"""
This module contains the `AngularSizeFinder` class, which is used to estimate the angular size of a set of radio galaxy
images on a 80x80 grid based on the component data extracted from PyBDSF catalogue FITS files.

The class processes the FITS files, filters the components based on total flux to remove any present noise islands, and
estimates the angular size of the sources by creating a shape from the components and calculating the maximum distance
between points on the convex hull of this shape.
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
from tqdm import tqdm

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from ..utils.plotting import paper_style
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer
from . import floodfill_size
from .make_shape import MakeShape


class AngularSizeFinder:
    """
    A class to estimate the angular size of a set of radio galaxy images on a 80x80 grid based on the component data
    extracted from PyBDSF catalogue FITS files.
    """
    def __init__(self,
                 root_dir: Path = paths.PYBDSF_CATALOG_PARENT / "dr2_cutouts_download",
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
            `paths.PYBDSF_CATALOG_PARENT / "dr2_cutouts_download"`.
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

        A staticmethod taking `flux_threshold` explicitly so it is picklable (i.e., shareable across processes) and can
        be dispatched to `RecursiveFileAnalyzer`'s process mode for parallel parsing.

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

    def _load_components_from_catalogue(self,
                                        fits_indices: np.ndarray,
                                        source_catalogue_path: str | Path = paths.STRIPPED_CATALOGUE_PATH,
                                        component_catalogue_path: str | Path = paths.COMPONENT_CATALOGUE_PATH
                                        ) -> np.ndarray:
        """
        Load the per-source components and indices from a pre-existing catalogue, bypassing the extraction step.
        
        Parameters
        ----------
        fits_indices : np.ndarray
            The source indices corresponding to the components in the catalogue.
        source_catalogue_path : str | Path
            Path to the source catalogue FITS file containing the DR2 catalogue information.
        component_catalogue_path : str | Path
            Path to the component catalogue FITS file containing the DR2 components.
        
        Returns
        -------
        components_list : np.ndarray
            The per-source filtered component lists.
        """
        # Get the Source_Name of each source from the source catalogue
        self.logger.info(f"Loading source names from {source_catalogue_path}")
        with fits.open(source_catalogue_path, memmap=False) as hdul:
            source_data = hdul[1].data
            source_names = source_data['Source_Name'][fits_indices]

        # Assemble the components for each source from the component catalogue, by matching Source_Name.
        self.logger.info(f"Loading components from {component_catalogue_path}")
        with fits.open(component_catalogue_path, memmap=False) as hdul:
            component_data = hdul[1].data

            # Pull the six columns we need once, as a single contiguous (n_components, 6) float array. The downstream
            # geometry (MakeShape) was written for PyBDSF cutout catalogues, whose DC_Maj/DC_Min are in DEGREES (it
            # multiplies them by 3600). This value-added component catalogue instead stores its axes in ARCSEC (and
            # carries no TUNIT to flag it), so convert them to degrees here to keep that convention.
            component_values = np.column_stack([
                np.asarray(component_data['Total_flux'], dtype=float),
                np.asarray(component_data['RA'], dtype=float),
                np.asarray(component_data['DEC'], dtype=float),
                np.asarray(component_data['DC_Maj'], dtype=float) / 3600.0,
                np.asarray(component_data['DC_Min'], dtype=float) / 3600.0,
                np.asarray(component_data['PA'], dtype=float),
            ])
            parent_source = np.asarray(component_data['Parent_Source'])

        # Group every component's row position by its Parent_Source in a single pass, as dicts (O(1) lookup).
        rows_by_source = pd.Series(np.arange(len(parent_source))).groupby(parent_source).indices

        empty = np.empty((0, 6), dtype=float)
        components_list = np.empty(len(source_names), dtype=object)
        missing = 0
        for i, source_name in tqdm(enumerate(source_names), desc="Matching components to sources..."):
            positions = rows_by_source.get(source_name)
            if positions is None:
                components_list[i] = empty
                missing += 1
            else:
                components_list[i] = component_values[positions]

        if missing:
            self.logger.warning(
                f"{missing} of {len(source_names)} sources had no components in the component catalogue.")

        return components_list

    def _load_or_extract_components(self,
                                    fits_dir: str | Path | None,
                                    pattern: str,
                                    components_cache: str | Path | None,
                                    load_from_catalogue: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        load_from_catalogue : bool
            If `True`, assemble components from the DR2 component catalogue by matching source names; otherwise extract
            them from the PyBDSF catalogue FITS files under `fits_dir`. By default `False`.

        Returns
        -------
        components_list : np.ndarray
            The per-source filtered component lists.
        fits_indices : np.ndarray
            The source indices corresponding to `components_list`.
        """
        if components_cache is not None:
            if os.path.exists(components_cache):
                self.logger.info(f"Loading consolidated components from {components_cache}")
                with open(components_cache, "rb") as f:
                    cached = pickle.load(f)
                return cached["components"], cached["indices"]
            self.logger.info(f"No consolidated components found at {components_cache}; extracting from FITS files")

        if load_from_catalogue:
            self.logger.info("Loading components from the DR2 catalogue")
            fits_indices = self.rfa.get_unwrapped_list(path=fits_dir,
                                                       pattern=pattern,
                                                       return_nums=True).numbers
            components_list = self._load_components_from_catalogue(fits_indices)
        else:
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
                               read_from_file: bool = False,
                               load_from_catalogue: bool = False,
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
        read_from_file : bool, optional
            If `True`, the method will attempt to read the estimated angular sizes from the output file, if it exists.
            If `False`, the method will always re-calculate the angular sizes and save them to the output file. By
            default `False`.
        load_from_catalogue : bool, optional
            If `True`, the method will load components from the DR2 catalogue instead of extracting them from FITS
            files or a cached file. If `False`, the method will extract components as usual. By default `False`.
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
        assert (read_from_file and output_file is not None) or not read_from_file, (
            "Cannot read from file if no output file is specified.")
        # If the output file already exists, read the sizes from the file and return them along with the corresponding
        # indices
        if read_from_file:
            if not os.path.exists(output_file):
                self.logger.error(f"Output file {output_file} does not exist. Cannot read estimated angular sizes from "
                                  "it. Recalculating sizes instead.")
            else:
                try:
                    self.logger.info(f"Reading estimated angular sizes from {output_file}")
                    fits_indices, ang_sizes = np.loadtxt(output_file, delimiter=',', skiprows=1, unpack=True)
                    fits_indices = fits_indices.astype(int)
                    ang_sizes = ang_sizes.astype(float)
                except Exception as e:
                    raise Exception(f"Failed to read {output_file}. Please check the file and try again: {e}") from e

                return fits_indices, ang_sizes

        # Extract (or reload consolidated) component data for each FITS file
        components_list, fits_indices = self._load_or_extract_components(fits_dir, pattern, components_cache,
                                                                         load_from_catalogue=load_from_catalogue)

        # Estimate the angular size of each image based on the component data
        ang_sizes = self._estimate_sizes(components_list)

        # Save the estimated angular sizes to a CSV file if an output file name is provided
        if output_file:
            self.logger.info(f"Saving estimated angular sizes and indices to {output_file}")
            # Create a DataFrame with the estimated angular sizes and FITS indices
            df = pd.DataFrame({
                "fits_index": fits_indices,
                "estimated_las_arcsec": ang_sizes,
            })
            df.to_csv(output_file, index=False, mode="w")

        return fits_indices, np.array(ang_sizes)

    # ---------- FLOOD-FILL (PIXEL-BASED) SIZING ----------
    def measure_floodfill(self,
                          image: np.ndarray | str | Path,
                          include: np.ndarray,
                          exclude: np.ndarray,
                          rms: float,
                          peak: float,
                          total_flux: float,
                          header: fits.Header | None = None,
                          component_size: float | None = None,
                          badflux: float = 0.0,
                          component_las_from: str = "Catalogue") -> dict:
        """
        Measure a source's pixel-based flood-fill size and flux from its image, delegating the geometry to
        `floodfill_size.measure_source`, and - when a component-based size is supplied - the final LAS via
        `select_angular_size`.

        Parameters
        ----------
        image : np.ndarray | str | Path
            The 2D cutout image (Jy/beam), or a path to a FITS file to read it (and its header) from.
        include : np.ndarray
            The source's own components as `(ra, dec, maj_deg, min_deg, pa_deg)` rows.
        exclude : np.ndarray
            Foreign components (same columns) to mask out; may be empty.
        rms, peak : float
            The source island rms and peak flux in Jy/beam (setting the threshold `max(4*rms, peak/50)`).
        total_flux : float
            The source total flux (same units as the measured flood-fill flux) for the adoption flux-match test.
        header : astropy.io.fits.Header, optional
            The 2D FITS header (WCS, `CDELT1/2`, `BMAJ`, `BMIN`). Required when `image` is an array; read from the file
            when `image` is a path.
        component_size : float, optional
            The component-based (catalogue) size in arcsec. When given, the returned dict also carries the hybrid `LAS`
            and `LAS_from`.
        badflux : float, optional
            The minimum credible flood-fill flux in Jy for the `Bad_flux` flag, by default 0.0. LoTSS uses
            `min(Total_flux)/2000` over the catalogue; the caller should pass that value.
        component_las_from : str, optional
            Provenance label reported for `LAS_from` where the flood-fill size is not adopted, by default "Catalogue".

        Returns
        -------
        dict
            `LM_size` (arcsec), `LM_flux` (Jy), `Bad_flux`, `Bad_image`, and - when `component_size` is given - `LAS`
            (arcsec) and `LAS_from`.
        """
        if isinstance(image, (str, Path)):
            with fits.open(image, memmap=False) as hdul:
                data = np.squeeze(hdul[0].data)
                header = hdul[0].header
        else:
            data = np.squeeze(np.asarray(image))
            if header is None:
                raise ValueError("A FITS header must be provided when `image` is an array.")

        result = floodfill_size.measure_source(data, header, include, exclude, rms, peak, badflux)

        # Select the final LAS if a component-based size is supplied, using the Hardcastle et al. (2023) rule.
        if component_size is not None:
            las, las_from = self.select_angular_size(
                component_size, result["LM_size"], result["LM_flux"], total_flux,
                result["Bad_flux"], result["Bad_image"], component_las_from=component_las_from)
            result["LAS"] = las
            result["LAS_from"] = las_from

        return result

    @staticmethod
    def select_angular_size(component_size: float | np.ndarray,
                            lm_size: float | np.ndarray,
                            lm_flux: float | np.ndarray,
                            total_flux: float | np.ndarray,
                            bad_flux: bool | np.ndarray,
                            bad_image: bool | np.ndarray,
                            component_las_from: str = "Catalogue"):
        """
        Choose the final angular size between the component-based estimate and the flood-fill estimate, applying the
        Hardcastle et al. (2023) rule: take the flood-fill size when it is unflagged, its flux matches the catalogue
        flux to within 20%, and the component-based size is between 30 and 600 arcsec; otherwise keep the
        component-based size. Works element-wise on arrays or on scalars.

        Parameters
        ----------
        component_size : array-like or float
            The component-based size in arcsec (`2*DC_Maj` or the `MakeShape` composite size).
        lm_size : array-like or float
            The flood-fill size in arcsec (see `measure_floodfill`).
        lm_flux, total_flux : array-like or float
            The flood-fill and catalogue total fluxes (same units); their ratio must lie in `(0.8, 1.2)`.
        bad_flux, bad_image : array-like or bool
            The flood-fill quality flags; a set flag blocks selection of the flood-fill size.
        component_las_from : str, optional
            Provenance label reported where the flood-fill size is not selected, by default "Catalogue".

        Returns
        -------
        las : np.ndarray or float
            The selected size in arcsec.
        las_from : np.ndarray or str
            "Flood-fill" where the pixel size is selected, else `component_las_from`.
        """
        component_size = np.asarray(component_size, dtype=float)
        lm_size = np.asarray(lm_size, dtype=float)
        flux_ratio = np.asarray(lm_flux, dtype=float) / np.asarray(total_flux, dtype=float)

        use_floodfill = (~np.asarray(bad_flux, dtype=bool)
                         & ~np.asarray(bad_image, dtype=bool)
                         & (flux_ratio > 0.8) & (flux_ratio < 1.2)
                         & (component_size > 30) & (component_size < 600))

        las = np.where(use_floodfill, lm_size, component_size)
        las_from = np.where(use_floodfill, "Flood-fill", component_las_from)

        if np.ndim(use_floodfill) == 0:
            return float(las), str(las_from)
        return las, las_from


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
    parser.add_argument("--read-from-file", action="store_true",
                        help="If set, the script will attempt to read the estimated angular sizes from the output file "
                             "if it exists, instead of recalculating them. Default is False.")
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
                        help="Optional path to a consolidated components file (.pkl). When given, extracted components "
                        "are cached to (and reloaded from) it, so re-runs skip re-parsing every catalogue FITS. "
                        "Default is None (no consolidation).")
    parser.add_argument("--load-from-catalogue", action="store_true",
                        help="If set, the script will load components from the DR2 catalogue instead of extracting "
                        "them from FITS files. Default is False.")
    parser.add_argument("--no-plot", action="store_true",
                        help="If set, the script will not generate plots of the estimated angular sizes and their "
                        "differences from the DR2 LAS values. Default is False (plots will be generated).")
    return parser


if __name__ == "__main__":
    _default_root = paths.PYBDSF_CATALOG_PARENT / "dr2_cutouts_download"

    parser = build_arg_parser()
    args = parser.parse_args()

    root = args.root_dir if args.root_dir else _default_root

    asf = AngularSizeFinder(root,
                            flux_threshold=args.flux_threshold,
                            n_points=args.num_points,
                            num_processes=args.num_processes)
    indices, sizes = asf.estimate_angular_sizes(output_file=args.output_file,
                                                components_cache=args.components_cache,
                                                load_from_catalogue=args.load_from_catalogue,
                                                read_from_file=args.read_from_file)

    # Check for estimated angular sizes that are above the outlier threshold - "outliers"
    if not args.no_plot:
        outliers = np.where(sizes > args.outlier_threshold)[0]
        asf.logger.warning(f"Found {len(outliers)} outliers with estimated angular sizes above {args.outlier_threshold}"
                                    f" arcseconds. These will be removed from the analysis.")
        indices = np.delete(indices, outliers)
        sizes = np.delete(sizes, outliers)

        for i in range(0, round(max(sizes)), 5):
            print(f"Size bin: {i} - {i+5} arcseconds")
            print(f"Number of sources in this size bin: {len(sizes[(sizes >= i) & (sizes < i+5)])}")

        # Plot a histogram of the estimated angular sizes
        with paper_style():
            plt.figure(figsize=(10, 6))
            plt.hist(sizes, bins=50, color='skyblue', edgecolor='black')
            plt.title('Distribution of Estimated Angular Sizes of Radio Sources')
            plt.xlabel('Estimated Angular Size (arcseconds)')
            plt.ylabel('Number of Sources')
            plt.grid(axis='y', alpha=0.75)
            plt.savefig(args.output_file.replace('.csv', '_distribution.png'))
            plt.show()

        if root == _default_root or args.load_from_catalogue:
            # On a separate figure, plot a histogram of the differences between these sizes and the LAS values in the
            # DR2 catalogue, for sources that have a LAS value.
            with paper_style():
                plt.figure(figsize=(10, 6))
                # Load the DR2 catalogue to get the LAS values
                with fits.open(paths.STRIPPED_CATALOGUE_PATH, memmap=False) as hdul:
                    dr2_data = hdul[1].data
                    las_values = dr2_data['LAS'][indices]  # Get LAS values for the sources we have sizes for

                # Calculate the differences between estimated sizes and LAS values, ignoring NaNs
                valid_indices = ~np.isnan(las_values)
                size_differences = sizes[valid_indices] - las_values[valid_indices]
                # filter to keep only 95% of absolute differences to avoid extreme outliers in the histogram
                lower_bound = np.percentile(size_differences, 2.5)
                upper_bound = np.percentile(size_differences, 97.5)
                size_differences = size_differences[(size_differences >= lower_bound) \
                                                    & (size_differences <= upper_bound)]

                plt.hist(size_differences, bins=100, color='lightcoral', edgecolor='black')
                plt.title('Differences Between Estimated Angular Sizes and DR2 LAS Values')
                plt.xlabel('Estimated Size - DR2 LAS (arcseconds)')
                plt.ylabel('Number of Sources')
                plt.grid(axis='y', alpha=0.75)
                plt.savefig(args.output_file.replace('.csv', '_size_difference_distribution.png'))
                plt.show()
