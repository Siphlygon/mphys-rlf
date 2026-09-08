"""
Loading and flux-filtering the radio-source components that `AngularSizeFinder` sizes.

`ComponentLoader` can extract and flux-filter components from PyBDSF catalogue FITS files, assemble them from the DR2
value-added component catalogue by matching source names, or reload a previously consolidated set from a pickle cache.
Every route returns the same per-source component lists (rows of `(Total_flux, RA, DEC, DC_Maj, DC_Min, PA)`) and their
source indices.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer


class ComponentLoader:
    """
    Load per-source radio components from PyBDSF catalogue FITS files, the DR2 component catalogue, or a pickle cache.
    """
    def __init__(self,
                 root_dir: Path = paths.PYBDSF_CATALOG_PARENT / "dr2_cutouts_download",
                 flux_threshold: float = 0.95,
                 num_processes: int = 1):
        """
        Parameters
        ----------
        root_dir : Path, optional
            The default root directory of PyBDSF catalogue FITS files, by default
            `paths.PYBDSF_CATALOG_PARENT / "dr2_cutouts_download"`.
        flux_threshold : float, optional
            The fraction of total flux to keep when filtering components, by default 0.95. PyBDSF can fit islands to
            noise, so components are sorted by flux and the dimmest are dropped while the cumulative flux of those kept
            stays above this fraction.
        num_processes : int, optional
            Worker processes for the (GIL-bound) FITS parsing step, by default 1 (threaded). Values > 1 parse across a
            process pool.
        """
        self.logger = get_logger("ComponentLoader", LoggingLevels.INFO.value)
        self.root_dir = root_dir
        self.flux_threshold = flux_threshold
        self.num_processes = num_processes
        self.rfa = RecursiveFileAnalyzer(self.root_dir)

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

        return ComponentLoader._filter_by_flux(components, flux_threshold)

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

    def _extract_components(self, fits_dir: str | Path | None, pattern: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract and flux-filter the components from every catalogue FITS under `fits_dir`.

        Reading a PyBDSF binary table is dominated by astropy parsing which binds the GIL, so when `num_processes > 1`
        this runs across worker processes (`mode="process"`), which the GIL-bound parse actually benefits from;
        otherwise it stays on the default threaded file mode.

        Parameters
        ----------
        fits_dir : str | Path | None
            The root directory of catalogue FITS files. If `None`, the loader's `root_dir` is used.
        pattern : str
            The regex pattern matching the catalogue files, with a capture group for the source index.

        Returns
        -------
        components_list : np.ndarray
            The per-source filtered component lists.
        fits_indices : np.ndarray
            The source indices corresponding to `components_list`.
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
        Assemble the per-source components from the DR2 catalogue, matching each source's components by `Source_Name`.

        Parameters
        ----------
        fits_indices : np.ndarray
            The source indices (positions in the source catalogue) to load components for.
        source_catalogue_path : str | Path
            Path to the source catalogue FITS file containing the DR2 catalogue information.
        component_catalogue_path : str | Path
            Path to the component catalogue FITS file containing the DR2 components.

        Returns
        -------
        components_list : np.ndarray
            The per-source component lists, one object-array entry per source index.
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

    def load(self,
             fits_dir: str | Path | None,
             pattern: str,
             components_cache: str | Path | None,
             load_from_catalogue: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the per-source components and indices, using a one-time consolidated cache when available.

        Extracting the components re-parses every catalogue FITS (the dominant cost of the whole pipeline). When
        `components_cache` is given, the extracted result is consolidated into that single file so subsequent runs -
        e.g. re-estimating with a different `n_points`, or recovering after a downstream failure - skip the parse
        entirely.

        Parameters
        ----------
        fits_dir : str | Path | None
            The root directory of catalogue FITS files. If `None`, the loader's `root_dir` is used.
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
            The per-source component lists.
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
