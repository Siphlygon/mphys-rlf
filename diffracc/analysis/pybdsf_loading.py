"""
A module designed to interface with PyBDSF's outputs (specifically the logs, catalogues, and components extracted from
the catalogues) and provide a convenient way to load and analyse them. Also accesses the resolved catalogue values to
compare against the PyBDSF outputs.

Specifically produces:
  * the extended PyBDSF log table  (built here from the per-cutout logs via `log_analyzer.extract_log_fields`),
  * the resolved catalogue values  (`peak_flux`, `rms` = `Isl_rms`, `total_flux`, in mJy),
  * the per-cutout PyBDSF component list  (the `dr2_cutouts_pybdsf.pkl` produced by the cutout run).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from ..las.component_loading import ComponentLoader
from ..utils import paths
from ..utils.logger import get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer
from . import log_analyzer as la

logger = get_logger("analysis.pybdsf_loading")

# Default locations of the three products - override if necessary
DEFAULT_LOG_DIR = paths.PYBDSF_LOG_PARENT / "dr2_cutouts_download"
DEFAULT_LOG_TABLE_CSV = paths.STORAGE_PARENT / "pybdsf_logs_extended.csv"
DEFAULT_COMPONENTS_PKL = paths.STORAGE_PARENT / "dr2_cutouts_pybdsf.pkl"

# Regex used to list the per-cutout logs and recover the cutout index from the filename (first capture group).
_CUTOUT_LOG_PATTERN = r".*?cutout(\d+)\.fits\.pybdsf\.log$"

# TODO: is this needed?
# Column layout of one component row in the pkl, in degrees for the axes.
_C_TFLUX, _C_RA, _C_DEC, _C_DC_MAJ, _C_DC_MIN, _C_PA = range(6)

# Detection threshold PyBDSF applied (thresh_pix); the island threshold thresh_isl was 4. Used only to report the
# implied 5-sigma peak level alongside a source, never to re-decide detection.
THRESH_PIX = 5.0


def percentiles(values: pd.Series | np.ndarray, points: tuple[int, ...] = (5, 25, 50, 75, 95)) -> dict[int, float]:
    """
    Return the given percentiles of `values`, ignoring NaNs.

    Parameters
    ----------
    values : pd.Series | np.ndarray
        The values to summarise.
    points : tuple[int, ...], optional
        The percentiles to compute, by default the 5th, 25th, 50th, 75th and 95th.

    Returns
    -------
    dict[int, float]
        Mapping of each percentile in `points` to its value, rounded to 4 significant places.
    """
    return {p: round(float(np.nanpercentile(values, p)), 4) for p in points}


def build_log_table(log_dir: Path | str = DEFAULT_LOG_DIR,
                    out_csv: Path | str | None = DEFAULT_LOG_TABLE_CSV,
                    overwrite: bool = False,
                    numeric_range: tuple[int, int] | None = None,
                    mode: str = "batch",
                    num_workers: int | None = None) -> pd.DataFrame:
    """
    Parse the per-cutout PyBDSF logs into one table, extended with the fields the a-trous and background-audit analyses
    need (all-scales model flux, main-stage Gaussian count and background mean, and the constant-rms / 1-D-warning
    flags) beyond the six the original prototype cached.
    
    The result is cached to `out_csv` -- pass `overwrite=True` to force a re-parse, or `numeric_range=(lo, hi)` to build
    only a slice of cutout indices (useful for a quick sample without touching the cache - pair it with `out_csv=None`.

    Parameters
    ----------
    log_dir : Path | str, optional
        Directory tree of `cutoutN.fits.pybdsf.log` files, by default `DEFAULT_LOG_DIR`.
    out_csv : Path | str | None, optional
        Where to cache the table. If `None`, the table is not written, by default `DEFAULT_LOG_TABLE_CSV`.
    overwrite : bool, optional
        Re-parse even if `out_csv` already exists, by default `False`.
    numeric_range : tuple[int, int] | None, optional
        A half-open `[lo, hi)` range of cutout indices to restrict the parse to, by default `None` (all).
    mode : str, optional
        `RecursiveFileAnalyzer` execution mode; `"batch"` (threaded) suits this light-parse, I/O-bound work, by default
        `"batch"`.
    num_workers : int | None, optional
        Worker count, by default `None` (the analyzer's per-mode default).

    Returns
    -------
    pd.DataFrame
        The per-cutout fields, indexed by cutout number (`index`), sorted, with failed-to-parse logs dropped.
    """
    if out_csv is not None and Path(out_csv).exists() and not overwrite:
        logger.info("Loading cached log table from %s", out_csv)
        return pd.read_csv(out_csv, index_col="index")

    logger.info("Parsing PyBDSF logs under %s", log_dir)
    result = RecursiveFileAnalyzer(log_dir).run_pipeline(function=la.extract_log_fields,
                                                         pattern=_CUTOUT_LOG_PATTERN,
                                                         return_nums=True,
                                                         numeric_range=numeric_range,
                                                         mode=mode,
                                                         num_workers=num_workers,
                                                         progress_bar_desc="Parsing PyBDSF logs")
    records, indices = list(result.results), result.numbers
    # A failed parse comes back as None (RecursiveFileAnalyzer swallows per-file errors); drop those, keeping alignment.
    keep = [(idx, rec) for idx, rec in zip(indices, records) if rec is not None]
    df = pd.DataFrame([rec for _, rec in keep], index=[idx for idx, _ in keep])
    df.index.name = "index"
    df = df.sort_index()

    if out_csv is not None:
        logger.info("Caching log table (%d rows) to %s", len(df), out_csv)
        df.to_csv(out_csv)
    return df


def load_catalogue_values(path: Path | str = paths.STRIPPED_CATALOGUE_PATH) -> pd.DataFrame:
    """
    Load the resolved catalogue values from a FITS file.

    Parameters
    ----------
    path : Path | str, optional
        Path to the catalogue FITS file, by default `paths.STRIPPED_CATALOGUE_PATH`.

    Returns
    -------
    pd.DataFrame
        Columns `peak_flux`, `rms`, `total_flux` of resolved sources from the catalogue, indexed by cutout number
        (`index`).
    """
    with fits.open(path) as hdul:
        d: np.recarray = hdul[1].data
        data = d[d["Resolved"]]
        df = pd.DataFrame({
            "peak_flux": data["Peak_flux"],
            "rms": data["Isl_rms"],
            "total_flux": data["Total_flux"],
        }, dtype=float)
    df.index.name = "index"
    return df


def _summarise_components(rows: np.ndarray) -> dict[str, float]:
    """
    Reduce one source's component array to the per-source scalars the analysis uses.

    A 1-D degenerate ("ridge") component has a deconvolved minor axis of exactly zero but a non-zero major axis - this
    is what is meant by "1-D" in the PyBDSF log. A point component has both axes zero.

    Parameters
    ----------
    rows : np.ndarray
        The source's components, shape `(n, 6)`: `Total_flux` (Jy), RA, DEC, `DC_Maj`, `DC_Min`, PA (degrees).

    Returns
    -------
    dict[str, float]
        `n_comp`, total `model_flux_mjy`, `model_flux_no_oned_mjy` (excluding ridge components), and the ridge/point
        counts `n_oned` / `n_point`.
    """
    rows = np.asarray(rows, dtype=float)
    is_ridge = (rows[:, _C_DC_MIN] == 0) & (rows[:, _C_DC_MAJ] > 0)
    is_point = (rows[:, _C_DC_MIN] == 0) & (rows[:, _C_DC_MAJ] == 0)
    return {
        "n_comp": len(rows),
        "model_flux_mjy": rows[:, _C_TFLUX].sum() * 1e3,
        "model_flux_no_oned_mjy": rows[~is_ridge, _C_TFLUX].sum() * 1e3,
        "n_oned": int(is_ridge.sum()),
        "n_point": int(is_point.sum()),
    }


def load_components(path: Path | str = DEFAULT_COMPONENTS_PKL) -> pd.DataFrame:
    """
    Load the per-cutout PyBDSF component list and reduce it to one summary row per detected source. Only cutouts with at
    least one fitted component appear (i.e. the detections).
    
    This differs from `load_raw_components` in that it returns a DataFrame of summary statistics, rather than the raw
    `(n, 6)` direct arrays.
    
    Parameters
    ----------
    path : Path | str, optional
        Path to the components pickle (a dict of `components` and `indices`), by default `DEFAULT_COMPONENTS_PKL`.

    Returns
    -------
    pd.DataFrame
        One row per detected source (columns from `_summarise_components`), indexed by cutout number (`index`).
    """
    if not Path(path).exists():
        logger.info(f"Components pickle {path} not found; building it from the PyBDSF catalogues.")

        # Load the components from the PyBDSF catalogues and save them to a pickle for future use.
        loader = ComponentLoader()
        loader.load_components(fits_dir=paths.PYBDSF_CATALOG_PARENT / "dr2_cutouts_download",
                               components_cache=path) # running this will create the pickle file at the specified path

    obj = pd.read_pickle(path)
    summaries = [_summarise_components(rows) for rows in obj["components"]]
    df = pd.DataFrame(summaries, index=np.asarray(obj["indices"]))
    df.index.name = "index"
    return df.sort_index()


def load_raw_components(path: Path | str = DEFAULT_COMPONENTS_PKL) -> dict[int, np.ndarray]:
    """
    Load the components as raw `(n, 6)` arrays keyed by cutout index, for inspecting a single source.

    Parameters
    ----------
    path : Path | str, optional
        Path to the components pickle, by default `DEFAULT_COMPONENTS_PKL`.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping of cutout index to its component array (`Total_flux`, RA, DEC, `DC_Maj`, `DC_Min`, PA).
    """
    obj = pd.read_pickle(path)
    return {int(idx): np.asarray(rows, dtype=float) for idx, rows in zip(obj["indices"], obj["components"])}


def flux_alignment_corr(joined: pd.DataFrame) -> float:
    """
    Log-space correlation between the image's summed pixel flux and the catalogue total flux, as an alignment check.

    A correct index alignment gives a strong positive correlation (~0.8); a value near zero means the log/catalogue join
    is pairing unrelated sources (e.g. a full-catalogue file joined against resolved-subset cutout indices). This is
    meant as a way to diagnose a misalignment before getting deep into the machinery.

    Parameters
    ----------
    joined : pd.DataFrame
        A join of the log table and catalogue values, carrying `sum_flux` (Jy) and `total_flux` (mJy).

    Returns
    -------
    float
        Pearson correlation of `log10(sum_flux)` and `log10(total_flux)` over rows where both are positive.
    """
    ok = (joined["sum_flux"] > 0) & (joined["total_flux"] > 0)
    x = np.log10(joined.loc[ok, "sum_flux"])
    y = np.log10(joined.loc[ok, "total_flux"])
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def join_logs_catalogue(logs: pd.DataFrame, cat: pd.DataFrame, warn_below: float = 0.5) -> pd.DataFrame:
    """
    Inner-join the log table and catalogue values on cutout index, warning if the flux alignment looks broken.

    Parameters
    ----------
    logs : pd.DataFrame
        The (extended) log table from `build_log_table`.
    cat : pd.DataFrame
        The catalogue values from `load_catalogue_values`.
    warn_below : float, optional
        Log a warning if `flux_alignment_corr` falls below this, by default `0.5`.

    Returns
    -------
    pd.DataFrame
        The joined table, indexed by cutout number.
    """
    joined = logs.join(cat, how="inner")
    corr = flux_alignment_corr(joined)
    if corr < warn_below:
        logger.warning("Flux-alignment correlation is %.3f (< %.2f): logs and catalogue may be misaligned.",
                      corr, warn_below)
    else:
        logger.info("Flux-alignment correlation %.3f over %d joined cutouts.", corr, len(joined))
    return joined
