"""
A module for analyzing pyBDSF log files. It provides functions to extract flux, mean, and rms values from the log files,
which can be used by other classes or functions in the diffracc package. The functions use regular expressions to search
for specific patterns in the log files and extract the relevant values.
"""
import re
from pathlib import Path

import numpy as np

# A run is delimited by a banner line of '=' characters; PyBDSF appends a new block per run.
_RUN_SEPARATOR = re.compile(r"^=+\s*$", re.MULTILINE)

# Regex pattern that matches a signed decimal or scientific-notation number, e.g. "0.2861", "0.0", "2.48e-04", "-1e-05".
_FLOAT = r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"

# Pre-compile regex patterns for each log field we want to extract. Each pattern captures the numeric value of interest
# in a single group, which we can then convert to float. They are designed around the specific pyBDSF log output format.
_LOG_PATTERNS = {
    "sum_flux":             re.compile(rf"Flux from sum of \(non-blank\) pixels \.+ : {_FLOAT} Jy"),
    "model_flux_main":      re.compile(rf"Total flux density in model \.+ : {_FLOAT} Jy"),
    "model_flux_allscales": re.compile(rf"Total flux density in model on all scales : {_FLOAT} Jy"),
    "raw_mean":             re.compile(rf"Raw mean \(Stokes I\) =  {_FLOAT} mJy"),
    "sigma_clipped_mean":   re.compile(rf"sigma clipped mean \(Stokes I\) =  {_FLOAT} mJy"),
    "raw_rms":              re.compile(rf"raw rms =  {_FLOAT} mJy"),
    "sigma_clipped_rms":    re.compile(rf"sigma clipped rms =  {_FLOAT} mJy"),
    "n_gauss_main":         re.compile(r"Total number of Gaussians fit to image \.+ : (\d+)"),
    "bg_mean_main":         re.compile(rf"Value of background mean \.+ : {_FLOAT} Jy/beam"),
}


# Presence-only flags:
# if the rms_box is too large for the stamp, PyBDSF falls back to a single constant background (const rms)
# if more than 50% of the Gaussians are 1-D, PyBDSF issues a degenerate-fit warning (oned warning)
_CONST_RMS_PATTERN = re.compile(r"Size of rms_box larger than 1/4 of image size")
_ONED_WARNING_PATTERN = re.compile(r"50% of Gaussians are 1-D")


def _latest_run(filedata: str) -> str:
    """
    Return the text of the most recent PyBDSF run in a (possibly appended-to) log file.
    
    PyBDSF appends new runs to the end of the log file. In normal usage the user hopefully shouldn't have to run PyBDSF
    multiple times on the same source, but if they do, this function ensures that we only analyze the most recent run.

    Parameters
    ----------
    filedata : str
        The text content of a PyBDSF log file, which may contain multiple runs appended together.
    
    Returns
    -------
    str
        The text content of the most recent run in the log file. If no run separator is found, the entire filedata is
        returned.
    """
    for block in reversed(_RUN_SEPARATOR.split(filedata)):
        if block.strip():
            return block
    return filedata


def _get_match(path: Path | str,
               pattern: str | list[str] | re.Pattern | list[re.Pattern]) -> re.Match[str] | list[re.Match[str]] | None:
    """
    A back-end function to open a log file and search for a pattern.
    
    This is the maximum shared functionality between those below, as sometimes on `match=None` we may want to print an
    error, other times it is expected and we want to return a default value. This function just returns the match
    object, or None if not found.

    Parameters
    ----------
    path: str
        The path to the log file
    pattern: str | list[str] | re.Pattern | list[re.Pattern]
        The regular expression pattern(s) to search for in the log file. If a list of patterns is provided, the function
        will return a list of match objects corresponding to each pattern. If a single pattern is provided, it will
        return a single match object.

    Returns
    -------
    match: re.Match[str] | list[re.Match[str]] | None
        The match object containing the extracted values from the log file, or a list of match objects if multiple
        patterns are provided, or None if no match is found.
    """
    assert isinstance(pattern, (str, list, re.Pattern)), "pattern must be a string, compiled pattern, or list thereof"

    with open(path, encoding='utf-8') as file:
        filedata = file.read()
    filedata = _latest_run(filedata)

    if isinstance(pattern, str):
        exp = re.compile(pattern)
        match = exp.search(filedata)
        return match

    if isinstance(pattern, re.Pattern):
        match = pattern.search(filedata)
        return match

    matches = []
    if isinstance(pattern, list):
        for p in pattern:
            if isinstance(p, str):
                exp = re.compile(p)
                match = exp.search(filedata)
            elif isinstance(p, re.Pattern):
                match = p.search(filedata)
            else:
                continue
            matches.append(match)
        return matches

    return None


def get_total_flux(path: Path)-> float:
    """
    A function to get the total flux of an image from a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    flux: float
        The total flux of the image in Jy or arbitrary units (because of 0-1 normalizaiton)
    """
    match = _get_match(path, _LOG_PATTERNS["sum_flux"])
    if match is None:
        print(str(path))
    total_flux = float(match.group(1))  # NOTE: this can be -0.000 Jy
    return total_flux


def get_model_flux(path: Path)-> float:
    """
    A function to get the model flux of a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    model_flux: float
        The flux of the model in Jy or arbitrary units (because of 0-1 normalizaiton)
    """
    match = _get_match(path, _LOG_PATTERNS["model_flux_main"])
    if match is None:
        return 0 # Log won't have this line if no flux is found - so set model flux to 0
    model_flux = float(match.group(1))
    return model_flux


def get_mean(path: Path) -> float:
    """
    A function to get the mean of a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    mean: float
        The raw mean of the image in mJy or arbitrary units
    """
    match = _get_match(path, _LOG_PATTERNS["raw_mean"])
    if match is None:
        print(str(path))
    mean = float(match.group(1))
    return mean


def get_sigma_clipped_mean(path: Path) -> float:
    """
    A function to get the sigma clipped mean of a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    mean: float
        The sigma clipped mean of the image in mJy or arbitrary units
    """
    match = _get_match(path, _LOG_PATTERNS["sigma_clipped_mean"])
    if match is None:
        print(str(path))
    mean = float(match.group(1))
    return mean


def get_rms(path: Path) -> float:
    """
    A function to get the rms of a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    rms: float
        The raw rms of the image in mJy or arbitrary units
    """
    match = _get_match(path, _LOG_PATTERNS["raw_rms"])
    if match is None:
        print(str(path))
    rms = float(match.group(1))
    return rms


def get_sigma_clipped_rms(path: Path) -> float:
    """
    A function to get the sigma clipped rms of a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    rms: float
        The sigma clipped rms of the image in mJy or arbitrary units
    """
    match = _get_match(path, _LOG_PATTERNS["sigma_clipped_rms"])
    if match is None:
        print(str(path))
    rms = float(match.group(1))
    return rms


def get_flux_mean_rms(path: Path)-> tuple[float, float, float]:
    """
    A function to combine getting the flux, mean, and rms of a log file at path

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    flux: float
        The flux of the image in Jy or arbitrary units (because of 0-1 normalizaiton)
    mean: float
        The raw mean of the image in mJy or arbitrary units
    rms: float
        The raw rms of the image in mJy or arbitrary units
    """
    with open(str(path), encoding='utf-8')as file:
        filedata = file.read()
    filedata = _latest_run(filedata)
    #include re.DOTALL to make the .*? able to expand over newlines
    exp = re.compile(
        r"Raw mean \(Stokes I\) =  (-?\d+\.\d+) mJy and raw rms =  (-?\d+\.\d+) mJy"
        r".*?Flux from sum of \(non-blank\) pixels ..... : (-?\d+\.\d+) Jy",
        re.DOTALL,
    )
    match = exp.search(filedata)
    if match is None:
        print(str(path))
    mean = float(match.group(1))
    rms = float(match.group(2))
    flux = float(match.group(3))
    return flux, mean, rms


def extract_log_fields(path: Path | str) -> dict[str, float]:
    """
    Extract every per-cutout field used by the PyBDSF validation analysis from a single log, in one file read.

    Numeric fields absent from the log are returned as `np.nan` (e.g. `model_flux_allscales` when a-trous added nothing,
    or certain fields when the source was not detected). The two flags are returned as `0.0`/`1.0`.

    Parameters
    ----------
    path : Path | str
        The path to the PyBDSF log file.

    Returns
    -------
    dict[str, float]
        The extracted fields: the keys of `_LOG_PATTERNS` plus `const_rms` and `oned_warning`. Fluxes are in Jy,
        means/rms in mJy (raw/clipped) or Jy/beam (`bg_mean_main`), matching the log's own units.
    """
    with open(path, encoding="utf-8") as file:
        filedata = file.read()
    filedata = _latest_run(filedata)

    fields = {}
    for name, pattern in _LOG_PATTERNS.items():
        match = pattern.search(filedata)
        fields[name] = float(match.group(1)) if match else np.nan
    fields["const_rms"] = float(_CONST_RMS_PATTERN.search(filedata) is not None)
    fields["oned_warning"] = float(_ONED_WARNING_PATTERN.search(filedata) is not None)
    return fields
