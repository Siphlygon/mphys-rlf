"""
A module for analyzing PyBDSF log files. It provides functions to extract flux, mean, and rms values from the log files,
which can be used by other classes or functions in the diffracc package. The functions use regular expressions to search
for specific patterns in the log files and extract the relevant values.
"""
import re
from pathlib import Path

# A run is delimited by a banner line of '=' characters; PyBDSF appends a new block per run.
_RUN_SEPARATOR = re.compile(r"^=+\s*$", re.MULTILINE)

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


def _get_match(path: Path | str, pattern: str) -> re.Match[str]:
    """
    A back-end function to open a log file and search for a pattern.
    
    This is the maximum shared functionality between those below, as sometimes on `match=None` we may want to print an
    error, other times it is expected and we want to return a default value. This function just returns the match
    object, or None if not found.

    Parameters
    ----------
    path: str
        The path to the log file
    pattern: str
        The regular expression pattern to search for in the log file

    Returns
    -------
    match: re.Match[str]
        The match object containing the extracted values from the log file
    """
    with open(path, encoding='utf-8') as file:
        filedata = file.read()
    filedata = _latest_run(filedata)

    exp = re.compile(pattern)
    match = exp.search(filedata)
    return match


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
    match = _get_match(path, r"Flux from sum of \(non-blank\) pixels ..... : (-?\d+\.\d+) Jy")
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
    match = _get_match(path, r"Total flux density in model ............. : (-?\d+\.\d+) Jy")
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
    match = _get_match(path, r"Raw mean \(Stokes I\) =  (-?\d+\.\d+) mJy")
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
    match = _get_match(path, r"sigma clipped mean \(Stokes I\) =  (-?\d+\.\d+) mJy")
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
    match = _get_match(path, r"raw rms =  (-?\d+\.\d+) mJy")
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
    match = _get_match(path, r"sigma clipped rms =  (-?\d+\.\d+) mJy")
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
