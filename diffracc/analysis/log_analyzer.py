import re
from pathlib import Path, PurePath

from ..utils import paths
from ..utils.logger import LoggingLevels
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer


def get_flux( path: Path ):
    """
    A function to get the flux of a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    flux: float
        The flux of the image in Jy or arbitrary units (because of 0-1 normalizaiton)
    """
    with open( str( path ), encoding='utf-8' ) as file:
        filedata = file.read()
    exp = re.compile( r"Flux from sum of \(non-blank\) pixels ..... : (\d+\.\d+) Jy" )
    match : re.Match[str] = exp.search( filedata )
    if match is None:
        print( str( path ) )
    flux = float( match.group( 1 ) )
    return flux


def get_model_flux( path: Path ):
    """
    A function to get the model flux of a log file at path.

    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    flux: float
        The flux of the model in Jy or arbitrary units (because of 0-1 normalizaiton)
    """
    with open( str( path ), encoding='utf-8' ) as file:
        filedata = file.read()
    exp = re.compile( r"Total flux density in model ............. : (\d+\.\d+) Jy" )
    match = exp.search( filedata )
    if match is None:
        flux = 0 # Log won't have this line if no flux is found - so set model flux to 0
    else:
        flux = float( match.group( 1 ) )
    return flux


def get_mean( path: Path ):
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
    with open( str( path ), encoding='utf-8' ) as file:
        filedata = file.read()
    exp = re.compile( r"Raw mean \(Stokes I\) =  (\d+\.\d+) mJy" )
    match = exp.search( filedata )
    mean = float( match.group( 1 ) )
    return mean


def get_sigma_clipped_mean( path: Path ):
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
    with open( str( path ), encoding='utf-8' ) as file:
        filedata = file.read()
    exp = re.compile( r"sigma clipped mean \(Stokes I\) =  -?(\d+\.\d+) mJy" )
    match = exp.search( filedata )
    if match is None:
        print( str( path ) )
    mean = float( match.group( 1 ) )
    return mean


def get_rms( path: Path ):
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
    with open( str( path ), encoding='utf-8' ) as file:
        filedata = file.read()
    exp = re.compile( r"raw rms =  (\d+\.\d+) mJy" )
    match = exp.search( filedata )
    if match is None:
        print( str( path ) )
    rms = float( match.group( 1 ) )
    return rms


def get_sigma_clipped_rms( path: Path ):
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
    with open( str( path ), encoding='utf-8' ) as file:
        filedata = file.read()
    exp = re.compile( r"sigma clipped rms =  (\d+\.\d+) mJy" )
    match = exp.search( filedata )
    if match is None:
        print( str( path ) )
    rms = float( match.group( 1 ) )
    return rms


def get_flux_mean_rms( path: Path ):
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
    with open( str( path ), encoding='utf-8' ) as file:
        filedata = file.read()
    #include re.DOTALL to make the .*? able to expand over newlines
    exp = re.compile( r"Raw mean \(Stokes I\) =  (\d+\.\d+) mJy and raw rms =  (\d+\.\d+) mJy.*?Flux from sum of \(non-blank\) pixels ..... : (\d+\.\d+) Jy", re.DOTALL )
    match = exp.search( filedata )
    if match is None:
        print( str( path ) )
    mean = float( match.group( 1 ) )
    rms = float( match.group( 2 ) )
    flux = float( match.group( 3 ) )
    return flux, mean, rms



class LogAnalyzer():
    """
    A class to analyze PyBDSF log files
    """
    def __init__( self,
                 subdir: PurePath | str,
                 log_file_dir: Path = paths.PYBDSF_LOG_PARENT,
                 log_level: int = LoggingLevels.INFO.value ):
        """
        Initialises a LogAnalyzer object, which is a recursive analyzer for log_file_dir/subdir, with additional
        functionality to extract flux, mean, and rms from the log files.

        Parameters
        ----------
        subdir : PurePath | str
            The subdirectory of this object, e.g. 'generated' or 'dataset'
        log_file_dir : Path = utils.paths.PYBDSF_LOG_PARENT
            The directory of the log files, without the subdirectory. By default, this is the PYBDSF_LOG_PARENT path in
            utils.paths
        log_level : int = logging.INFO
            The logging level to use for this object. By default, this is logging.INFO
        """
        self.rfa = RecursiveFileAnalyzer( log_file_dir / subdir, log_level )
        self.subdir = subdir if isinstance( subdir, PurePath ) else PurePath( subdir )
