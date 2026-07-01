"""
This module provides utility functions for logging, notably a function to set up a logger with a specified name and
logging level. It also defines an enumeration for logging levels to facilitate easy configuration of the logger.
"""
import logging
from enum import Enum

from tqdm import tqdm


class LoggingLevels(Enum):
    """
    An enum to represent the different logging levels. This is used to set the logging level for the logger.
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def get_logger(name: str, level: int = LoggingLevels.INFO.value) -> logging.Logger:
    """
    Set up a logger with the given name and level. If the logger already has
    handlers, clear and reset them.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level, by default LoggingLevels.INFO.value

    Returns
    -------
    logging.Logger
        The logger with the given name and level.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # Check if the logger already has handlers
        logger.handlers.clear()  # Clear the default handlers
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s (%(name)s): %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


pbar, last_loaded = None, 0
def show_dl_progress(block_num : int, block_size : int, total_size : int):
    """
    Designed as report_hook argument for urllib.request.urlretrieve. Displays
    a progress bar for the download.

    Parameters
    ----------
    block_num : int
        Number of blocks downloaded so far.
    block_size : int
        Size of blocks in bytes.
    total_size : int
        Total size of the download in bytes.

    Comments
    --------
    I didn't specifically check whether the arguments are actually float, so
    if your life depends on it, don't make your life depend on it.
    """
    global pbar, last_loaded
    if pbar is None:
        pbar = tqdm(total=total_size, unit="Bytes", unit_scale=True)

    downloaded = block_num * block_size
    increment = downloaded - last_loaded
    last_loaded = downloaded
    if downloaded < total_size:
        pbar.update(increment)
    else:
        pbar.close()
        pbar, last_loaded = None, 0
