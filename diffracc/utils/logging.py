import logging
from tqdm import tqdm


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and level. If the logger already has
    handlers, clear and reset them.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level, by default logging.INFO

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


def show_dl_progress(block_num, block_size, total_size):
    """
    Designed as report_hook argument for urllib.request.urlretrieve. Displays
    a progress bar for the download.

    Parameters
    ----------
    block_num : float
        Number of blocks downloaded so far.
    block_size : float
        Size of blocks in bytes.
    total_size : float
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
