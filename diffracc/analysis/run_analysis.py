"""
This module provides a command-line interface for analyzing FITS files using the ImageAnalyzer class. It allows users to
specify subdirectories and an optional input directory for FITS files. The analysis results, including Gaussian model
and residual images, are exported in the specified catalog format. The module also includes logging for tracking the
analysis process.
"""
import argparse
from pathlib import Path

from ..analysis.image_analyzer import ImageAnalyzer
from ..utils import paths
from ..utils.logger import get_logger

logger = get_logger(__name__)


def analyze(subdirs: list[str], fits_input_dir: str | Path | None = None):
    """
    A function to analyze fits files in the given subdirectories.

    Parameters
    ----------
    subdirs : list[str]
        A list of subdirectories to analyze
    fits_input_dir : str | Path | None, optional
        The directory containing the FITS files to analyze, by default None
    """
    if fits_input_dir is None:
        fits_input_dir = paths.FITS_PARENT

    for subdir in subdirs:
        analyzer = ImageAnalyzer(subdir=subdir,
                                 fits_input_dir=fits_input_dir,
                                 export_images=['gaus_model', 'gaus_resid'],
                                 catalog_format='fits')
        analyzer.analyze_all_fits_in_input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python run_analysis.py',
                                     description='A program to analyze fits files with PyBDSF')
    parser.add_argument("--input-dir",
                        help="FITS input directory, default utils.paths.FITS_PARENT", type=str, default=None)
    parser.add_argument("SUBDIRS",
                        help="Any number of subdirectories, or 0 to use utils.paths.SUBDIRS", nargs='*')
    args = parser.parse_args()

    if len(args.SUBDIRS) > 0:
        logger.info(f"Analyzing {len(args.SUBDIRS)} custom subdirs")
        for subdir in args.SUBDIRS:
            logger.info(f"    {subdir}")
        logger.info(f"fits input directory {args.input_dir}")
        analyze(args.SUBDIRS, args.input_dir)
    else:
        logger.info("Analyzing default subdirectories from utils.paths.SUBDIRS")
        analyze(paths.SUBDIRS)
