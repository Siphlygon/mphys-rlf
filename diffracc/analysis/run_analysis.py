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


def analyze(subdirs: list[str],
            fits_input_dir: str | Path | None = None,
            n_cpus: int | None = None,
            max_tasks_per_child: int | None = 50):
    """
    A function to analyze fits files in the given subdirectories.

    Parameters
    ----------
    subdirs : list[str]
        A list of subdirectories to analyze
    fits_input_dir : str | Path | None, optional
        The directory containing the FITS files to analyze, by default None
    n_cpus : int | None, optional
        Number of worker processes. None resolves to the N_CPUS environment variable if set, else os.cpu_count().
    max_tasks_per_child : int | None, optional
        Recycle each worker after this many files to bound PyBDSF's per-image memory growth. None disables recycling
        (useful on 'spawn' platforms where each recycle re-imports bdsf). By default 50.
    """
    if fits_input_dir is None:
        fits_input_dir = paths.FITS_PARENT

    for subdir in subdirs:
        analyzer = ImageAnalyzer(subdir=subdir,
                                 fits_input_dir=fits_input_dir,
                                 export_images=['gaus_model', 'gaus_resid'],
                                 catalog_format='fits')
        analyzer.analyze_all_fits_in_input(n_cpus=n_cpus, max_tasks_per_child=max_tasks_per_child)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python run_analysis.py',
                                     description='A program to analyze fits files with PyBDSF')
    parser.add_argument("--input-dir",
                        help="FITS input directory, default utils.paths.FITS_PARENT", type=str, default=None)
    parser.add_argument("-j", "--n-cpus", type=int, default=None,
                        help="Number of worker processes. Default: $N_CPUS if set, else os.cpu_count().")
    parser.add_argument("--max-tasks-per-child", type=int, default=50,
                        help="Recycle each worker after N files to bound PyBDSF memory growth. Pass 0 to disable "
                             "recycling (useful on 'spawn' platforms where each recycle re-imports bdsf). Default: 50.")
    parser.add_argument("SUBDIRS",
                        help="Any number of subdirectories, or 0 to use utils.paths.SUBDIRS", nargs='*')
    args = parser.parse_args()

    # argparse can't express "0 means None"; translate here so --max-tasks-per-child 0 disables recycling.
    max_tasks_per_child = args.max_tasks_per_child or None

    if len(args.SUBDIRS) > 0:
        logger.info(f"Analyzing {len(args.SUBDIRS)} custom subdirs")
        for subdir in args.SUBDIRS:
            logger.info(f"    {subdir}")
        logger.info(f"fits input directory {args.input_dir}")
        analyze(args.SUBDIRS, args.input_dir, n_cpus=args.n_cpus, max_tasks_per_child=max_tasks_per_child)
    else:
        logger.info("Analyzing default subdirectories from utils.paths.SUBDIRS")
        analyze(paths.SUBDIRS, n_cpus=args.n_cpus, max_tasks_per_child=max_tasks_per_child)
