"""
This file was created by Ashley and Luna. It provides a complete application that can be used to sample images according
to the LOFAR model with certain parameters, and it provides a function that can be used to access that application
through other files. This application can be distributed across multiple nodes.
"""

import argparse
import configparser
import dataclasses
import math
from pathlib import Path, PurePath
from typing import Callable

import numpy as np
import scipy.stats
import torch

from ..analysis.image_analyzer import ImageAnalyzer, RecursiveFileAnalyzer
from ..model import sampler
from ..utils import paths
from ..utils.distributed import DistributedUtils
from ..utils.logger import LoggingLevels, get_logger
from ..utils.power_transform import PeakFluxPowerTransformer

# scipy distribution used for each non-'dataset' value of SampleArgs.distribution
_FPEAK_DISTRIBUTIONS = {
    'uniform': scipy.stats.uniform,
    'loguniform': scipy.stats.loguniform,
}

# bounds (arcsec) for the largest-angular-size context value, when las_conditioning_enabled is set
_LAS_LOWER_BOUND = 6
_LAS_UPPER_BOUND = 120


@dataclasses.dataclass(frozen=True)
class SampleArgs:
    """
    Typed, validated arguments for `sample`, loaded from a section of diffracc/config.ini.
    """
    generated_subdir: str
    batch_size: int
    n_samples: int
    folder_size: int
    timesteps: int
    use_cpu: bool
    preserve_values: bool
    model_name: str
    upper_bound: float
    lower_bound: float
    distribution: str
    las_conditioning_enabled: bool

    _INT_FIELDS = ('batch_size', 'n_samples', 'folder_size', 'timesteps')
    _FLOAT_FIELDS = ('upper_bound', 'lower_bound')
    _BOOL_FIELDS = ('use_cpu', 'preserve_values', 'las_conditioning_enabled')
    _VALID_DISTRIBUTIONS = ('dataset', 'uniform', 'loguniform')

    def __post_init__(self) -> None:
        if self.distribution not in self._VALID_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {self.distribution!r}, expected one of {self._VALID_DISTRIBUTIONS}")

    @classmethod
    def from_config(cls, config: configparser.ConfigParser, section: str) -> "SampleArgs":
        """
        Build a SampleArgs from a section of a ConfigParser.

        config.ini's [DEFAULT] section is shared by scripts across the project (dataset, completeness, RLF, ...),
        so section items are filtered down to just this class's fields before construction.

        Parameters
        ----------
        config : configparser.ConfigParser
            The parsed config file
        section : str
            The section to read arguments from

        Returns
        -------
        SampleArgs
            The parsed and validated arguments

        Raises
        ------
        ValueError
            If a required key is missing from the section, or 'distribution' is not a recognised value
        """
        field_names = {f.name for f in dataclasses.fields(cls)}
        raw = {k: v for k, v in config.items(section) if k in field_names}

        missing = field_names - raw.keys()
        if missing:
            raise ValueError(f"Missing required config keys in section {section!r}: {sorted(missing)}")

        for field in cls._INT_FIELDS:
            raw[field] = int(raw[field])
        for field in cls._FLOAT_FIELDS:
            raw[field] = float(raw[field])
        for field in cls._BOOL_FIELDS:
            raw[field] = raw[field] == 'True'

        return cls(**raw)  # __post_init__ validates 'distribution'


def get_path_from_index(index: int,
                        subdir: str,
                        bin_size: int) -> tuple[Path, PurePath]:
    """
    Given an index, a subdirectory, and a bin size, this function returns the full path to the FITS file corresponding
    to that index, as well as the postfix path (the part of the path after the subdirectory).

    Parameters
    ----------
    index : int
        The index of the image
    subdir : str
        The subdirectory containing the FITS files
    bin_size : int
        The size of each bin

    Returns
    -------
    Path
        The full path to the FITS file corresponding to the given index
    PurePath
        The postfix path (the part of the path after the subdirectory)
    """
    lower_bound = int(math.floor((index) / bin_size) * bin_size)
    upper_bound = int(math.ceil((index + 1) / bin_size) * bin_size) - 1
    postfix = PurePath(*[f"{lower_bound}-{upper_bound}", f"image{index}.fits"])
    full_image_path = (paths.FITS_PARENT / subdir) / postfix
    return full_image_path, postfix


def _count_existing_samples(generated_subdir: str, bin_start: int, bin_end: int) -> int:
    """
    Count how many FITS files already exist in [bin_start, bin_end) for generated_subdir, so a resumed run doesn't
    regenerate samples that are already on disk.
    
    Parameters
    ----------
    generated_subdir : str
        The subdirectory containing the FITS files
    bin_start : int
        The start of the bin to count
    bin_end : int
        The end of the bin to count
    
    Returns
    -------
    int
        The number of FITS files already existing in the specified bin
    """
    generated_images_dir = paths.FITS_PARENT / generated_subdir
    if not generated_images_dir.exists():
        return 0
    analyzer = RecursiveFileAnalyzer(generated_images_dir)
    return len(analyzer.get_unwrapped_list(None, r'.*?image(\d+)\.fits$', (bin_start, bin_end)).paths)


def _get_fpeak_dist(args: SampleArgs,
                     model_sampler: sampler.Sampler,
                     pt: PeakFluxPowerTransformer) -> Callable[[int], np.ndarray]:
    """
    Get the peak-flux sampling function for args.distribution, mapping raw values through the power transform.
    
    Parameters
    ----------
    args : SampleArgs
        The arguments controlling sampling
    model_sampler : sampler.Sampler
        The model sampler to use for 'dataset' distribution
    pt : PeakFluxPowerTransformer
        The power transformer to use for mapping raw values to the transformed space
    
    Returns
    -------
    Callable[[int], np.ndarray]
        A function that takes an integer n and returns an array of n sampled peak fluxes in the transformed space
    """
    assert args.distribution in _FPEAK_DISTRIBUTIONS or args.distribution == 'dataset', \
        f"Unknown distribution {args.distribution!r}, expected one of {list(_FPEAK_DISTRIBUTIONS.keys()) + ['dataset']}"

    if args.distribution == 'dataset':
        return model_sampler.get_fpeak_model_dist(
            train_set_path=None,
            max_vals=paths.NP_ARRAY_PARENT / args.generated_subdir / paths.MAXVALS)

    scipy_dist = _FPEAK_DISTRIBUTIONS[args.distribution]

    def fpeak_model_dist(n: int) -> np.ndarray:
        values = scipy_dist.rvs(args.lower_bound, args.upper_bound, size=n)
        return pt.transform(np.asarray(values))
    return fpeak_model_dist


def _normalise_image(image: np.ndarray) -> np.ndarray:
    """
    Rescale image to [0, 1], clipping negative values to 0 first. Returns the image unchanged if it's constant,
    since min-max scaling is undefined (would divide by zero) in that case.
    
    Parameters
    ----------
    image : np.ndarray
        The image to normalise
    
    Returns
    -------
    np.ndarray
        The normalised image
    """
    im_max = np.max(image)
    im_min = np.min(image)
    if im_max == im_min:
        return image
    if im_min < 0:
        image = np.where(image > 0, image, 0)
    return (image - im_min) / (im_max - im_min)


def _save_generated_image(image_analyzer: ImageAnalyzer,
                          sample_index: int,
                          image: np.ndarray,
                          context_row: np.ndarray,
                          args: SampleArgs) -> int:
    """
    Normalise (if requested) and save a single generated image to the next free FITS path at or after sample_index.

    Parameters
    ----------
    image_analyzer : ImageAnalyzer
        The image analyzer to use for saving the image
    sample_index : int
        The index of the image to save
    image : np.ndarray
        The image to save
    context_row : np.ndarray
        The context values for the image, used to set FITS headers
    args : SampleArgs
        The arguments controlling sampling, used to determine the subdirectory and bin size for saving

    Returns
    -------
    int
        The index the image was actually saved at (may be greater than sample_index, see the exists() loop below)
    """
    if not args.preserve_values:
        image = _normalise_image(image)

    extra_headers = {'FXSCLD': context_row[0]}
    if args.las_conditioning_enabled:
        extra_headers['LASIZE'] = context_row[1]

    full_image_path, postfix = get_path_from_index(sample_index, args.generated_subdir, args.folder_size)
    # Resuming a bin picks up where a previous run left off; safe because each SLURM task/node owns a
    # disjoint [bin_start, bin_end) range, so no two processes ever write into the same bin concurrently.
    while full_image_path.exists():
        sample_index += 1
        full_image_path, postfix = get_path_from_index(sample_index, args.generated_subdir, args.folder_size)
    image_analyzer.save_image_to_fits(image, postfix, **extra_headers)
    return sample_index


def sample(args: SampleArgs):
    """
    A function to sample images according to the LOFAR model with certain parameters. This function can be distributed
    across multiple nodes, and it will save the generated images to disk as they are created.

    Parameters
    ----------
    args : SampleArgs
        The arguments controlling sampling
    """
    logger = get_logger(__name__, LoggingLevels.DEBUG.value)

    #Do a sampling loop of batch_size samples and save them to the disk as they're generated, until we reach n_samples
    model_sampler = sampler.Sampler(n_samples=args.batch_size, timesteps=args.timesteps)

    # SLURM distribution w/ batching
    du = DistributedUtils()
    n_samples = args.n_samples
    bin_start = du.get_bin_start(n_samples)
    bin_end = du.get_bin_end(n_samples)
    logger.debug('bin_end=%i, bin_start=%i, n_samples=%i', bin_end, bin_start, n_samples)

    # Figure out initial count based on number of fits files already in the directory
    logger.debug('Getting initial count...')
    initial_count = _count_existing_samples(args.generated_subdir, bin_start, bin_end)
    n_samples_in_bin = bin_end - bin_start
    logger.debug('Got initial count %i, requested samples in this bin %i', initial_count, n_samples_in_bin)

    n_samples_to_generate = n_samples_in_bin - initial_count
    if n_samples_to_generate <= 0:
        logger.info('Skipping bin %i-%i, nothing to do', bin_start, bin_end)
        return

    # Get the power transformer for the peak fluxes and the appropriate distribution function
    pt = PeakFluxPowerTransformer(args.generated_subdir)
    fpeak_model_dist = _get_fpeak_dist(args, model_sampler, pt)

    # Generate/Sample the samples
    sample_generated_count = 0
    sample_index = bin_start
    image_analyzer = ImageAnalyzer(args.generated_subdir)
    while sample_generated_count < n_samples_to_generate:
        # to not double-generate at the borders
        batch_size = min(args.batch_size, n_samples_to_generate - sample_generated_count)
        context = fpeak_model_dist(batch_size)[:, np.newaxis]
        # if las conditioning is enabled, add it to context
        if args.las_conditioning_enabled:
            las_values = scipy.stats.uniform.rvs(_LAS_LOWER_BOUND, _LAS_UPPER_BOUND, size=batch_size)
            context = np.concatenate((context, las_values[:, np.newaxis]), axis=1)

        samples = model_sampler.quick_sample(f"{args.model_name}",
                                             context=torch.from_numpy(context),
                                             n_samples=batch_size,
                                             distribute_model=not args.use_cpu)
        sample_generated_count += batch_size

        for i in range(samples.shape[0]):
            sample_index = _save_generated_image(
                image_analyzer, sample_index, samples[i, -1, 0, :, :], context[i], args)

            if sample_index > bin_end:
                logger.error('Sample index %i has gone outside allowed value %i', sample_index, bin_end)
            elif sample_index == bin_end:
                logger.info('Sample index %i has reached bin end %i - generated sample count %i/%i',
                            sample_index, bin_end, sample_generated_count, n_samples_to_generate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help=f"Which config to use for image generation, as defined in {paths.PROGRAM_CONFIG.name}",
                        type=str)
    cli_args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(paths.PROGRAM_CONFIG)
    args = SampleArgs.from_config(config, cli_args.config)

    sample(args)
