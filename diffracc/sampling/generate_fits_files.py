"""
This file was created by Ashley and Luna. It provides a complete application that can be used to sample images according
to the LOFAR model with certain parameters, and it provides a function that can be used to access that application
through other files. This application can be distributed across multiple nodes.
"""

import argparse
import configparser
import dataclasses
import json
import math
from pathlib import Path, PurePath
from typing import Callable

import h5py
import numpy as np
import scipy.stats
import torch
from sklearn.preprocessing import PowerTransformer

from ..analysis.image_analyzer import ImageAnalyzer, RecursiveFileAnalyzer
from ..model import model_utils, sampler
from ..utils import device_utils, paths
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
    # Path to the training h5, used to fit the LAS standardisation transform the model was trained with. Required
    # when las_conditioning_enabled is set.
    train_data_path: str | None = None

    _INT_FIELDS = ('batch_size', 'n_samples', 'folder_size', 'timesteps')
    _FLOAT_FIELDS = ('upper_bound', 'lower_bound')
    _BOOL_FIELDS = ('use_cpu', 'preserve_values', 'las_conditioning_enabled')
    _OPTIONAL_STR_FIELDS = ('train_data_path',)
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

        required = {f.name for f in dataclasses.fields(cls) if f.default is dataclasses.MISSING}
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"Missing required config keys in section {section!r}: {sorted(missing)}")

        for field in cls._INT_FIELDS:
            raw[field] = int(raw[field])
        for field in cls._FLOAT_FIELDS:
            raw[field] = float(raw[field])
        for field in cls._BOOL_FIELDS:
            raw[field] = raw[field] == 'True'
        for field in cls._OPTIONAL_STR_FIELDS:
            if raw.get(field) == 'None':
                raw[field] = None

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


def _get_las_transformer(train_data_path: str) -> PowerTransformer:
    """
    Fit the LAS standardisation transform the model was trained with, so physical LAS prompts (arcsec) can be mapped
    into the standardised (~N(0,1)) space the model expects as context.

    Mirrors ``TrainDatasetNoScale.transform_las_vals``: fit on the training set's ``cat_info['LAS']`` column with
    Box-Cox (Yeo-Johnson fallback for non-positive values). Fitting on the same values training used makes the fit
    deterministic, so the resulting transform is identical to the training-time one.

    Parameters
    ----------
    train_data_path : str
        Path to the training h5 file containing ``cat_info['LAS']``

    Returns
    -------
    PowerTransformer
        The fitted transformer; use ``.transform`` to map physical LAS values into the model's context space
    """
    # Logged either side of the open: on NFS without HDF5_USE_FILE_LOCKING=FALSE this call blocks on the file lock
    # indefinitely, and without these lines the run just stops silently after the PeakFluxPowerTransformer fit.
    logger = get_logger(__name__)
    logger.info('Fitting LAS standardisation transform from %s ...', train_data_path)
    with h5py.File(train_data_path, "r") as f:
        # float32 first to match the values training saw (set_las_values), then float64 so sklearn fits at the
        # same precision it uses for training's torch-tensor input - this makes the fitted lambdas identical
        las_values = np.ascontiguousarray(f["cat_info"][:]["LAS"], dtype=np.float32).astype(np.float64)
    method = "yeo-johnson" if (las_values <= 0).any() else "box-cox"
    pt = PowerTransformer(method=method)
    pt.fit(las_values.reshape(-1, 1))
    logger.info('Fitted LAS standardisation transform on %i values (method=%s)', las_values.size, method)
    return pt


def _get_model_flux_transform(model_name: str) -> dict | None:
    """
    Read the global flux transform recorded in a trained model's saved config, if any. The trainer records the
    transform that was applied to the training pixels, and the Sampler inverts it to map generated images back to
    physical Jy/beam.

    Parameters
    ----------
    model_name : str
        The model name, as stored in model_results/<NAME>/config_<NAME>.json

    Returns
    -------
    dict | None
        The flux transform parameter dict, or None if the config records no transform (raw-pixel models)
    """
    logger = get_logger(__name__)
    config_file = paths.MODEL_PARENT / model_name / f"config_{model_name}.json"
    if not config_file.exists():
        logger.warning('No saved model config found at %s; assuming no flux transform to invert.', config_file)
        return None
    with open(config_file, "r", encoding="utf-8") as f:
        flux_transform = json.load(f).get("flux_transform")
    if flux_transform is None:
        logger.warning("No flux transform recorded in %s; samples will be saved in the model's raw output space.",
                       config_file)
    return flux_transform


def _check_context_matches_model(model_config, args: SampleArgs) -> None:
    """
    Check that the number of context values this run will prompt with matches the number the model was trained on.

    The saved config records the context columns the model was trained with, e.g. ``["max_values_tr"]`` for a peak-flux
    only model and ``["max_values_tr", "las_values_tr"]`` when LAS conditioning was used. ``sample()`` builds a context
    of width 1 or 2 from ``las_conditioning_enabled``, so a config.ini section whose ``LAS_CONDITIONING_ENABLED``
    disagrees with the model prompts it with the wrong number of conditioning values. Failing here turns that into an
    immediate, actionable error rather than a shape mismatch deep inside the UNet.

    Parameters
    ----------
    model_config : ModelConfig
        The trained model's saved config
    args : SampleArgs
        The arguments controlling sampling

    Raises
    ------
    ValueError
        If the model's context width and args.las_conditioning_enabled disagree
    """
    logger = get_logger(__name__)
    context = getattr(model_config, "context", None)
    if not context:
        logger.warning("Model %r records no context columns; skipping the context-width check.", args.model_name)
        return

    expected = 2 if args.las_conditioning_enabled else 1
    if len(context) != expected:
        raise ValueError(
            f"Model {args.model_name!r} was trained with {len(context)} context value(s) {list(context)}, but this "
            f"config prompts with {expected} (las_conditioning_enabled={args.las_conditioning_enabled}). Set "
            f"LAS_CONDITIONING_ENABLED={'True' if len(context) == 2 else 'False'} in this config.ini section "
            f"(and TRAIN_DATA_PATH too, if enabling it)."
        )


def _load_sampling_model(args: SampleArgs, model_sampler: sampler.Sampler) -> torch.nn.Module:
    """
    Load the trained model once and place it on its device, so the sampling loop can reuse a single resident model.

    ``Sampler.quick_sample`` loads the model itself when passed ``model=None``, and moves it back to the CPU afterwards
    when it did the placement. Calling it per batch therefore repeats load_model -> torch.load -> ``.to(cuda)`` on every
    iteration - dozens of redundant reads of the parameter file for a single bin. This function loads the model once,
    and passes it to ``quick_sample`` with ``distribute_model=False`` so it doesn't move it back to the CPU after each
    batch.

    Parameters
    ----------
    args : SampleArgs
        The arguments controlling sampling; ``model_name`` selects the model and ``use_cpu`` suppresses GPU placement
    model_sampler : sampler.Sampler
        The sampler whose device settings the placement should match

    Returns
    -------
    torch.nn.Module
        The loaded model, placed and in eval mode, ready to be passed to ``quick_sample(model=...)``
    """
    model, model_config = model_utils.load_model(args.model_name, return_config=True)
    _check_context_matches_model(model_config, args)
    if not args.use_cpu:
        model, _ = device_utils.distribute_model(model, model_sampler.settings["n_devices"])
    return model.eval()


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
                          header_row: np.ndarray,
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
    header_row : np.ndarray
        The prompt values for the image, used to set FITS headers: FXSCLD is the peak flux in the transformed
        (power-transform) space, LASIZE the physical LAS in arcsec
    args : SampleArgs
        The arguments controlling sampling, used to determine the subdirectory and bin size for saving

    Returns
    -------
    int
        The index the image was actually saved at (may be greater than sample_index, see the exists() loop below)
    """
    if not args.preserve_values:
        image = _normalise_image(image)

    extra_headers = {'FXSCLD': header_row[0]}
    if args.las_conditioning_enabled:
        extra_headers['LASIZE'] = header_row[1]

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
    #The flux transform recorded at training time (if any) is inverted by the Sampler, so images land in Jy/beam
    model_sampler = sampler.Sampler(n_samples=args.batch_size, timesteps=args.timesteps,
                                    flux_transform=_get_model_flux_transform(args.model_name))

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

    # LAS prompts must be standardised with the same power transform used at training time - the model was trained on
    # las_values_tr ~ N(0,1), so feeding raw arcsec (~6-120) would be far outside the trained context distribution
    if args.las_conditioning_enabled:
        assert args.train_data_path is not None, \
            "las_conditioning_enabled requires train_data_path, to fit the LAS standardisation transform"
        las_transformer = _get_las_transformer(args.train_data_path)

    # Loaded once, after the nothing-to-do early return above, and reused for every batch below
    model = _load_sampling_model(args, model_sampler)

    # Generate/Sample the samples
    sample_generated_count = 0
    sample_index = bin_start
    image_analyzer = ImageAnalyzer(args.generated_subdir)
    while sample_generated_count < n_samples_to_generate:
        # to not double-generate at the borders
        batch_size = min(args.batch_size, n_samples_to_generate - sample_generated_count)
        context = fpeak_model_dist(batch_size)[:, np.newaxis]
        # header_context keeps the values stored in the FITS headers: FXSCLD stays in the transformed peak-flux
        # space (downstream code inverts it), LASIZE stays in physical arcsec
        header_context = context
        # if las conditioning is enabled, add it to context (standardised for the model, physical for the header)
        if args.las_conditioning_enabled:
            las_physical = scipy.stats.uniform.rvs(
                _LAS_LOWER_BOUND, _LAS_UPPER_BOUND - _LAS_LOWER_BOUND, size=batch_size)
            las_standardised = las_transformer.transform(las_physical.reshape(-1, 1))
            context = np.concatenate((context, las_standardised), axis=1)
            header_context = np.concatenate((header_context, las_physical[:, np.newaxis]), axis=1)

        # distribute_model=False because the model is already loaded and placed - it also stops quick_sample
        # moving the model back to the CPU after each batch
        samples = model_sampler.quick_sample(f"{args.model_name}",
                                             model=model,
                                             context=torch.from_numpy(context),
                                             n_samples=batch_size,
                                             distribute_model=False)
        sample_generated_count += batch_size

        for i in range(samples.shape[0]):
            sample_index = _save_generated_image(
                image_analyzer, sample_index, samples[i, -1, 0, :, :], header_context[i], args)

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
