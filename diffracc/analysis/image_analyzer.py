"""
This module contains the ImageAnalyzer class, which is a general-purpose class to turn data into, and run PyBDSF
analysis on, FITS images. This class supports both single-node multiprocessing and multi-node distributed processing
through batching the images, and supports the use of a TOML configuration file to set PyBDSF parameters.
"""
import multiprocessing.pool
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping

# PyBDSF is parallelised across images by a process Pool (see ImageAnalyzer.analyze_all_fits_in_input), so each worker
# should run numpy/scipy/BLAS single-threaded: otherwise every one of the N pool workers spawns its own BLAS thread pool
# and a node ends up with N x cores threads contending for N cores. These must be set before numpy/scipy/bdsf are
# imported, as BLAS reads them once at load time; setdefault so an explicit environment value (e.g. from SLURM) wins.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import bdsf
import bdsf.image
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # for Python <3.11
    import tomli as tomllib
from astropy.io import fits

from ..utils import paths
from ..utils.distributed import DistributedUtils
from ..utils.logger import LoggingLevels, get_logger
from ..utils.paths import cast_to_path
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer


#Neccesary pool extention - PyBDSF uses daemon processes but only sometimes, and we want to batch the files themselves
# Courtesy of
# https://stackoverflow.com/questions/52948447/error-group-argument-must-be-none-for-now-in-multiprocessing-pool
class NonDaemonPool(multiprocessing.pool.Pool):
    """
    A multiprocessing pool that uses non-daemon processes, allowing for nested parallelism.
    """
    def Process(self, *args, **kwds):
        """
        Create a non-daemon process for the pool.

        Returns
        -------
        multiprocessing.Process
            A non-daemon process that can be used in the pool.
        """
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""
            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc



@dataclass(frozen=True)
class ProcessArgs:
    """
    A dataclass to hold the arguments for the process_image function in PyBDSF.
    """
    # Default config
    beam: tuple[float, float, float] = (0.00166667, 0.00166667, 0.0)
    frequency: float = 144e6
    mean_map: str = "zero"
    rms_map: bool = True
    rms_box: tuple[int, int] = (60, 15)
    thresh: str = "hard"
    thresh_isl: float = 4.0
    thresh_pix: float = 5.0

    # Adaptive box config
    adaptive_rms_box: bool = True
    adaptive_thresh: int = 150
    rms_box_bright: tuple[int, int] = (60, 15)

    # Advanced config
    advanced_opts: bool = True
    group_by_isl: bool = False
    group_tol: int = 10
    ini_method: str = "intensity"

    # A trous config
    atrous_do: bool = True
    atrous_jmax: int = 4

    # Parallelism config. `ncores` is how many cores PyBDSF uses within a *single* image (a-trous, fitting). We instead
    # parallelise across images with a process Pool (see ImageAnalyzer.analyze_all_fits_in_input), so each image runs
    # single-threaded to avoid nesting cores-within-workers oversubscription. Left out of the TOML deliberately (it is a
    # runtime, not a science, choice); override for a specific run with process_ncores=N.
    ncores: int = 1


    @classmethod
    def from_toml(cls, toml_path: str | Path) -> "ProcessArgs":
        """
        Create a ProcessArgs object from a TOML file.

        Parameters
        ----------
        toml_path : str | Path
            The path to the TOML file.

        Returns
        -------
        ProcessArgs
            A ProcessArgs object with the parameters from the TOML file.
        """
        toml_path = cast_to_path(toml_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"TOML file {toml_path} does not exist")

        with open(toml_path, 'rb') as f:
            config = tomllib.load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None = None) -> "ProcessArgs":
        """
        Create a ProcessArgs object from a dictionary.

        Parameters
        ----------
        raw : Mapping[str, Any] | None, optional
            The dictionary representation of the configuration, by default None

        Returns
        -------
        ProcessArgs
            The initialized ProcessArgs instance

        Raises
        ------
        ValueError
            If there are unknown sections in the provided dictionary that do not correspond to any of the expected
            configuration sections.
        """
        raw = raw or {}

        valid_sections = {"default", "adaptivebox", "advanced", "atrous"}
        unknown_sections = set(raw) - valid_sections

        if unknown_sections:
            raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

        return cls(
            **raw.get("default", {}),
            **raw.get("adaptivebox", {}),
            **raw.get("advanced", {}),
            **raw.get("atrous", {})
        )


    def to_dict(self) -> dict:
        """
        Convert the ProcessArgs object to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the ProcessArgs object.
        """
        return asdict(self)


class ImageAnalyzer:
    """
    A class to analyze images of radio galaxies using PyBDSF, with LOFAR defaults
    """
    def __init__(self,
                 subdir: str | PurePath,
                 fits_input_dir: str | Path = paths.FITS_PARENT,
                 log_dir: str | Path = paths.PYBDSF_LOG_PARENT,
                 catalog_dir: str | Path = paths.PYBDSF_CATALOG_PARENT,
                 img_dir: str | Path = paths.PYBDSF_EXPORT_IMAGE_PARENT,
                 write_catalog: bool = True,
                 export_images: list[str] | None = None,
                 log_level: int = LoggingLevels.INFO.value,
                 **kwargs: dict):
        """
        Initialises an ImageAnalyzer object, which can be used to analyze images of radio galaxies using PyBDSF.

        Parameters
        ----------
        subdir : str
            The subdirectory appended to all root directories, to separate different use cases,
            i.e. files are read from "[fits_input_dir]/[subdir]/\*\*.fits",
            catalogs are written to "[catalog_dir]/[subdir]/\*\*.fits",
            and images are written to "[img_dir]/[subdir]/[img_type]/\*\*.fits",
            where "\*\*" implies the files are recursively searched for, allowing the data to be segmented into folder
            bins
        fits_input_dir : str | Path = utils.paths.FITS_PARENT
            Root directory of all fits input files. Images are taken from "[fits_input_dir]/[subdir]/\*\*.fits",
            where the postfix after subdir is used as the postfix for img_dir and catalog_dir
        catalog_dir : str | Path = utils.paths.PYBDSF_ANALYSIS_PARENT
            Root directory of all catalogs. Catalogs are written to "[catalog_dir]/[subdir]/\*\*.fits"
        img_dir : str | Path = utils.paths.EXPORT_IMAGE_PARENT
            Root directory of all images exported by PyBDSF. Images are written to
            "[img_dir]/[subdir]/[img_type]/\*\*.fits"
        write_catalog : bool = True
            Whether or not to write a catalog. If true, arguments can be passed to bdsf.write_catalog
            by prefixing them with 'catalog_' in kwargs. For example, the default args are "catalog_type='srl', 
            catalog_clobber=True" which is the equivalent of passing "catalog_type='srl', clobber=True" to 
            bdsf.write_catalog. Of note, the pybdsf argument 'catalog_type' is shortened here to simply 'type' 
            for the avoidance of writing the argument 'catalog_catalog_type'. If 'catalog_catalog_type' is passed 
            as an argument it will be ignored. If false, catalog_ args are ignored. Outfile cannot be specified - see
            catalog_dir or subdir for output directory structure.
        export_images : list[str]
            Types of pybdsf images to export. See pybdsf documentation for the types of images that can be exported.
            Arguments can be passed to bdsf.export_image by prefixing them with the image type in export_images followed
            by an underscore. For example, to write pybdsf's 'gaus_model' image, pass 'gaus_model' to export_images as
            a free string parameter, then to overwrite if the file exists with pybdsf's 'clobber' parameter, pass
            "gaus_model_clobber=True" in to kwargs. Outfile cannot be specified - see subdir for output directory
            structure
        **kwargs : dict
            All arguments to pass to bdsf.export_catalog, bdsf.export_image, and bdsf.process_image. The method for
            formatting arguments to export_catalog and export_image are explained above in write_catalog and
            *export_images. To pass arguments to bdsf.process_image, pass the arguments with the prefix 'process_'. For
            example, to process an image with the defaults used in this project, pass
            "process_beam = (0.00166667, 0.00166667, 0.0), process_thresh_isl = 5, process_thresh_pix = 0.5,
            process_mean_map = 'const', process_rms_map = True, process_thresh = 'hard', process_frequency = 144e6".
            For PyBDSF, beam and frequency must be present.
        """
        self.logger = get_logger("ImageAnalyzer", log_level)

        #Ensure all types are paths
        self.log_dir = cast_to_path(log_dir)
        self.catalog_dir = cast_to_path(catalog_dir)
        self.img_dir = cast_to_path(img_dir)
        self.fits_input_dir = cast_to_path(fits_input_dir)

        # Ensure subdir is a PurePath, so it can join paths without worrying about OS-specific path separators
        self.subdir = subdir if isinstance(subdir, PurePath) else PurePath(subdir)

        self.write_catalog = write_catalog
        export_images = export_images if export_images is not None else []
        self.export_images = export_images

        self.rfa = RecursiveFileAnalyzer(self.fits_input_dir / self.subdir, log_level)

        self.process_args = {}
        self.catalog_args = {}
        self.export_img_args = {} #elements will be (str, img args dict)
        for img_type in export_images:
            self.export_img_args[img_type] = {"img_type": img_type}

        # Loop through kwargs and sort arguments into catalog, export_img, or process
        # Catalog args are prefixed with 'catalog_' and export_img args are prefixed with '[img_type]_'
        for key, val in kwargs.items():
            arg_used = False

            #  Handle catalog args first
            if key.find('catalog_') > -1:
                if write_catalog:
                    new_key = key[len('catalog_'):]

                    #expand type to catalog_type and skip if new_key is already catalog_type
                    if new_key == 'type':
                        new_key = 'catalog_type'
                    elif new_key == 'catalog_type':
                        self.logger.info('Skipping argument catalog_catalog_type: please refer to write_catalog in'
                                         ' ImageAnalyzer docstring')
                        continue

                    self.catalog_args[new_key] = val
                else:
                    self.logger.warning('WARNING - argument %s passed with catalog prefix but write_catalog is false',
                                        key)
                arg_used = True

            # Handle export_img args next
            for img_type in export_images:
                if key.find(f'{img_type}_') > -1:
                    self.export_img_args[img_type][key[len(f'{img_type}_'):]] = val
                    arg_used = True

            # Handle process args last
            if key.find('process_') > -1:
                self.process_args[key[len('process_'):]] = val
                arg_used = True

            if not arg_used:
                self.logger.warning('WARNING - argument %s passed but not used (are you passing all neccesary strings'
                                    ' for export_images?)', key)

        #Clobber by default if nothing passed
        if write_catalog:
            self.catalog_args['clobber'] = self.catalog_args.get('clobber', True)
        for img_type in export_images:
            self.export_img_args[img_type]['clobber'] = self.export_img_args[img_type].get('clobber', True)

        #Set process arg defaults to project defaults if nothing passed
        process_args_defaults = ProcessArgs.from_toml(paths.PYBDSF_CONFIG).to_dict()
        for key, val in process_args_defaults.items():
            self.process_args[key] = self.process_args.get(key, val)


    def get_postfix(self, path: Path) -> PurePath:
        """
        Simple function to get the 'postfix' of a path relative to subdir.
        
        This function looks for the last occurrence of the last element of subdir.parts in path.parts, and returns all
        of path.parts after its index, giving the relative path to the path from subdir, referred to here as the
        'postfix'.

        Parameters
        ----------
        path : Path
            The path to get the postfix of.

        Returns
        -------
        PurePath
            The parts for the postfix as a `PurePath`.

        Raises
        ------
        ValueError
            If `self.subdir.parts[-1]` is not present in `path.parts`.
        """
        last_index_of_subdir = len(path.parts) - 1 - path.parts[::-1].index(self.subdir.parts[-1])
        return PurePath(*path.parts[(last_index_of_subdir + 1):])


    def analyze_all_fits_in_input(self, n_cpus: int | None = None, max_tasks_per_child: int | None = 50):
        """
        Recursively analyze all of "[fits_input_dir]/[subdir]/**.fits" with PyBDSF.

        Spawns `n_cpus` worker processes. If `n_cpus` is None it falls back to the N_CPUS environment variable and if
        that is unset to `os.cpu_count()` so a bare local run uses the whole machine. When distributed across an array
        of tasks, the sorted input file list is split across tasks by interleaving (this task takes
        `files[task_id :: task_count]`), so files still needing work are shared evenly across tasks rather than
        concentrated in a few contiguous ranges. `analyze_fits_at_path` skips any file whose PyBDSF outputs already exist.

        Parameters
        ----------
        n_cpus : int | None = None
            Number of worker processes. None resolves to N_CPUS if set, else os.cpu_count().
        max_tasks_per_child : int | None = 50
            Recycle each worker after this many files to bound PyBDSF's per-image memory growth. Recycling is cheap
            under the `fork` start method (Linux/WSL: forked workers inherit the already-imported bdsf) but costly under
            `spawn` (native Windows/macOS: each replacement re-imports bdsf from scratch), so pass None to disable
            recycling on a spawn platform where memory growth is not a concern (e.g. small local runs).
        """
        du = DistributedUtils()

        if n_cpus is None:
            env_n_cpus = os.environ.get("N_CPUS")
            n_cpus = int(env_n_cpus) if env_n_cpus else (os.cpu_count() or 1)
        self.logger.info("Using %i cpu" + ("s" if n_cpus != 1 else ""), n_cpus)
        input_subdir = self.fits_input_dir / self.subdir

        # Sort first so every task partitions an identical, stable order (scandir order is not guaranteed consistent
        # across tasks), then interleave: each file's owning task is a fixed function of its rank
        files = sorted(self.rfa.get_unwrapped_list(path=input_subdir, pattern=r'.*?\.fits$').paths)
        files = files[du.get_task_id() :: du.get_task_count()]

        # One process per CPU, each running PyBDSF single-threaded (ProcessArgs.ncores), so the pool is the sole source
        # of parallelism.
        # imap_unordered with chunksize=1 hands out files one at a time, keeping workers load-balanced despite the wide
        # spread in per-image PyBDSF runtime; maxtasksperchild recycles workers so PyBDSF's per-image memory growth
        # can't accumulate across a long task.
        with NonDaemonPool(processes=n_cpus, maxtasksperchild=max_tasks_per_child) as p:
            for _ in p.imap_unordered(self.analyze_fits_at_path, files, chunksize=1):
                pass


    def analyze_fits_at_path(self, path: Path | str):
        """
        Function to analyze a single fits file at a given path

        Parameters
        ----------
        path : Path | str
            the path to the file to analyze
        """
        if not isinstance(path, Path):
            path = Path(path)

        assert path.exists(), f"Path {str(path)} does not exist"

        if path.is_dir():
            self.logger.error('ERROR - Cannot analyze %s as fits file, is directory', str(path))
            raise ValueError(f'Cannot analyze {str(path)} as fits file, is directory')

        if path.suffix != ".fits":
            self.logger.error('ERROR - Cannot analyze %s as fits file, is not fits file', str(path))
            raise ValueError(f'Cannot analyze {str(path)} as fits file, is not fits file')

        # First see if we have any work to do - check for a flag file instead of actual output because sometimes PyBDSF
        # doesn't write output, e.g. when running analysis on data that has no findable sources.
        postfix = self.get_postfix(path)
        flag_postfix = PurePath(*(postfix.parts[:-1] + (postfix.parts[-1] + '.flag',)))

        log_file = self.log_dir / self.subdir / f"{postfix}.pybdsf.log"

        write_catalog = self.write_catalog
        if write_catalog:
            catalog_outfile_flag = self.catalog_dir / self.subdir / flag_postfix
            if catalog_outfile_flag.exists():
                write_catalog = False # Don't write catalog if it already exists

        export_images: list[str] = []
        for img_type in self.export_images:
            image_outfile_flag = self.img_dir / self.subdir / img_type / flag_postfix
            if not image_outfile_flag.exists():
                export_images.append(img_type)

        if (not write_catalog) and (len(export_images) == 0) and (log_file.exists()):
            self.logger.info(f"Skipping {path}, no work to do")
            return #nothing to do
        self.logger.info(f"Processing {path}:")

        #Something to do, process the image
        failed_to_process = False
        # Pre-create the PyBDSF output (log) directory ourselves. PyBDSF's set_up_output_paths does a racy check with no
        # exist_ok. Creating it here with exist_ok=True is a solution, and once it exists PyBDSF skips its own makedirs.
        outdir = self.log_dir / self.subdir / postfix.parent
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            img: bdsf.image.Image = bdsf.process_image(
                str(path),
                outdir=outdir,
                **self.process_args
           )
        except ValueError:
            self.logger.error(f'Image {str(path)} failed to process')
            failed_to_process = True
        except RuntimeError:
            self.logger.error(f'Image {str(path)} had unphysical RMS, couldn\'t process')
            failed_to_process = True

        # Write the output files
        for img_type in export_images:
            image_outfile = self.img_dir / self.subdir / img_type / postfix
            image_outfile_flag = self.img_dir / self.subdir / img_type / flag_postfix
            image_outfile.parent.mkdir(parents=True, exist_ok=True)
            if not failed_to_process:
                img.export_image(outfile=str(image_outfile), **self.export_img_args[img_type])
            image_outfile_flag.touch()

        # Write the catalog if requested
        if write_catalog:
            catalog_outfile = self.catalog_dir / self.subdir / postfix
            catalog_outfile_flag = self.catalog_dir / self.subdir / flag_postfix
            catalog_outfile.parent.mkdir(parents=True, exist_ok=True)
            if not failed_to_process:
                img.write_catalog(outfile=str(catalog_outfile), **self.catalog_args)
            catalog_outfile_flag.touch()

    # todo: make this static or extract it into a different program
    def save_image_to_fits(self,
                           image: np.ndarray,
                           postfix: PurePath,
                           overwrite: bool = True,
                           **kwargs):
        """
        Save a numpy 2d array to a fits file under "[fits_input_dir]/[subdir]/"

        Parameters
        ----------
        image : np.ndarray (2D)
            the pixel values that represent the image (should be 80x80)

        postfix : str
            postfix for the fits file. Can either be the name of the fits file (e.g. "example.fits") or the name
            and location under "[fits_input_dir]/[subdir]/" to store it in (e.g. "example_bin/example.fits")

        overwrite : bool = True
            Whether or not to overwrite the file if it already exists

        **kwargs
            Various information to include in the FITS header. The dictionary will be added to hdu.header such that the
            names in the header are the arguments passed. This is usually to pass model context parameters, like
            FXSCLD=fscaled
        """
        # HACK - add some random noise to the image so pybdsf doesn't fail
        #z = np.random.normal(0, scale=1e-5, size=image.shape)
        #image += z
        hdu = fits.PrimaryHDU(image)
        hdu.header["CTYPE1"] = "RA---SIN"
        hdu.header["CTYPE2"] = "DEC--SIN"
        hdu.header["CDELT1"] = 1.5 * 0.00027778
        hdu.header["CDELT2"] = 1.5 * 0.00027778
        hdu.header["CUNIT1"] = "deg"
        hdu.header["CUNIT2"] = "deg"
        for key, value in kwargs.items():
            hdu.header[key] = value
        hdul = fits.HDUList([hdu])
        outfile = self.fits_input_dir / self.subdir / postfix
        outfile.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(str(outfile), overwrite=overwrite)


    def analyze_image(self, image: np.ndarray, fscaled: float, postfix: str):
        """
        Save a numpy 2d array to a fits file under "[fits_input_dir]/[subdir]/" and analyze it, storing
        the output in "[catalog_dir]/[subdir]/[postfix]" or "[img_dir]/[subdir]/[img_type]/[postfix]" depending
        on ImageAnalyzer parameters

        Parameters
        ----------
        image : np.ndarray (2D)
            the pixel values that represent the image (should be 80x80)
        
        fscaled : float
            box-cox transformed peak flux of the image

        postfix : str
            postfix for the fits file. Can either be the name of the fits file (e.g. "example.fits") or the name
            and location under "[fits_input_dir]/[subdir]/" to store it in (e.g. "example_bin/example.fits")
        """
        self.save_image_to_fits(image, postfix, FXSCLD=fscaled)
        self.analyze_fits_at_path(self.fits_input_dir / self.subdir / postfix)
