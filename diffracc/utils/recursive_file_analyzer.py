"""
This is a file created by Ashley and Luna. It defines the RecursiveFileAnalyzer and the HistogramErrorDrawer. The main
purpose of the RecursiveFileAnalyzer is the method to get an unwrapped list of all the files in its directory
recursively, as this is useful for multiprocessing or other scenareos where knowing the total number of files is
helpful. The HistogramErrorDrawer is a utility class to house its Draw function, which draws a histogram and calculates
its errors with astropy.stats.poisson_conf_interval
"""
import os
import re
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Generator, Generic, Iterator, Literal, NamedTuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from tqdm import tqdm

from .logger import LoggingLevels, get_logger

_module_logger = get_logger("RecursiveFileAnalyzer", LoggingLevels.DEBUG.value)




def _pad_to_shape(array: npt.NDArray, target_shape: tuple[int, ...]) -> npt.NDArray:
    """
    Pads a numpy array with zeros to match a target shape.

    Parameters
    ----------
    array : npt.NDArray
        The input array to be padded.
    target_shape : tuple[int, ...]
        The desired shape of the output array.

    Returns
    -------
    npt.NDArray
        The padded array with the specified target shape.
    """
    pad_width = [(0, max(0, ts - s)) for s, ts in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=0)


# Utility functions for for_each
def get_fits_primaryhdu_data( path: Path, expected_shape: tuple[int, ...] | None = None ) -> fits.FITS_rec:
    """
    A function to get the primary HDU data from a FITS file.

    Parameters
    ----------
    path : Path
        The path to the FITS file
    expected_shape : tuple[int, ...] | None, optional
        The shape the data is expected to have once leading size-1 dimensions are stripped, by default None. Some
        FITS images (e.g. PyBDSF outputs for the dr2 cutouts) are occasionally corrupt/inhomogeneous; when
        expected_shape is given, data not matching it is replaced with a zero-filled array of that shape instead of
        being returned as-is. This keeps shape normalisation next to the read itself, so callers collecting many of
        these via RecursiveFileAnalyzer.run_pipeline get back a directly stackable array without a separate fix-up
        pass. If None, no check is performed.

    Returns
    -------
    fits.FITS_rec
        The primary HDU data from the FITS file
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    # Get rid of leading 1s in shape, e.g. (1,1,n,n) -> (n,n), but preserve 2 dimensions for single pixel images
    while len(data.shape) > 2 and data.shape[0] == 1:
        data = data[0]
    if expected_shape is not None and data.shape != expected_shape:
        _module_logger.warning("%s has shape %s instead of expected %s, substituting a zero-filled array", path,
                               data.shape, expected_shape)
        data = _pad_to_shape(data, expected_shape)
    return data


def get_fits_primaryhdu_header( path: Path, key: str | None = None ) -> fits.Header | str:
    """
    A function to get the primary HDU header from a FITS file, or a specific key from the header.

    Parameters
    ----------
    path : Path
        The path to the FITS file
    key : str | None, optional
        The key of the header value to retrieve, by default None

    Returns
    -------
    fits.Header | str
        The primary HDU header from the FITS file, or a specific key from the header
    """
    with fits.open(path, memmap=False) as hdul:
        if key is not None:
            header = hdul[0].header[key]
        else:
            header = hdul[0].header
    return header



# Constrained to exactly these two shapes: numbers is a NumberArray when return_nums=True, else None. Letting
# ScanResult/PipelineResult be generic over this lets the @overload signatures below narrow `.numbers` to a
# non-Optional NumberArray whenever return_nums=True is passed as a literal, instead of every caller needing to
# handle a spurious `NumberArray | None`.
NumberArray = npt.NDArray[np.int_]
NumbersT = TypeVar("NumbersT", NumberArray, None)

# numpy has no dtype for arbitrary Python objects, so a "1D array of Any" that may hold ragged/heterogeneous
# per-file results is, in practice, an object-dtype array when it can't be stacked into a proper dtype.
ResultArray = npt.NDArray[Any]


def _to_array(items: Sequence[Any]) -> ResultArray:
    """
    Builds a numpy array from a sequence of per-file results. When the results are homogeneous (e.g. scalars, or
    array-likes of identical shape - the latter expected to already be normalised by the reading function itself,
    e.g. get_fits_primaryhdu_data's expected_shape), this produces a properly stacked, properly-dtyped array
    directly usable by callers. When they are not homogeneous (e.g. a ragged per-file result), falls back to a 1D
    object-dtype array of the raw items instead of raising.

    Parameters
    ----------
    items : Sequence[Any]
        The items to place into the array, in order.

    Returns
    -------
    ResultArray
        A numpy array containing `items`.
    """
    # Classic numpy as array
    try:
        return np.array(items)
    # Fall back if there's inhomogenity
    except ValueError:
        array = np.empty(len(items), dtype=object)
        array[:] = items
        return array


class ScanResult(NamedTuple, Generic[NumbersT]):
    """
    The result of a directory scan.

    Attributes
    ----------
    paths : list[Path]
        The matched file paths.
    numbers : NumberArray | None
        The numbers extracted from each file name via the pattern's capture group, in the same order as `paths`,
        or None if numbers were not requested (return_nums=False).
    """
    paths: list[Path]
    numbers: NumbersT


class PipelineResult(NamedTuple, Generic[NumbersT]):
    """
    The result of a processing pipeline run.

    Attributes
    ----------
    results : ResultArray
        The per-file (or per-batch, flattened) results of applying the pipeline function, as a numpy array.
    numbers : NumberArray | None
        The numbers extracted from each file name via the pattern's capture group, in the same order as `results`,
        or None if numbers were not requested (return_nums=False).
    """
    results: ResultArray
    numbers: NumbersT



class RecursiveFileAnalyzer:
    """
    A class to recursively analyse files in a given directory. It provides methods to get an unwrapped list of all files
    in the directory, optionally matching a regex pattern and filtering by a numeric range extracted from the file
    names. It also provides methods to process files in parallel using either file mode (one file per task) or batch
    mode (one batch per task), with options for progress display and output to a file.
    """
    def __init__(self, path: Path | str, log_level: int = LoggingLevels.INFO.value):
        """
        Initialises the RecursiveFileAnalyzer with a given path and log level.

        Parameters
        ----------
        path: Path | str
            The root directory to recursively search under
        log_level: int = LoggingLevels.INFO.value
            The log level for the recursive file analyzer logger. When set to DEBUG, will log a message to the console
            when a directory is entered or a file read, with an index associated with each read file. Useful for slow
            operations to provide feedback on progress. Default LoggingLevels.INFO.value
        """
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        self.logger = get_logger("RecursiveFileAnalyzer", LoggingLevels.DEBUG.value)
        self.logger.setLevel(log_level)


    @overload
    def get_unwrapped_list(self,
                           path: Path | str | None = None,
                           pattern: str | None = None,
                           numeric_range: tuple[int, int] | None = None,
                           return_nums: Literal[False] = False) -> ScanResult[None]: ...
    @overload
    def get_unwrapped_list(self,
                           path: Path | str | None = None,
                           pattern: str | None = None,
                           numeric_range: tuple[int, int] | None = None,
                           *,
                           return_nums: Literal[True]) -> ScanResult[NumberArray]: ...
    @overload
    def get_unwrapped_list(self,
                           path: Path | str | None = None,
                           pattern: str | None = None,
                           numeric_range: tuple[int, int] | None = None,
                           *,
                           return_nums: bool) -> ScanResult[NumberArray] | ScanResult[None]: ...
    def get_unwrapped_list(self,
                           path: Path | str | None = None,
                           pattern: str | None = None,
                           numeric_range: tuple[int, int] | None = None,
                           return_nums: bool = False) -> ScanResult:
        """
        A method to recursively unwrap all files in a directory, with optional regex pattern matching and numeric range
        filtering on the pattern capture group.

        Parameters
        ----------
        path: Path | str | None = None
            The path to scan. If None, defaults to the root path of the RecursiveFileAnalyzer.
        pattern: str | None = None
            A regex pattern to filter files. If None, all files are yielded. If provided, only files whose names match
            the pattern are yielded.
        numeric_range: tuple[int, int] | None = None
            A range of numbers to filter files. If None, no filtering is applied.
        return_nums: bool = False
            Whether to also extract file numbers.

        Returns
        -------
        ScanResult
            The matched file paths, and their extracted numbers in `.numbers` if return_nums is True, else None.
        """
        if return_nums:
            file_paths, idxs = map(list, zip(*self._quick_scan(path=path,
                                                               pattern=pattern,
                                                               numeric_range=numeric_range,
                                                               return_nums=return_nums)))
            # sort paths and idxs by idxs
            idxs, file_paths = map(list, zip(*sorted(zip(idxs, file_paths))))
            return ScanResult(paths=file_paths, numbers=np.array(idxs))

        file_paths = list(self._quick_scan(path=path, pattern=pattern, numeric_range=numeric_range))
        return ScanResult(paths=file_paths, numbers=None)


    @overload
    def _quick_scan(self,
                    path: Path | str | None = None,
                    pattern: str | None = None,
                    numeric_range: tuple[int, int] | None = None,
                    return_nums: Literal[False] = False) -> Generator[Path, None, None]: ...
    @overload
    def _quick_scan(self,
                    path: Path | str | None = None,
                    pattern: str | None = None,
                    numeric_range: tuple[int, int] | None = None,
                    *,
                    return_nums: Literal[True]) -> Generator[tuple[Path, int], None, None]: ...
    @overload
    def _quick_scan(self,
                    path: Path | str | None = None,
                    pattern: str | None = None,
                    numeric_range: tuple[int, int] | None = None,
                    *,
                    return_nums: bool) -> Generator[Path | tuple[Path, int], None, None]: ...
    def _quick_scan( self,
                    path: Path | str | None = None,
                    pattern: str | None = None,
                    numeric_range: tuple[int, int] | None = None,
                    return_nums: bool = False) -> Generator[Path | tuple[Path, int], None, None]:
        """
        A method to recursively scan a directory and yield file paths, with optional regex pattern matching and numeric
        range filtering on the pattern capture group.

        Parameters
        ----------
        path : Path | str | None, optional
            The path to scan. If None, defaults to the root path of the RecursiveFileAnalyzer, by default None
        pattern : str | None, optional
            A regex pattern to filter files. If None, all files are yielded. If provided, only files whose names match
            the pattern are yielded, by default None
        numeric_range : tuple[int, int] | None, optional
            A range of numbers to filter files. If None, no filtering is applied, by default None
        return_nums : bool, optional
            Whether to return file numbers. If True, returns a tuple of (file_path, file_number) for each file. The
            file_number is extracted from the file name using the first capture group in the regex pattern. If False,
            only the file_path is returned, by default False

        Yields
        ------
        Generator[Path | tuple[Path, int], None, None]
            A generator of file paths, and optionally a tuple of (file_path, file_number) if return_nums is True

        Raises
        ------
        ValueError
            If return_nums is True and the regex pattern does not contain a capture group to extract numbers from the
            file names
        """
        assert not (return_nums and pattern is None), (
            "If return_nums is True, a regex pattern must be provided to extract the numbers")
        assert not (numeric_range and pattern is None), (
            "If numeric_range is provided, a regex pattern must be provided to extract the numbers")

        if path is None:
            path = self.path

        with os.scandir(path) as it:
            for entry in it:
                # Use follow_symlinks=False to avoid infinite loops from circular symlinks.
                if entry.is_dir(follow_symlinks=False):
                    yield from self._quick_scan(entry.path,
                                                pattern=pattern,
                                                numeric_range=numeric_range,
                                                return_nums=return_nums)
                elif pattern is None:
                    yield Path(entry)

                elif re.match(pattern, entry.name):
                    if return_nums:
                        # Extract file number using the first capture group in the regex pattern
                        try:
                            idx = int(re.search(pattern, entry.name).group(1))
                        except AttributeError as exc:
                            raise ValueError(
                                f"Pattern '{pattern}' does not match any characters in '{entry.name}'") from exc
                        except IndexError as exc:
                            raise ValueError(f"Pattern '{pattern}' does not contain a capture group to extract "
                                             f"numbers for file '{entry.name}'") from exc
                        if numeric_range is not None:
                            if idx < numeric_range[0] or idx >= numeric_range[1]:
                                continue
                        yield (Path(entry), idx)
                    else:
                        # save some processing power by only doing regex match if numeric_range is provided
                        if numeric_range is not None:
                            try:
                                idx = int(re.search(pattern, entry.name).group(1))
                            except IndexError as exc:
                                raise ValueError(f"Pattern '{pattern}' does not contain a capture group to extract "
                                                 f"numbers for file '{entry.name}'") from exc
                            if idx < numeric_range[0] or idx >= numeric_range[1]:
                                continue
                        yield Path(entry)


    def _batcher(self, iterable: Iterable, batch_size: int) -> Iterator[list]:
        """
        A generator function to yield batches of a specified size from an iterable.
        
        Parameters
        ----------
        iterable : list
            The iterable to batch.
        batch_size : int
            The size of each batch.

        Yields
        ------
        list
            A batch of items from the iterable.
        """
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


    def _process_file(self, path: str | Path, function: Callable) -> Any:
        """
        Processes a single filewith the given function, handling exceptions and logging warnings if any occur.
        
        Parameters
        ----------
        path : str | Path
            The path to the file to be processed.
        function : Callable
            The function (or partial) to apply to the file.
            
        Returns
        -------
        Any
            The result of applying the function to the file, or None if an exception occurred.
        """
        try:
            result = function(path)
            return result
        except Exception as e:
            self.logger.warning("Error processing %s: %s", path, e)
            return None


    def _process_batch(self, file_batch: Sequence[str | Path], function: Callable) -> list[Any]:
        """
        Processes a batch of files with the given function, handling exceptions and logging warnings if any occur.

        Parameters
        ----------
        file_batch : list[str  |  Path]
            A list of file paths to be processed.
        function : Callable
            The function (or partial) to apply to each file.

        Returns
        -------
        list[Any]
            A list of results from applying the function to each file in the batch.
        """
        results = []
        for path in file_batch:
            results.append(self._process_file(path, function))
        return results


    def _run_file_mode(self,
                       *args,
                       function: Callable,
                       num_workers: int = 16,
                       output_file: str | Path | None = None,
                       progress_bar_desc: str | None = "default",
                       file_paths: Sequence[str | Path],
                       **kwargs) -> list[Any]:
        """
        Process files by scheduling one file per task.

        Parameters
        ----------
        function : Callable
            The function to apply to each file.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 16
        output_file : str | Path | None, optional
            Optional path to a file where results will be written. If None, results are kept in memory, by default None
        progress_bar_desc : str | None, optional
            Description for the tqdm progress bar, by default "default". If None, no progress bar is shown. If
            "default", a default description is used.
        file_paths : Sequence[str | Path]
            A list of file paths to be processed.

        Returns
        -------
        list[Any]
            A list of results from applying the function to each file.
        """
        results = []
        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        if progress_bar_desc == "default":
            progress_bar_desc = f"Processing files (file mode, workers={num_workers})"

        with open(output_file, "a", encoding="utf-8") if output_file else nullcontext() as out_handle:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                self.logger.info("Processing %d files with %d workers", len(file_paths), num_workers)
                iterator = executor.map(self._process_file, file_paths, repeat(func_with_args))
                if progress_bar_desc is not None:
                    iterator = tqdm(iterator, total=len(file_paths), min_interval=1.0,desc=progress_bar_desc)

                for result in iterator:
                    if out_handle:
                        out_handle.write(f"{result}\n")
                    else:
                        results.append(result)

        return results


    def _run_batch_mode(self,
                        *args,
                        function: Callable,
                        num_workers: int = 8,
                        batch_size: int = 500,
                        output_file: str | Path | None = None,
                        progress_bar_desc: str | None = "default",
                        file_paths: Sequence[str | Path],
                        **kwargs) -> list[Any]:
        """
        Process files by scheduling one batch of batch_size per task.

        Parameters
        ----------
        function : Callable
            The function to apply to each file.
        file_paths : Sequence[str | Path]
            A list of file paths to be processed.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 8
        batch_size : int, optional
            The number of files to process in each batch, by default 500
        output_file : str | Path | None, optional
            Optional path to a file where results will be written. If None, results are kept in memory, by default None
        progress_bar_desc : str | None, optional
            Description for the tqdm progress bar, by default "default". If None, no progress bar is shown. If
            "default", a default description is used.

        Returns
        -------
        list[Any]
            A list of results from applying the function to each file.
        """
        results = []
        batches = list(self._batcher(file_paths, batch_size))

        if progress_bar_desc == "default":
            progress_bar_desc = f"Processing files (batch mode, workers={num_workers}, batch size={batch_size})"

        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        with open(output_file, "a", encoding="utf-8") if output_file else nullcontext() as out_handle:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                self.logger.info("Processing %d files in %d batches with batch size %d using %d workers",
                                 len(file_paths), len(batches), batch_size, num_workers)
                iterator = executor.map(self._process_batch, batches, repeat(func_with_args))
                if progress_bar_desc is not None:
                    iterator = tqdm(iterator, total=len(batches), desc=progress_bar_desc)

                for batch_results in iterator:
                    if out_handle:
                        for result in batch_results:
                            out_handle.write(f"{result}\n")
                    else:
                        results.extend(batch_results)

        return results


    @overload
    def run_pipeline(
        self,
        *args,
        function: Callable,
        return_nums: Literal[False] = False,
        numeric_range: tuple[int, int] | None = None,
        root_dir: Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        batch_size: int = 500,
        num_workers: int | None = None,
        output_file: str | Path | None = None,
        mode: str = "batch",
        progress_bar_desc: str | None = None,
        file_paths_override: Sequence[str | Path] | None = None,
        **kwargs) -> PipelineResult[None]: ...
    @overload
    def run_pipeline(
        self,
        *args,
        function: Callable,
        return_nums: Literal[True],
        numeric_range: tuple[int, int] | None = None,
        root_dir: Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        batch_size: int = 500,
        num_workers: int | None = None,
        output_file: str | Path | None = None,
        mode: str = "batch",
        progress_bar_desc: str | None = None,
        file_paths_override: Sequence[str | Path] | None = None,
        **kwargs) -> PipelineResult[NumberArray]: ...
    @overload
    def run_pipeline(
        self,
        *args,
        function: Callable,
        return_nums: bool = ...,
        numeric_range: tuple[int, int] | None = None,
        root_dir: Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        batch_size: int = 500,
        num_workers: int | None = None,
        output_file: str | Path | None = None,
        mode: str = "batch",
        progress_bar_desc: str | None = None,
        file_paths_override: Sequence[str | Path] | None = None,
        **kwargs) -> PipelineResult[NumberArray] | PipelineResult[None]: ...
    def run_pipeline(
        self,
        *args,
        function: Callable,
        return_nums: bool = False,
        numeric_range: tuple[int, int] | None = None,
        root_dir: Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        batch_size: int = 500,
        num_workers: int | None = None,
        output_file: str | Path | None = None,
        mode: str = "batch",
        progress_bar_desc: str | None = None,
        file_paths_override: Sequence[str | Path] | None = None,
        **kwargs) -> PipelineResult:
        """
        A method to run a processing pipeline on files in the directory, with options for file mode or batch mode,
        progress display, and output to a file. It can also return file numbers extracted from the file names using a
        regex pattern.

        Parameters
        ----------
        args : list[Any]
            Positional arguments to pass to the function.
        function : Callable
            The function to apply to each file.
        return_nums : bool, optional
            Whether to also extract file numbers from the file names, by default False
        numeric_range : tuple[int,int] | None, optional
            The range of numeric values to consider, by default None
        root_dir : Path | str | None, optional
            The root directory to search for files, by default None
        pattern : str | None, optional
            The regex pattern to match files, by default r".*?\.fits$"
        batch_size : int, optional
            The number of files to process in each batch, by default 500
        num_workers : int | None, optional
            The number of worker processes to use, by default None
        output_file : str | Path | None, optional
            The file to write output to, by default None
        mode : str, optional
            The mode to run the pipeline in, by default "batch"
        progress_bar_desc : str | None, optional
            Description for the tqdm progress bar, by default None. If None, no progress bar is shown. If "default", a
            default description is used.
        file_paths_override : Sequence[str | Path] | None, optional
            A sequence of file paths to override the default file search, by default None. Cannot be combined with
            return_nums=True, since numbers cannot be derived from an overridden file list.

        Returns
        -------
        PipelineResult
            The per-file results, and their extracted numbers in `.numbers` if return_nums is True, else None.
        """
        assert mode in ("file", "batch"), "Mode must be either 'file' or 'batch'"

        if root_dir is None:
            root_dir = self.path

        if file_paths_override is not None:
            assert not return_nums, (
                "file_paths_override cannot be combined with return_nums=True, since numbers cannot be derived "
                "from an overridden file list")
            self.logger.info("Using provided list of file paths with %d entries", len(file_paths_override))
            file_paths = file_paths_override
            numbers = None
        else:
            scan_result = self.get_unwrapped_list(path=root_dir,
                                                   pattern=pattern,
                                                   return_nums=return_nums,
                                                   numeric_range=numeric_range)
            file_paths, numbers = scan_result.paths, scan_result.numbers
            self.logger.info("Found %d files matching pattern '%s' in %s", len(file_paths), pattern, root_dir)

        assert file_paths, "No files found to process. Check the root_dir and pattern parameters."

        if mode == "file":
            if num_workers is None:
                num_workers = 16
            return_values = self._run_file_mode(
                *args,
                function=function,
                num_workers=num_workers,
                output_file=output_file,
                progress_bar_desc=progress_bar_desc,
                file_paths=file_paths,
                **kwargs
            )

        else:
            if num_workers is None:
                num_workers = 8
            return_values = self._run_batch_mode(
                *args,
                function=function,
                num_workers=num_workers,
                batch_size=batch_size,
                output_file=output_file,
                progress_bar_desc=progress_bar_desc,
                file_paths=file_paths,
                **kwargs
            )

        # numbers' shape (NumberArray vs None) always matches return_nums by construction above, but that
        # correlation isn't statically provable without duplicating the branch, hence the type: ignore.
        return PipelineResult(results=_to_array(return_values), numbers=numbers)  # type: ignore[arg-type]


    def benchmark_pipeline(
        self,
        function: Callable,
        *args,
        return_nums: bool = False,
        root_dir: Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        numeric_range: tuple[int, int] | None = None,
        worker_options: tuple[int, ...] = (8, 16, 24, 32),
        batch_size_options: tuple[int, ...] = (25, 50, 100, 250, 500),
        sample_size: int | None = 5000,
        repeats: int = 1,
        output_csv: str | Path | None = None,
        progress_bar_desc: str | None = None,
        **kwargs) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Runs a benchmark on the provided function using different numbers of workers and batch sizes, and returns the
        results as a list of dictionaries and the best result as a dictionary. Optionally, the results can be written
        to a CSV file.

        Parameters
        ----------
        function : Callable
            The function to benchmark. It should accept a file path as its first argument, followed by any additional
            arguments and keyword arguments provided in *args and **kwargs.
        return_nums : bool, optional
            Whether to return the number of files processed, by default False
        root_dir : Path | str | None, optional
            The root directory to search for files, by default None
        pattern : str | None, optional
            The regex pattern to match files, by default r".*?\.fits$"
        numeric_range : tuple[int,int] | None, optional
            The range of numeric values to consider, by default None
        worker_options : tuple[int, ...], optional
            The options for the number of workers to use, by default (8, 16, 24, 32)
        batch_size_options : tuple[int, ...], optional
            The options for the batch size to use, by default (25, 50, 100, 250, 500)
        sample_size : int | None, optional
            The number of files to sample for benchmarking, by default 5000
        repeats : int, optional
            The number of times to repeat each benchmark, by default 1
        output_csv : str | Path | None, optional
            The path to the CSV file to write the results to, by default None
        progress_bar_desc : str | None, optional
            Description for the tqdm progress bar, by default None. If None, no progress bar is shown. If "default", a
            default description is used.

        Returns
        -------
        tuple[list[dict[str, Any]], dict[str, Any]]
            A list of dictionaries containing the benchmark results and a dictionary containing the best result

        Raises
        ------
        ValueError
            If no files are found for benchmarking.
        """
        if root_dir is None:
            root_dir = self.path

        # Numbers are not used in the return of this function, but return_nums is accepted here so callers can
        # benchmark the performance of the number-extraction code path too.
        file_paths = self.get_unwrapped_list(path=root_dir,
                                            pattern=pattern,
                                            return_nums=return_nums,
                                            numeric_range=numeric_range).paths

        # Limit files to a subset for benchmarking if specified
        if sample_size is not None:
            file_paths = file_paths[:sample_size]

        if not file_paths:
            raise ValueError("No files found for benchmarking.")

        rows = []
        total_files = len(file_paths)

        self.logger.info("Benchmarking %d files", total_files)

        for workers in worker_options:
            best_seconds = float("inf")

            # Running file mode benchmark
            for _ in range(repeats):
                t0 = time.perf_counter()
                self.run_pipeline(
                    function,
                    *args,
                    root_dir=root_dir,
                    num_workers=workers,
                    mode="file",
                    progress_bar_desc=progress_bar_desc,
                    file_paths_override=file_paths,
                    **kwargs
                )
                elapsed = time.perf_counter() - t0
                best_seconds = min(best_seconds, elapsed)

            rows.append(
                {
                    "mode": "file",
                    "workers": workers,
                    "batch_size": None,
                    "seconds": best_seconds,
                    "files_per_second": total_files / best_seconds,
                }
            )

            # Running batch mode benchmarks for each batch size
            for batch_size in batch_size_options:
                best_seconds = float("inf")
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    self.run_pipeline(
                        function,
                        *args,
                        root_dir=root_dir,
                        batch_size=batch_size,
                        num_workers=workers,
                        mode="batch",
                        progress_bar_desc=progress_bar_desc,
                        file_paths_override=file_paths,
                        **kwargs
                    )
                    elapsed = time.perf_counter() - t0
                    best_seconds = min(best_seconds, elapsed)

                rows.append(
                    {
                        "mode": "batch",
                        "workers": workers,
                        "batch_size": batch_size,
                        "seconds": best_seconds,
                        "files_per_second": total_files / best_seconds,
                    }
                )

        rows_sorted = sorted(rows, key=lambda x: x["seconds"])
        best_row = rows_sorted[0]

        if output_csv:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                f.write("mode,workers,batch_size,seconds,files_per_second\n")
                for row in rows_sorted:
                    batch_value = "" if row["batch_size"] is None else row["batch_size"]
                    f.write(
                        f"{row['mode']},{row['workers']},{batch_value},"
                        f"{row['seconds']:.6f},{row['files_per_second']:.3f}\n"
                    )

        return rows_sorted, best_row
