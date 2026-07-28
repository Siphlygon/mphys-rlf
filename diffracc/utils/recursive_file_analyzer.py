"""
This file defines the RecursiveFileAnalyzer class, which is used to analyze files in a directory recursively.

It provides methods to get an unwrapped list of all files in the directory, optionally matching a regex pattern and
filtering by a numeric range extracted from the file names. It also provides methods to process files in parallel using
either file mode (one file per task) or batch mode (one batch per task), with options for progress display and output to
a file.
"""
from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from tqdm import tqdm

from .logger import LoggingLevels, get_logger

_module_logger = get_logger("RecursiveFileAnalyzer", LoggingLevels.DEBUG.value)




def _safe_call(function: Callable, path: str | Path) -> Any:
    """
    Apply `function` to `path`, returning `None` on any error.

    This function is module-level (and therefore picklable) so it can wrap the mapped function inside a
    `ProcessPoolExecutor` worker. Per-file errors return `None` so one bad file cannot abort the whole run - mirroring
    `_process_file`, but without the cross-process logging (which does not propagate cleanly from worker processes).

    Parameters
    ----------
    function : Callable
        The (already argument-bound) function to apply to the file.
    path : str | Path
        The path to the file to process.

    Returns
    -------
    Any
        The result of `function(path)`, or `None` if it raised.
    
    Raises
    ------
    Exception
        Any exception raised by `function(path)` is caught and logged, and `None` is returned instead of propagating the
        exception.
    """
    try:
        return function(path)
    except Exception:
        return None


def _pad_to_shape(array: npt.NDArray, target_shape: tuple[int, ...]) -> npt.NDArray:
    """
    Pads a numpy array with NaNs to match a target shape.

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
    return np.pad(array, pad_width, mode='constant', constant_values=np.nan)


# Utility functions for for_each
def get_fits_primaryhdu_data(path: Path, expected_shape: tuple[int, ...] | None = None) -> fits.FITS_rec:
    """
    A function to get the primary HDU data from a FITS file, with optional shape normalisation.

    Some FITS images (e.g. PyBDSF outputs for the LoTSS-DR2 cutouts) are occasionally inhomogeneous, which prevents them
    from being stacked into a single numpy array. When `expected_shape` is given, data not matching it is replaced with
    a NaN-filled array of that shape instead of being returned as-is.

    Parameters
    ----------
    path : Path
        The path to the FITS file
    expected_shape : tuple[int, ...] | None, optional
        The shape the data is expected to have once leading size-1 dimensions are stripped, by default `None`.

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
        _module_logger.warning("%s has shape %s instead of expected %s, substituting a NaN-filled array", path,
                               data.shape, expected_shape)
        data = _pad_to_shape(data, expected_shape)
    return data


def get_fits_primaryhdu_header(path: Path, key: str | None = None) -> fits.Header | str:
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


# Constrained results to exactly these two shapes: numbers is a NumberArray when return_nums=True, else None.
NumberArray = npt.NDArray[np.int_]
NumbersT = TypeVar("NumbersT", NumberArray, None)
ResultArray = npt.NDArray[Any]


def _to_array(items: Sequence[Any]) -> ResultArray:
    """
    Builds a numpy array from a sequence of per-file results.
    
    When the results are homogeneous (e.g. scalars, or array-likes of identical shape e.g. `get_fits_primaryhdu_data`'s
    expected_shape), this produces a properly stacked, properly-dtyped array directly usable by callers. When they are
    not homogeneous (e.g. a ragged per-file result), falls back to a 1D object-dtype array of the raw items instead of
    raising.

    Parameters
    ----------
    items : Sequence[Any]
        The items to place into the array, in order.

    Returns
    -------
    ResultArray
        A numpy array containing `items`.
    """
    try:
        return np.array(items)
    # Fall back if there's inhomogenity
    except ValueError:
        array = np.empty(len(items), dtype=object)
        array[:] = items
        return array


# Generic NamedTuples (`class X(NamedTuple, Generic[T])`) are only valid at runtime on Python 3.11+; on 3.10 the class
# statement itself raises TypeError. Type checkers understand generic NamedTuples regardless of the running interpreter,
# so we expose the generic definitions to them here -- preserving the @overload narrowing of `.numbers` to NumberArray
# vs None -- while the runtime (below) uses plain NamedTuples. This keeps the module importable on Python 3.10 with no
# loss of static typing and no change to runtime behaviour (still real, tuple-unpackable NamedTuples).
if TYPE_CHECKING:
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
else:
    class ScanResult(NamedTuple):
        paths: list
        numbers: object

    class PipelineResult(NamedTuple):
        results: object
        numbers: object



class RecursiveFileAnalyzer:
    """
    A class to recursively analyse files in a given directory. It provides methods to get an unwrapped list of all files
    in the directory, optionally matching a regex pattern and filtering by a numeric range extracted from the file
    names. It also provides methods to process files in parallel using file mode (threaded; one file per task), batch
    mode (threaded; one batch per task), and process mode (uses processes), with options for progress display and output
    to a file.
    """
    def __init__(self, path: Path | str, log_level: int = LoggingLevels.INFO.value):
        """
        Initialises the `RecursiveFileAnalyzer` class with a given path and log level.

        Parameters
        ----------
        path: Path | str
            The root directory to recursively search under if no path is specified in its function calls.
        log_level: int, default=LoggingLevels.INFO.value
            The log level for the class logger. Default `LoggingLevels.INFO.value`.
        """
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        self.logger = get_logger("RecursiveFileAnalyzer", log_level)


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
        filtering on the pattern capture group, returning a `ScanResult` containing the matched file paths and,
        optionally, the extracted numbers.

        Parameters
        ----------
        path: Path | str | None = None
            The path to scan. If `None`, defaults to the root path of the `RecursiveFileAnalyzer`. By default `None`.
        pattern: str | None = None
            A regex pattern to filter files. If `None`, all files are yielded. If provided, only files whose names match
            the pattern are yielded. By default `None`.
        numeric_range: tuple[int, int] | None = None
            A range of numbers to filter files. If `None`, no filtering is applied. By default `None`.
        return_nums: bool = False
            Whether to also extract file numbers. If `True`, returns a `ScanResult` with the matched file paths and
            their extracted numbers in `.numbers`. If `False`, returns a `ScanResult` with the matched file paths and
            `None` in `.numbers`. By default `False`. 

        Returns
        -------
        ScanResult
            The matched file paths, and their extracted numbers in `.numbers` if `return_nums=True`, else `None`.
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
    def _quick_scan(self,
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
            The path to scan. If `None`, defaults to the root path of the `RecursiveFileAnalyzer`. By default `None`.
        pattern : str | None, optional
            A regex pattern to filter files. If `None`, all files are yielded. If provided, only files whose names match
            the pattern are yielded. By default `None`.
        numeric_range : tuple[int, int] | None, optional
            A range of numbers to filter files. If `None`, no filtering is applied, by default `None.`
        return_nums : bool, optional
            Whether to return file numbers. If `True`, returns a tuple of `(file_path, file_number)` for each file. The
            file number is extracted from the file name using the first capture group in the regex pattern. If `False`,
            only the file_path is returned, by default `False.`

        Yields
        ------
        Generator[Path | tuple[Path, int], None, None]
            A generator of file paths, and optionally a tuple of `(file_path, file_number)` if `return_nums=True`

        Raises
        ------
        ValueError
            If `return_nums=True` and the regex pattern does not contain a capture group to extract numbers from the
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
                        if numeric_range is not None and (idx < numeric_range[0] or idx >= numeric_range[1]):
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
        A generator function to yield batches of a specified size from an iterable, necessary for `_run_batch_mode` to
        schedule one batch per task.

        Parameters
        ----------
        iterable : Iterable
            The iterable to batch.
        batch_size : int
            The size of each batch.

        Yields
        ------
        Iterator[list]
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
        Processes a single file with the given function, handling exceptions and logging warnings if any occur.
        
        The given function may be a partial function with args and kwargs passed into `run_pipeline`, or a simple
        function that takes a single path argument. Any exception raised by the function is caught and logged, and
        `None` is returned instead of propagating the exception.
        
        Parameters
        ----------
        path : str | Path
            The path to the file to be processed.
        function : Callable
            The function (or partial) to apply to the file.
            
        Returns
        -------
        Any
            The result of applying `function` to the file at `path`, or `None` if an exception occurred.
        
        Raises
        ------
        Exception
            Any exception raised by `function(path)` is caught and logged, and `None` is returned instead of propagating
            the exception.
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
        
        Runs `_process_file` on each file in the batch, which handles exceptions and logging. Returns a list of results.

        Parameters
        ----------
        file_batch : Sequence[str | Path]
            A list of file paths to be processed.
        function : Callable
            The function (or partial) to apply to each file. See `_process_file` for details on the function.

        Returns
        -------
        list[Any]
            A list of results from applying `function` to each file in the batch.
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
        Process files by scheduling one file per task, using a thread pool for concurrent processing.
        
        The function is combined with any provided positional and keyword arguments using `functools.partial`, and then
        applied to each file in `file_paths`. Results are collected and optionally written to an output file. A progress
        bar can be displayed using `tqdm`.
        
        This is the simplest mode, and is appropriate for I/O-bound or light-parse work (e.g. reading image data blocks
        or small text logs). For CPU-bound per-file work dominated by GIL-holding Python (e.g. astropy parsing of
        many-column FITS binary tables), use `_run_process_mode` instead.

        Parameters
        ----------
        *args : list[Any]
            Positional arguments to pass to `function`.
        function : Callable
            The function to apply to each file.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 16.
        output_file : str | Path | None, optional
            Optional path to a file where results will be written. If `None`, results are not written to a file. By
            default `None`.
        progress_bar_desc : str | None, optional
            Description for the `tqdm` progress bar. If `None`, no progress bar is shown. If `"default"`, a basic
            description is used. By default `"default"`.
        file_paths : Sequence[str | Path]
            A list of file paths to be processed.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to `function`.

        Returns
        -------
        list[Any]
            A list of results from applying `function` to each file in `file_paths`.
        """
        results = []
        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        if progress_bar_desc == "default":
            progress_bar_desc = f"Processing files (file mode, workers={num_workers})"

        with (open(output_file, "a", encoding="utf-8") if output_file else nullcontext()) as out_handle, \
        ThreadPoolExecutor(max_workers=num_workers) as executor:
            self.logger.info("Processing %d files with %d workers", len(file_paths), num_workers)
            iterator = executor.map(self._process_file, file_paths, repeat(func_with_args))
            if progress_bar_desc is not None:
                iterator = tqdm(iterator, total=len(file_paths), mininterval=1.0, desc=progress_bar_desc)

            for result in iterator:
                if out_handle:
                    out_handle.write(f"{result}\n")
                else:
                    results.append(result)

        return results


    def _run_process_mode(self,
                          *args,
                          function: Callable,
                          num_workers: int = 8,
                          chunksize: int = 64,
                          output_file: str | Path | None = None,
                          progress_bar_desc: str | None = "default",
                          file_paths: Sequence[str | Path],
                          **kwargs) -> list[Any]:
        """
        Process files by scheduling them across worker processes rather than threads.

        This is appropriate for CPU-bound per-file work dominated by GIL-holding Python (e.g. astropy parsing of
        many-column FITS binary tables), which threads cannot parallelise because of the GIL. Threads remain the right
        choice for I/O-bound or light-parse work (e.g. reading image data blocks or small text logs), where process
        startup and the pickling of arguments/return values would cost more than they save.

        `function` (with any bound *args/**kwargs) and its return value must be picklable, and `function` must be
        importable by qualified name - a module-level function or a static/classmethod, not a local closure or lambda.

        Parameters
        ----------
        *args : list[Any]
            Positional arguments to pass to `function`.
        function : Callable
            The function to apply to each file.
        num_workers : int, optional
            The number of worker processes to use. On a shared cluster node this should be set to the job's core
            allocation, not the node's total core count. By default 8.
        chunksize : int, optional
            The number of files handed to each worker per dispatch. Larger values spreads the per-task IPC overhead over
            more (small) files. By default 64.
        output_file : str | Path | None, optional
            Optional path to a file where results will be written. If `None`, results are not written to a file. By
            default `None`.
        progress_bar_desc : str | None, optional
            Description for the `tqdm` progress bar. If `None`, no progress bar is shown. If `"default"`, a default
            description is used. By default `"default"`.
        file_paths : Sequence[str | Path]
            A list of file paths to be processed.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to `function`.

        Returns
        -------
        list[Any]
            A list of results from applying the function to each file, with `None` in place of any file that errored.
        """
        results = []
        # Bind the caller's args/kwargs, then wrap in _safe_call so a single bad file returns None instead of
        # propagating out of a worker and tearing down the whole pool.
        func_with_args = partial(function, *args, **kwargs)
        call = partial(_safe_call, func_with_args)

        if progress_bar_desc == "default":
            progress_bar_desc = f"Processing files (process mode, workers={num_workers})"

        with (open(output_file, "a", encoding="utf-8") if output_file else nullcontext()) as out_handle, \
        ProcessPoolExecutor(max_workers=num_workers) as executor:
            self.logger.info("Processing %d files with %d worker processes", len(file_paths), num_workers)
            iterator = executor.map(call, file_paths, chunksize=chunksize)
            if progress_bar_desc is not None:
                iterator = tqdm(iterator, total=len(file_paths), mininterval=1.0, desc=progress_bar_desc)

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
        Process files by scheduling one batch of batch_size per task, using a thread pool for concurrent processing.
        
        This can be faster than file mode for many small files, because it reduces the per-task scheduling overhead. It
        is also a thread-based mode, so it is appropriate for I/O-bound or light-parse work (e.g. reading image data
        blocks or small text logs). For CPU-bound per-file work dominated by GIL-holding Python (e.g. astropy parsing of
        many-column FITS binary tables), use `_run_process_mode` instead.

        Parameters
        ----------
        *args : list[Any]
            Positional arguments to pass to `function`.
        function : Callable
            The function to apply to each file.
        file_paths : Sequence[str | Path]
            A list of file paths to be processed, fed in batches to `function`.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 8.
        batch_size : int, optional
            The number of files to process in each batch, by default 500.
        output_file : str | Path | None, optional
            Optional path to a file where results will be written. If `None`, results are not written to a file. By
            default `None`.
        progress_bar_desc : str | None, optional
            Description for the `tqdm` progress bar. If `None`, no progress bar is shown. If `"default"`, a basic
            description is used. By default `"default"`.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to `function`.

        Returns
        -------
        list[Any]
            A list of results from applying `function` to each file in `file_paths`.
        """
        results = []
        batches = list(self._batcher(file_paths, batch_size))

        if progress_bar_desc == "default":
            progress_bar_desc = f"Processing files (batch mode, workers={num_workers}, batch size={batch_size})"

        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        with (open(output_file, "a", encoding="utf-8") if output_file else nullcontext()) as out_handle, \
        ThreadPoolExecutor(max_workers=num_workers) as executor:
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
        progress_bar_desc: str | None = "default",
        file_paths_override: Sequence[str | Path] | None = None,
        **kwargs) -> PipelineResult:
        """
        A method to run a processing pipeline, applying `function` to files found in the `root_dir`, with options for
        `"file"`, `"batch"`, or `"process"` mode, setting a progress display, and output to a `output_file`. It can also
        return file numbers extracted from the file names using a regex `pattern`.

        Parameters
        ----------
        args : list[Any]
            Positional arguments to pass to `function`.
        function : Callable
            The function to apply to each file.
        return_nums : bool, optional
            Whether to also extract file numbers from the file names, by default `False`.
        numeric_range : tuple[int,int] | None, optional
            The range of numeric values to consider, by default `None`, which considers all values.
        root_dir : Path | str | None, optional
            The root directory to search for files, by default `None`, which searches the directory in `self.path`.
        pattern : str | None, optional
            The regex pattern to match files, by default `r".*?\.fits$"`.
        batch_size : int, optional
            The number of files to process in each batch if using batch mode, by default 500.
        num_workers : int | None, optional
            The number of worker processes to use, by default `None`. If `None`, defaults to 16 for file mode, or 8 for
            batch and process modes.
        output_file : str | Path | None, optional
            The file to write output to, by default `None`, which doesn't write to a file.
        mode : str, optional
            The mode to run the pipeline in, by default `"batch"`. `"file"` and `"batch"` schedule work across threads
            (one task per file, or per batch of files); `"process"` schedules one task per file across worker processes,
            for CPU-bound per-file work that the GIL prevents threads from parallelising (see `_run_process_mode` for
            the picklability requirements it imposes on `function`).
        progress_bar_desc : str | None, optional
            Description for the `tqdm` progress bar, by default `"default"`. If `None`, no progress bar is shown. If
            `"default"`, a basic description is used.
        file_paths_override : Sequence[str | Path] | None, optional
            A sequence of file paths to override the default file search, by default `None`. Cannot be combined with
            `return_nums=True`, since numbers cannot be derived from an overridden file list.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to the function.

        Returns
        -------
        PipelineResult
            The per-file results, and their extracted numbers in `.numbers` if `return_nums=True`, else `None`.
        """
        assert mode in ("file", "batch", "process"), "Mode must be 'file', 'batch', or 'process'"

        if root_dir is None:
            root_dir = self.path

        if file_paths_override is not None:
            #todo: functionality for the below can be implemented if needed, but we rarely use file_paths_override
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

        assert file_paths, "No files found to process. Check the root_dir and pattern (if specified) parameters."

        if num_workers is None:
            match mode:
                case "file":
                    num_workers = 16
                case "process":
                    num_workers = 8
                case "batch":
                    num_workers = 8

        if mode == "file":
            return_values = self._run_file_mode(
                *args,
                function=function,
                num_workers=num_workers,
                output_file=output_file,
                progress_bar_desc=progress_bar_desc,
                file_paths=file_paths,
                **kwargs
           )

        elif mode == "process":
            return_values = self._run_process_mode(
                *args,
                function=function,
                num_workers=num_workers,
                output_file=output_file,
                progress_bar_desc=progress_bar_desc,
                file_paths=file_paths,
                **kwargs
           )

        else:
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

        # numbers' shape (NumberArray vs None) always matches return_nums by construction above, but that correlation
        # isn't statically provable without duplicating the branch, hence the type: ignore.
        return PipelineResult(results=_to_array(return_values), numbers=numbers)  # type: ignore[arg-type]


    # deprecated: was used to benchmark the pipeline with different numbers of workers and batch sizes, but is no longer
    # used in the current codebase.
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
