"""
Recursively analyse files in a directory, with optional regex pattern matching and numeric range filtering.

This module provides the `RecursiveFileAnalyzer` class, which can be used to recursively scan a directory for files,
optionally filtering them based on a regex pattern and a numeric range extracted from the file names. It also provides
methods to process files in parallel using different modes ('file' mode and 'process' mode), with options for progress
display and output to a file.

'file' mode uses `ThreadPoolExecutor` to process one file per task, which is very efficient for I/O-bound or light-parse
work. 'process' mode uses `ProcessPoolExecutor` to process one file per task, which is suitable for CPU-bound work that 
eleases the GIL but has more overhead due to process creation and inter-process communication. 

This module is used extensively throughout the diffracc codebase for file access and processing.
"""
from __future__ import annotations

import os
import re
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

    def _iter_file_mode(self,
                        *args,
                        function: Callable,
                        num_workers: int = 8,
                        file_paths: Sequence[str | Path],
                        **kwargs) -> Iterator[Any]:
        """
        Stream per-file results in submission order, scheduling one file per task across a thread pool.

        This is a streaming primitive that yields results as they are completed in the same order as 'file_paths',
        regardless of completion order. Any caller using this function can therefore e.g., copy the result into a 
        preallocated array to avoid duplicating the list in memory. `_run_file_mode` instead materialises the results
        into a list which is useful for e.g., when the full number of results is not known, or when one wants to avoid
        handling the collection loop manually.

        The `ThreadPoolExecutor` stays open for the lifetime of the generator. A caller that stops before exhausting it
        should close the generator (or let it be garbage-collected) so the pool is shut down.

        Parameters
        ----------
        *args : list[Any]
            Positional arguments to pass to `function`.
        function : Callable
            The function to apply to each file.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 8.
        file_paths : Sequence[str | Path]
            A list of file paths to be processed.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to `function`.

        Yields
        ------
        Any
            The result of applying `function` to each file, in `file_paths` order, with `None` in place of any file
            that errored (see `_process_file`).
        """
        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            self.logger.info("Processing %d files with %d workers", len(file_paths), num_workers)
            yield from executor.map(self._process_file, file_paths, repeat(func_with_args))

    def _run_file_mode(self,
                       *args,
                       function: Callable,
                       num_workers: int = 8,
                       output_file: str | Path | None = None,
                       progress_bar_desc: str | None = "default",
                       file_paths: Sequence[str | Path],
                       **kwargs) -> list[Any]:
        """
        Process files by scheduling one file per task, using a thread pool for concurrent processing.

        The function is combined with any provided positional and keyword arguments using `functools.partial`, and then
        applied to each file in `file_paths`. Results are materialised from the `_iter_file_mode` stream into a list and
        optionally written to an output file. A progress bar can be displayed using `tqdm`.

        This is the simplest mode, and is appropriate for I/O-bound or light-parse work (e.g. reading image data blocks
        or small text logs). For CPU-bound per-file work dominated by GIL-holding Python (e.g. astropy parsing of
        many-column FITS binary tables), use `_run_process_mode` instead. To consume results one at a time without
        holding the whole list in memory, use `iter_pipeline` (the streaming counterpart of `run_pipeline`) instead.

        Parameters
        ----------
        *args : list[Any]
            Positional arguments to pass to `function`.
        function : Callable
            The function to apply to each file.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 8.
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
        if progress_bar_desc == "default":
            progress_bar_desc = f"Processing files (file mode, workers={num_workers})"

        iterator: Iterable[Any] = self._iter_file_mode(*args,
                                                       function=function,
                                                       num_workers=num_workers,
                                                       file_paths=file_paths,
                                                       **kwargs)
        if progress_bar_desc is not None:
            iterator = tqdm(iterator, total=len(file_paths), mininterval=1.0, desc=progress_bar_desc)

        results = []
        with (open(output_file, "a", encoding="utf-8") if output_file else nullcontext()) as out_handle:
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
        many-column FITS binary tables -- NOT simple image reading), which threads cannot parallelise because of the
        GIL. Threads remain the right choice for I/O-bound or light-parse work (e.g. reading image data blocks or small
        text logs), where process startup and the pickling of arguments/return values would cost more than they save.

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

    def _resolve_file_paths(self,
                            *,
                            return_nums: bool,
                            numeric_range: tuple[int, int] | None,
                            root_dir: Path | str | None,
                            pattern: str | None,
                            file_paths_override: Sequence[str | Path] | None
                            ) -> tuple[Sequence[str | Path], NumberArray | None]:
        """
        Resolve the files to process (and their extracted numbers) for `run_pipeline` and `iter_pipeline`.

        Either scans `root_dir` for files matching `pattern` (extracting numbers when `return_nums=True`) or uses a
        caller-supplied `file_paths_override`. Shared by the materialising (`run_pipeline`) and streaming
        (`iter_pipeline`) entry points so the two resolve their inputs identically.

        Parameters
        ----------
        return_nums : bool
            Whether to also extract file numbers from the file names.
        numeric_range : tuple[int, int] | None
            The range of numeric values to consider, or `None` to consider all values.
        root_dir : Path | str | None
            The root directory to search, or `None` to use `self.path`.
        pattern : str | None
            The regex pattern to match files.
        file_paths_override : Sequence[str | Path] | None
            An explicit list of files to use instead of scanning. Cannot be combined with `return_nums=True`, since
            numbers cannot be derived from an overridden file list.

        Returns
        -------
        tuple[Sequence[str | Path], NumberArray | None]
            The files to process, and their extracted numbers if `return_nums=True`, else `None`.
        """
        if root_dir is None:
            root_dir = self.path

        if file_paths_override is not None:
            #todo: functionality for the below can be implemented if needed, but we rarely use file_paths_override
            assert not return_nums, (
                "file_paths_override cannot be combined with return_nums=True, since numbers cannot be derived "
                "from an overridden file list")
            self.logger.info("Using provided list of file paths with %d entries", len(file_paths_override))
            return file_paths_override, None

        scan_result = self.get_unwrapped_list(path=root_dir,
                                              pattern=pattern,
                                              return_nums=return_nums,
                                              numeric_range=numeric_range)
        self.logger.info("Found %d files matching pattern '%s' in %s", len(scan_result.paths), pattern, root_dir)
        return scan_result.paths, scan_result.numbers

    @overload
    def run_pipeline(
        self,
        *args,
        function: Callable,
        return_nums: Literal[False] = False,
        numeric_range: tuple[int, int] | None = None,
        root_dir: Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        num_workers: int = 8,
        output_file: str | Path | None = None,
        mode: str = "file",
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
        num_workers: int = 8,
        output_file: str | Path | None = None,
        mode: str = "file",
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
        num_workers: int = 8,
        output_file: str | Path | None = None,
        mode: str = "file",
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
        num_workers: int = 8,
        output_file: str | Path | None = None,
        mode: str = "file",
        progress_bar_desc: str | None = "default",
        file_paths_override: Sequence[str | Path] | None = None,
        **kwargs) -> PipelineResult:
        """
        A method to run a processing pipeline, applying `function` to files found in the `root_dir`, with options for
        `"file"` (multiple threads, one process under GIL) or `"process"` (worker processes sidestepping GIL) mode,
        setting a progress display, and output to a `output_file`. It can also return file numbers extracted from the
        file names using a regex `pattern`.

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
        num_workers : int, optional
            The number of worker processes to use, by default 8.
        output_file : str | Path | None, optional
            The file to write output to, by default `None`, which doesn't write to a file.
        mode : str, optional
            The mode to run the pipeline in, must be `"file"` or `"process"`, by default `"file"`. See `_run_file_mode`
            and `_run_process_mode` for details on the differences between the two modes.
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
        assert mode in ("file", "process"), "Mode must be 'file or 'process'"

        file_paths, numbers = self._resolve_file_paths(return_nums=return_nums,
                                                       numeric_range=numeric_range,
                                                       root_dir=root_dir,
                                                       pattern=pattern,
                                                       file_paths_override=file_paths_override)

        assert file_paths, "No files found to process. Check the root_dir and pattern (if specified) parameters."

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
        else:
            return_values = self._run_process_mode(
                *args,
                function=function,
                num_workers=num_workers,
                output_file=output_file,
                progress_bar_desc=progress_bar_desc,
                file_paths=file_paths,
                **kwargs
           )

        # numbers' shape (NumberArray vs None) always matches return_nums by construction above, but that correlation
        # isn't statically provable without duplicating the branch, hence the type: ignore.
        return PipelineResult(results=_to_array(return_values), numbers=numbers)  # type: ignore[arg-type]


    @overload
    def iter_pipeline(self,
                      *args,
                      function: Callable,
                      return_nums: Literal[False] = False,
                      numeric_range: tuple[int, int] | None = None,
                      root_dir: Path | str | None = None,
                      pattern: str | None = r".*?\.fits$",
                      num_workers: int = 8,
                      progress_bar_desc: str | None = None,
                      file_paths_override: Sequence[str | Path] | None = None,
                      **kwargs) -> Iterator[Any]: ...
    @overload
    def iter_pipeline(self,
                      *args,
                      function: Callable,
                      return_nums: Literal[True],
                      numeric_range: tuple[int, int] | None = None,
                      root_dir: Path | str | None = None,
                      pattern: str | None = r".*?\.fits$",
                      num_workers: int = 8,
                      progress_bar_desc: str | None = None,
                      file_paths_override: Sequence[str | Path] | None = None,
                      **kwargs) -> Iterator[tuple[int, Any]]: ...
    def iter_pipeline(self,
                      *args,
                      function: Callable,
                      return_nums: bool = False,
                      numeric_range: tuple[int, int] | None = None,
                      root_dir: Path | str | None = None,
                      pattern: str | None = r".*?\.fits$",
                      num_workers: int = 8,
                      progress_bar_desc: str | None = None,
                      file_paths_override: Sequence[str | Path] | None = None,
                      **kwargs) -> Iterator[Any] | Iterator[tuple[int, Any]]:
        r"""
        Stream per-file results in file-number order without materialising them all in memory.

        The streaming counterpart to `run_pipeline`; file mode only (threaded). Yields each result as it completes, in
        `file_paths` order. When `return_nums=True` it yields `(number, result)` pairs, where `number` is taken from
        the pattern's capture group, so a caller can place each result by index - e.g. into a preallocated,
        index-aligned array - and never hold the whole stack at once.This is the memory-safe path for very large runs
        whose results are folded away as they arrive; for a materialised `results` array plus its `numbers`, or for
        process modes, use `run_pipeline` instead.

        The underlying thread pool stays open for the lifetime of the generator, so fully consume it (or close it) to
        shut the pool down.

        Parameters
        ----------
        args : list[Any]
            Positional arguments to pass to `function`.
        function : Callable
            The function to apply to each file.
        return_nums : bool, optional
            Whether to yield `(number, result)` pairs rather than bare results, by default `False`. Requires `pattern`
            to contain a capture group, and cannot be combined with `file_paths_override`.
        numeric_range : tuple[int, int] | None, optional
            The range of numeric values to consider, by default `None`, which considers all values.
        root_dir : Path | str | None, optional
            The root directory to search for files, by default `None`, which searches the directory in `self.path`.
        pattern : str | None, optional
            The regex pattern to match files, by default `r".*?\.fits$"`.
        num_workers : int, optional
            The number of worker threads to use, by default `8`.
        progress_bar_desc : str | None, optional
            Description for the `tqdm` progress bar, by default `None`, which shows no bar. If `"default"`, a basic
            description is used. The bar's total is the number of files found.
        file_paths_override : Sequence[str | Path] | None, optional
            A sequence of file paths to override the default file search, by default `None`. Cannot be combined with
            `return_nums=True`, since numbers cannot be derived from an overridden file list.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to `function`.

        Yields
        ------
        Any | tuple[int, Any]
            Each per-file result in `file_paths` order, or `(number, result)` if `return_nums=True`. Failed files
            yield `None` in the result position (see `_process_file`).
        """
        file_paths, numbers = self._resolve_file_paths(return_nums=return_nums,
                                                       numeric_range=numeric_range,
                                                       root_dir=root_dir,
                                                       pattern=pattern,
                                                       file_paths_override=file_paths_override)

        assert file_paths, "No files found to process. Check the root_dir and pattern (if specified) parameters."

        results = self._iter_file_mode(*args,
                                       function=function,
                                       num_workers=num_workers,
                                       file_paths=file_paths,
                                       **kwargs)

        # numbers is a NumberArray exactly when return_nums=True (guaranteed by _resolve_file_paths), so the zip below
        # is only reached with a real array; the type: ignore covers the None-typed branch the checker can't rule out.
        stream: Iterable[Any] = zip(numbers, results) if return_nums else results  # type: ignore[arg-type]

        if progress_bar_desc is not None:
            if progress_bar_desc == "default":
                progress_bar_desc = f"Processing files (streaming, workers={num_workers})"
            stream = tqdm(stream, total=len(file_paths), mininterval=1.0, desc=progress_bar_desc)

        yield from stream
