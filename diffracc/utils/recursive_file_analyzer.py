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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Iterator

import astropy.stats
import numpy as np
from astropy.io import fits
from matplotlib.axes import Axes
from tqdm import tqdm

from .logger import LoggingLevels, get_logger


# Utility functions for for_each
def get_fits_primaryhdu_data( path: Path ) -> np.ndarray:
    """
    A function to get the primary HDU data from a FITS file.

    Parameters
    ----------
    path : Path
        The path to the FITS file

    Returns
    -------
    np.ndarray
        The primary HDU data from the FITS file
    """
    with fits.open( str( path ), memmap=False ) as hdul:
        data = hdul[ 0 ].data
    # Get rid of leading 1s in shape, e.g. (1,1,n,n) -> (n,n), but preserve 2 dimensions for single pixel images
    while ( len( data.shape ) > 2 ) and ( data.shape[ 0 ] == 1 ):
        data = data[ 0 ]
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
    with fits.open( str( path ), memmap=False ) as hdul:
        if key is not None:
            header = hdul[ 0 ].header[ key ]
        else:
            header = hdul[ 0 ].header
    return header


class RecursiveFileAnalyzer:
    """
    A class to recursively analyse files in a given directory. It provides methods to get an unwrapped list of all files
    in the directory, optionally matching a regex pattern and filtering by a numeric range extracted from the file
    names. It also provides methods to process files in parallel using either file mode (one file per task) or batch
    mode (one batch per task), with options for progress display and output to a file.
    """
    def __init__( self, path: Path | str, log_level: int = LoggingLevels.INFO.value ):
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
        if path is not Path:
            path = Path( path )
        self.path = path
        self.logger = get_logger( self.__class__.__name__ )
        self.logger.setLevel( log_level )


    def get_unwrapped_list(self,
                           path: Path | str | None = None,
                           pattern: str | None = None,
                           numeric_range: tuple[int,int] | None = None,
                           return_nums: bool = False) -> list[Path] | tuple[list[Path], list[int]]:
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
        numeric_range: tuple[int,int] | None = None
            A range of numbers to filter files. If None, no filtering is applied.
        return_nums: bool = False
            Whether to return file numbers.

        Returns
        -------
        list[Path] | tuple[list[Path], list[int]]
            A list of file paths, and optionally a list of file numbers if return_nums is True.
        """
        if return_nums:
            file_paths, idxs = map( list, zip( *self._quick_scan( path, pattern, numeric_range, return_nums ) ) )
            # sort paths and idxs by idxs
            idxs, file_paths = map(list, zip( *sorted( zip( idxs, file_paths ) ) ))
            return file_paths, idxs

        file_paths = list( self._quick_scan( path, pattern, numeric_range ) )
        return file_paths


    def _quick_scan( self,
                    path: Path | str | None = None,
                    pattern: str | None = None,
                    numeric_range: tuple[int,int] | None = None,
                    return_nums: bool = False) -> Iterator[Path | tuple[Path, int]]:
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
        numeric_range : tuple[int,int] | None, optional
            A range of numbers to filter files. If None, no filtering is applied, by default None
        return_nums : bool, optional
            Whether to return file numbers. If True, returns a tuple of (file_path, file_number) for each file. The
            file_number is extracted from the file name using the first capture group in the regex pattern. If False,
            only the file_path is returned, by default False

        Yields
        ------
        Iterator[Path | tuple[Path, int]]
            An iterator of file paths, and optionally a tuple of (file_path, file_number) if return_nums is True

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

                elif pattern is None or re.match(pattern, entry.name):
                    if return_nums:
                        # Extract file number using the first capture group in the regex pattern
                        try:
                            idx = int( re.search( pattern, entry.name ).group( 1 ) )
                        except IndexError as exc:
                            raise ValueError(f"Pattern '{pattern}' does not contain a capture group to extract "
                                             f"numbers for file '{entry.name}'") from exc
                        if numeric_range is not None:
                            if idx < numeric_range[ 0 ] or idx >= numeric_range[ 1 ]:
                                continue
                        yield (Path(entry), idx)
                    else:
                        # save some processing power by only doing regex match if numeric_range is provided
                        if numeric_range is not None:
                            try:
                                idx = int( re.search( pattern, entry.name ).group( 1 ) )
                            except IndexError as exc:
                                raise ValueError(f"Pattern '{pattern}' does not contain a capture group to extract "
                                                 f"numbers for file '{entry.name}'") from exc
                            if idx < numeric_range[ 0 ] or idx >= numeric_range[ 1 ]:
                                continue
                        yield Path(entry)


    def _batcher(self, iterable : list, batch_size : int) -> Iterator[list]:
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


    def _process_file(self, path : str | Path, function : Callable) -> Any:
        """
        Processes a single file with the given function, handling exceptions and logging warnings if any occur.
        
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


    def _process_batch(self, file_batch : list[str | Path], function : Callable) -> list[Any]:
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
            try:
                results.append(self._process_file(path, function))
            except Exception as e:
                self.logger.warning("Error with %s: %s", path, e)
        return results


    def _run_file_mode(self,
                       *args,
                       function : Callable,
                       num_workers : int = 16,
                       output_file : str | Path | None = None,
                       show_progress : bool = True,
                       file_paths : list[str | Path],
                       **kwargs,
                       ) -> list[Any]:
        """
        Process files by scheduling one file per task.

        Parameters
        ----------
        function : Callable
            The function to apply to each file.
        file_paths : list[str  |  Path]
            A list of file paths to be processed.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 16
        output_file : str | Path | None, optional
            Optional path to a file where results will be written. If None, results are kept in memory, by default None
        show_progress : bool, optional
            Whether to display a tqdm progress bar, by default True

        Returns
        -------
        list[Any]
            A list of results from applying the function to each file.
        """
        results = []
        out_handle = open(output_file, "a", encoding="utf-8") if output_file else None

        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                self.logger.info("Processing %d files with %d workers", len(file_paths), num_workers)
                iterator = executor.map(self._process_file, file_paths, repeat(func_with_args))
                if show_progress:
                    iterator = tqdm(iterator, total=len(file_paths),
                        desc=f"Processing files (file mode, workers={num_workers})",
                    )
                for result in iterator:
                    if out_handle:
                        out_handle.write(f"{result}\n")
                    else:
                        results.append(result)
        finally:
            if out_handle:
                out_handle.close()

        return results


    def _run_batch_mode(self,
                        *args,
                        function : Callable,
                        num_workers : int = 8,
                        batch_size : int = 500,
                        output_file : str | Path | None = None,
                        show_progress : bool = True,
                        file_paths : list[str | Path],
                        **kwargs) -> list[Any]:
        """
        Process files by scheduling one batch of batch_size per task.

        Parameters
        ----------
        function : Callable
            The function to apply to each file.
        file_paths : list[str  |  Path]
            A list of file paths to be processed.
        num_workers : int, optional
            The number of worker threads to use for concurrent processing, by default 8
        batch_size : int, optional
            The number of files to process in each batch, by default 500
        output_file : str | Path | None, optional
            Optional path to a file where results will be written. If None, results are kept in memory, by default None
        show_progress : bool, optional
            Whether to display a tqdm progress bar, by default True

        Returns
        -------
        list[Any]
            A list of results from applying the function to each file.
        """
        results = []
        batches = list(self._batcher(file_paths, batch_size))
        out_handle = open(output_file, "a", encoding="utf-8") if output_file else None

        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                self.logger.info("Processing %d files in %d batches with batch size %d using %d workers",
                                 len(file_paths), len(batches), batch_size, num_workers)
                iterator = executor.map(self._process_batch, batches, repeat(func_with_args))
                if show_progress:
                    iterator = tqdm(iterator, total=len(batches),
                        desc=(
                            f"Processing files (batch mode, workers={num_workers}, batch size={batch_size})"
                        ),
                    )

                for batch_results in iterator:
                    if out_handle:
                        for result in batch_results:
                            out_handle.write(f"{result}\n")
                    else:
                        results.extend(batch_results)
        finally:
            if out_handle:
                out_handle.close()

        return results


    def run_pipeline(
        self,
        *args,
        function : Callable,
        return_nums : bool = False,
        numeric_range: tuple[int,int] | None = None,
        root_dir : Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        batch_size : int = 500,
        num_workers : int | None = None,
        output_file : str | Path | None = None,
        mode : str = "batch",
        show_progress : bool = True,
        file_paths_override : list[str | Path] | None = None,
        **kwargs
        ) -> list[Any] | tuple[list[Any], list[int]]:
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
            Whether to return file numbers extracted from the file names, by default False
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
        show_progress : bool, optional
            Whether to show progress, by default True
        file_paths_override : list[str  |  Path] | None, optional
            A list of file paths to override the default file search, by default None

        Returns
        -------
        list[Any] | tuple[list[Any], list[int]]
            The results of the pipeline execution, and optionally a list of file numbers if return_nums is True
        """
        assert mode in ("file", "batch"), "Mode must be either 'file' or 'batch'"

        if root_dir is None:
            root_dir = self.path

        if file_paths_override is not None:
            self.logger.info("Using provided list of file paths with %d entries", len(file_paths_override))
            file_paths = file_paths_override
        else:
            file_paths = self.get_unwrapped_list(path=root_dir,
                                                 pattern=pattern,
                                                 return_nums=return_nums,
                                                 numeric_range=numeric_range)
            if return_nums:
                file_paths, numbers = file_paths
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
                show_progress=show_progress,
                file_paths=file_paths,
                **kwargs
            )

        if mode == "batch":
            if num_workers is None:
                num_workers = 8
            return_values = self._run_batch_mode(
                *args,
                function=function,
                num_workers=num_workers,
                batch_size=batch_size,
                output_file=output_file,
                show_progress=show_progress,
                file_paths=file_paths,
                **kwargs
            )

        return (return_values, numbers) if return_nums else return_values


    def benchmark_pipeline(
        self,
        function : Callable,
        *args,
        return_nums : bool = False,
        root_dir : Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        numeric_range: tuple[int,int] | None = None,
        worker_options : tuple[int, ...] = (8, 16, 24, 32),
        batch_size_options : tuple[int, ...] = (25, 50, 100, 250, 500),
        sample_size : int | None = 5000,
        repeats : int = 1,
        output_csv : str | Path | None = None,
        show_progress : bool = True,
        **kwargs
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        show_progress : bool, optional
            Whether to show a progress bar, by default True

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

        file_paths = self.get_unwrapped_list(path=root_dir,
                                            pattern=pattern,
                                            return_nums=return_nums,
                                            numeric_range=numeric_range)
        if isinstance(file_paths, tuple):
            # You can test performance of returning numbers, but it is not used in the return of this function.
            file_paths, _ = file_paths

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
                    show_progress=show_progress,
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
                        show_progress=show_progress,
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


class HistogramErrorDrawer:
    """
    Purely utility class to draw histograms with error bars
    """
    def __init__( self ):
        pass

    def draw( self,
             data: np.ndarray,
             ax: Axes,
             bins: int,
             range: tuple[ float, float ],
             label: str,
             color: str,
             density: bool,
             relative: bool ):
        """
        Utility function to draw a histogram with error bars according to astropy.stats.poisson_conf_interval with 
        sigma=1.0

        Parameters
        ----------
        data: np.ndarray
            The data to plot
        ax: Axes
            The axes to plot the histogram and error bars on
        bins: int
            Number of bins to sort the data into
        range: tuple[ float, float ]
            Range to plot the histogram on (neccesary parameter to compare histograms of slightly different data)
        label: str
            How to label the data
        color: str
            How to color the data
        density: bool
            Whether or not to make the histogram (and associated error bars) a density plot
        relative: bool
            Whether or not to draw a relative frequency histogram. Mutually exclusive with density.
        """
        if density and relative:
            raise RuntimeError( "Cannot have a histogram be both density and relative frequency" )

        hist, _ = np.histogram( data, bins=bins, range=range )
        conf_interval = astropy.stats.poisson_conf_interval( hist, sigma=1.0, interval='frequentist-confidence' )


        drawn_histogram, bin_edges = np.histogram( data, bins=bins, range=range, density=density )
        if relative:
            drawn_histogram = drawn_histogram / data.shape[ 0 ]
        bin_centres = ( bin_edges[ :-1 ] + bin_edges[ 1: ] )/2.0
        ax.step( bin_edges, np.append( drawn_histogram, np.zeros( 1 ) ), label=label, color=color, where='post' )

        yerr = conf_interval[ 1 ] - conf_interval[ 0 ]

        #poisson_conf_interval needs the raw data to be accurate, so we do it on the unweighted histogram and weight it
        # afterward here
        if density:
            yerr = yerr / np.sum( data )
        elif relative:
            yerr = yerr / data.shape[ 0 ]

        ax.errorbar( bin_centres, drawn_histogram, yerr, fmt='.', color=color )
