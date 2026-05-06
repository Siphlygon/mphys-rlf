# This is a file created by Ashley and Luna. It defines the RecursiveFileAnalyzer and the HistogramErrorDrawer.
# The main purpose of the RecursiveFileAnalyzer is the method to get an unwrapped list of all the files in
# its directory recursively, as this is useful for multiprocessing or other scenareos where knowing the total
# number of files is helpful. The HistogramErrorDrawer is a utility class to house its Draw function, which
# draws a histogram and calculates its errors with astropy.stats.poisson_conf_interval

import astropy.stats
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from utils.logging import get_logger
from tqdm import tqdm
import re
import os
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import time
from functools import partial
from itertools import repeat   

# Utility functions for for_each
def get_fits_primaryhdu_data( path: Path ):
    with fits.open( str( path ), memmap=False ) as hdul:
        data = hdul[ 0 ].data
    # Get rid of leading 1s in shape, e.g. (1,1,n,n) -> (n,n), but preserve 2 dimensions for single pixel images
    while ( len( data.shape ) > 2 ) and ( data.shape[ 0 ] == 1 ):
        data = data[ 0 ]
    return data

def get_fits_primaryhdu_header( path: Path, key: str | None = None ):
    with fits.open( str( path ), memmap=False ) as hdul:
        if key is not None:
            header = hdul[ 0 ].header[ key ]
        else:
            header = hdul[ 0 ].header
    return header

class RecursiveFileAnalyzer:
    """
    A base class to act a function and return its value as a 1D list on all files (optionally matching an extension) under a given root directory

    If the OS environment variable 

    Parameters
    ----------
    path: Path | str
        The root directory to recursively search under
    log_level: int = logging.INFO
        The log level for the recursive file analyzer logger. When set to DEBUG, will log a message to the console when a directory is entered
        or a file read, with an index associated with each read file. Useful for slow operations to provide feedback on progress. Default logging.INFO
    """
    def __init__( self, path: Path | str, log_level: int = logging.INFO ):
        if path is not Path:
            path = Path( path )
        self.path = path
        self.logger = get_logger( self.__class__.__name__ )
        self.logger.setLevel( log_level )

    def get_unwrapped_list( self,
                            path: Path | str | None = None,
                            pattern: str | None = None,
                            numeric_range: tuple[int,int] | None = None,
                            return_nums: bool = False ):
        """A method to recursively unwrap all files in a directory, with optional regex pattern matching and numeric range filtering on the pattern capture group.

        Args:
            path (Path | str | None, optional): The path to scan. Defaults to None.
            pattern (str | None, optional): A regex pattern to filter files. Defaults to None.
            numeric_range (tuple[int,int] | None, optional): A range of numbers to filter files. Defaults to None.
            return_nums (bool, optional): Whether to return file numbers. Defaults to False.

        Returns:
            list (Path) | tuple(list[Path], list[int]) : A list of file paths, and optionally a list of file numbers if return_nums is True.
        """
        if return_nums:
            paths, idxs = map( list, zip( *self._quick_scan( path, pattern, numeric_range, return_nums ) ) )
            return paths, idxs
        else:
            paths = list( self._quick_scan( path, pattern, numeric_range ) )
            return paths
    
    def _quick_scan( self, 
            path: Path | str | None = None,
            pattern: str | None = None,
            numeric_range: tuple[int,int] | None = None,
            return_nums: bool = False):
        """
        A generator function to recursively scan through all files in a directory, 
        yielding the path of each file. If a regex pattern is provided, only files matching the pattern are yielded.

        :param path: The path to scan. If None, defaults to the root path of the RecursiveFileAnalyzer.
        :param pattern: A regex pattern to filter files. If None, all files are yielded. If provided, only files whose names match the pattern are yielded.
        :param numeric_range: Range on which to only return values if their index (from the capture group in pattern) is is within said range, or None to do no filtering
        :param return_nums: If True, returns a tuple of (file_path, file_number) for each file. The file_number is extracted from the file name using the first capture group in the regex pattern. If False, only the file_path is returned.
        :return: A generator yielding paths of files that match the criteria, and optionally their extracted numbers.
        """
        if return_nums:
            assert (pattern is not None), "If return_nums is True, a regex pattern must be provided to extract the numbers"
        
        if numeric_range:
            assert (pattern is not None), "If numeric_range is provided, a regex pattern must be provided to extract the numbers"
        
        if path is None:
            path = self.path
        
        with os.scandir(path) as it:
            for entry in it:
                # Use follow_symlinks=False to avoid infinite loops from circular symlinks.
                if entry.is_dir(follow_symlinks=False):
                    yield from self._quick_scan(entry.path, pattern=pattern, numeric_range=numeric_range, return_nums=return_nums)
                
                elif pattern is None or re.match(pattern, entry.name):
                    if return_nums:
                        # Extract file number using the first capture group in the regex pattern
                        idx = int( re.search( pattern, entry.name ).group( 1 ) )
                        if numeric_range is not None:
                            if idx < numeric_range[ 0 ] or idx >= numeric_range[ 1 ]:
                                continue
                        yield (entry.path, idx)
                    else:
                        # save some processing power by only doing regex match if numeric_range is provided
                        if numeric_range is not None:
                            idx = int( re.search( pattern, entry.name ).group( 1 ) )
                            if idx < numeric_range[ 0 ] or idx >= numeric_range[ 1 ]:
                                continue
                        yield entry.path

    def _batcher(self, iterable : list, batch_size : int):
        """
        Batches an iterable into chunks of a specified size.
        
        :param iterable: An iterable to be batched.
        :param batch_size: The size of each batch.
        :return: A generator yielding batches of the iterable.
        """
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _process_file(self, path : str | Path, function : Callable):
        """
        Processes a single file with the given function.
        
        :param path: The path to the file to be processed.
        :param function: The function (or partial) to apply to the file.
        :return: The result of applying the function to the file.
        """
        try:
            result = function(path)
            return result
        except Exception as e:
            self.logger.warning("Error processing %s: %s", path, e)
            return None

    def _process_batch(self, file_batch : list[str | Path], function : Callable):
        """     
        Processes a batch of files with the given function.
        
        :param file_batch: A list of file paths to be processed.
        :param function: The function (or partial) to apply to each file.
        :return: A list of results from applying the function to each file in the batch.
        """
        results = []
        for path in file_batch:
            try:
                results.append(self._process_file(path, function))
            except Exception as e:
                self.logger.warning("Error with %s: %s", path, e)
        return results

    def _run_file_mode(self,
                       file_paths : list[str | Path],
                       function : Callable,
                       num_workers : int,
                       output_file : str | Path | None = None,
                       show_progress : bool = True,
                       *args, **kwargs):
        """
        Process files by scheduling one file per task.
        
        :param file_paths: A list of file paths to process.
        :param function: The function to apply to each file.
        :param num_workers: The number of worker threads to use for concurrent processing.
        :param output_file: Optional path to a file where results will be written. If None, results are kept in memory.
        :param show_progress: Whether to display a tqdm progress bar.
        :return: A list of results if output_file is None, otherwise None.
        """
        results = []
        out_handle = open(output_file, "a") if output_file else None
        
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

        return results if not output_file else None

    def _run_batch_mode(self,
                        file_paths : list[str | Path],
                        num_workers : int,
                        batch_size : int,
                        function : Callable,
                        output_file : str | Path | None = None,
                        show_progress : bool = True,
                        *args, **kwargs):
        """
        Process files by scheduling one batch per task.
        
        :param file_paths: A list of file paths to process.
        :param num_workers: The number of worker threads to use for concurrent processing.
        :param batch_size: The number of files to process in each batch.
        :param function: The function to apply to each file.
        :param output_file: Optional path to a file where results will be written. If None, results are kept in memory.
        :param show_progress: Whether to display a tqdm progress bar.
        :return: A list of results if output_file is None, otherwise None.
        """
        results = []
        batches = list(self._batcher(file_paths, batch_size))
        out_handle = open(output_file, "a") if output_file else None
        
        # Create a partial function with the provided args and kwargs
        func_with_args = partial(function, *args, **kwargs)

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                self.logger.info("Processing %d files in %d batches with batch size %d using %d workers", len(file_paths), len(batches), batch_size, num_workers)
                iterator = executor.map(self._process_batch, batches, repeat(func_with_args))
                if show_progress:
                    iterator = tqdm(iterator, total=len(batches),
                        desc=(
                            "Processing files "
                            f"(batch mode, workers={num_workers}, batch size={batch_size})"
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

        return results if not output_file else None

    def run_pipeline(
        self,
        function : Callable,
        return_nums : bool = False,
        numeric_range: tuple[int,int] | None = None,
        root_dir : Path | str | None = None,
        pattern: str | None = r".*?\.fits$",
        batch_size : int = 500,
        num_workers : int = 8,
        output_file : str | Path | None = None,
        mode : str = "batch",
        show_progress : bool = True,
        file_paths_override : list[str | Path] | None = None,
        *args, **kwargs
    ):
        """
        Runs the complete pipeline to process files in a directory with the specified function, using either file mode or batch mode for concurrent processing.
        
        :param function: The function to apply to each file.
        :param return_nums: If True, returns a tuple of (file_path, file_number) for each file. The file_number is extracted from the file name using the first capture group in the regex pattern. If False, only the file_path is returned.
        :param numeric_range: Range on which to only return values if their index (from the capture group in pattern) is is within said range, or None to do no filtering
        :param root_dir: The root directory to scan for files to process.
        :param pattern: A regex pattern to filter files. Only files matching the pattern will be processed. Default is r".*?\.fits$" to match FITS files.
        :param batch_size: The number of files to process in each batch.
        :param num_workers: The number of worker threads to use for concurrent processing.
        :param output_file: Optional path to a file where results will be written. If None, results are kept in memory.
        :param mode: Either "file" (one file per task) or "batch" (one batch per task).
        :param show_progress: Whether to display tqdm progress bars.
        :param file_paths_override: Optional precomputed list of file paths.
        :param args: Additional positional arguments to pass to the processing function.
        :param kwargs: Additional keyword arguments to pass to the processing function.
        
        :return: A list of results if output_file is None, otherwise None. If return_nums is True, also returns a list of file numbers extracted from the file names.
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
            return_values = self._run_file_mode(
                file_paths=file_paths,
                function=function,
                num_workers=num_workers,
                output_file=output_file,
                show_progress=show_progress,
                *args, **kwargs
            )

        if mode == "batch":
            return_values = self._run_batch_mode(
                file_paths=file_paths,
                function=function,
                num_workers=num_workers,
                batch_size=batch_size,
                output_file=output_file,
                show_progress=show_progress,
                *args, **kwargs
            )

        return (return_values, numbers) if return_nums else return_values

    def benchmark_pipeline(
        self,
        function : Callable,
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
        *args, **kwargs
    ):
        """
        Benchmark throughput for different worker counts and batch sizes.

        Includes:
        - file mode: one file per task (batch_size recorded as None)
        - batch mode: one batch per task for each batch size in batch_size_options

        :param function: The function to apply to each file.
        :param return_nums: If True, returns a tuple of (file_path, file_number) for each file. The file_number is extracted from the file name using the first capture group in the regex pattern. If False, only the file_path is returned.
        :param root_dir: Root directory containing files to process.
        :param pattern: A regex pattern to filter files. Only files matching the pattern will be processed. Default is r".*?\.fits$" to match FITS files.
        :param numeric_range: Range on which to only return values if their index (from the capture group in pattern) is is within said range, or None to do no filtering
        :param worker_options: Iterable of worker counts to test.
        :param batch_size_options: Iterable of batch sizes to test in batch mode.
        :param sample_size: Number of files to benchmark (None for all files).
        :param repeats: Repetitions per config; best time is kept.
        :param output_csv: Optional path to write benchmark results as CSV.
        :param show_progress: Whether to display tqdm progress bars during benchmarking.
        :return: Tuple (all_rows, best_row)
        """
        if root_dir is None:
            root_dir = self.path
        
        file_paths = self.get_unwrapped_list(path=root_dir, 
                                            pattern=pattern,
                                            return_nums=return_nums,
                                            numeric_range=numeric_range)
        if return_nums:
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
                    root_dir=root_dir,
                    num_workers=workers,
                    mode="file",
                    show_progress=show_progress,
                    file_paths_override=file_paths,
                    function=function,
                    *args, **kwargs
                )
                elapsed = time.perf_counter() - t0
                if elapsed < best_seconds:
                    best_seconds = elapsed

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
                        root_dir=root_dir,
                        batch_size=batch_size,
                        num_workers=workers,
                        mode="batch",
                        show_progress=show_progress,
                        file_paths_override=file_paths,
                        function=function,
                        *args, **kwargs
                    )
                    elapsed = time.perf_counter() - t0
                    if elapsed < best_seconds:
                        best_seconds = elapsed

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
            with open(csv_path, "w", newline="") as f:
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

    def draw( self, data: np.ndarray, ax: plt.Axes, bins: int, range: tuple[ float, float ], label: str, color: str, density: bool, relative: bool ):
        """
        Utility function to draw a histogram with error bars according to astropy.stats.poisson_conf_interval with sigma=1.0

        Parameters
        ----------
        data: np.ndarray
            The data to plot
        ax: plt.Axes
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

        #poisson_conf_interval needs the raw data to be accurate, so we do it on the unweighted histogram and weight it afterward here
        if density:
            yerr = yerr / np.sum( data )
        elif relative:
            yerr = yerr / data.shape[ 0 ]

        ax.errorbar( bin_centres, drawn_histogram, yerr, fmt='.', color=color )
