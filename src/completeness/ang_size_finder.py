from typing import Any
from pathlib import Path

from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os 
import logging
import utils.paths as paths
import utils.logging
from utils.recursive_file_analyzer import RecursiveFileAnalyzer
import time
import numpy as np
import matplotlib.pyplot as plt

class AngularSizeFinder:
    """
    A class to estimate the angular size of a radio galaxy image on a 80x80 grid.
    """
    def __init__(self):
        self.logger = utils.logging.get_logger("AngularSizeFinder", logging.DEBUG)
    
    def scan_dir(self, path):
        """
        Recursively scans a directory for .fits files.
        
        :param path: The root directory to start scanning from.
        :return: A generator yielding paths to .fits files.
        """
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    yield from self.scan_dir(entry.path)
                elif entry.name.endswith(".fits"):
                    yield entry.path

    def batcher(self, iterable, batch_size):
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

    def process_file(self, path):
        """
        Processes a single .fits file to estimate the index and  angular size of the source.
        
        :param path: The path to the .fits file to be processed.
        :return: A tuple containing the index and estimated angular size for the source in the file.
        """
        # Assuming filename format is image[index].fits, extract index
        # index = int(os.path.basename(path).split("image")[1].split(".fits")[0])
        index = int(os.path.basename(path)[5:-5]) # quicker
        data: Any = fits.getdata(path, ext=1, memmap=False)
        # return len(np.unique(data["Isl_id"]))
        return index

    def process_batch(self, file_batch):
        """     
        Processes a batch of .fits files to estimate angular sizes.
        """
        results = []
        for path in file_batch:
            try:
                results.append(self.process_file(path))
            except Exception as e:
                self.logger.warning("Error with %s: %s", path, e)
        return results

    def _run_file_mode(self, file_paths, num_workers, output_file=None, show_progress=True):
        """
        Process files by scheduling one file per task.
        """
        results = []
        out_handle = open(output_file, "a") if output_file else None

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                iterator = executor.map(self.process_file, file_paths)
                if show_progress:
                    iterator = tqdm(iterator, total=len(file_paths),
                        desc=f"Processing FITS files (file mode, workers={num_workers})",
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

    def _run_batch_mode(self, file_paths, num_workers, batch_size, output_file=None, show_progress=True):
        """
        Process files by scheduling one batch per task.
        """
        results = []
        batches = list(self.batcher(file_paths, batch_size))
        out_handle = open(output_file, "a") if output_file else None

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                iterator = executor.map(self.process_batch, batches)
                if show_progress:
                    iterator = tqdm(iterator, total=len(batches),
                        desc=(
                            "Processing FITS files "
                            f"(batch mode, workers={num_workers}, batch={batch_size})"
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
        root_dir,
        batch_size=100,
        num_workers=8,
        output_file=None,
        mode="file",
        show_progress=True,
        file_paths_override=None,
    ):
        """
        Runs the complete pipeline to estimate angular sizes from .fits files in a directory.
        
        :param root_dir: The root directory to scan for .fits files.
        :param batch_size: The number of files to process in each batch.
        :param num_workers: The number of worker threads to use for concurrent processing.
        :param output_file: Optional path to a file where results will be written. If None, results are kept in memory.
        :param mode: Either "file" (one file per task) or "batch" (one batch per task).
        :param show_progress: Whether to display tqdm progress bars.
        :param file_paths_override: Optional precomputed list of file paths.
        :return: A list of estimated angular sizes if output_file is None, otherwise None.
        """
        if file_paths_override is not None:
            self.logger.info("Using provided list of file paths with %d entries", len(file_paths_override))
            file_paths = file_paths_override
        else:
            file_paths = list(RecursiveFileAnalyzer(root_dir).quick_scan(pattern=r".*?\.fits$"))

        if mode == "file":
            return self._run_file_mode(
                file_paths=file_paths,
                num_workers=num_workers,
                output_file=output_file,
                show_progress=show_progress,
            )

        if mode == "batch":
            return self._run_batch_mode(
                file_paths=file_paths,
                num_workers=num_workers,
                batch_size=batch_size,
                output_file=output_file,
                show_progress=show_progress,
            )

        raise ValueError(f"Unsupported mode '{mode}'. Use 'file' or 'batch'.")

    def benchmark_pipeline(
        self,
        root_dir,
        worker_options=(8, 16, 24, 32),
        batch_size_options=(25, 50, 100, 250, 500),
        sample_size=5000,
        repeats=1,
        output_csv=None,
    ):
        """
        Benchmark throughput for different worker counts and batch sizes.

        Includes:
        - file mode: one file per task (batch_size recorded as None)
        - batch mode: one batch per task for each batch size in batch_size_options

        :param root_dir: Root directory containing FITS files.
        :param worker_options: Iterable of worker counts to test.
        :param batch_size_options: Iterable of batch sizes to test in batch mode.
        :param sample_size: Number of files to benchmark (None for all files).
        :param repeats: Repetitions per config; best time is kept.
        :param output_csv: Optional path to write benchmark results as CSV.
        :return: Tuple (all_rows, best_row)
        """
        file_paths = list(RecursiveFileAnalyzer(root_dir).quick_scan(pattern=r".*?\.fits$"))
        if sample_size is not None:
            file_paths = file_paths[:sample_size]

        if not file_paths:
            raise ValueError("No FITS files found for benchmarking.")

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
                    show_progress=True,
                    file_paths_override=file_paths,
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
                        show_progress=True,
                        file_paths_override=file_paths,
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

if __name__ == "__main__":
    root = paths.STORAGE_PARENT / "src/completeness/retrained_loguniform"
    # root = paths.DATASET_PARENT / "dr2_cutouts_download"
    ang_size_finder = AngularSizeFinder()
    
    # There is functionality to run a benchmark to find optimum parameters; however, on local testing,
    # there seems to be little difference between any such configurations
    run_benchmark = False

    if run_benchmark:
        rows, best = ang_size_finder.benchmark_pipeline(
            root_dir=root,
            worker_options=(8, 16, 24, 32),
            batch_size_options=(25, 50, 100, 250, 500),
            sample_size=10000,
            repeats=1,
            output_csv="src/completeness/benchmark_results.csv",
        )
        print("Top 5 configs:")
        for row in rows[:5]:
            print(row)
        print(f"Best config: {best}")
    else:
        results = ang_size_finder.run_pipeline(
            root_dir=root,
            batch_size=500,
            num_workers=8,
            output_file="src/completeness/results.txt",
            mode="batch",
        )
        
        # print(f"Processed {len(results)} files. Sample results: {results[:10]}")
        # # Create a histogram of results
        # plt.hist(results, bins=30, density=False)
        # plt.xlabel('(number of unique Island_id values)')
        # plt.ylabel('Frequency')
        # plt.title('Number of Islands According to PyBDSF')
        # plt.grid(True)
        # # plt.savefig('angular_size_histogram.png')
        # plt.show()

