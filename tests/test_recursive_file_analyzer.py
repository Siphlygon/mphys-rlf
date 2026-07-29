"""Unit tests for diffracc/utils/recursive_file_analyzer.py."""
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from diffracc.utils.recursive_file_analyzer import (
    RecursiveFileAnalyzer,
    _pad_to_shape,
    _safe_call,
    _to_array,
    get_fits_primaryhdu_data,
    get_fits_primaryhdu_header,
)


class TestPadToShape:
    """Tests for the _pad_to_shape function, which pads smaller arrays to a target shape with NaNs."""

    def test_pads_with_nans_to_target_shape(self):
        """Test that a smaller array is padded to the target shape with NaNs."""
        arr = np.ones((2, 2))
        padded = _pad_to_shape(arr, (4, 4))
        assert padded.shape == (4, 4)
        np.testing.assert_array_equal(padded[:2, :2], 1.0)
        np.testing.assert_array_equal(padded[2:, :], np.nan)
        np.testing.assert_array_equal(padded[:, 2:], np.nan)

    def test_no_padding_needed_when_already_target_shape(self):
        """Test that an array that already matches the target shape is returned unchanged."""
        arr = np.ones((3, 3))
        padded = _pad_to_shape(arr, (3, 3))
        np.testing.assert_array_equal(padded, arr)

    def test_only_pads_dimensions_that_are_smaller(self):
        """
        Test that only dimensions smaller than the target shape are padded, while larger dimensions remain unchanged.
        """
        arr = np.ones((5, 2))
        padded = _pad_to_shape(arr, (5, 4))
        assert padded.shape == (5, 4)


class TestGetFitsPrimaryhduData:
    """Tests for the get_fits_primaryhdu_data function, which reads data from the primary HDU of a FITS file."""

    def test_reads_2d_data_unchanged(self, tmp_path):
        """Test that 2D data in a FITS file is read back unchanged."""
        data = np.arange(25, dtype=np.float32).reshape(5, 5)
        path = tmp_path / "img.fits"
        fits.PrimaryHDU(data=data).writeto(path)
        result = get_fits_primaryhdu_data(path)
        np.testing.assert_allclose(result, data)

    def test_strips_leading_singleton_dimensions(self, tmp_path):
        """Test that leading singleton dimensions are stripped from the data when read from a FITS file."""
        data = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
        path = tmp_path / "img.fits"
        fits.PrimaryHDU(data=data).writeto(path)
        result = get_fits_primaryhdu_data(path)
        assert result.shape == (5, 5)
        np.testing.assert_allclose(result, data[0, 0])

    def test_preserves_2d_minimum_even_if_leading_dim_is_1(self, tmp_path):
        """Test that a 2D array with a leading singleton dimension is preserved as 2D when read from a FITS file."""
        data = np.array([[7.0]], dtype=np.float32)
        path = tmp_path / "single_pixel.fits"
        fits.PrimaryHDU(data=data).writeto(path)
        result = get_fits_primaryhdu_data(path)
        assert result.shape == (1, 1)

    def test_nan_pads_when_shape_does_not_match_expected(self, tmp_path):
        """Test that the data is NaN-padded to the expected shape when it does not match."""
        data = np.ones((3, 3), dtype=np.float32)
        path = tmp_path / "small.fits"
        fits.PrimaryHDU(data=data).writeto(path)
        result = get_fits_primaryhdu_data(path, expected_shape=(5, 5))
        assert result.shape == (5, 5)
        np.testing.assert_array_equal(result[:3, :3], 1.0)
        np.testing.assert_array_equal(result[3:, :], np.nan)

    def test_matching_expected_shape_is_unchanged(self, tmp_path):
        """Test that an array with the expected shape is returned unchanged."""
        data = np.ones((5, 5), dtype=np.float32)
        path = tmp_path / "matches.fits"
        fits.PrimaryHDU(data=data).writeto(path)
        result = get_fits_primaryhdu_data(path, expected_shape=(5, 5))
        np.testing.assert_array_equal(result, data)


class TestGetFitsPrimaryhduHeader:
    """Tests for the get_fits_primaryhdu_header function, which reads the header from the primary HDU of a FITS file."""

    def test_returns_full_header_when_no_key(self, tmp_path):
        """Test that the full header is returned when no specific key is requested."""
        hdu = fits.PrimaryHDU(data=np.zeros((2, 2)))
        hdu.header["MYKEY"] = "hello"
        path = tmp_path / "h.fits"
        hdu.writeto(path)
        header = get_fits_primaryhdu_header(path)
        assert header["MYKEY"] == "hello"

    def test_returns_specific_key_value(self, tmp_path):
        """Test that the value of a specific key is returned when requested."""
        hdu = fits.PrimaryHDU(data=np.zeros((2, 2)))
        hdu.header["MYKEY"] = "hello"
        path = tmp_path / "h.fits"
        hdu.writeto(path)
        value = get_fits_primaryhdu_header(path, key="MYKEY")
        assert value == "hello"


class TestToArray:
    """Tests for the _to_array function, which converts a list of items into a numpy array."""

    def test_homogeneous_scalars_become_a_proper_array(self):
        """Test that a list of homogeneous scalar values is converted into a proper numpy array."""
        result = _to_array([1, 2, 3])
        assert result.dtype != object
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_homogeneous_same_shape_arrays_are_stacked(self):
        """Test that a list of homogeneous arrays with the same shape is stacked into a higher-dimensional array."""
        items = [np.zeros((2, 2)), np.ones((2, 2))]
        result = _to_array(items)
        assert result.shape == (2, 2, 2)

    def test_ragged_items_fall_back_to_object_dtype(self):
        """Test that a list of items with different shapes falls back to an object dtype array."""
        items = [np.zeros((2, 2)), np.zeros((3, 3))]
        result = _to_array(items)
        assert result.dtype == object
        assert result.shape == (2,)
        assert result[0].shape == (2, 2)
        assert result[1].shape == (3, 3)


def _touch_files(root: Path, *names: str):
    """Helper function to create empty files at the specified paths under the root directory."""
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()


class TestInit:
    """Tests for the initialisation of the RecursiveFileAnalyzer class."""

    def test_accepts_a_string_path_and_converts_to_path(self, tmp_path):
        """Test that a string path is accepted and converted to a Path object."""
        rfa = RecursiveFileAnalyzer(str(tmp_path))
        assert rfa.path == tmp_path
        assert isinstance(rfa.path, Path)


class TestGetUnwrappedList:
    """Tests for the get_unwrapped_list method of the RecursiveFileAnalyzer class."""

    def test_lists_all_files_recursively_with_no_pattern(self, tmp_path):
        """Test that all files are listed recursively when no pattern is provided."""
        _touch_files(tmp_path, "a.txt", "sub/b.txt", "sub/nested/c.txt")
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.get_unwrapped_list()
        assert len(result.paths) == 3
        assert result.numbers is None

    def test_filters_by_regex_pattern(self, tmp_path):
        """Test that only files matching the provided regex pattern are returned."""
        _touch_files(tmp_path, "image1.fits", "image2.fits", "notes.txt")
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.get_unwrapped_list(pattern=r".*\.fits$")
        assert len(result.paths) == 2

    def test_return_nums_extracts_and_sorts_by_capture_group(self, tmp_path):
        """Test that when return_nums=True, the numbers are extracted from the capture group and sorted."""
        _touch_files(tmp_path, "image10.fits", "image2.fits", "image1.fits")
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.get_unwrapped_list(pattern=r".*?(\d+)\.fits$", return_nums=True)
        assert list(result.numbers) == [1, 2, 10]
        assert [p.name for p in result.paths] == ["image1.fits", "image2.fits", "image10.fits"]

    def test_numeric_range_filters_matched_files(self, tmp_path):
        """Test that when a numeric range is provided, only files with numbers within that range are returned."""
        _touch_files(tmp_path, "image1.fits", "image5.fits", "image10.fits")
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.get_unwrapped_list(pattern=r".*?(\d+)\.fits$", return_nums=True, numeric_range=(2, 10))
        assert list(result.numbers) == [5]

    def test_return_nums_without_pattern_raises(self, tmp_path):
        """Test that if return_nums=True is specified without a pattern, an AssertionError is raised."""
        rfa = RecursiveFileAnalyzer(tmp_path)
        with pytest.raises(AssertionError):
            rfa.get_unwrapped_list(return_nums=True)

    def test_pattern_without_capture_group_raises_value_error_when_return_nums(self, tmp_path):
        """
        Test that if return_nums=True is specified with a pattern that has no capture group, a ValueError is raised.
        """
        _touch_files(tmp_path, "image1.fits")
        rfa = RecursiveFileAnalyzer(tmp_path)
        with pytest.raises(ValueError):
            rfa.get_unwrapped_list(pattern=r".*\.fits$", return_nums=True)

    def test_numeric_range_filters_without_return_nums(self, tmp_path):
        """
        Test that numeric_range can be used without return_nums, filtering based on the capture group but not returning
        the numbers.
        """
        _touch_files(tmp_path, "image1.fits", "image5.fits", "image10.fits")
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.get_unwrapped_list(pattern=r".*?(\d+)\.fits$", return_nums=False, numeric_range=(2, 10))
        assert [p.name for p in result.paths] == ["image5.fits"]
        assert result.numbers is None

    def test_numeric_range_without_capture_group_raises_value_error(self, tmp_path):
        """Test that if numeric_range is specified with a pattern that has no capture group, a ValueError is raised."""
        _touch_files(tmp_path, "image1.fits")
        rfa = RecursiveFileAnalyzer(tmp_path)
        with pytest.raises(ValueError):
            rfa.get_unwrapped_list(pattern=r".*\.fits$", return_nums=False, numeric_range=(0, 10))


class Test_Batcher:
    """Tests for the _batcher method of the RecursiveFileAnalyzer class, which splits an iterable into batches."""

    def _get_batcher(self, tmp_path):
        """Helper method to create a RecursiveFileAnalyzer instance and return its _batcher method."""
        return RecursiveFileAnalyzer(tmp_path)._batcher

    def test_splits_into_exact_batches(self, tmp_path):
        """
        Test that an iterable is split into batches of the specified size when the total number of items is divisible by
        the batch size.
        """
        batcher = self._get_batcher(tmp_path)
        batches = list(batcher(range(6), 2))
        assert batches == [[0, 1], [2, 3], [4, 5]]

    def test_includes_final_partial_batch(self, tmp_path):
        """Test that the final batch is included even if it has fewer items than the specified batch size."""
        batcher = self._get_batcher(tmp_path)
        batches = list(batcher(range(5), 2))
        assert batches == [[0, 1], [2, 3], [4]]

    def test_empty_iterable_yields_no_batches(self, tmp_path):
        """Test that an empty iterable yields no batches."""
        batcher = self._get_batcher(tmp_path)
        assert list(batcher([], 3)) == []

    def test_batch_size_one_yields_singletons(self, tmp_path):
        """Test that a batch size of one yields each item in its own batch."""
        batcher = self._get_batcher(tmp_path)
        assert list(batcher([1, 2], 1)) == [[1], [2]]


class TestProcessFileAndBatch:
    """Tests for the _process_file and _process_batch methods of the RecursiveFileAnalyzer class."""

    def test_process_file_returns_function_result(self, tmp_path):
        """Test that _process_file returns the result of applying the provided function to the file path."""
        rfa = RecursiveFileAnalyzer(tmp_path)
        assert rfa._process_file("x", lambda p: p.upper()) == "X"

    def test_process_file_returns_none_and_logs_on_exception(self, tmp_path, caplog):
        """Test that _process_file returns None and logs an error when the function raises an exception."""
        rfa = RecursiveFileAnalyzer(tmp_path)

        def _raise(p):
            raise ValueError("boom")

        result = rfa._process_file("x", _raise)
        assert result is None

    def test_process_batch_applies_function_to_each_and_preserves_order(self, tmp_path):
        """Test that _process_batch applies the function to each item and preserves the order of results."""
        rfa = RecursiveFileAnalyzer(tmp_path)
        results = rfa._process_batch([1, 2, 3], lambda x: x * 10)
        assert results == [10, 20, 30]

    def test_process_batch_returns_none_for_failing_items_without_stopping(self, tmp_path):
        """Test that _process_batch returns None for items where the function raises an exception, without stopping."""
        rfa = RecursiveFileAnalyzer(tmp_path)

        def _fn(x):
            if x == 2:
                raise ValueError("boom")
            return x

        results = rfa._process_batch([1, 2, 3], _fn)
        assert results == [1, None, 3]


class TestRunPipeline:
    """Tests for the run_pipeline method of the RecursiveFileAnalyzer class."""

    def _make_text_files(self, tmp_path, contents: dict[str, str]):
        """Helper method to create text files with specified contents in the temporary directory."""
        for name, content in contents.items():
            (tmp_path / name).write_text(content)

    def test_batch_mode_applies_function_to_every_matched_file(self, tmp_path):
        """Test that in batch mode, the function is applied to every matched file and results are returned."""
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2", "c.txt": "3"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=lambda p: int(p.read_text()), pattern=r".*\.txt$",
                                  mode="batch", progress_bar_desc=None)
        assert sorted(result.results.tolist()) == [1, 2, 3]
        assert result.numbers is None

    def test_file_mode_applies_function_to_every_matched_file(self, tmp_path):
        """Test that in file mode, the function is applied to every matched file and results are returned."""
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=lambda p: int(p.read_text()), pattern=r".*\.txt$",
                                  mode="file", progress_bar_desc=None)
        assert sorted(result.results.tolist()) == [1, 2]

    def test_return_nums_true_gives_results_aligned_with_sorted_numbers(self, tmp_path):
        """Test that when return_nums=True, the results are aligned with the sorted numbers extracted from filenames."""
        self._make_text_files(tmp_path, {"item2.txt": "b", "item1.txt": "a", "item3.txt": "c"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=lambda p: p.read_text(), pattern=r".*?(\d+)\.txt$",
                                  return_nums=True, mode="batch", num_workers=1, progress_bar_desc=None)
        assert list(result.numbers) == [1, 2, 3]
        assert list(result.results) == ["a", "b", "c"]

    def test_output_file_writes_results_instead_of_returning_them(self, tmp_path):
        """Test that when an output file is specified, results are written to the file instead of being returned."""
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        out_file = tmp_path / "out.log"
        result = rfa.run_pipeline(function=lambda p: int(p.read_text()), pattern=r".*\.txt$",
                                  mode="batch", output_file=out_file, progress_bar_desc=None)
        assert result.results.tolist() == []
        written = {line.strip() for line in out_file.read_text().splitlines()}
        assert written == {"1", "2"}

    def test_file_paths_override_bypasses_scanning(self, tmp_path):
        """
        Test that when file_paths_override is provided, the scanning step is bypassed and only those files are
        processed.
        """
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2", "c.txt": "3"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        override = [tmp_path / "a.txt", tmp_path / "b.txt"]
        result = rfa.run_pipeline(function=lambda p: int(p.read_text()), file_paths_override=override,
                                  mode="batch", progress_bar_desc=None)
        assert sorted(result.results.tolist()) == [1, 2]

    def test_file_paths_override_with_return_nums_raises(self, tmp_path):
        """
        Test that if file_paths_override is provided with return_nums=True, an AssertionError is raised since numbers
        cannot be extracted from arbitrary file paths.
        """
        rfa = RecursiveFileAnalyzer(tmp_path)
        with pytest.raises(AssertionError):
            rfa.run_pipeline(function=lambda p: p, file_paths_override=[tmp_path / "a.txt"], return_nums=True)

    def test_no_matching_files_raises_assertion_error(self, tmp_path):
        """Test that if no files match the pattern, an AssertionError is raised."""
        rfa = RecursiveFileAnalyzer(tmp_path)
        with pytest.raises(AssertionError):
            rfa.run_pipeline(function=lambda p: p, pattern=r".*\.doesnotexist$", progress_bar_desc=None)

    def test_invalid_mode_raises_assertion_error(self, tmp_path):
        """Test that if an invalid mode is specified, an AssertionError is raised."""
        self._make_text_files(tmp_path, {"a.txt": "1"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        with pytest.raises(AssertionError):
            rfa.run_pipeline(function=lambda p: p, pattern=r".*\.txt$", mode="not_a_mode", progress_bar_desc=None)

    def test_batch_mode_default_progress_bar_desc_still_returns_results(self, tmp_path):
        """Test that in batch mode, using the default progress_bar_desc still returns results correctly."""
        # progress_bar_desc="default" exercises both the default-description string and the tqdm-wrapped
        # iterator path in _run_batch_mode (run_pipeline's own default is None, which skips both).
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=lambda p: int(p.read_text()), pattern=r".*\.txt$", mode="batch",
                                  progress_bar_desc="default")
        assert sorted(result.results.tolist()) == [1, 2]

    def test_file_mode_default_progress_bar_desc_still_returns_results(self, tmp_path):
        """Test that in file mode, using the default progress_bar_desc still returns results correctly."""
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=lambda p: int(p.read_text()), pattern=r".*\.txt$", mode="file",
                                  progress_bar_desc="default")
        assert sorted(result.results.tolist()) == [1, 2]

    def test_output_file_writes_results_in_file_mode(self, tmp_path):
        """
        Test that when an output file is specified in file mode, results are written to the file instead of being
        returned.
        """
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        out_file = tmp_path / "out.log"
        result = rfa.run_pipeline(function=lambda p: int(p.read_text()), pattern=r".*\.txt$", mode="file",
                                  output_file=out_file, progress_bar_desc=None)
        assert result.results.tolist() == []
        written = {line.strip() for line in out_file.read_text().splitlines()}
        assert written == {"1", "2"}

    def test_output_file_with_default_progress_bar_desc_in_batch_mode(self, tmp_path):
        """
        Test that when an output file is specified in batch mode with the default progress_bar_desc, results are written
        to the file instead of being returned.
        """
        self._make_text_files(tmp_path, {"a.txt": "1", "b.txt": "2"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        out_file = tmp_path / "out.log"
        rfa.run_pipeline(function=lambda p: int(p.read_text()), pattern=r".*\.txt$", mode="batch",
                         output_file=out_file, progress_bar_desc="default")
        written = {line.strip() for line in out_file.read_text().splitlines()}
        assert written == {"1", "2"}


class TestSafeCall:
    """Tests for the module-level _safe_call helper, which wraps the mapped function inside a process worker."""

    def test_returns_function_result(self):
        """Test that _safe_call returns the result of applying the function to the path."""
        assert _safe_call(str.upper, "abc") == "ABC"

    def test_returns_none_on_exception(self):
        """Test that _safe_call swallows any exception and returns None, so one bad file cannot abort the run."""
        def _raise(_):
            raise ValueError("boom")

        assert _safe_call(_raise, "x") is None


class TestRunPipelineProcessMode:
    """
    Tests for run_pipeline's process mode.

    The mapped function must be picklable for the spawn-based ProcessPoolExecutor (and importable by the worker
    processes), so - unlike the thread-mode tests - these use importable module-level functions (os.path.getsize,
    get_fits_primaryhdu_data), not lambdas. os.path.getsize is convenient here because a file's byte count is a
    predictable, order-checkable integer.
    """

    def _make_sized_files(self, tmp_path, contents: dict[str, str]):
        """Create text files whose byte length equals len(content), so os.path.getsize is predictable."""
        for name, content in contents.items():
            (tmp_path / name).write_text(content)

    def test_applies_function_to_every_matched_file(self, tmp_path):
        """Test that in process mode the function is applied to every matched file and results are returned."""
        self._make_sized_files(tmp_path, {"a.txt": "1", "b.txt": "22", "c.txt": "333"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=os.path.getsize, pattern=r".*\.txt$",
                                  mode="process", num_workers=2, progress_bar_desc=None)
        assert sorted(result.results.tolist()) == [1, 2, 3]
        assert result.numbers is None

    def test_return_nums_gives_results_aligned_with_sorted_numbers(self, tmp_path):
        """Test that process-mode results stay aligned with the sorted numbers extracted from filenames."""
        # byte lengths are chosen to equal the number in each filename, so alignment is directly checkable.
        self._make_sized_files(tmp_path, {"item2.txt": "22", "item1.txt": "1", "item3.txt": "333"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=os.path.getsize, pattern=r".*?(\d+)\.txt$",
                                  return_nums=True, mode="process", num_workers=2, progress_bar_desc=None)
        assert list(result.numbers) == [1, 2, 3]
        assert list(result.results) == [1, 2, 3]

    def test_writes_to_output_file_instead_of_returning(self, tmp_path):
        """Test that in process mode, results are written to the output file instead of being returned."""
        self._make_sized_files(tmp_path, {"a.txt": "1", "b.txt": "22"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        out_file = tmp_path / "out.log"
        result = rfa.run_pipeline(function=os.path.getsize, pattern=r".*\.txt$", mode="process",
                                  num_workers=2, output_file=out_file, progress_bar_desc=None)
        assert result.results.tolist() == []
        written = {line.strip() for line in out_file.read_text().splitlines()}
        assert written == {"1", "2"}

    def test_returns_none_for_files_that_error_without_aborting(self, tmp_path):
        """Test that a file whose function raises yields None (via _safe_call) without killing the whole run."""
        fits.PrimaryHDU(data=np.ones((5, 5), dtype=np.float32)).writeto(tmp_path / "good.fits")
        (tmp_path / "bad.fits").write_text("this is not a FITS file")
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=get_fits_primaryhdu_data, pattern=r".*\.fits$",
                                  mode="process", num_workers=2, progress_bar_desc=None)
        results = list(result.results)
        assert sum(r is None for r in results) == 1
        assert any(getattr(r, "shape", None) == (5, 5) for r in results)

    def test_default_progress_bar_desc_still_returns_results(self, tmp_path):
        """Test that process mode with the default progress_bar_desc still returns results correctly."""
        self._make_sized_files(tmp_path, {"a.txt": "1", "b.txt": "22"})
        rfa = RecursiveFileAnalyzer(tmp_path)
        result = rfa.run_pipeline(function=os.path.getsize, pattern=r".*\.txt$", mode="process",
                                  num_workers=2, progress_bar_desc="default")
        assert sorted(result.results.tolist()) == [1, 2]
