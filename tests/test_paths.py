"""Unit tests for diffracc/utils/paths.py's cast_to_path and rename_files."""
from pathlib import Path

import pytest

from diffracc.utils.paths import cast_to_path, rename_files


class TestCastToPath:
    """Tests for the cast_to_path function, which converts input to a Path object if necessary."""

    def test_path_input_is_returned_as_is(self):
        """Test that if the input is already a Path object, it is returned unchanged."""
        p = Path("some/path")
        assert cast_to_path(p) is p

    def test_string_input_is_converted_to_path(self):
        """Test that if the input is a string, it is converted to a Path object."""
        result = cast_to_path("some/path")
        assert isinstance(result, Path)
        assert result == Path("some/path")

    def test_other_types_raise_type_error(self):
        """Test that if the input is neither a Path nor a string, a TypeError is raised."""
        with pytest.raises(TypeError):
            cast_to_path(123)

    def test_none_raises_type_error(self):
        """Test that if the input is None, a TypeError is raised."""
        with pytest.raises(TypeError):
            cast_to_path(None)


class TestRenameFiles:
    """Tests for the rename_files function, which renames files in a directory based on a model name."""

    def test_renames_files_matching_directory_name_by_default(self, tmp_path):
        """Test that files containing the directory name are renamed to contain the new model name."""
        root = tmp_path / "old_model"
        root.mkdir()
        (root / "old_model_a.txt").touch()
        (root / "old_model_b.pt").touch()
        (root / "unrelated.txt").touch()

        rename_files(root, "new_model")

        names = {p.name for p in root.iterdir()}
        assert names == {"new_model_a.txt", "new_model_b.pt", "unrelated.txt"}

    def test_respects_explicit_old_model_name(self, tmp_path):
        """Test that the function respects an explicitly specified old model name."""
        root = tmp_path / "some_dir"
        root.mkdir()
        (root / "checkpoint_v1.pt").touch()

        rename_files(root, "v2", model_name_old="v1")

        names = {p.name for p in root.iterdir()}
        assert names == {"checkpoint_v2.pt"}

    def test_recurses_into_subdirectories_using_the_original_old_name(self, tmp_path):
        """Test that the function recurses into subdirectories and renames files using the original old model name."""
        root = tmp_path / "old_model"
        root.mkdir()
        (root / "old_model_top.txt").touch()
        sub = root / "subdir_with_different_name"
        sub.mkdir()
        (sub / "old_model_nested.txt").touch()

        rename_files(root, "new_model")

        assert (root / "new_model_top.txt").exists()
        assert (sub / "new_model_nested.txt").exists()

    def test_file_with_no_matching_substring_is_unchanged(self, tmp_path):
        """Test that files with no matching substring remain unchanged."""
        root = tmp_path / "old_model"
        root.mkdir()
        (root / "completely_different.txt").touch()

        rename_files(root, "new_model")

        assert (root / "completely_different.txt").exists()
