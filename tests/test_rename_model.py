"""
Unit tests for diffracc/scripts/rename_model.py.

All filesystem-only (no GPU/model), built against tmp_path so no real model_results/ folder is touched. Covers both
entry points the script is designed to handle; renaming a directory that hasn't moved yet, and fixing internal filenames
when the directory has already been moved (the exact scenario that motivated this script).
"""
import json

import pytest

from diffracc.scripts.rename_model import rename_model


def _make_model_dir(parent, name, extra_files=(), model_name_in_json=None):
    """Helper to create a minimal model results directory with a config JSON and the given extra templated files."""
    model_dir = parent / name
    model_dir.mkdir(parents=True)
    (model_dir / f"config_{name}.json").write_text(
        json.dumps({"model_name": model_name_in_json or name, "iterations": 5000}), encoding="utf-8")
    for fname in extra_files:
        (model_dir / fname).write_text("dummy", encoding="utf-8")
    return model_dir


class TestRenameModelDirectoryNotYetMoved:
    """Unit tests for the case where parent_dir/old_name still exists (the script performs the move itself)."""

    def test_directory_is_moved(self, tmp_path):
        """Test that the old-name directory is renamed to the new-name directory."""
        _make_model_dir(tmp_path, "old")
        result = rename_model("old", "new", parent_dir=tmp_path)
        assert result == tmp_path / "new"
        assert not (tmp_path / "old").exists()
        assert (tmp_path / "new").is_dir()

    def test_internal_files_are_renamed(self, tmp_path):
        """Test that templated files (parameters_, losses_train_, etc.) are renamed to the new name."""
        _make_model_dir(tmp_path, "old", extra_files=[
            "parameters_old.pt", "losses_train_old.csv", "losses_val_old.csv",
            "training_log_old.log", "power_ema_old.pt",
        ])
        rename_model("old", "new", parent_dir=tmp_path)
        new_dir = tmp_path / "new"
        for fname in ("config_new.json", "parameters_new.pt", "losses_train_new.csv",
                     "losses_val_new.csv", "training_log_new.log", "power_ema_new.pt"):
            assert (new_dir / fname).exists(), f"{fname} missing after rename"
        for fname in ("parameters_old.pt", "losses_train_old.csv"):
            assert not (new_dir / fname).exists(), f"{fname} should have been renamed away"

    def test_model_name_field_is_patched(self, tmp_path):
        """Test that the config JSON's model_name field is updated to the new name."""
        _make_model_dir(tmp_path, "old")
        rename_model("old", "new", parent_dir=tmp_path)
        with open(tmp_path / "new" / "config_new.json", encoding="utf-8") as f:
            assert json.load(f)["model_name"] == "new"

    def test_other_config_keys_are_preserved(self, tmp_path):
        """Test that non-model_name keys in the config JSON survive the patch untouched."""
        _make_model_dir(tmp_path, "old")
        rename_model("old", "new", parent_dir=tmp_path)
        with open(tmp_path / "new" / "config_new.json", encoding="utf-8") as f:
            assert json.load(f)["iterations"] == 5000

    def test_missing_optional_files_are_skipped(self, tmp_path):
        """Test that templated files that were never created (e.g. no power_ema) don't raise - they're just absent."""
        _make_model_dir(tmp_path, "old")  # no extra_files
        result = rename_model("old", "new", parent_dir=tmp_path)  # should not raise
        assert not (result / "power_ema_new.pt").exists()

    def test_snapshot_files_are_left_untouched(self, tmp_path):
        """Test that snapshot files don't embed the model name and must not be touched by the rename."""
        model_dir = _make_model_dir(tmp_path, "old")
        (model_dir / "snapshots").mkdir()
        (model_dir / "snapshots" / "snapshot_iter_00002000.pt").write_text("dummy", encoding="utf-8")
        rename_model("old", "new", parent_dir=tmp_path)
        assert (tmp_path / "new" / "snapshots" / "snapshot_iter_00002000.pt").exists()

    def test_wandb_run_id_file_is_left_untouched(self, tmp_path):
        """Test that wandb_run_id.txt doesn't embed the model name and must survive the rename unchanged."""
        model_dir = _make_model_dir(tmp_path, "old")
        (model_dir / "wandb_run_id.txt").write_text("abc123", encoding="utf-8")
        rename_model("old", "new", parent_dir=tmp_path)
        assert (tmp_path / "new" / "wandb_run_id.txt").read_text(encoding="utf-8") == "abc123"


class TestRenameModelDirectoryAlreadyMoved:
    """Unit tests for the exact scenario that motivated this script: the directory was already `mv`-ed by hand."""

    def test_only_internal_files_are_fixed(self, tmp_path):
        """
        Test that only internal files are fixed when parent_dir/new_name already exists (files still old-named inside).
        """
        _make_model_dir(tmp_path, "new", extra_files=["parameters_old.pt"], model_name_in_json="old")
        (tmp_path / "new" / f"config_old.json").write_text(
            json.dumps({"model_name": "old", "iterations": 100}), encoding="utf-8")
        (tmp_path / "new" / "config_new.json").unlink()  # only the old-named config should exist pre-fix

        result = rename_model("old", "new", parent_dir=tmp_path)

        assert result == tmp_path / "new"
        assert (tmp_path / "new" / "config_new.json").exists()
        assert (tmp_path / "new" / "parameters_new.pt").exists()
        with open(tmp_path / "new" / "config_new.json", encoding="utf-8") as f:
            assert json.load(f)["model_name"] == "new"


class TestRenameModelErrors:
    """Unit tests for rename_model's error handling."""

    def test_same_name_raises(self, tmp_path):
        """Test that renaming a model to its own name is a no-op request and should raise, not silently succeed."""
        _make_model_dir(tmp_path, "same")
        with pytest.raises(ValueError):
            rename_model("same", "same", parent_dir=tmp_path)

    def test_neither_directory_existing_raises(self, tmp_path):
        """Test that if neither the old nor new directory exists, there is nothing to rename."""
        with pytest.raises(FileNotFoundError):
            rename_model("ghost_old", "ghost_new", parent_dir=tmp_path)

    def test_both_directories_existing_raises(self, tmp_path):
        """Test that both old and new directories existing is ambiguous and must not be silently resolved either way."""
        _make_model_dir(tmp_path, "old")
        _make_model_dir(tmp_path, "new")
        with pytest.raises(FileExistsError):
            rename_model("old", "new", parent_dir=tmp_path)

    def test_clobbering_an_existing_target_file_raises(self, tmp_path):
        """Test that if both the old- and new-named version of a templated file exist, refuse to silently overwrite."""
        _make_model_dir(tmp_path, "old", extra_files=["parameters_old.pt", "parameters_new.pt"])
        with pytest.raises(FileExistsError):
            rename_model("old", "new", parent_dir=tmp_path)


class TestRenameModelDryRun:
    """Unit tests for the dry_run functionality."""

    def test_dry_run_makes_no_changes(self, tmp_path):
        """Test that with dry_run=True, no directory move, file rename, or JSON edit actually happens."""
        _make_model_dir(tmp_path, "old", extra_files=["parameters_old.pt"])
        rename_model("old", "new", parent_dir=tmp_path, dry_run=True)

        assert (tmp_path / "old").is_dir()
        assert not (tmp_path / "new").exists()
        with open(tmp_path / "old" / "config_old.json", encoding="utf-8") as f:
            assert json.load(f)["model_name"] == "old"
