"""
Unit tests for diffracc/training/output_manager.py.

OutputManager is filesystem-only (no GPU/model), so these run everywhere. Every OutputManager is built against a
tmp_path parent directory so no real model_results/ folder is touched. Covered: model-name collision renaming, the
config JSON round-trip (save_config -> read_iter_count), loss-CSV headers/appending, checkpoint saving, and the
write_output guard rails.
"""
import csv
import json

import pytest
import torch
from torch import nn

from diffracc.training.output_manager import OutputManager


def _om(tmp_path, name="mymodel", **kwargs):
    """Build an OutputManager rooted at tmp_path with output writing on unless overridden."""
    kwargs.setdefault("write_output", True)
    return OutputManager(name, parent_dir=tmp_path, **kwargs)


def _read_csv(path):
    """Read all rows of a CSV. Files are opened with newline="" so no spurious blank rows appear on any platform."""
    with open(path, newline="") as f:
        return list(csv.reader(f))


class TestCheckRenameModel:
    """Tests for the model-name collision handling in _check_rename_model (via _setup_files)."""

    def test_fresh_name_is_kept(self, tmp_path):
        """Testing a new model name creates the directory and keeps the name as-is."""
        om = _om(tmp_path, name="fresh")
        assert om.model_name == "fresh"
        assert (tmp_path / "fresh").is_dir()

    def test_existing_name_gets_suffixed(self, tmp_path):
        """Testing a second model with the same name gets a _1 suffix."""
        _om(tmp_path, name="dup")
        second = _om(tmp_path, name="dup")
        assert second.model_name == "dup_1"

    def test_suffix_increments_past_existing_suffixes(self, tmp_path):
        """Testing a third collision skips the taken _1 and becomes _2 (single increment, no _1_1 nesting)."""
        _om(tmp_path, name="dup")
        _om(tmp_path, name="dup")
        third = _om(tmp_path, name="dup")
        assert third.model_name == "dup_2"

    def test_override_does_not_rename(self, tmp_path):
        """Testing with override=True the existing name is reused rather than renamed."""
        _om(tmp_path, name="keep")
        again = _om(tmp_path, name="keep", override=True)
        assert again.model_name == "keep"


class TestConfigRoundTrip:
    """Tests for save_config() / read_iter_count()."""

    def test_iterations_override_is_written_and_read_back(self, tmp_path):
        """Testing save_config(iterations=N) stores N under 'iterations', and read_iter_count returns it."""
        om = _om(tmp_path)
        om.save_config({"model_name": "mymodel", "lr": 1e-4}, iterations=1234)
        assert om.read_iter_count() == 1234

    def test_other_params_are_preserved(self, tmp_path):
        """Testing non-iteration params in the dict are written verbatim to the JSON."""
        om = _om(tmp_path)
        om.save_config({"model_name": "mymodel", "batch_size": 64, "effective_batch_size": 256}, iterations=1)
        with open(om.config_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["batch_size"] == 64
        assert saved["effective_batch_size"] == 256


class TestExpectedFilePaths:
    """
    Tests for expected_config_file()/expected_parameters_file() and read_iter_count()'s use of them.

    Note: under DDP, only the primary rank calls set_writing_status(True) (which runs _setup_files() and sets
    self.config_file/self.parameters_file). Every rank calls read_iter_count() / load_state() during pickup though, so a
    non-primary rank hit AttributeError: 'OutputManager' object has no attribute 'config_file'. These methods must work
    purely from model_name/parent_dir, with no dependency on _setup_files() having run.
    """

    def test_expected_config_file_available_without_setup_files(self, tmp_path):
        """Testing expected_config_file() works even when _setup_files() was never called (write_output=False)."""
        om = OutputManager("mymodel", parent_dir=tmp_path, write_output=False)
        assert om.expected_config_file() == tmp_path / "mymodel" / "config_mymodel.json"

    def test_expected_parameters_file_available_without_setup_files(self, tmp_path):
        """Testing expected_parameters_file() works even when _setup_files() was never called."""
        om = OutputManager("mymodel", parent_dir=tmp_path, write_output=False)
        assert om.expected_parameters_file() == tmp_path / "mymodel" / "parameters_mymodel.pt"

    def test_read_iter_count_works_without_setup_files(self, tmp_path):
        """Testing read_iter_count() succeeds on an OutputManager that never ran _setup_files() - the exact
        situation a non-primary DDP rank is in during pickup."""
        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        (model_dir / "config_mymodel.json").write_text(json.dumps({"iterations": 34000}), encoding="utf-8")

        om = OutputManager("mymodel", parent_dir=tmp_path, write_output=False)
        assert om.read_iter_count() == 34000

    def test_expected_paths_match_the_real_setup_files_paths(self, tmp_path):
        """Testing expected_config_file()/expected_parameters_file() agree with the paths _setup_files() actually
        uses, so primary and non-primary ranks resolve to the identical files."""
        om = _om(tmp_path, name="mymodel")  # write_output=True -> _setup_files() runs
        assert om.expected_config_file() == om.config_file
        assert om.expected_parameters_file() == om.parameters_file


class TestLossFiles:
    """Tests for loss-CSV initialisation and appending."""

    def test_create_initialises_headers(self, tmp_path):
        """Testing init_training_loop() (create path) writes the expected headers to both loss files."""
        om = _om(tmp_path)
        om.init_training_loop()
        with open(om.train_loss_file, newline="") as f:
            assert next(csv.reader(f)) == ["iteration", "loss"]
        with open(om.val_loss_file, newline="") as f:
            assert next(csv.reader(f)) == ["iteration", "loss", "ema_loss"]

    def test_write_train_losses_appends_rows(self, tmp_path):
        """Testing write_train_losses appends [iteration, loss] rows after the header."""
        om = _om(tmp_path)
        om.init_training_loop()
        om.write_train_losses([[1, 0.5], [2, 0.25]])
        # No blank-row filtering: OutputManager opens its CSVs with newline="", so the output is clean on every
        # platform. Asserting the exact rows here guards against that newline="" regressing.
        rows = _read_csv(om.train_loss_file)
        # write_train_losses routes through _write_losses (writerows), which serialises values as-is - no :.2e
        # formatting (that is only in the single-value _write_loss path).
        assert rows == [["iteration", "loss"], ["1", "0.5"], ["2", "0.25"]]

    def test_write_val_losses_appends_triples(self, tmp_path):
        """Testing write_val_losses appends [iteration, loss, ema_loss] rows, with no spurious blank rows."""
        om = _om(tmp_path)
        om.init_training_loop()
        om.write_val_losses([[5, 0.4, 0.3]])
        rows = _read_csv(om.val_loss_file)
        assert rows == [["iteration", "loss", "ema_loss"], ["5", "0.4", "0.3"]]

    def test_create_raises_if_output_file_exists(self, tmp_path):
        """Testing without override, init_training_loop_create refuses to clobber an existing output file."""
        om = _om(tmp_path)
        om.config_file.touch()  # a pre-existing output file
        with pytest.raises(FileExistsError):
            om.init_training_loop_create()


class TestSaveParams:
    """Tests for save_params()'s checkpoint contents."""

    def test_saves_model_ema_optimizer_keys(self, tmp_path):
        """
        Testing the checkpoint holds state dicts under model/ema_model/optimizer, with None passed through as None.
        """
        om = _om(tmp_path)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        om.save_params(model, None, optimizer)

        ckpt = torch.load(om.parameters_file, map_location="cpu", weights_only=False)
        assert set(["model", "ema_model", "optimizer"]).issubset(ckpt.keys())
        assert ckpt["ema_model"] is None
        assert "weight" in ckpt["model"]

    def test_power_ema_models_are_saved_under_gamma_keys(self, tmp_path):
        """Testing Power-EMA models are stored under power_ema_<gamma> keys."""
        om = _om(tmp_path)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        om.save_params(model, None, optimizer, power_ema_models=[nn.Linear(2, 2)], gammas=[6.94])

        ckpt = torch.load(om.parameters_file, map_location="cpu", weights_only=False)
        assert "power_ema_6.94" in ckpt


class TestWriteGuards:
    """Tests for the write_output guard rails."""

    def test_write_train_losses_asserts_when_output_disabled(self, tmp_path):
        """Testing a manager built with write_output=False must refuse to write losses."""
        om = OutputManager("noout", parent_dir=tmp_path, write_output=False)
        with pytest.raises(AssertionError):
            om.write_train_losses([[1, 0.5]])

    def test_set_writing_status_initialises_files(self, tmp_path):
        """Testing turning writing on after the fact runs the file setup that __init__ skipped."""
        om = OutputManager("later", parent_dir=tmp_path, write_output=False)
        assert not om.ready_to_write
        om.set_writing_status(True)
        assert om.ready_to_write
        assert (tmp_path / "later").is_dir()
