"""
Unit tests for diffracc/scripts/sample_snapshot_grid.py.

sample_grid is exercised end-to-end against a real (but tiny) EDMPrecond+Unet model via the make_tiny_model_dir
fixture - fast enough (~0.1-0.5s per call) that faking diffusion.edm_sampling/the network isn't worth the
indirection, and it means these tests also catch integration mistakes (wrong shapes/kwargs at the call boundary)
that a mocked model would hide.
"""
import pytest
import torch

from diffracc.scripts import sample_snapshot_grid as ssg
from diffracc.utils import paths


class TestFindSnapshot:
    """
    Unit tests for the `_find_snapshot` function in sample_snapshot_grid.py, which is responsible for locating the
    appropriate snapshot file in a model's snapshots directory.
    """

    def test_missing_snapshots_dir_raises_file_not_found(self, tmp_path):
        """Test that if the snapshots directory is missing, a FileNotFoundError is raised."""
        with pytest.raises(FileNotFoundError):
            ssg._find_snapshot(tmp_path / "nomodel", None)

    def test_empty_snapshots_dir_raises_file_not_found(self, tmp_path):
        """Test that if the snapshots directory exists but is empty, a FileNotFoundError is raised."""
        (tmp_path / "snapshots").mkdir()
        with pytest.raises(FileNotFoundError):
            ssg._find_snapshot(tmp_path, None)

    def test_none_picks_the_highest_iteration(self, tmp_path):
        """Test that if the iteration is None, the function picks the snapshot with the highest iteration number."""
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "snapshot_iter_00000005.pt").touch()
        (snap_dir / "snapshot_iter_00000020.pt").touch()
        (snap_dir / "snapshot_iter_00000010.pt").touch()

        result = ssg._find_snapshot(tmp_path, None)

        assert result.name == "snapshot_iter_00000020.pt"

    def test_specific_iteration_is_selected(self, tmp_path):
        """Test that if a specific iteration is provided, the function selects the corresponding snapshot."""
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "snapshot_iter_00000005.pt").touch()
        (snap_dir / "snapshot_iter_00000010.pt").touch()

        result = ssg._find_snapshot(tmp_path, 5)

        assert result.name == "snapshot_iter_00000005.pt"

    def test_missing_specific_iteration_raises_file_not_found(self, tmp_path):
        """Test that if a specific iteration is requested but does not exist, a FileNotFoundError is raised."""
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "snapshot_iter_00000005.pt").touch()

        with pytest.raises(FileNotFoundError):
            ssg._find_snapshot(tmp_path, 99)


class TestLoadModel:
    """
    Unit tests for the `_load_model` function in sample_snapshot_grid.py, which is responsible for loading a model's
    state_dict from a snapshot file.
    """

    def test_key_model_loads_state_dict_unmodified(self, make_tiny_model_dir, tmp_path):
        """Test that when the key is 'model', the state_dict is loaded without any modifications."""
        model_dir, reference = make_tiny_model_dir()
        snapshot_path = tmp_path / "snap.pt"
        torch.save({"model": reference.state_dict()}, snapshot_path)

        loaded, config = ssg._load_model(model_dir, snapshot_path, "model")

        torch.testing.assert_close(loaded.state_dict()["model.init_conv.weight"],
                                   reference.state_dict()["model.init_conv.weight"])
        assert loaded.training is False  # returned in eval mode

    def test_ema_key_strips_module_prefix_when_present(self, make_tiny_model_dir, tmp_path):
        """Test that when the key is 'ema_model', any 'module.' prefix in the state_dict keys is stripped."""
        model_dir, reference = make_tiny_model_dir()
        prefixed = {f"module.{k}": v for k, v in reference.state_dict().items()}
        snapshot_path = tmp_path / "snap.pt"
        torch.save({"ema_model": prefixed}, snapshot_path)

        loaded, _ = ssg._load_model(model_dir, snapshot_path, "ema_model")

        torch.testing.assert_close(loaded.state_dict()["model.init_conv.weight"],
                                   reference.state_dict()["model.init_conv.weight"])

    def test_ema_key_loads_state_dict_without_module_prefix(self, make_tiny_model_dir, tmp_path):
        """
        Test that when the key is 'ema_model', the state_dict is loaded correctly even if there is no 'module.'
        prefix.
        """
        model_dir, reference = make_tiny_model_dir()
        snapshot_path = tmp_path / "snap.pt"
        torch.save({"ema_model": reference.state_dict()}, snapshot_path)

        loaded, _ = ssg._load_model(model_dir, snapshot_path, "ema_model")

        torch.testing.assert_close(loaded.state_dict()["model.init_conv.weight"],
                                   reference.state_dict()["model.init_conv.weight"])

    def test_falls_back_to_model_when_ema_model_absent(self, make_tiny_model_dir, tmp_path, capsys):
        """
        Test that if the 'ema_model' key is absent, the function falls back to loading from the 'model' key and logs a
        message.
        """
        model_dir, reference = make_tiny_model_dir()
        snapshot_path = tmp_path / "snap.pt"
        torch.save({"model": reference.state_dict()}, snapshot_path)  # no "ema_model" key

        loaded, _ = ssg._load_model(model_dir, snapshot_path, "ema_model")

        torch.testing.assert_close(loaded.state_dict()["model.init_conv.weight"],
                                   reference.state_dict()["model.init_conv.weight"])
        assert "falling back to 'model'" in capsys.readouterr().out


@pytest.fixture
def patched_sample_grid_env(monkeypatch, tmp_path):
    """Fixture to patch the environment for sample_grid tests, ensuring MODEL_PARENT points to a temporary directory."""
    monkeypatch.setattr(paths, "MODEL_PARENT", tmp_path)
    return tmp_path


def _save_snapshot(model_dir, model, iteration=10):
    """Helper function to save a snapshot of the model's state_dict in the appropriate snapshots directory."""
    snap_dir = model_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"snapshot_iter_{iteration:08d}.pt"
    torch.save({"model": model.state_dict(), "ema_model": model.state_dict()}, path)
    return path


class TestSampleGrid:
    """
    Unit tests for the `sample_grid` function in sample_snapshot_grid.py, which generates a grid of samples from a
    model.
    """

    def test_runs_and_saves_expected_default_filename(self, make_tiny_model_dir, patched_sample_grid_env):
        """Test that sample_grid runs without errors and saves the output to the expected default filename."""
        model_dir, model = make_tiny_model_dir("tinymodel")
        _save_snapshot(model_dir, model, iteration=10)

        out_path = ssg.sample_grid("tinymodel", n=2, timesteps=2)

        assert out_path.exists()
        assert out_path.name == "sample_grid_ema_model_iter10_jy.png"

    def test_unconditional_model_uses_no_context(self, make_tiny_model_dir, patched_sample_grid_env):
        """Test that an unconditional model (no context) is built with context_dim=0 and runs without errors."""
        model_dir, model = make_tiny_model_dir("tinymodel")  # no "context" -> context_dim=0
        _save_snapshot(model_dir, model)
        assert model.model.context_dim == 0

        out_path = ssg.sample_grid("tinymodel", n=2, timesteps=2)
        assert out_path.exists()

    def test_conditional_model_builds_zero_context(self, make_tiny_model_dir, patched_sample_grid_env):
        """Test that a conditional model (with context) is built with context_dim=1 and runs without errors."""
        model_dir, model = make_tiny_model_dir("tinymodel", context=["max_values_tr"])
        _save_snapshot(model_dir, model)
        assert model.model.context_dim == 1

        out_path = ssg.sample_grid("tinymodel", n=2, timesteps=2)
        assert out_path.exists()

    def test_invert_no_leaves_raw_output_and_no_transform_message(self, make_tiny_model_dir, patched_sample_grid_env,
                                                                   capsys):
        """Test that when invert="no", the output is left in raw units and no transform message is printed."""
        model_dir, model = make_tiny_model_dir("tinymodel")
        _save_snapshot(model_dir, model)

        ssg.sample_grid("tinymodel", n=2, timesteps=2, invert="no")

        assert "Jy/beam (no transform)" in capsys.readouterr().out

    def test_invert_yes_without_recorded_transform_raises_value_error(self, make_tiny_model_dir,
                                                                       patched_sample_grid_env):
        """Test that when invert="yes" but no recorded transform is present, a ValueError is raised."""
        model_dir, model = make_tiny_model_dir("tinymodel")  # no flux_transform recorded
        _save_snapshot(model_dir, model)

        with pytest.raises(ValueError):
            ssg.sample_grid("tinymodel", n=2, timesteps=2, invert="yes")

    def test_invert_auto_applies_recorded_transform(self, make_tiny_model_dir, patched_sample_grid_env, capsys):
        """
        Test that when invert="auto" and a recorded transform is present, the transform is applied and a message is
        printed.
        """
        transform_dict = {"name": "linear", "k": 2.0}
        model_dir, model = make_tiny_model_dir("tinymodel", flux_transform=transform_dict)
        _save_snapshot(model_dir, model)

        ssg.sample_grid("tinymodel", n=2, timesteps=2, invert="auto")

        assert "transform inverted" in capsys.readouterr().out

    def test_explicit_out_path_is_respected(self, make_tiny_model_dir, patched_sample_grid_env, tmp_path):
        """Test that when an explicit out_path is provided, the output is saved to that path."""
        model_dir, model = make_tiny_model_dir("tinymodel")
        _save_snapshot(model_dir, model)
        custom_path = tmp_path / "custom_out.png"

        result = ssg.sample_grid("tinymodel", n=2, timesteps=2, out_path=custom_path)

        assert result == custom_path
        assert custom_path.exists()
