"""Unit tests for diffracc/program_prep/copy_configs.py's copy_configs()."""
from diffracc.program_prep import copy_configs as cc


class TestCopyConfigs:
    """Tests for the copy_configs() function in copy_configs.py."""

    def _patch_paths(self, tmp_path, monkeypatch, model_names):
        """Helper function to monkeypatch paths in copy_configs.py to point to temporary directories."""
        config_parent = tmp_path / "configs"
        config_parent.mkdir()
        model_parent = tmp_path / "model_results"
        model_parent.mkdir()
        monkeypatch.setattr(cc.paths, "CONFIG_PARENT", config_parent)
        monkeypatch.setattr(cc.paths, "MODEL_PARENT", model_parent)
        monkeypatch.setattr(cc.paths, "MODEL_NAMES", model_names)
        return config_parent, model_parent

    def test_copies_config_when_missing(self, tmp_path, monkeypatch):
        """Test that copy_configs() copies the config file to the model folder when it doesn't already exist."""
        config_parent, model_parent = self._patch_paths(tmp_path, monkeypatch, ["LOFAR_model"])
        (config_parent / "LOFAR_model.json").write_text('{"key": "value"}', encoding="utf-8")
        (model_parent / "LOFAR_model").mkdir()

        cc.copy_configs()

        dest = model_parent / "LOFAR_model" / "config_LOFAR_model.json"
        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == '{"key": "value"}'

    def test_skips_copy_when_destination_already_exists(self, tmp_path, monkeypatch):
        """Test that copy_configs() does not overwrite the destination config file if it already exists."""
        # Deliberately don't create a source file at all - if the guard didn't work, this would raise
        # FileNotFoundError from shutil.copy.
        config_parent, model_parent = self._patch_paths(tmp_path, monkeypatch, ["LOFAR_model"])
        dest_dir = model_parent / "LOFAR_model"
        dest_dir.mkdir()
        dest = dest_dir / "config_LOFAR_model.json"
        dest.write_text("existing content", encoding="utf-8")

        cc.copy_configs()  # must not raise, must not overwrite

        assert dest.read_text(encoding="utf-8") == "existing content"

    def test_handles_each_model_name_independently(self, tmp_path, monkeypatch):
        """Test that copy_configs() handles multiple model names independently."""
        config_parent, model_parent = self._patch_paths(tmp_path, monkeypatch, ["LOFAR_model", "FIRST_model"])
        for name in ["LOFAR_model", "FIRST_model"]:
            (config_parent / f"{name}.json").write_text(f'{{"name": "{name}"}}', encoding="utf-8")
            (model_parent / name).mkdir()

        cc.copy_configs()

        for name in ["LOFAR_model", "FIRST_model"]:
            dest = model_parent / name / f"config_{name}.json"
            assert dest.exists()
            assert name in dest.read_text(encoding="utf-8")
