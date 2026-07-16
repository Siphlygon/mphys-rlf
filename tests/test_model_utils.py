"""Unit tests for diffracc/model/model_utils.py."""
import json

import pytest
import torch
import torch.nn as nn

from diffracc.model import model_utils
from diffracc.model.unet import EDMPrecond
from diffracc.utils import paths

_TINY_CONFIG = {
    "init_channels": 32,  # divisible by 32, needed by ResidualLinearAttention's hardcoded GroupNorm(32, ...)
    "channel_mults": [1, 2],
    "image_channels": 1,
    "norm_groups": 1,
    "attention_levels": 0,
    "num_res_blocks": 1,
}


def _write_model_dir(root, model_name):
    """Helper to write a config_{name}.json (and return the dir) for a tiny, fast-constructing model."""
    model_dir = root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"config_{model_name}.json").write_text(json.dumps(_TINY_CONFIG))
    return model_dir


def _save_parameters(model_dir, model_name, model, key="model"):
    """Helpers for saving a model's state_dict to a parameters_{name}.pt file."""
    path = model_dir / f"parameters_{model_name}.pt"
    torch.save({key: model.state_dict()}, path)
    return path


class TestLoadModel:
    """Unit tests for the `load_model` function in diffracc.model.model_utils."""

    def test_path_to_config_file_is_used_directly(self, tmp_path):
        """Test that providing a path to a config file directly loads the model without errors."""
        config_file = tmp_path / "my_config.json"
        config_file.write_text(json.dumps(_TINY_CONFIG))
        model = model_utils.load_model(config_file, load_weights=False)
        assert isinstance(model, EDMPrecond)

    def test_path_to_file_with_wrong_suffix_raises_assertion_error(self, tmp_path):
        """Test that providing a path to a file with an unexpected suffix raises an AssertionError."""
        bad_file = tmp_path / "my_config.txt"
        bad_file.write_text("{}")
        with pytest.raises(AssertionError):
            model_utils.load_model(bad_file, load_weights=False)

    def test_path_to_directory_looks_up_config_by_directory_name(self, tmp_path):
        """Test that providing a path to a directory looks up the config file by the directory name."""
        model_dir = _write_model_dir(tmp_path, "mymodel")
        model = model_utils.load_model(model_dir, load_weights=False)
        assert isinstance(model, EDMPrecond)

    def test_string_source_looks_up_model_parent_by_default(self, tmp_path, monkeypatch):
        """Test that providing a string source looks up the model in MODEL_PARENT by default."""
        monkeypatch.setattr(paths, "MODEL_PARENT", tmp_path)
        _write_model_dir(tmp_path, "mymodel")
        model = model_utils.load_model("mymodel", load_weights=False)
        assert isinstance(model, EDMPrecond)

    def test_string_source_uses_pretrained_parent_when_requested(self, tmp_path, monkeypatch):#
        """Test that providing a string source with from_pretrained=True looks up the model in PRETRAINED_PARENT."""
        monkeypatch.setattr(paths, "PRETRAINED_PARENT", tmp_path)
        _write_model_dir(tmp_path, "mymodel")
        model = model_utils.load_model("mymodel", load_weights=False, from_pretrained=True)
        assert isinstance(model, EDMPrecond)

    def test_invalid_source_type_raises_value_error(self):
        """Test that providing an invalid source type raises a ValueError."""
        with pytest.raises(ValueError):
            model_utils.load_model(123, load_weights=False)

    def test_load_weights_false_skips_loading_state_dict(self, tmp_path):
        """
        Test that when load_weights=False, the function does not attempt to load a state_dict, even if no parameters
        file exists.
        """
        # No parameters_*.pt file exists at all - load_weights=False should never need it.
        model_dir = _write_model_dir(tmp_path, "mymodel")
        model = model_utils.load_model(model_dir, load_weights=False)
        assert isinstance(model, EDMPrecond)

    def test_return_config_true_also_returns_the_model_config(self, tmp_path):
        """Test that when return_config=True, the function returns both the model and its configuration."""
        model_dir = _write_model_dir(tmp_path, "mymodel")
        model, config = model_utils.load_model(model_dir, load_weights=False, return_config=True)
        assert isinstance(model, EDMPrecond)
        assert config.init_channels == 32

    def test_loads_weights_from_the_default_final_model_file(self, tmp_path):
        """Test that when load_weights=True, the function loads weights from the default parameters_{name}.pt file."""
        model_dir = _write_model_dir(tmp_path, "mymodel")
        reference = EDMPrecond.from_config(model_utils.ModelConfig.from_preset(model_dir / "config_mymodel.json"))
        _save_parameters(model_dir, "mymodel", reference, key="model")

        loaded = model_utils.load_model(model_dir, load_weights=True, key="model")

        torch.testing.assert_close(
            loaded.state_dict()["model.init_conv.weight"], reference.state_dict()["model.init_conv.weight"]
        )

    def test_snapshot_iter_loads_the_snapshot_file(self, tmp_path):
        """Test that when snapshot_iter is provided, the function loads weights from the corresponding snapshot file."""
        # Regression test for the `iter`-builtin-vs-`snapshot_iter` typo, which made this path always raise
        # TypeError before formatting the filename could even happen.
        model_dir = _write_model_dir(tmp_path, "mymodel")
        reference = EDMPrecond.from_config(model_utils.ModelConfig.from_preset(model_dir / "config_mymodel.json"))
        (model_dir / "snapshots").mkdir()
        snapshot_path = model_dir / "snapshots" / "snapshot_iter_00000005.pt"
        torch.save({"model": reference.state_dict()}, snapshot_path)

        loaded = model_utils.load_model(model_dir, load_weights=True, key="model", snapshot_iter=5)

        torch.testing.assert_close(
            loaded.state_dict()["model.init_conv.weight"], reference.state_dict()["model.init_conv.weight"]
        )

    def test_missing_snapshot_file_raises_file_not_found_error(self, tmp_path):
        """
        Test that when snapshot_iter is provided but the corresponding file does not exist, a FileNotFoundError is
        raised.
        """
        model_dir = _write_model_dir(tmp_path, "mymodel")
        with pytest.raises(FileNotFoundError):
            model_utils.load_model(model_dir, load_weights=True, key="model", snapshot_iter=99)


class TestLoadParameters:
    """Tests for the `load_parameters` function in diffracc.model.model_utils."""

    def _tiny_model(self):
        """Helper to construct a tiny model for testing."""
        return EDMPrecond.from_config(model_utils.ModelConfig(**_TINY_CONFIG))

    def test_key_model_loads_state_dict_unmodified(self, tmp_path):
        """Test that when key="model", the function loads the state_dict without modification."""
        reference = self._tiny_model()
        path = _save_parameters(tmp_path, "m", reference, key="model")
        target = self._tiny_model()

        model_utils.load_parameters(target, path, key="model")

        torch.testing.assert_close(target.state_dict()["model.init_conv.weight"],
                                   reference.state_dict()["model.init_conv.weight"])

    def test_ema_key_strips_module_prefix_when_present(self, tmp_path):
        """Test that when key="ema_model", the function strips the "module." prefix from state_dict keys if present."""
        reference = self._tiny_model()
        prefixed_state_dict = {f"module.{k}": v for k, v in reference.state_dict().items()}
        path = tmp_path / "parameters_m.pt"
        torch.save({"ema_model": prefixed_state_dict}, path)
        target = self._tiny_model()

        model_utils.load_parameters(target, path, key="ema_model")

        torch.testing.assert_close(target.state_dict()["model.init_conv.weight"],
                                   reference.state_dict()["model.init_conv.weight"])

    def test_ema_key_loads_state_dict_without_module_prefix(self, tmp_path):
        """Test that when key="ema_model", the function loads the state_dict without the "module." prefix."""
        reference = self._tiny_model()
        path = _save_parameters(tmp_path, "m", reference, key="ema_model")
        target = self._tiny_model()

        model_utils.load_parameters(target, path, key="ema_model")

        torch.testing.assert_close(target.state_dict()["model.init_conv.weight"],
                                   reference.state_dict()["model.init_conv.weight"])


class TestIsModel:
    """Tests for the `isModel` function in diffracc.model.model_utils."""

    class _OtherModel(nn.Module):
        """A dummy model class for testing purposes."""
        def forward(self, x):
            return x

    def test_true_for_matching_class(self):
        """Test that isModel returns True for an instance of the specified class."""
        model = nn.Linear(2, 2)
        assert model_utils.isModel(model, nn.Linear) is True

    def test_false_for_non_matching_class(self):
        """Test that isModel returns False for an instance of a different class."""
        model = nn.Linear(2, 2)
        assert model_utils.isModel(model, self._OtherModel) is False

    def test_true_for_dataparallel_wrapped_matching_class(self):
        """Test that isModel returns True for a DataParallel-wrapped instance of the specified class."""
        model = nn.DataParallel(nn.Linear(2, 2))
        assert model_utils.isModel(model, nn.Linear) is True

    def test_false_for_dataparallel_wrapped_non_matching_class(self):
        """Test that isModel returns False for a DataParallel-wrapped instance of a different class."""
        model = nn.DataParallel(nn.Linear(2, 2))
        assert model_utils.isModel(model, self._OtherModel) is False
