"""Unit tests for diffracc/model/config.py's ModelConfig."""
import json

import pytest

from diffracc.model.config import ModelConfig
from diffracc.utils import paths


class TestInit:
    """Unit tests for the `__init__` method of ModelConfig."""

    def test_kwargs_become_both_param_dict_and_attributes(self):
        """Test that keyword arguments passed to ModelConfig are stored in both param_dict and as attributes."""
        cfg = ModelConfig(a=1, b="two")
        assert cfg.param_dict == {"a": 1, "b": "two"}
        assert cfg.a == 1
        assert cfg.b == "two"

    def test_no_kwargs_gives_empty_param_dict(self):
        """Test that initializing ModelConfig with no keyword arguments results in an empty param_dict."""
        cfg = ModelConfig()
        assert cfg.param_dict == {}


class TestSetattr:
    """Unit tests for the `__setattr__` method of ModelConfig."""

    def test_setting_a_new_attribute_after_init_updates_param_dict(self):
        """
        Test that setting a new attribute on a ModelConfig instance after initialization updates the param_dict
        accordingly.
        """
        cfg = ModelConfig(a=1)
        cfg.c = 3
        assert cfg.param_dict["c"] == 3

    def test_overwriting_an_existing_attribute_updates_param_dict(self):
        """Test that overwriting an existing attribute on a ModelConfig instance updates the param_dict accordingly."""
        cfg = ModelConfig(a=1)
        cfg.a = 2
        assert cfg.param_dict["a"] == 2

    def test_param_dict_itself_is_not_added_as_a_key_of_param_dict(self):
        """Test that the param_dict attribute itself is not added as a key in the param_dict."""
        cfg = ModelConfig(a=1)
        assert "param_dict" not in cfg.param_dict


class TestUpdate:
    """Unit tests for the `update` method of ModelConfig."""

    def test_updates_both_param_dict_and_attributes(self):
        """Test that the update method updates both the param_dict and the corresponding attributes."""
        cfg = ModelConfig(a=1, b=2)
        cfg.update({"b": 20, "c": 3})
        assert cfg.param_dict == {"a": 1, "b": 20, "c": 3}
        assert cfg.b == 20
        assert cfg.c == 3


class TestFromPreset:
    """Unit tests for the `from_preset` method of ModelConfig."""

    def test_path_to_directory_looks_up_config_named_after_the_directory(self, tmp_path):
        """Test that providing a path to a directory looks up the config file named after the directory."""
        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        (model_dir / "config_mymodel.json").write_text(json.dumps({"a": 1}))

        cfg = ModelConfig.from_preset(model_dir)

        assert cfg.a == 1

    def test_path_to_file_is_used_directly(self, tmp_path):
        """Test that providing a path to a config file directly loads the configuration without errors."""
        config_file = tmp_path / "some_config.json"
        config_file.write_text(json.dumps({"a": 1, "b": 2}))

        cfg = ModelConfig.from_preset(config_file)

        assert cfg.param_dict == {"a": 1, "b": 2}

    def test_string_looks_up_model_configs_by_name(self, tmp_path, monkeypatch):
        """Test that providing a string looks up the model configuration in MODEL_CONFIGS by name."""
        config_file = tmp_path / "named_config.json"
        config_file.write_text(json.dumps({"a": 1}))
        monkeypatch.setattr(paths, "MODEL_CONFIGS", {"my_model_name": config_file})

        cfg = ModelConfig.from_preset("my_model_name")

        assert cfg.a == 1

    def test_invalid_type_raises_value_error(self):
        """Test that providing an invalid type (neither str nor Path) raises a ValueError."""
        with pytest.raises(ValueError):
            ModelConfig.from_preset(123)

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        """Test that providing a path to a non-existent file raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ModelConfig.from_preset(tmp_path / "does_not_exist.json")


class TestConstruct:
    """
    Tests for the `construct` method of ModelConfig, which instantiates a class using the configuration parameters.
    """

    class _Target:
        """A dummy target class for testing the construct method of ModelConfig."""
        def __init__(self, a, b=10, c=20):
            self.a = a
            self.b = b
            self.c = c

    def test_filters_param_dict_to_matching_constructor_kwargs(self):
        """
        Test that the construct method filters the param_dict to only include keys that match the target class's
        constructor parameters.
        """
        cfg = ModelConfig(a=1, b=2, unrelated_key="ignored")
        obj = cfg.construct(self._Target)
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 20  # not in param_dict, keeps its own default

    def test_positional_args_are_passed_through(self):
        """Test that positional arguments passed to construct are forwarded to the target class's constructor."""
        cfg = ModelConfig(b=99)
        obj = cfg.construct(self._Target, 5)  # a=5 positional
        assert obj.a == 5
        assert obj.b == 99

    def test_config_values_take_precedence_over_explicit_kwargs_for_matching_names(self):
        """
        Test that values from param_dict take precedence over explicitly passed keyword arguments for matching names.
        """
        # Documents actual precedence: construct() merges as `kwargs | config_kwargs`, so a param_dict entry
        # overrides an explicitly-passed kwarg of the same name.
        cfg = ModelConfig(b=99)
        obj = cfg.construct(self._Target, a=1, b=2)
        assert obj.b == 99
