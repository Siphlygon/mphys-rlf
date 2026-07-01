import json
from typing import Any
from pathlib import Path
from inspect import signature

import utils.paths as paths


class ModelConfig(object):
    """
    A class representing the configuration for a model. The purpose of this
    class is to provide a single, flexible object that can be passed to a model
    constructor, rather than passing a large number of arguments. This class stores
    the arguments needed to construct a model in a dictionary, but also as attributes,
    inspired by pandas DataFrame.

    Args:
        **kwargs: Keyword arguments representing the configuration parameters.

    Attributes:
        param_dict (dict): A dictionary containing the configuration parameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.param_dict = kwargs
        self.__dict__.update(self.param_dict)

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        Set an attribute value.

        Args:
            __name (str): The name of the attribute.
            __value (Any): The value to be set.
        """
        super().__setattr__(__name, __value)
        if __name != "param_dict":
            self.param_dict[__name] = __value

    @classmethod
    def from_preset(self, preset: Path | str) -> "ModelConfig":
        match preset:

            # If path is a file, it is either the config file or its parent:
            case Path():
                # If path is a directory:
                if preset.is_dir():
                    config_file = preset / f"config_{preset.name}.json"

                # If path is the file itself, or unexisting (will raise error later):
                else:
                    config_file = preset

            # If path is a string, it is assumed to be a model name:
            case str():
                config_file = paths.MODEL_CONFIGS[preset]

            # Anything else is invalid.
            case _:
                raise ValueError(f"Invalid model identifier: {preset}")

        # Check if config file exists
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")

        # Load config file
        with open(config_file, "r") as f:
            config = json.load(f)

        return self(**config)

    def update(self, update_dict: dict) -> None:
        """
        Update the configuration parameters with a dictionary.

        Args:
            update_dict (dict): A dictionary containing the parameters to be updated.
        """
        self.param_dict.update(update_dict)
        self.__dict__.update(self.param_dict)

    def construct(self, cls: type, *args: Any, **kwargs: Any) -> Any:
        """
        Construct an object of a class using the configuration object.

        Args:
            cls (class): The class to be instantiated.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            An instance of the class constructed using the configuration object.
        """
        # Extract valid kwargs from hyperparams
        config_kwargs = {
            k: v
            for k, v in self.param_dict.items()
            if k in signature(cls).parameters.keys()
        }
        return cls(*args, **(kwargs | config_kwargs))
