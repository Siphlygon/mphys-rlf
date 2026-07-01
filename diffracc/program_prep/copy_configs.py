"""
Copies configuration files for each model to the sampling directory.
"""
import shutil

from ..utils import paths


def copy_configs():
    """Copies configuration files for each model to the sampling directory."""
    # Copy sample lofar/first config to sampling directory
    for name in paths.MODEL_NAMES:
        sampling_config_path = paths.MODEL_PARENT / f"{name}" / f"config_{name}.json"
        if not sampling_config_path.exists():
            shutil.copy(paths.CONFIG_PARENT / f"{name}.json", sampling_config_path)


if __name__ == '__main__':
    copy_configs()
