"""
Copies configuration files for each model to the sampling directory.
"""
import shutil
import utils.paths as pth


def copy_configs():
    """Copies configuration files for each model to the sampling directory."""
    # Copy sample lofar/first config to sampling directory
    for name in pth.MODEL_NAMES:
        sampling_config_path = pth.MODEL_PARENT / f"{name}" / f"config_{name}.json"
        if not sampling_config_path.exists():
            shutil.copy(pth.CONFIG_PARENT / f"{name}.json", sampling_config_path)


if __name__ == '__main__':
    copy_configs()
