import shutil
import utils.paths as pth

def copy_configs():
    # Copy sample lofar/first config to sampling directory
    for name in pth.NAMES:
        SAMPLING_CONFIG_PATH = pth.MODEL_PARENT / f"{name}_model" / f"config_{name}_model.json"
        if not SAMPLING_CONFIG_PATH.exists():
            shutil.copy(pth.CONFIG_PARENT / f"{name}_Model.json", SAMPLING_CONFIG_PATH)

if __name__ == '__main__':
    copy_configs()