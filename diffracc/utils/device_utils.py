import os
import nvidia_smi

import torch
import pandas as pd
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP




def physical_gpu_df():
    """
    Get a DataFrame with information about GPU memory usage. The DataFrame has the
    following columns:
    - memory.used (MiB)
    - memory.free (MiB)
    - memory.total (MiB)

    Returns
    -------
    gpu_df : pd.DataFrame
        DataFrame with GPU memory usage information.

    Raises
    ------
    RuntimeError
        If the device the software is running on has no GPUs.
    """
    # Get device count from nvidia-smi
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    # Raise error if no GPUs are found
    if not deviceCount:
        raise RuntimeError("No GPUs found")

    # Prepare dictionary to store memory information, will be used to create a
    # DataFrame
    memory_dict = {
        "memory.used (MiB)": [],
        "memory.free (MiB)": [],
        "memory.total (MiB)": [],
    }

    # Get memory information for each GPU
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        memory_dict["memory.used (MiB)"].append(mem.used / 1024**2)
        memory_dict["memory.free (MiB)"].append(mem.free / 1024**2)
        memory_dict["memory.total (MiB)"].append(mem.total / 1024**2)

    # Create DataFrame
    gpu_df = pd.DataFrame(memory_dict)

    return gpu_df


def visible_gpus_by_space(renumber=True):
    """
    Get a list of visible GPUs sorted by free memory.

    Parameters
    ----------
    renumber : bool, optional
        Whether to renumber the GPUs from 0 to N-1 for N visible devices,
        by default True.

    Returns
    -------
    list of int
        List of IDs for visible GPUs sorted by free memory.
    """

    # Get GPU memory information
    gpu_df = physical_gpu_df()

    # Sort by free memory
    gpu_df.sort_values(by="memory.free (MiB)", inplace=True, ascending=False)

    # Reduce to available GPUs by CUDA_VISIBLE_DEVICES (if set)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        vis = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        gpu_df = gpu_df.loc[vis]

        # Renumber if requested, i.e. reducing the index to the visible GPUs
        if renumber:
            gpu_df["renumbering"] = [vis.index(i) for i in gpu_df.index]
            gpu_df.set_index("renumbering", inplace=True)

    return list(gpu_df.index)


def distribute_model(model, n_devices=1, device_ids=None):
    """
    Distribute a model across multiple GPUs. If n_devices is 1, the model is
    moved to the first GPU in device_ids. If n_devices is greater than 1, the
    model is wrapped in a torch.nn.DataParallel object and distributed to the GPUs
    specified in device_ids.

    Parameters
    ----------
    model : nn.Module
        The model to distribute.
    n_devices : int, optional
        The number of devices to distribute the model to, by default 1.
    device_ids : list of int, optional
        The device IDs to distribute the model to, by default None.
        If None, the 'n_devices' GPUs with the most free memory are selected.

    Returns
    -------
    model : nn.Module or DataParallel
        The distributed model.
    device_ids : list of int
        The device IDs the model was distributed to.
    """
    device_ids = device_ids or visible_gpus_by_space()[:n_devices]
    if n_devices == 1:
        model.to(torch.device("cuda", device_ids[0]))

    else:
        model.to(torch.device("cuda", device_ids[0]))
        model = DDP(model, device_ids=device_ids)
    return model, device_ids


def set_visible_devices(gpu_spec):
    """
    Set the visible CUDA devices using the CUDA_VISIBLE_DEVICES environment variable.

    Parameters
    ----------
    gpu_spec : int or list of int
        The number of GPUs to use or a list of GPU IDs to use. If an integer is
        provided, the first 'n' GPUs with the most free memory are selected.

    Returns
    -------
    dev_ids : list of int
        The device IDs that were set as visible.

    Raises
    ------
    TypeError
        If gpu_spec is not an int or a list of ints.
    """
    match gpu_spec:
        case int():
            n_gpu = gpu_spec
            dev_ids = None
        case list():
            n_gpu = len(gpu_spec)
            dev_ids = gpu_spec
        case _:
            raise TypeError("gpu_spec must be int or list of ints")

    # Check that n_gpu is valid
    max_dev = len(physical_gpu_df())
    assert 0 <= n_gpu <= max_dev, f"n_gpu must be between 0 and {max_dev}"

    # Unset CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    # Set visible devices
    dev_ids = dev_ids or visible_gpus_by_space()[:n_gpu]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in dev_ids])
    return dev_ids
