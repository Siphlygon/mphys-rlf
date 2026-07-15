import torch
from torch.nn.parallel import DistributedDataParallel as DDP


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
