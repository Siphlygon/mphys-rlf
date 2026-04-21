from copy import deepcopy

import torch
from torch.utils.data import DataLoader


def sample_sigmas(
    img_batch,
    P_mean=-1.2,
    P_std=1.2,
):
    """
    Sample noise levels from a log-normal distribution. used during training for
    adding noise to the input images.

    Parameters
    ----------
    img_batch : torch.Tensor
        Input image batch, used to infer shape.
    P_mean : float, optional
        log(mean) parameter for the log-normal distribution, by default -1.2
    P_std : float, optional
        log(std) parameter for the log-normal distribution, by default 1.2

    Returns
    -------
    _type_
        _description_
    """
    rnd_normal = torch.randn([img_batch.shape[0], 1, 1, 1], device=img_batch.device)
    sigmas = (rnd_normal * P_std + P_mean).exp()
    return sigmas


def edm_loss(
    model,
    img_batch,
    sigma_data=0.5,
    P_mean=-1.2,
    P_std=1.2,
    sigmas=None,
    noise=None,
    context=None,
    class_labels=None,
    return_output=False,
    mean=True,
):
    """
    Calculates the EDM (Expected Denoising MSE) loss between the denoised image and the original image.

    Parameters
    ----------
    model : nn.Module
        The denoising model used for denoising the image.
    img_batch : torch.Tensor
        The batch of input images to be denoised.
    sigma_data : float, optional
        The assumed standard deviation of the noise in the training data, by default 0.5.
    P_mean : float, optional
        The log-mean of the log-normal distribution used for sampling sigmas, by default -1.2.
    P_std : float, optional
        The log-standard deviation of the log-normal distribution used for sampling sigmas, by default 1.2.
    sigmas : torch.Tensor, optional
        The noise levels for each image in the batch, by default None.
        If None, they are sampled from a log-normal distribution.
    noise : torch.Tensor, optional
        The noise vector to be added to the input images, by default None.
        If None, it is sampled from a normal distribution with noise levels given by 'sigmas'.
    context : object, optional
        The context information for the denoising model, by default None.
    class_labels : object, optional
        The class labels for the input images, by default None.
    return_output : bool, optional
        Whether to return the denoised image along with the loss, by default False.
    mean : bool, optional
        Whether to compute the mean loss across the batch, by default True.

    Returns
    -------
    torch.Tensor or tuple
        If `return_output` is True, returns a tuple containing the loss and the denoised image.
        If `return_output` is False, returns only the loss.

    Raises
    ------
    AssertionError
        If `noise` is provided but `sigmas` is not provided.

    Notes
    -----
    The EDM loss is calculated as the weighted mean squared error between the denoised image and the original image.
    The weight coefficient for the loss is computed based on the noise levels and the standard deviation of the noise in the input images.
    The denoised image is obtained by adding the noise vector to the input images and passing them through the denoising model.
    """

    # Set noise vector
    if noise is not None:
        assert sigmas is not None, "If noise is provided, sigmas must be provided."
        n = noise
    else:
        sigmas = sigmas or sample_sigmas(img_batch, P_mean, P_std)
        n = torch.randn_like(img_batch) * sigmas

    # Weight coefficient for loss, as introduced in EDM paper
    weight = (sigmas**2 + sigma_data**2) / (sigmas * sigma_data) ** 2

    # Compute denoised image with forward model pass
    D_yn = model(img_batch + n, sigmas, context=context, class_labels=class_labels)

    # Compute loss
    loss = weight * (D_yn - img_batch) ** 2
    if mean:
        loss = loss.mean()

    return (loss, D_yn) if return_output else loss


class use_ema:
    """
    Context manager to temporarily use the EMA model during training.
    """

    def __init__(self, model, ema_model):
        """
        Initialize the context manager.

        Parameters
        ----------
        model : nn.Module
            The model to used during training.
        ema_model : nn.Module
            The EMA model from which parameters are temporarily copied
        """
        self.model = model
        self.ema_model = ema_model

    def __enter__(self):
        """
        Temporarily load the EMA model parameters into the model.
        """
        self.model_state = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema_model.module.state_dict())

    def __exit__(self):
        """
        Restore the original model parameters.
        """
        self.model.load_state_dict(self.model_state)


def get_power_ema_avg_fn(gamma):
    """
    Returns a function that computes the Power-EMA update for a given gamma.

    Parameters
    ----------
    gamma : float
        The exponent of the power-EMA update.

    Returns
    -------
    function
        The function that computes the power-EMA update.

    References
    ----------
    [1] https://arxiv.org/abs/2312.02696
    """

    @torch.no_grad()
    def ema_update(ema_param: torch.Tensor, current_param: torch.Tensor, num_averaged):
        """
        Compute the Power-EMA update for a given gamma.

        Parameters
        ----------
        ema_param : torch.Tensor
            Parameters of the EMA model.
        current_param : torch.Tensor
            Updated parameters of the model.
        num_averaged : int
            Number of averaging steps already made so far.

        Returns
        -------
        torch.Tensor
            Updated power-EMA parameters.
        """
        beta = (1 - 1 / num_averaged) ** (gamma + 1)
        return beta * ema_param + (1 - beta) * current_param

    return ema_update


def load_data(dataset, batch_size, shuffle=True):
    """
    Convenience function to continuously load data from a dataset. Will not stop
    until manually interrupted. A dataloader is created with the given dataset
    and batch size, and data is yielded from it. Basically, this returns an
    infinite DataLoader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to load data from

    batch_size : int
        The batch size to use for loading data
    shuffle : bool, optional
        Whether to shuffle the data, by default True

    Yields
    ------
    torch.Tensor or tuple
        The next batch of data from the dataset
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
    )
    while True:
        yield from loader
