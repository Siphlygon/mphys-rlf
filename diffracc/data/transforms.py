import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def safe_to_tensor(img: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Safely convert an image to a torch tensor, preserving its data type and range.

    Parameters
    ----------
    img : ndarray or torch.Tensor
        The input image, which can be a numpy array or a torch tensor.

    Returns
    -------
    torch.Tensor
        The image converted to a torch tensor, preserving its original data type and range.
    """
    if isinstance(img, torch.Tensor):
        return img
    return TF.to_tensor(img)


def ToTensor() -> T.Lambda:
    """
    Returns a torchvision transform that safely converts an image to a torch tensor.

    Returns
    -------
    torchvision.transforms.Lambda
        A transform that safely converts an image to a torch tensor.
    """
    return T.Lambda( safe_to_tensor )


def TrainTransformNoScale(image_size: int = 80) -> T.Compose:
    """
    Returns a torchvision transform for training that includes safe conversion to tensor,
    center cropping, single channel conversion, random horizontal and vertical flips,
    and random rotation by 90 degrees.
    """
    transform = T.Compose(
        [
            T.Lambda(safe_to_tensor),
            T.CenterCrop(image_size),
            T.Lambda(single_channel),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(random_rotate_90_tensor),
        ]
    )
    return transform


def single_channel(img: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Convert image to single channel if it has multiple channels.

    Parameters
    ----------
    img : ndarray or torch.Tensor
        The input image, which can be a numpy array or a torch tensor.

    Returns
    -------
    ndarray or torch.Tensor
        The image converted to a single channel, preserving its original data type and range.
    """
    if len(img.shape) == 3:
        return img[:1, :, :]

    if len(img.shape) == 2:
        return img.unsqueeze(0) if type(img) == torch.Tensor else img[None, :, :]

    # If the image is already single channel, return it as is
    return img


# def random_rotate_90(img: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
#     """
#     Randomly rotate an image by a multiple of 90 degrees.

#     Parameters
#     ----------
#     img : ndarray or torch.Tensor
#         The input image, which can be a numpy array or a torch tensor.

#     Returns
#     -------
#     ndarray or torch.Tensor
#         The rotated image, preserving its original data type and range.
#     """
#     angle = random.choice([0, 90, 180, 270])
#     img = TF.to_pil_image(img)
#     img = TF.rotate(img, angle)
#     return TF.to_tensor(img)  # type: ignore[arg-type]


def random_rotate_90_tensor(img: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Rotate by a multiple of 90 degrees using tensor ops.

    This is value-preserving (no interpolation) and safe for arbitrary float ranges, including negative values.
    
    Parameters
    ----------
    img : torch.Tensor or np.ndarray
        The input image, which can be a numpy array or a torch tensor.
    
    Returns
    -------
    torch.Tensor or np.ndarray
        The rotated image, preserving its original data type and range.
    """
    if not isinstance(img, torch.Tensor):
        img = safe_to_tensor(img)
    if img.ndim == 2:
        img = img.unsqueeze(0)
    k = random.choice([0, 1, 2, 3])
    return torch.rot90(img, k=k, dims=(-2, -1))
