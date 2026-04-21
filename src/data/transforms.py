import random
import torch

import torchvision.transforms as T
import torchvision.transforms.functional as TF

def safe_to_tensor(img):
    if isinstance(img, torch.Tensor):
        return img
    return TF.to_tensor(img)

def ToTensor():
    return T.Lambda( safe_to_tensor )

def TrainTransform(image_size):
    transform = T.Compose(
        [
            T.Lambda( safe_to_tensor ),
            T.CenterCrop(image_size),
            T.Lambda(single_channel),  # Only one channel
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(random_rotate_90),
            T.Lambda(train_scale),  # Scale to [-1, 1]
        ]
    )
    return transform


def EvalTransform(image_size):
    transform = T.Compose(
        [
            T.Lambda(single_channel),  # Only one channel
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.CenterCrop(image_size),
        ]
    )
    return transform


def single_channel(img):
    if len(img.shape) == 3:
        return img[:1, :, :]

    elif len(img.shape) == 2:
        return img.unsqueeze(0) if type(img) == torch.Tensor else img[None, :, :]


def train_scale(img):
    return img * 2 - 1


def minmax_scale(img):
    if img.max() == img.min():
        return torch.zeros_like(img)

    return (img - img.min()) / (img.max() - img.min())


def random_rotate_90(img):
    angle = random.choice([0, 90, 180, 270])
    img = TF.to_pil_image(img)
    img = TF.rotate(img, angle)
    return TF.to_tensor(img)
