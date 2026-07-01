from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from astropy.stats import sigma_clipped_stats


def clip_image(image, f_clip=1.5):
    """
    Clip the image to f_clip*stddev.
    """
    if f_clip is None:
        return np.array(image)

    elif f_clip == 0:
        return np.clip(image, 0, np.inf)

    _, _, stddev = sigma_clipped_stats(data=image, sigma=3.0, maxiters=10)
    image = np.clip(image, f_clip * stddev, np.inf)
    return image


def resize_image(image, size=128):
    """
    Resize the image to size*size.
    """
    if size == image.shape[-1]:
        return image
    image = resize(image, (size, size))
    return image


def minmax_scale(image):
    """
    Scale the image to [0, 1].
    """
    image = (image - image.min()) / (image.max() - image.min())
    return image


def sigma_mask(img, threshold=5):
    _, med, std = sigma_clipped_stats(img)
    return img > med + threshold * std


def sigma_snr(img, threshold=5):
    img = minmax_scale(img)
    mask = sigma_mask(img, threshold)

    if mask.sum() == 0:
        return sigma_snr(img, threshold - 0.5)

    if img[~mask].sum() == 0:
        print("No pixels below threshold.")
        return -1

    return img[mask].sum() / img[~mask].sum() * (~mask).sum() / mask.sum()


def sigma_snr_set(images, threshold=5):
    return np.array([sigma_snr(img, threshold) for img in tqdm(images, desc="\t")])


def check_nans(images, names):
    print("Checking for nans...")
    for i, img in tqdm(enumerate(images), desc="\t", total=len(images)):
        if np.isnan(img).any():
            print(f"\t{names[i]} has nans.")


def edge_max(img):
    return max(img[0].max(), img[-1].max(), img[1:-1, 0].max(), img[1:-1, -1].max())


def edge_relative_max(images):
    edge_max_arr = np.array([edge_max(img) for img in tqdm(images, desc="\t")])
    return edge_max_arr / images.max(axis=(1, 2))


def edge_threshold_mask(images, threshold=0.8):
    return edge_relative_max(images) > threshold


def broken_image_mask(images):
    # This will NOT work on clipped images!!
    images_collapsed = images.reshape(images.shape[0], -1)
    minvals = images_collapsed.min(axis=1)
    return (images_collapsed == np.expand_dims(minvals, 1)).sum(axis=1) > 1


def threshold_mask(
    catalog: pd.DataFrame,
    label: str,
    threshold: int | float | Sequence,
):
    """
    Create a mask for a catalog based on a threshold value. Returns a mask
    where True indicates that the source should be removed.

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog to use.
    label : str
        The column label to use.
    threshold : int | float | Sequence
        The threshold value(s) to use. If a sequence is provided, the mask will
        be created using the values in the sequence as thresholds. If a single
        value is provided, the mask will be created using that value as the
        threshold.

    Returns
    -------
    mask : pd.Series
        The mask for the catalog.
    """
    match threshold:
        case Sequence():
            mask = catalog[label].between(threshold[0], threshold[1])

        case int() | float() if (threshold > 0):
            mask = catalog[label] > threshold

        case int() | float() if (threshold == 0.0):
            print(f"\tZero {label} threshold, no cuts applied.")
            return pd.Series(False, index=catalog.index)

        case _:
            raise ValueError(f"Invalid {label} threshold value: {threshold}")

    print(f"\t{(~mask).sum():_} sources below {label} threshold.\n")
    return ~mask
