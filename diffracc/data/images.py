import numbers
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.paths import cast_to_Path
from data.cutouts import save_images_hpy5
from data.image_utils import clip_image, threshold_mask


def clip_cutouts(
    cutout_file: str | Path,
    f_clip: numbers.Number | list | tuple = 0,
):
    # Load cutouts and catalog from file
    print(f"Reading file {cutout_file.name}...")
    cutout_file = cast_to_Path(cutout_file)
    with h5py.File(cutout_file, "r") as f:
        images = np.array(f["cutouts"])
    catalog = pd.read_hdf(cutout_file, key="catalog")
    catalog_flag = ~catalog["Problem_cutout"]

    assert len(images) == sum(
        catalog_flag
    ), f"Number of images {len(images)} and catalog entries {sum(catalog_flag)} do not match."

    # Clip images
    if isinstance(f_clip, numbers.Number):
        f_clip = [f_clip]

    for clip in f_clip:
        images_clipped = []
        clip_flag = np.zeros(len(catalog[catalog_flag]), dtype=bool)
        sigma_snr_vals = -1 * np.ones(len(catalog[catalog_flag]), dtype=bool)

        # Loop through images for clipping and sigma-snr calculation
        for i, image in enumerate(tqdm(images, desc=f"Clipping {clip}-sigma...")):
            # Clipping
            try:
                image = clip_image(image, f_clip=clip)
                clip_flag[i] = True
                images_clipped.append(image)
            except Exception:
                pass

        # Write values into catalog
        print("Updating catalog...")
        catalog[f"Clipped_{clip}sigma"] = np.zeros(len(catalog), dtype=bool)
        catalog.loc[catalog_flag, f"Clipped_{clip}sigma"] = clip_flag

        # Save clipped images to same file as new dataset
        print("Saving images...")
        save_images_hpy5(
            images_clipped,
            cutout_file,
            dset_name=f"cutouts_{clip}sigma",
        )

    # Replace catalog in file
    print("Saving catalog...")
    catalog.to_hdf(
        cutout_file,
        key="catalog",
        mode="a",
    )
    print("Done.")


def create_subset(
    cutouts_file,
    subset_name,
    clip_sigma=None,
    las_thr=0.0,
    flux_thr=0.0,
    peak_flux=False,
    SNR_thr=5,
    edge_thr=0.8,
):
    # Read cutouts file
    img_container = h5py.File(cutouts_file, "r")

    # Specify dataset tag
    tag = f"cutouts_{clip_sigma}sigma" if clip_sigma is not None else "cutouts"

    # See if dataset is available. If not, clip images first to create.
    if tag not in img_container:
        # Close img container
        img_container.close()
        print(f"{tag} dataset not found. Clipping images...")
        clip_cutouts(cutouts_file, f_clip=clip_sigma)
        img_container = h5py.File(cutouts_file, "r")

    # Read images
    print(f"Reading images: {tag}...")
    images = np.array(img_container[tag])
    print(f"\t{len(images):_} images in catalog.\n")

    # Get the catalog
    print(f"Reading catalog...")
    catalog = pd.read_hdf(cutouts_file, key="catalog")
    if clip_sigma is not None:
        catalog = catalog[catalog[f"Clipped_{clip_sigma}sigma"]]
    else:
        catalog = catalog[~catalog["Problem_cutout"]]
    print(f"\t{len(catalog):_} sources in catalog.\n")
    n0 = len(catalog)

    print("Creating catalog masks...")

    # Problematic sources
    print("\tProblematic sources:")
    p_cols = [c for c in catalog.columns if "Problem" in c or "Nans" in c]
    p_cat = catalog[p_cols]
    print(*[f"\t{el}\n" for el in p_cat.sum().items()])
    problem_mask = p_cat.sum(axis=1).astype(bool)
    print(f"\t{problem_mask.sum():_} sources with problems.\n")

    # LAS Threshold
    print("\tLAS threshold:")
    las_mask = threshold_mask(catalog, "LAS", las_thr)
    print(f"\t{las_mask.sum():_} sources below threshold.\n")

    # Flux threshold
    print("\tFlux threshold:")
    flux_mask = threshold_mask(
        catalog, "Peak_flux" if peak_flux else "Total_flux", flux_thr
    )
    print(f"\t{flux_mask.sum():_} sources below threshold.\n")

    # SNR threshold
    SNR_mask = np.zeros(len(images), dtype=bool)
    if SNR_thr is not None:
        print("\tSNR threshold:")
        SNR_mask = catalog[f"Sigma_SNR"] < SNR_thr
        print(f"\t{SNR_mask.sum():_} sources below threshold.\n")

    # Edge threshold
    print("\tEdge pixel threshold:")
    edge_mask = catalog["Edge_max"] > edge_thr
    print(f"\t{edge_mask.sum():_} sources with edge pixels above threshold.\n")

    # Broken images
    print("\tBroken images:")
    broken_mask = catalog["Broken_cutout"]
    print(f"\t{broken_mask.sum():_} sources with broken image.\n")

    # Combine the masks to create the 'cleaned' catalog mask
    print("\tCombining masks...")
    subset_mask = ~(
        problem_mask | las_mask | flux_mask | SNR_mask | edge_mask | broken_mask
    )
    n_bad = (~subset_mask).sum()
    print(f"\t{n_bad:_} sources will be removed. {n0 - n_bad:_} sources in subset.\n")

    # Filter dataset
    print("Applying mask...")
    images_subset = images[subset_mask]
    catalog_subset = catalog[subset_mask]

    # Create the subset in new file
    print(f"Saving subset to file...")
    subset_file = Path(cutouts_file).parent.parent / f"subsets/{subset_name}.hdf5"
    with h5py.File(subset_file, "w") as f:
        img_dataset = f.create_dataset("images", data=images_subset)
        img_dataset.attrs["las_threshold"] = las_thr
        img_dataset.attrs["flux_threshold"] = flux_thr
        img_dataset.attrs["peak_flux"] = peak_flux
        img_dataset.attrs["processed_file"] = str(cutouts_file)
        processed_attrs = {
            key: value
            for dset in img_container.values()
            for (key, value) in dset.attrs.items()
        }
        img_dataset.attrs.update(processed_attrs)

    print("Saving catalog to file...")
    catalog_subset.to_hdf(subset_file, key="catalog", mode="a")

    print(f"Subset file created:\n\t{subset_file}\n")
    img_container.close()
