from pathlib import Path

import h5py
import wget
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

import data.image_utils as imgutil
from utils.paths import cast_to_Path, MOSAIC_DIR, CUTOUTS_DIR, LOFAR_RES_CAT


def single_cutout(
    data,
    header,
    wcs,
    name,
    ra,
    dec,
    size_px=None,
    las=None,
    f_las=1.5,
    export_dir=None,
    mask_nan=True,
):
    """
    Create a cutout of the mosaic for a given source.
    """

    # Set size of cutout in pixels
    pixel_size = 1.5  # arcsec
    assert (size_px is not None) or (las is not None)
    if size_px is None:
        if las == 0.0:
            las = 15
        elif las < 10:
            las = 10
        size_px = round(f_las * las / pixel_size)  # In pixels

    # Create cutout
    center = SkyCoord(ra=[ra], dec=[dec], unit="deg")
    cutout = Cutout2D(data, center, size_px, wcs=wcs, mode="strict")

    # Check for NaNs, replace with nanmin if mask_nan is True
    if contain_nan := np.isnan(cutout.data).any():
        if mask_nan:
            print(
                f"Nan values present in {name} cutout.\n" " Nan values set to nanmin."
            )
            cutout.data[np.isnan(cutout.data)] = np.nanmin(cutout.data)

    # Export cutout as .fits file if export_dir is passed
    if export_dir is not None:
        header.update(cutout.wcs.to_header())
        fits.writeto(
            cast_to_Path(export_dir) / f"{name}.fits",
            cutout.data,
            header,
            overwrite=True,
        )

    return cutout.data, contain_nan


def cutout_from_catalog(catalog, ind, mask_nan=False, size_px=None, opt_c=True):
    """
    Create a cutout from the mosaic for a given source in catalog.
    """
    # Get mosaic data and WCS from fits file
    mosaic = catalog["Mosaic_ID"].iloc[ind]
    file_name = f"../data/mosaics_public/{mosaic}/mosaic-blanked.fits"
    with fits.open(file_name) as hdul:
        data, header = hdul[0].data, hdul[0].header
        wcs = WCS(header)
    name = catalog["Source_Name"].iloc[ind]

    # Get optical coordinates if desired/available,
    # otherwise use mean coordinates.
    if opt_c:
        ra = catalog["optRA"].iloc[ind]
        dec = catalog["optDec"].iloc[ind]
    if not opt_c or np.isnan(ra):
        ra = catalog["RA"].iloc[ind]
        dec = catalog["DEC"].iloc[ind]

    # Get LAS (largest angular size)
    las = catalog["LAS"].iloc[ind]
    las_type = catalog["LAS_from"].iloc[ind]  # for printing
    print(f"Source with index {ind} has a LAS of" f"{las:.3f} arcsec from {las_type}.")

    # Create cutout
    return single_cutout(
        data, header, wcs, name, ra, dec, las=las, mask_nan=mask_nan, size_px=size_px
    )


def get_cutouts(
    size_px=80,
    f=1.5,
    opt_c=True,
    fname_comment="",
    catalog=None,
    catalog_file=LOFAR_RES_CAT,
    export_dir=CUTOUTS_DIR,
    mosaic_dir=MOSAIC_DIR,
):
    """
    Create cutouts for all the sources in the catalogue.
    """
    # Check mosaic directory
    mosaic_dir = cast_to_Path(mosaic_dir)
    assert mosaic_dir.exists(), f"Mosaic directory {mosaic_dir} does not exist."
    assert len(mosaic_dir.glob("*.fits")), f"No mosaics found in {mosaic_dir}."

    # Create export directory
    export_dir = cast_to_Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Name for outfile:
    out_name = f"cutouts"
    out_name += f"_{size_px}p" if size_px is not None else f"_{f}las".replace(".", "p")
    out_name += "_optC" if opt_c else "_radioC"
    out_name += "_" * int(bool(len(fname_comment))) + fname_comment
    out_file = export_dir / f"{out_name}.hdf5"

    # Check if out_file already exists
    assert not out_file.exists(), f"{out_file} already exists. Aborted for safety."

    # Read catalog
    catalog = catalog if catalog is not None else pd.read_csv(catalog_file)

    # Create columns in catalog for problems,
    # will be set to True if there is a problem
    problem_cutouts = np.zeros(len(catalog), dtype=bool)
    nans_cutouts = np.zeros(len(catalog), dtype=bool)

    # Extract unique mosaic IDs from catalogue
    mosaic_ids = catalog["Mosaic_ID"].unique()
    catalog.set_index("Mosaic_ID", inplace=True)

    # Assert that all mosaics are present
    assert all(
        (mosaic_dir / f"{mosaic}/mosaic-blanked.fits").exists() for mosaic in mosaic_ids
    ), (
        "Not all mosaics are present in the mosaic directory."
        "Make sure to download all mosaics with the given catalogue."
        f"Missing mosaics: {set(mosaic_ids) - set(mosaic_dir.glob('*'))}"
    )

    # Loop through unique mosaic IDs in catalogue and collect images
    images = []
    for mosaic in tqdm(mosaic_ids, desc="Looping through mosaics..."):

        # Filter catalogue for sources in mosaic
        catalog_mosaic = catalog.loc[mosaic]
        if isinstance(catalog_mosaic, pd.Series):
            catalog_mosaic = pd.concat(
                [
                    pd.DataFrame([i], columns=[name])
                    for name, i in catalog_mosaic.items()
                ],
                axis=1,
            )

        # Assign variables from catalogue:
        # Names
        names = catalog_mosaic["Source_Name"].values
        # Coordinates
        if opt_c:
            ras_opt = catalog_mosaic["optRA"].values
            decs_opt = catalog_mosaic["optDec"].values
            ras_mean = catalog_mosaic["RA"].values
            decs_mean = catalog_mosaic["DEC"].values
            ras = np.where(np.isnan(ras_opt), ras_mean, ras_opt)
            decs = np.where(np.isnan(decs_opt), decs_mean, decs_opt)
        else:
            ras = catalog_mosaic["RA"].values
            decs = catalog_mosaic["DEC"].values
        # LAS (largest angular size)
        lass = catalog_mosaic["LAS"].values

        # Get mosaic data and WCS from fits file
        with fits.open(mosaic_dir / f"{mosaic}/mosaic-blanked.fits") as hdu:
            data = hdu[0].data
            header = hdu[0].header
        wcs = WCS(header)

        # Loop through sources in mosaic and create cutouts
        for name, ra, dec, las in zip(names, ras, decs, lass):

            # Boolean mask for accessing source entries in catalogue
            source_mask = catalog["Source_Name"] == name

            try:
                # Create cutout
                img_data, contain_nan = single_cutout(
                    data,
                    header,
                    wcs,
                    name,
                    ra,
                    dec,
                    size_px=size_px,
                    las=las,
                    f_las=f,
                    export_dir=None,
                )
                images.append(img_data)

                # Set Nans_cutout to True if there are NaNs
                nans_cutouts[source_mask] = contain_nan

            except Exception as exc:
                print(f"Problem with {name} cutout.\n{exc}")
                problem_cutouts[source_mask] = True

    # Update catalog
    print("Updating catalog...")
    catalog.reset_index(inplace=True)
    catalog["Problem_cutout"] = problem_cutouts
    catalog["Nans_cutout"] = nans_cutouts

    print("Saving images...")
    save_images_hpy5(
        np.array(images) if size_px is not None else images,
        out_file,
        dset_name="cutouts",
        src_names=None,
        attributes={
            "size_px": size_px,
            "f_las": f,
            "opt_c": opt_c,
            "comment": fname_comment,
        },
        dtype=h5py.vlen_dtype(images[0].dtype) if size_px is None else None,
    )

    print("Saving catalog... (this might take a while)")
    catalog.to_hdf(
        out_file,
        key="catalog",
        format="table",
        mode="a",
    )
    print("Done.")
    return catalog, out_file


def scan_cutouts(cutout_file):
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

    # Add broken flag
    print("Adding broken flag...")
    broken_flag = imgutil.broken_image_mask(images)
    catalog["Broken_cutout"] = False
    catalog.loc[catalog_flag, "Broken_cutout"] = broken_flag

    # Add edge relative max
    print("Adding edge relative max...")
    edge_relative_max = imgutil.edge_relative_max(images)
    catalog["Edge_max"] = 1
    catalog.loc[catalog_flag, "Edge_max"] = edge_relative_max

    # Add Sigma-SNR
    print("Adding Sigma-SNR...")
    sigma_snr_vals = imgutil.sigma_snr_set(images)
    catalog["Sigma_SNR"] = -1
    catalog.loc[catalog_flag, "Sigma_SNR"] = sigma_snr_vals

    # Update catalog in file
    print("Saving catalog...")
    catalog.to_hdf(
        cutout_file,
        key="catalog",
        mode="a",
    )


def save_images_hpy5(
    img_array, save_path, dset_name="images", src_names=None, attributes={}, dtype=None
):
    out_file = h5py.File(save_path, "a")

    # Variable length
    if h5py.check_vlen_dtype(dtype) is not None:
        img_dset = out_file.create_dataset(dset_name, (len(img_array),), dtype=dtype)
        shapes = []
        for i, img in enumerate(img_array):
            img_dset[i] = img.flatten()
            shapes.append(img.shape)
        img_dset.attrs["flattened"] = True
        out_file.create_dataset(
            f"{dset_name}_shapes",
            data=np.array(shapes),
        )

    # Fixed length
    else:
        img_dset = out_file.create_dataset(dset_name, data=img_array, dtype=dtype)

    # Add source names if passed:
    if src_names is not None:
        out_file.create_dataset(
            f"{dset_name}_names",
            data=src_names.astype(object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    # Add attributes:
    for key, value in attributes.items():
        img_dset.attrs[key] = value

    out_file.close()


def download_mosaics(
    mosaic_dir: Path = MOSAIC_DIR,
    catalog: pd.DataFrame | None = LOFAR_RES_CAT,
    mosaic_id: str | None = None,
    url_base: str | None = None,
):
    """
    Download the mosaics from the LoTSS DR2.
    """
    # Set default url_base for downloading mosaics
    url_base = url_base or "https://lofar-webdav.grid.surfsara.nl:2881/"

    # Create mosaic directory if it doesn't exist
    mosaic_dir = cast_to_Path(mosaic_dir)
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    # Get unique mosaic IDs from catalog if passed
    if catalog is not None:
        mosaic_ids = catalog["Mosaic_ID"].unique()
    # Else, set mosaic ID if passed
    elif mosaic_id is not None:
        assert isinstance(mosaic_id, str), "mosaic_id must be a string."
        mosaic_ids = [mosaic_id]
    # Else, raise error
    else:
        raise ValueError("Either catalog or mosaic_id must be passed.")

    # Loop through mosaic IDs
    print(f"Downloading {len(mosaic_ids)} mosaics...")
    for mosaic in tqdm(mosaic_ids):

        # Set mosaic file path
        mosaic_file = mosaic_dir / f"{mosaic}/mosaic-blanked.fits"
        mosaic_file.parent.mkdir(parents=True, exist_ok=True)

        # Download mosaic if it doesn't exist
        if not mosaic_file.exists():
            print(f"Downloading {mosaic}...")
            wget.download(
                url_base + f"{mosaic}/mosaic-blanked.fits",
                out=str(mosaic_file.parent),
            )

        # Print message if mosaic already exists
        else:
            print(f"{mosaic} already exists under {mosaic_file}.")

    return mosaic_ids


if __name__ == "__main__":
    get_cutouts()
