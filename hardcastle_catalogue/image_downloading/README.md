# Downloading Source Files

The source files for this project come from the LOFAR data releases. Speciffically, we refer to data release 2 (DR2) and
the radio-optical crossmatching of DR2 sources by Hardcastle et al. (2023). The catalogue file provided on the server is downloaded,
and contains enough information to prompt the LOFAR cutout server and download every image individually, which is necessary
for the next stage of pre-processing. Not every image is used for the project - we only care about resolved sources in this catalogue.

The mechanics of this process is contained within `hardcastle_catalogue_creator.py`, which runs the necessary scripts described below.

This subdir contains scripts to:
1. Download the radio-optical crossmatching catalogue file from the server and prepare a list of coordinates for resolved sources: `hardcastle_catalogue_downloader.py`
2. Prompt the cutout server to download the images for these sources (~300k), and save them in a folder structure inside 'dr2_cutouts': `cutout_downloader.py`
3. Run a download verifying script, as some downloads can be corrupted or otherwise incomplete, and re-download any missing or corrupted files: `download_verifier.py`


## Links

[The LOFAR DR2 release page.](https://lofar-surveys.org/dr2_release.html)<br>
[The Hardcastle et al. (2023) paper describing the radio-optical crossmatching catalogue](https://arxiv.org/abs/2309.00102)<br>
[The resultant catalogue file described by the paper](https://lofar-surveys.org/public/DR2/catalogues/combined-release-v1.2-LM_opt_mass.fits)


## References
M. J. Hardcastle et al. The LOFAR Two-Metre Sky Survey: VI. Optical identifications for the second data release. Astronomy & Astrophysics, 678:A151, October 2023. ISSN 1432-0746. doi: 10.1051/0004-6361/202347333.