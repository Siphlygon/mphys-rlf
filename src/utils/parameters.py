# This file has been created by Ashley and Luna. It defines the parameters used in the program,
# allowing easy tweaking for running on different devices. Specifically, it attempts to import utils.parameters_local_override
# to overwrite these values, a file included in .gitignore

local_overrides = True
try:
    import utils.parameters_local_override
except ImportError:
    local_overrides = False

DATASET_FITS_SUBDIR = "dataset"
DATASET_FITS_SUBDIR = getattr( utils.parameters_local_override, 'DATASET_FITS_SUBDIR', DATASET_FITS_SUBDIR ) if local_overrides else DATASET_FITS_SUBDIR

# Default bins sizes to sort data from the dataset into when converting h5 to fits
BINS_ARRAY = [ 10000 ]
BINS_ARRAY = getattr( utils.parameters_local_override, 'BINS_ARRAY', BINS_ARRAY ) if local_overrides else BINS_ARRAY

# Default parameters for fits file generation are placed here for ease of maintenance
FITS_SAMPLING_ARGS = {
    "batch_size": 100,
    "n_samples": 50000,
    "timesteps": 25,
    "bin_size": 10000,
}
FITS_SAMPLING_ARGS = getattr( utils.parameters_local_override, 'FITS_SAMPLING_ARGS', FITS_SAMPLING_ARGS ) if local_overrides else FITS_SAMPLING_ARGS

# Default number of fits images to convert, None converts all
LOFAR_FITS_COUNT_CUTOFF = None
LOFAR_FITS_COUNT_CUTOFF = getattr( utils.parameters_local_override, 'LOFAR_FITS_COUNT_CUTOFF', LOFAR_FITS_COUNT_CUTOFF ) if local_overrides else LOFAR_FITS_COUNT_CUTOFF


# Whether or not to clean the lofar fits images directory each run
CLEAN_LOFAR_FITS_IMAGES = False
CLEAN_LOFAR_FITS_IMAGES = getattr( utils.parameters_local_override, 'CLEAN_LOFAR_FITS_IMAGES', CLEAN_LOFAR_FITS_IMAGES ) if local_overrides else CLEAN_LOFAR_FITS_IMAGES