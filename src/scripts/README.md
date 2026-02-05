<h1>Namespace scripts</h1>

<hr>

The namespace scripts contains multiple python files that can be run as a main file to perform various tasks.

This namespace contains the following scripts:
 - completeness_estim.py
    - This scripts calls analysis.analysis.analyze_everything() to prepare the neccesary data to estimate the completeness correction. It then uses code helpfully provided by Wara to estimate the completeness correction, using residual images in place of random gaussian noise and generated images in place of the FIRST images.
 - fits_viewer.py
    - This script defines the FitsViewer class, which contains read_from_files() (called in \_\_init\_\_()) and show_image_grid(). When using as a class, the number of str arguments to show_image_grid should match the number of files stored int he instance of the class. This file can be run as a main file, and when done so it plots FITS images passed to it, using the filename as the plot title. Command line arguments when run as a main file are listed in --help. This program is NOT designed to be distributed across multiple nodes, but batching can be implemented outside of the program to draw one figure for each node, or utils.distributed can be used to execute this script on only one node.
 - flux_vs_residual.py
    - This script calls analysis.analysis.analyze_everything() to prepare the neccesary data for the flux vs residual scatterplot, then creates the scatterplot pairing gaus_resid image sums with the corresponding scaled fluxes and discarding items without a match. It is designed to be able to run on multiple nodes to do the analysis - the final scatterplot will be created only on the last node.
 - lofar_get_cutout_coords.py
    - This is a simple script which attempts to get a LOFAR cutout from the RA/DEC info in LOFAR_dataset.h5. The RA/DEC values, however, appear to give a different galaxy than the one in the dataset, and the cutout vs the dataset image are displayed with an os.system call to scripts.fits_viewer. The image number is currently controlled by a hardcoded parameter in the script <b>image_num</b>
 - plot_histograms.py
    - This script calls analysis.analysis.analyze_everything() to prepare the neccesary data for the histograms (refer to the analysis namespace README.md for details). It takes one command line argument (-v, --verbose) which controls the printing level - if verbose set log_level to logging.DEBUG otherwise logging.INFO. This script generates four histograms on a single figure - the flux histogram (arbitrary Jy units), the mean histogram (arbitrary mJy units), the rms histogram (arbitrary mJy units), and the pixel values histogram (0-1). Error bars are calculated using the analysis.HistogramErrorDrawer class, which uses astropy.stats.poisson_conf_interval(). It is designed to be able to run on multiple nodes for the analysis - the final histograms will be created only on the last node.
