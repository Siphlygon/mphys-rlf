import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from ..analysis.log_analyzer import get_model_flux
from ..completeness.completeness_estimator import CompletenessEstimator
from ..completeness.size_binned_completeness import SizeBinnedCompleteness
from ..utils import paths
from ..utils.functions import erf01, richards01, sigmoid, sigmoid01
from ..utils.power_transform import PeakFluxPowerTransformer
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer, get_fits_primaryhdu_header

RMS_LOFAR = 95e-6 * 1e3
BEAM_WIDTH_LOFAR = (0.00166667, 0.00166667, 0.0)
BEAM_AREA_LOFAR = BEAM_WIDTH_LOFAR[ 0 ] * BEAM_WIDTH_LOFAR[ 1 ]


shimwell_data = np.array( [
    [ 0.20,   0.00000 ],
    [ 0.22,   0.015625 ],
    [ 0.24,   0.015625 ],
    [ 0.27,   0.03125 ],
    [ 0.30,   0.06250 ],
    [ 0.34,   0.12500 ],
    [ 0.38,   0.18750 ],
    [ 0.42,   0.28125 ],
    [ 0.46,   0.34375 ],
    [ 0.52,   0.46875 ],
    [ 0.58,   0.53125 ],
    [ 0.64,   0.62500 ],
    [ 0.72,   0.71875 ],
    [ 0.80,   0.78125 ],
    [ 0.88,   0.81250 ],
    [ 0.98,   0.87500 ],
    [ 1.10,   0.90625 ],
    [ 1.19,   0.96875 ],
    [ 1.25,   1.00000 ],
] ).transpose()


dejong_data = np.array( [
    [ 1, 0.025 ],
    [ 2, 0.1 ],
    [ 3, 0.15 ],
    [ 4, 0.2 ],
    [ 7, 0.3 ],
    [ 10, 0.35 ],
    [ 20, 0.4 ],
    [ 30, 0.425 ],
    [ 40, 0.45 ],
    [ 100, 0.475 ],
    [ 300, 0.6 ],
    [ 1000, 0.7 ],
    [ 10000, 0.8 ],
    [ 20000, 0.8 ] 
]).transpose()

kondapally_data = np.array( [
    [ 0.09,	0.143,	0  ,	0 ],
    [ 0.11,	0.221,	0  ,	0 ],
    [ 0.13,	0.351,	0.193,	0 ],
    [ 0.16,	0.553,	0.319,	0 ],
    [ 0.19,	0.68,	0.43,	0.211 ],
    [ 0.23,	0.742,	0.57,	0.322 ],
    [ 0.28,	0.778,	0.664,	0.486 ],
    [ 0.33,	0.802,	0.751,	0.624 ],
    [ 0.40,	0.813,	0.779,	0.717 ],
    [ 0.63,	0.849,	0.841,	0.816 ],
    [ 1.01,	0.884,	0.872,	0.871 ],
    [ 1.59,	0.921,	0.907,	0.912 ],
    [ 2.52,	0.951,	0.941,	0.952 ],
    [ 4.00,	0.961,	0.959,	0.971 ],
    [ 6.34,	0.973,	0.967,	0.982 ],
    [ 10.05,0.975,	0.974,	0.981 ],
    [ 15.92,0.978,	0.973,	0.98 ],
    [ 25.24,0.984,	0.978,	0.989 ],
    [ 40.0,	0.987,	0.978,	0.985 ],
]).transpose()


def plot_completeness(config_str: str,
                      which_dataset: str = 'GENERATED_SUBDIR',
                      override_data: bool = True):
    """
    Plot the completeness curve based on the provided configuration string. If the completeness estimates file does not
    exist, it will be created using the CompletenessEstimator class.

    Parameters
    ----------
    config_str : str
        The configuration string for the completeness estimator.
    which_dataset : str, optional
        The dataset to use for the completeness estimation, by default 'GENERATED_SUBDIR'
    override_data : bool, optional
        Whether to override existing data, by default True
    """
    # Check if the completeness estimate file exists, and if not, create it
    if not os.path.exists( 'completeness_estimate.csv' ):
        estimator = CompletenessEstimator(config_str, which_dataset=which_dataset, override_data=override_data)
        root = paths.STORAGE_PARENT / "diffracc/completeness/"

        model_images_path = root / 'model_images.npy'
        peak_fluxes_path = root / 'peak_fluxes.npy'
        model_fluxes_path = root / 'model_fluxes.npy'

        all_exist = model_images_path.exists() and peak_fluxes_path.exists() and model_fluxes_path.exists()

        if all_exist:
            model_fluxes = np.load( root / 'model_fluxes.npy' )
            peak_fluxes = np.load( root / 'peak_fluxes.npy' )
            model_images = np.load( root / 'model_images.npy' )

        else:
            if which_dataset == 'GENERATED_SUBDIR':
                path_to_subdir = paths.NP_ARRAY_PARENT / estimator.config["GENERATED_SUBDIR"]
            else:
                path_to_subdir = paths.NP_ARRAY_PARENT / estimator.config["DATASET_SUBDIR"]

            folder_name = "snr15_loguniform_nolas"
            paths_to_use=[root / (folder_name + "_fits_images"),
                         root / (folder_name + "_images/gaus_model"),
                         root / (folder_name + "_logs")]

            maxvals = np.load( root / 'maxvals.npy' )

            #paths_to_use = [paths.PYBDSF_CATALOG_PARENT / folder_name,
            #                 paths.PYBDSF_EXPORT_IMAGE_PARENT / folder_name / "gaus_model",
            #                 paths.PYBDSF_LOG_PARENT / folder_name]

            # Get model images
            def read_model_images(path: Path):
                return fits.getdata(path, 0)

            rfa = RecursiveFileAnalyzer(paths_to_use[0])

            image_files, mi_indices = rfa.get_unwrapped_list(path=paths_to_use[1],
                                                             pattern=r'.*?\D+(\d+)\.fits$',
                                                             return_nums=True)
            fits_files, fi_indices = rfa.get_unwrapped_list(path=paths_to_use[0],
                                                            pattern=r'.*?\D+(\d+)\.fits$',
                                                            return_nums=True)
            log_files, mf_indices = rfa.get_unwrapped_list(path=paths_to_use[2],
                                                           pattern=r'.*?\D+(\d+)\.fits.pybdsf.log$',
                                                           return_nums=True)

            image_files_pt1 = image_files[:len(image_files) // 2]
            image_files_pt2 = image_files[len(image_files) // 2:]
            log_files_pt1 = log_files[:len(log_files) // 2]
            log_files_pt2 = log_files[len(log_files) // 2:]
            fits_files_pt1 = fits_files[ :len(fits_files) // 2 ]
            fits_files_pt2 = fits_files[ len(fits_files) // 2: ]

            print( 'getting peak fluxes...' )

            def get_fxscld( path ):
                return get_fits_primaryhdu_header( path, 'FXSCLD' )

            peak_fluxes_tr_pt1 = rfa.run_pipeline( function=get_fxscld,
                                                  mode="file",
                                                  file_paths_override=fits_files_pt1,
                                                  show_progress=True).results

            print( peak_fluxes_tr_pt1 )
            peak_fluxes_tr_pt2 = rfa.run_pipeline( function=get_fxscld,
                                                  mode="file",
                                                  file_paths_override=fits_files_pt2,
                                                  show_progress=True).results
            print( peak_fluxes_tr_pt2 )

            print("Getting model images...")
            model_images_pt1 = rfa.run_pipeline(function=read_model_images,
                                                mode="file",
                                                file_paths_override=image_files_pt1,
                                                show_progress=True).results
            model_images_pt2 = rfa.run_pipeline(function=read_model_images,
                                                mode="file",
                                                file_paths_override=image_files_pt2,
                                                show_progress=True).results

            pt = PeakFluxPowerTransformer( 'nobody_cares', maxvals=maxvals )
            peak_fluxes_pt1 = pt.inverse_transform( np.array( peak_fluxes_tr_pt1 ) )
            peak_fluxes_pt2 = pt.inverse_transform( np.array( peak_fluxes_tr_pt2 ) )


            expected_shape = ( 1, 1, 80, 80 )
            for arr_list in [ model_images_pt1, model_images_pt2 ]:
                for i in tqdm(range( len( arr_list ) - 1, -1, -1 ),
                              desc='Checking pybdsf image homogeneity', unit='image'):
                    if arr_list[ i ].shape != expected_shape:
                        print( f'Image at index {i} has shape {arr_list[ i ].shape} instead of expected '
                              f'{expected_shape}, removing from arrays' )
                        arr_list[i] = np.zeros( expected_shape )

            model_images = np.concatenate([model_images_pt1, model_images_pt2])
            model_images *= 1e3 # convert from Jy/beam to mJy/beam

            # Get model fluxes
            print("Getting model fluxes...")
            model_fluxes_pt1 = rfa.run_pipeline(function=get_model_flux,
                                                mode="file",
                                                file_paths_override=log_files_pt1,
                                                show_progress=True).results
            model_fluxes_pt2 = rfa.run_pipeline(function=get_model_flux,
                                                mode="file",
                                                file_paths_override=log_files_pt2,
                                                show_progress=True).results
            model_fluxes = np.concatenate([model_fluxes_pt1, model_fluxes_pt2])
            model_fluxes *= 1e3  # convert from Jy/beam to mJy/beam

            peak_fluxes = np.concatenate([peak_fluxes_pt1, peak_fluxes_pt2])
            peak_fluxes *= 1e3  # convert from Jy/beam to mJy/beam

            # Filter the sizes, model images, and model fluxes to only include those with matching indices across all
            # three datasets
            common_indices = np.intersect1d(mi_indices, mf_indices)
            common_indices = np.intersect1d(common_indices, fi_indices)
            print(f"Number of sources with matching indices across all datasets: {len(common_indices)}")
            model_images = model_images[common_indices]
            model_fluxes = model_fluxes[common_indices]
            peak_fluxes = peak_fluxes[common_indices]

            # Filter so only peak_fluxes > 1 are included, and apply the same filter to model_images
            #peak_fluxes = np.concatenate([peak_fluxes_pt1, peak_fluxes_pt2])
            #mf_indices = np.array(mf_indices)[peak_fluxes > 1.0]
            model_fluxes = model_fluxes[peak_fluxes > 1.0]
            model_images = model_images[peak_fluxes > 1.0]
            peak_fluxes = peak_fluxes[peak_fluxes > 1.0]

            np.save( root / 'model_fluxes.npy', model_fluxes )
            np.save( root / 'peak_fluxes.npy', peak_fluxes )
            np.save( root / 'model_images.npy', model_images )

        if override_data:
            estimator.data.model_images = model_images
            estimator.data.model_fluxes = model_fluxes
            print(max(model_fluxes), min(model_fluxes), np.median(model_fluxes))
            print(max(estimator.data.model_fluxes),
                  min(estimator.data.model_fluxes),
                  np.median(estimator.data.model_fluxes))

            #estimator.data.model_images = np.load( path_to_subdir / 'model_images.npy', allow_pickle=True )
            #estimator.data.model_fluxes = np.load( path_to_subdir / 'model_fluxes.npy', allow_pickle=True )

        functions_to_use = [sigmoid, sigmoid01, richards01, erf01]

        for func in functions_to_use:
            lb_centres, completeness, yerr, fitted_params, pcov = estimator.estimate_completeness(func)
            print(f"Estimated completeness curve parameters: {fitted_params} with errors {np.sqrt(np.diag(pcov))}")

            # save the completeness curve data to a file
            np.savetxt('completeness_estimate_final.csv',
                       np.array([lb_centres, completeness, yerr]).transpose(),
                       delimiter=',', header='log_flux,completeness,yerr', comments='' )

            # save the fitted parameters to a file and erros from pcov
            np.savetxt(f'completeness_fit_params_{func.__name__}.csv',
                       np.array([fitted_params, np.sqrt(np.diag(pcov))]).transpose(),
                       delimiter=',', header='parameter,error', comments='' )

            # Plot data from other papers
            plt.plot(shimwell_data[0], shimwell_data[1], marker='p', color='r',
                     label='shimwell et al. 2022 data (approximate)' )
            plt.plot(dejong_data[0], dejong_data[1], marker='s', color='m',
                     label='de jong et al. 2023 data (approximate)' )
            kondapally_markers = ['<', '>', '^']
            kondapally_fields = ['ELAIS-N1', 'Lockman Hole', 'Boötes']
            for i, marker, field in zip(range(1, kondapally_data.shape[ 0 ]), kondapally_markers, kondapally_fields):
                plt.plot(kondapally_data[0][kondapally_data[i] > 0],
                         kondapally_data[i][kondapally_data[i] > 0],
                         color='k', marker=marker, label=f'kondapally et al. 2022 - {field}')

            # Plot our estimated completeness curve from a file
            plt.errorbar(lb_centres, completeness, yerr, fmt='.', color='g', label='data')
            plt.plot(lb_centres, completeness, marker='.', linestyle='None', color='g')

            # Generate smooth curve for plotting on log scale
            log_flux_smooth = np.logspace(lb_centres.min(), lb_centres.max(), 200)
            completeness_fit = func(log_flux_smooth, *fitted_params)
            plt.plot(log_flux_smooth, completeness_fit, color='c', label=f'{func.__name__} fit')

            plt.xscale('log')
            plt.ylim(0, 1.05)
            plt.xlim(left=0.01, right=100)
            plt.xlabel("Flux Density (mJy)")
            plt.ylabel("Completeness")
            plt.title("Flux Density Completeness Curve")
            plt.grid(True)
            plt.legend(loc='lower right')
            plt.show()
            plt.savefig(f'cplestim_{config_str}_{func.__name__}_full_plot.png' )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config",
                        help="Which config to to use for Dataset/Generated subdirs, as defined in "
                        f" {paths.PROGRAM_CONFIG.name}", type=str)
    args = parser.parse_args()

    plot_completeness(args.config)
