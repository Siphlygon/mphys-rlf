from scipy.optimize import curve_fit
import numpy as np
import scipy.stats
from tqdm import tqdm
import astropy.stats
import pandas as pd
import matplotlib.pyplot as plt
from analysis.image_analyzer import ImageAnalyzer
from utils.img_data_arrays import ImageDataArrays
import scipy.signal

rms_LOFAR = 95e-6 * 1e3
beam_width_LOFAR = ImageAnalyzer.LOFAR_process_arg_defaults[ 'process_beam' ][ :-1 ]
beam_area_LOFAR = beam_width_LOFAR[ 0 ] * beam_width_LOFAR[ 1 ]

shimwell_data = np.array( [
    [ 0.2,   0.0 ],
    [ 0.3,   0.1 ],
    [ 0.4,   0.4 ],
    [ 0.5,   0.6 ],
    [ 0.6,   0.7 ],
    [ 0.7,   0.8 ],
    [ 0.8,   0.85 ],
    [ 0.9,   0.9 ],
    [ 1.0,   0.93 ],
    [ 1.1,   0.94 ],
    [ 1.2,   0.96 ]
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


def get_noise(data):
    """
    from Cyril Tasse/kMS, courtesy of Wara
    """
    maskSup = 1e-7
    m = data[np.abs(data) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)
    for _ in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold)//rmsold) < diff: break
        rmsold = rms
    return rms

def masking(fits_data, threshold_level = 5.0):
    mean, median, std_dev = astropy.stats.sigma_clipped_stats(fits_data, sigma=3.0)
    
    # Calculate the threshold
    threshold = threshold_level * std_dev
    
    # Create a mask for values less than the threshold
    mask = fits_data < threshold
    
    # Set values less than the threshold to zero
    fits_data_nr = np.where(mask, 0, fits_data)

    return fits_data_nr

def create_noise_LOFAR(shape=(80,80), rms=rms_LOFAR):
    """
    Create a 2D patch of Gaussian noise with given RMS.
    """
    # Add beam-correllated noise

    # Source - https://stackoverflow.com/a/63868276
    # Posted by Igor
    # Retrieved 2026-02-12, License - CC BY-SA 4.0

    # Compute filter kernel with radius correlation_scale
    correlation_scale = 6 / 1.5 #( 6 arcsec / beam ) / ( 1.5 arcsec / pix )
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    dist = dist[np.newaxis, :, :]
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))

    noise = np.random.normal( loc=0.0, scale=rms, size=shape )
    noise = scipy.signal.fftconvolve( noise, filter_kernel, mode='same' )

    return noise

def get_completeness_estim():
    plt.figure(figsize = (8, 5))
    N_NOISE_PATCHES = 5
    for subdir in [ "generated_loguniform" ]:
        images, resid_images, m_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays()

        mock_fluxes = np.empty( (images.shape[ 0 ]*N_NOISE_PATCHES), dtype=float )
        detectable = np.empty( (images.shape[ 0 ]*N_NOISE_PATCHES), dtype=bool )


        for i in tqdm( range( images.shape[ 0 ] ), desc='Calculating mock images' ):
            #rms = image_rmss_actual[ random_image ]
            #noise_patch = resid_images[ random_image ]

            # Using rms=image_rmss_actual[ random_image ] is technically correct yet utterly useless
            # because the majority of the noise is from the artificial 1% noise added for pybdsf
            # TODO: Use raw LOFAR data so we can get rms locally based on strength of source, potential code commented above
            rms = rms_LOFAR
            mock_fluxes[ i:(i+N_NOISE_PATCHES) ] = model_fluxes[ i ][ np.newaxis ]
            noise_patches = create_noise_LOFAR( shape=(N_NOISE_PATCHES,80,80), rms=rms )
            sim_data = noise_patches + images[ i ][ np.newaxis, :, : ]

            peak_fluxes = np.max( sim_data, axis=(1,2) )
            threshold = 5 * rms
            detectable[ i:(i+N_NOISE_PATCHES) ] = peak_fluxes >= threshold


        test_mock = pd.DataFrame()
        #test_mock['mock_flux'] = mock_fluxes.ravel()
        #test_mock['detectable'] = detectable.ravel()
        test_mock['mock_flux'] = mock_fluxes
        test_mock['detectable'] = detectable

        # Define flux bins
        flux_bins = np.logspace( -2, 2, num=25 )
        bin_centers = 0.5 * (flux_bins[1:] + flux_bins[:-1])

        # Bin and count
        completeness = []   # to store completeness per bin
        total_counts = []   # optional: for diagnostics

        for i in range(len(flux_bins) - 1):
            # Select sources in this flux bin
            in_bin = (mock_fluxes >= flux_bins[i]) & (mock_fluxes < flux_bins[i + 1])

            n_detect = test_mock[ (test_mock['mock_flux'] >= flux_bins[i]) & (test_mock['mock_flux'] < flux_bins[i + 1]) ]
            
            if np.sum(in_bin) > 0:
                frac_recovered = np.sum(n_detect['detectable']) / np.sum(in_bin)
            else:
                frac_recovered = 0  

            completeness.append(frac_recovered)
            total_counts.append(np.sum(in_bin))

        # Handle confidence interval with poisson_conf_interval for total_counts = 0
        total_counts = np.array( total_counts )
        zero_counts = total_counts == 0
        total_counts = np.where( zero_counts, 1e-10, total_counts )
        conf_interval = astropy.stats.poisson_conf_interval( np.array( completeness ) * total_counts, interval='frequentist-confidence', sigma=1.0 )
        conf_interval /= total_counts
        conf_interval[ :, zero_counts ] = 0
        yerr = conf_interval[ 1 ] - conf_interval[ 0 ]

        # Plot completeness curve

        plt.errorbar( bin_centers, completeness, yerr, fmt='.', color='g' )
        plt.plot(bin_centers, completeness, marker='.', label = f'{subdir} completeness', color='g' )

    # Plot data from other papers
    plt.plot( shimwell_data[ 0 ], shimwell_data[ 1 ], marker='p', color='r', label='shimwell et al. 2022 data (approximate)' )
    plt.plot( dejong_data[ 0 ], dejong_data[ 1 ], marker='s', color='m', label='de jong et al. 2023 data (approximate)' )

    # Fit a curve (claude code)
    # Fit sigmoid to completeness curve
    def sigmoid(x, x0, k, a, b):
        """Sigmoid function: a / (1 + exp(-k*(x-x0))) + b"""
        return a / (1 + np.exp(-k * (x - x0))) + b
    
    # Use log of flux for fitting since we're on a log scale
    log_bin_centers = np.log10(bin_centers)
    
    # Initial parameter guesses: x0 (midpoint), k (steepness), a (amplitude), b (offset)
    initial_guess = [0.5, 2.0, 1.0, 0.0]
    
    try:
        # Fit the sigmoid
        popt, pcov = curve_fit(sigmoid, log_bin_centers, completeness, p0=initial_guess, maxfev=10000)
        
        # Generate smooth curve for plotting
        log_flux_smooth = np.linspace(log_bin_centers.min(), log_bin_centers.max(), 200)
        completeness_fit = sigmoid(log_flux_smooth, *popt)
        
        # Convert back to linear scale for plotting
        flux_smooth = 10**log_flux_smooth
        plt.plot(flux_smooth, completeness_fit, 'r--', linewidth=2, label=f'Sigmoid fit', alpha=0.7)
        
        print(f"Sigmoid fit parameters:")
        print(f"  x0 (log midpoint): {popt[0]:.3f} (flux: {10**popt[0]:.3f} mJy)")
        print(f"  k (steepness): {popt[1]:.3f}")
        print(f"  a (amplitude): {popt[2]:.3f}")
        print(f"  b (offset): {popt[3]:.3f}")
    except Exception as e:
        print(f"Warning: Sigmoid fit failed: {e}")


    kondapally_markers = [ '<', '>', '^' ]
    kondapally_fields = [ 'ELAIS-N1',	'Lockman Hole',	'Boötes' ]
    for i, marker, field in zip( range( 1, kondapally_data.shape[ 0 ] ), kondapally_markers, kondapally_fields ):
        plt.plot( kondapally_data[ 0 ][ kondapally_data[ i ] > 0 ], kondapally_data[ i ][ kondapally_data[ i ] > 0 ], color='k', marker=marker, label=f'kondapally et al. 2022 - {field}' )




    plt.xscale('log')
    plt.ylim(0, 1.1)
    plt.xlim( left=0.5, right=100 )
    plt.xlabel("Flux Density (mJy)")
    plt.ylabel("Completeness")
    plt.title("Flux Density Completeness Curve")
    plt.grid(True)
    plt.legend( loc='lower right' )
    plt.show()
    plt.savefig( 'cplestim.png' )


if __name__ == "__main__":
    get_completeness_estim()

