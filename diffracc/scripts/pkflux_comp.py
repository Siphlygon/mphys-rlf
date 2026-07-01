import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from ..utils import paths
from ..utils.power_transform import PeakFluxPowerTransformer

img_directory = paths.FITS_PARENT / 'retrained_loguniform'
power_transformer = PeakFluxPowerTransformer( 'retrained_loguniform' )

peak_fluxes_conditioned = []
peak_fluxes_from_imgs = []

for img_path in img_directory.rglob( "*.fits" ):
    with fits.open( img_path ) as hdul:
        flux_scaled = hdul[ 0 ].header[ 'FXSCLD' ]
        peak_flux_from_img = np.max( hdul[ 0 ].data )
    peak_flux_conditioned = power_transformer.inverse_transform( np.array( [ flux_scaled ] ) )

    peak_fluxes_from_imgs.append( peak_flux_from_img )
    peak_fluxes_conditioned.append( peak_flux_conditioned )

plt.scatter( peak_fluxes_conditioned, peak_fluxes_from_imgs, s=0.01 )

min_peakflux = min( np.min( peak_fluxes_conditioned ), np.min( peak_fluxes_from_imgs ) ) if len( peak_fluxes_from_imgs ) > 0 else np.min( peak_fluxes_conditioned )
max_peakflux = max( np.max( peak_fluxes_conditioned ), np.max( peak_fluxes_from_imgs ) ) if len( peak_fluxes_from_imgs ) > 0 else np.max( peak_fluxes_conditioned )

x = np.geomspace( min_peakflux, max_peakflux )
plt.plot( x, x, color='r' )
plt.grid()

plt.title( 'Peak flux from conditioning vs image' )
plt.xlabel( 'Conditioned peak flux, Jy/beam' )
plt.ylabel( 'Image peak flux, Jy/beam' )
plt.xscale( 'log' )
plt.yscale( 'log' )
plt.savefig( 'peak_flux_conditioning.png' )
plt.show()
