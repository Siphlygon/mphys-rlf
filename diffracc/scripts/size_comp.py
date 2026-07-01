import numpy as np
from utils.power_transform import PeakFluxPowerTransformer
import utils.paths as pth
from astropy.io import fits
import matplotlib.pyplot as plt


img_directory = pth.FITS_PARENT / 'snr15_loguniform'
power_transformer = PeakFluxPowerTransformer( 'snr15_loguniform' )

lass = []
lass_from_img = []

for img_path in img_directory.rglob( "*.fits" ):
    with fits.open( img_path ) as hdul:
        las = hdul[ 0 ].header[ 'LASIZE' ]
        img = hdul[ 0 ].data
        # not actually largest angular size, just a basic proxy with no physical motivation
        las_from_img = np.sqrt( (img > np.median( img ) + 3 * np.std( img )).sum() )
    lass.append( las )
    lass_from_img.append( las_from_img )

plt.scatter( lass, lass_from_img, s=0.01 )

minval = min( np.min( lass ), np.min( lass_from_img ) ) if len( lass ) > 0 else np.min( lass )
maxval = max( np.max( lass ), np.max( lass_from_img ) ) if len( lass ) > 0 else np.max( lass )

x = np.geomspace( minval, maxval )
plt.plot( x, x, color='r' )
plt.grid()

plt.title( 'Size conditioning test' )
plt.xlabel( 'Largest Angular Size (arcsec)' )
plt.ylabel( 'Random Size Proxy (a.u.)' )
plt.xscale( 'log' )
plt.yscale( 'log' )
plt.savefig( 'size_comp.png' )
plt.show()
