import sys
import os

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import matplotlib.pyplot as plt
import model.sampler
import plotting.image_plots
import numpy as np
import matplotlib.animation as animation

model_sampler = model.sampler.Sampler()

samples = model_sampler.sample(
    "LOFAR_model",
    distribute_model=False,
    n_samples=4,
)

square_size = int( np.ceil( np.sqrt( samples.shape[ 0 ] ) ) )

i = 0
j = 0
fig = plt.figure( figsize=(8, 8) )
gs = fig.add_gridspec(square_size, square_size,
                      left=0.05, right=0.95, bottom=0.1, top=0.95,
                      wspace=0.5, hspace=0.5)

images = []
while ( i + j * square_size ) < samples.shape[ 0 ]:
    sample = samples[ i + j * square_size ]
    ax = fig.add_subplot( gs[ i, j ] )
    ax.set_title( 'Sample ' + str( i + j * square_size ) )
    img = ax.imshow( sample[ 0, 0, :, : ] )
    img.set( animated=True )
    ax.set_xlabel( "dimX" )
    ax.set_ylabel( "dimY" )
    i += 1
    if ( i == square_size ):
        i = 0
        j += 1
    images.append( img )

def Animate( frame ):
    axes_plainlist = []
    for i in range( len( images ) ):
        image = images[ i ]
        sample = samples[ i ]
        axes_plainlist.append( image )
        image.set_data( sample[ frame, 0, :, : ] )
        image.set_clim( np.min( sample[ frame, 0, :, : ] ), np.max( sample[ frame, 0, :, : ] ) )
    return images

ani = animation.FuncAnimation( fig, Animate, frames=samples.shape[ 1 ], interval=100, blit=True )
