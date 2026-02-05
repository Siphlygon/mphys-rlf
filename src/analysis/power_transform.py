from utils.logging import get_logger
from utils.distributed import DistributedUtils
import utils.paths as pth
import h5py
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PowerTransformer

# DEPENDENCIES
# make_folders_and_copy_config
# download_dataset
# maxvals


class PeakFluxPowerTransformer:
    """
    A utility class to easily power transform peak flux values based on the max values in the dataset, without
    having to constantly validate files exist
    """
    def __init__( self ):
        # Get a distribution of scaled max fluxes from the lofar data
        self.logger = get_logger( __name__ )
        self.du = DistributedUtils()


        if not pth.MAXVALS.exists():
            raise FileNotFoundError( f'Could not find {pth.MAXVALS} - please make sure all dependencies are satisfied' )

        self.pt = PowerTransformer( method="box-cox" )
        self.pt.fit( np.load( pth.MAXVALS ).reshape(-1, 1) )

    def transform( self, array: np.ndarray ):
        return self.pt.transform( array.reshape( -1, 1 ) )[ :, 0 ]

    def inverse_transform( self, array: np.ndarray ):
        return self.pt.inverse_transform( array.reshape( -1, 1 ) )[ :, 0 ]
