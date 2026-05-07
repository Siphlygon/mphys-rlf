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
    def __init__( self, subdir: str, maxvals: np.ndarray | None = None ):
        # Get a distribution of scaled max fluxes from the lofar data
        self.logger = get_logger( __name__ )
        self.logger.info( 'Init PeakFluxPowerTransformer for subdir ' + subdir )
        self.du = DistributedUtils()

        self.subdir = subdir
        self.maxvals_path = pth.NP_ARRAY_PARENT / subdir / pth.MAXVALS

        if not self.maxvals_path.exists():
            if maxvals is None:
                raise FileNotFoundError( f'Could not find {self.maxvals_path} - please make sure all dependencies are satisfied' )
            
            self.logger.warning( f'Maxvals not found at {self.maxvals_path}, using maxvals argument to populate...' )
            np.save( self.maxvals_path, maxvals )
        

        self.pt = PowerTransformer( method="box-cox" )
        self.pt.fit( np.load( self.maxvals_path ).reshape(-1, 1) )
        self.logger.info( 'PeakFluxPowerTransformer for subdir ' + subdir + ' fit successfully' )

    def transform( self, array: np.ndarray ):
        return self.pt.transform( array.reshape( -1, 1 ) )[ :, 0 ]

    def inverse_transform( self, array: np.ndarray ):
        return self.pt.inverse_transform( array.reshape( -1, 1 ) )[ :, 0 ]
