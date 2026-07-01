import numpy as np
from sklearn.preprocessing import PowerTransformer

import utils.paths as pth
from utils.distributed import DistributedUtils
from utils.logging import get_logger

# DEPENDENCIES
# make_folders_and_copy_config
# download_dataset
# maxvals


class PeakFluxPowerTransformer:
    """
    A utility class to easily power transform peak flux values based on the max values in the dataset, without having to
    constantly validate files exist
    """
    def __init__( self, subdir: str, maxvals: np.ndarray | None = None ):
        """
        Initialises the PeakFluxPowerTransformer by loading the maxvals from a numpy file, fitting a PowerTransformer to
        them, and providing methods to transform and inverse transform peak flux values.

        Parameters
        ----------
        subdir : str
            The subdirectory to use for the maxvals file
        maxvals : np.ndarray | None, optional
            The maxvals to use if the file is not found, by default None

        Raises
        ------
        FileNotFoundError
            If the maxvals file is not found and no maxvals are provided
        """
        # Get a distribution of scaled max fluxes from the lofar data
        self.logger = get_logger( __name__ )
        self.logger.info( 'Init PeakFluxPowerTransformer for subdir ' + subdir )
        self.du = DistributedUtils()

        self.subdir = subdir
        self.maxvals_path = pth.NP_ARRAY_PARENT / subdir / pth.MAXVALS

        if not self.maxvals_path.exists():
            if maxvals is None:
                raise FileNotFoundError(
                    f'Could not find {self.maxvals_path} - please make sure all dependencies are satisfied' )

            self.logger.warning( f'Maxvals not found at {self.maxvals_path}, using maxvals argument to populate...' )
            np.save( self.maxvals_path, maxvals )


        self.pt = PowerTransformer( method="box-cox" )
        self.pt.fit( np.load( self.maxvals_path ).reshape(-1, 1) )
        self.logger.info( 'PeakFluxPowerTransformer for subdir ' + subdir + ' fit successfully' )


    def transform( self, array: np.ndarray ) -> np.ndarray:
        """
        Transforms the given array of peak flux values using the fitted PowerTransformer.

        Parameters
        ----------
        array : np.ndarray
            The array of peak flux values to transform

        Returns
        -------
        np.ndarray
            The transformed array of peak flux values
        """
        return self.pt.transform( array.reshape( -1, 1 ) )[ :, 0 ]


    def inverse_transform( self, array: np.ndarray ) -> np.ndarray:
        """
        Inverse transforms the given array of transformed peak flux values back to the original scale using the fitted
        PowerTransformer.

        Parameters
        ----------
        array : np.ndarray
            The array of transformed peak flux values to inverse transform

        Returns
        -------
        np.ndarray
            The inverse transformed array of peak flux values
        """
        return self.pt.inverse_transform( array.reshape( -1, 1 ) )[ :, 0 ]
