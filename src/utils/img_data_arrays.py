from utils.old_rfa import RecursiveFileAnalyzer
from analysis.log_analyzer import LogAnalyzer
import numpy as np
import utils.old_rfa as rfa
import analysis.log_analyzer as la
import utils.paths as pth
from utils.distributed import DistributedUtils
from utils.power_transform import PeakFluxPowerTransformer
from utils.logging import get_logger
from functools import reduce
import utils.paths
import argparse
import h5py
import configparser

from astropy.io import fits
import pandas as pd
import logging
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import utils.logging
from tqdm import tqdm 
import matplotlib.pyplot as plt

## THIS IS SUPER OUT OF PLACE AND HACKY PLEASE MOVE THIS AS SOON AS YOU CAN AND UPDATE TO NEW RFA ##
def get_las( file_path: str ):
    component_list = extract_component_data( file_path )
    # If there's only one component, either originaly or after filtering, we return an angular size based on DC_Maj
    if len(component_list) == 1:
        return 2 * component_list[0][3] * 3600 # DC_Maj is the 4th element in the component tuple
    else:            
        angular_size = fit_shape_and_estimate_size(component_list)
        return angular_size

def fit_shape_and_estimate_size(components: list[tuple]) -> float:
    """
    Create a shape representing the source from the filtered components and estimate the angular size of the source based on this shape.

    Args:
        components (list[tuple]): A list of tuples representing the filtered components, where each tuple contains the component's island ID, total flux, RA, DEC, major axis, minor axis, and position angle.
    Returns:
        float: The estimated angular size of the source in arcseconds.
    """
    assert components, "No components to create shape from. Check the filtering step and the input data."
    
    # Create a shape representing the source from the filtered components, and estimate the angular size of the source based on this shape. The shape is created by taking the union of ellipses representing each component, where the ellipses are defined by the major and minor axes and position angle of the components. The angular size is estimated as the maximum distance between any two points on the convex hull of the combined shape.
    shape = MakeShape(pd.DataFrame(components, columns=['Total_flux', 'RA', 'DEC', 'DC_Maj', 'DC_Min', 'PA']))
    return shape.length()

def extract_component_data(file_path: str) -> tuple[str, list[tuple]]:
    """
    Process a single FITS file to extract the component data necessary for estimating the angular size of the source.
    
    Args:
        file_path (Path): The path to the FITS file to be processed.
    """
    # index = int(re.search(r"\D*(\d+)", file_path).group(1)) - this doesn't work, the re pattern isn't write for the full path. ask luna
    components = []
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        for row in data:
            components.append((row["Total_flux"], row["RA"], row["DEC"], row["DC_Maj"], row["DC_Min"], row["PA"]))
    return filter_components( components )

def filter_components(components: list[tuple], flux_threshold: float = 0.95) -> list[tuple]:
    """
    Filter the components based on their fractional total flux, keeping only those components that contribute to a specified fraction of the total flux of the source.

    Args:
        components (list[tuple]): A list of tuples representing the components, where each tuple contains the component's island ID, total flux, RA, DEC, major axis, minor axis, and position angle.

    Returns:
        list[tuple]: A list of tuples representing the filtered components.
    """
    assert components, "No components found in the data. Check the FITS file and the expected column names."
    
    # Sort components by total flux in descending order
    components.sort(key=lambda x: x[0], reverse=True)

    # Calculate the total flux of the source by summing the total flux of all components
    sum_flux = sum(component[0] for component in components)
    if sum_flux == 0:
        return []

    filtered_components = []
    cumulative_flux = 0
    for component in components:
        cumulative_flux += component[0]
        filtered_components.append(component)
        # We are still below the flux threshold, so keep this component
        if cumulative_flux / sum_flux >= flux_threshold:
            break

    return filtered_components

class MakeShape(object):
    """
    Code in this class is adapted from the LoTSS-Catalogue GitHub, which contains the code to create the optically-identified LoTSS catalogues. 
    https://github.com/mhardcastle/lotss-catalogue/blob/master/dr2_catalogue/make_catalogue.py
    """
    def __init__(self,
                 clist: pd.DataFrame,
                 show_progress: bool = False):
        """
        A class to create a shape representing a radio source given a list of its components, and to calculate the angular size of the source based on this shape. The shape is created by taking the union of ellipses representing each component, where the ellipses are defined by the major and minor axes and position angle of the components. The angular size is estimated as the maximum distance between any two points on the convex hull of the combined shape.

        Args:
            clist (pd.DataFrame): A DataFrame containing the component information for the source, with columns 'RA', 'DEC', 'DC_Maj', 'DC_Min', and 'PA' representing the right ascension, declination, major axis, minor axis, and position angle of each component, respectively.
            show_progress (bool, optional): Whether to show a progress bar when processing the components. Defaults to False.
        """
        self.logger = utils.logging.get_logger("MakeShape", logging.DEBUG)
        self.show_progress = show_progress
        
        # Set the RA and DEC of the source to the mean RA and DEC of its components, which will be used as the reference point for calculating the angular size
        ra = np.mean(clist['RA'])
        dec = np.mean(clist['DEC'])
        self.ra = ra
        self.dec = dec

        # Create an ellipse for each component in the list, and take the union of these ellipses to form a shape representing the source
        ellist = self.create_ellipse_list(clist)
        self.combined_polygon = unary_union(ellist)

        # Calculate the convex hull of the combined shape, and find the maximum distance between any two points on the convex hull, which will be used as an estimate of the angular size of the source
        self.hull = self.combined_polygon.convex_hull
        hull_points = np.asarray(self.hull.exterior.coords)
        
        self.bestcoords, self.mdist2 = self.find_furthest_points(hull_points)
        self.hull_points = hull_points

    def ellipse(self,
                x0 : float,
                y0 : float,
                a : float,
                b : float,
                pa : float,
                n : int = 200) -> Polygon:
        """
        A function to create an ellipse given its center (x0, y0), semi-major axis a, semi-minor axis b, and position angle pa. The function returns a Polygon object representing the ellipse.
        
        Args:
            x0 (float): The x-coordinate of the ellipse center.
            y0 (float): The y-coordinate of the ellipse center.
            a (float): The length of the semi-major axis.
            b (float): The length of the semi-minor axis.
            pa (float): The position angle of the ellipse.
            n (int, optional): The number of points to use in the polygon representation. Defaults to 200.

        Returns:
            Polygon: A Shapely Polygon object representing the ellipse.
        """
        # Generate n points evenly spaced around a unit circle
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        st = np.sin(theta)
        ct = np.cos(theta)
        
        # Convert the position angle from degrees to radians and adjust it by 90 degrees to align with the standard mathematical convention for ellipses
        pa = np.deg2rad(pa + 90)
        sa = np.sin(pa)
        ca = np.cos(pa)
        
        # Calculate the coordinates of points using the parametric equations of an ellipse
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        return Polygon(p)

    def create_ellipse_list(self, clist):
        """
        A function to create a list of Polygon objects representing the ellipses for each component in the component list.

        Args:
            clist (pd.DataFrame): A DataFrame containing the component information for the source, with columns 'RA', 'DEC', 'DC_Maj', 'DC_Min', and 'PA' representing the right ascension, declination, major axis, minor axis, and position angle of each component, respectively.

        Returns:
            list[Polygon]: A list of Shapely Polygon objects representing the ellipses for each component.
        """
        ellist = []
        for component in tqdm(clist.iterrows(), desc="Creating ellipses for components...", disable=not self.show_progress):
            ra_n = component[1]['RA']
            dec_n = component[1]['DEC']

            # Convert the RA and DEC differences to arcseconds, accounting for the cosine of the declination for the RA component
            x = 3600 * np.cos(self.dec * np.pi/180.0) * (self.ra - ra_n)
            y = 3600 * (dec_n - self.dec)
            
            # Get the major and minor axes of the ellipse representing the component, and convert to arcseconds.
            dc_maj_n = component[1]['DC_Maj'] * 3600
            dc_min_n = component[1]['DC_Min'] * 3600
            
            # Add a small buffer (0.1 arcseconds) to the major and minor axes to ensure that the ellipses overlap and form a single connected shape, even if the components are very close together
            new_polygon = self.ellipse(x, y, dc_maj_n + 0.1, dc_min_n + 0.1, component[1]['PA'])
            ellist.append(new_polygon)
        return ellist

    def find_furthest_points(self, points: np.ndarray) -> tuple[tuple[tuple[float, float], tuple[float, float]], float]:
        """
        A function to find the pair of points in a given set of points that are furthest apart, and return these points as a tuple.

        Args:
            points (np.ndarray): An array of shape (n, 2) containing the coordinates of the points.
        Returns:
            tuple[tuple[float, float], tuple[float, float]]: A tuple containing the pair of points that are furthest apart, where each point is represented as a tuple of (x, y) coordinates.
        """
        mdist2 = 0
        bestcoords = None
        for point in tqdm(points, desc="Finding furthest points...", disable=not self.show_progress):
            # Calculate the squared distance from the current point to all other points in the set, and find the maximum distance
            dist2 = (points[:, 0] - point[0])**2.0 + (points[:, 1] - point[1])**2.0
            idist = np.argmax(dist2)
            mdist = dist2[idist]
            
            # Update the maximum distance and the corresponding pair of points
            if mdist > mdist2:
                mdist2 = mdist
                bestcoords = (point, points[idist]) # (point, furthest_point)

        return bestcoords, mdist2

    def plot(self):
        """
        A method to plot the combined shape of the source and its convex hull, along with the points on the convex hull and the pair of points that are furthest apart, which are used to estimate the angular size of the source.
        """
        plt.figure(figsize=(8, 8))
        
        # Some sources are combined together, into a MultiPolygon
        if isinstance(self.combined_polygon, MultiPolygon):
            for geom in self.combined_polygon.geoms:
                x, y = geom.exterior.xy
                plt.plot(x, y, label='Combined Shape', color='blue')
        else:
            x, y = self.combined_polygon.exterior.xy
            plt.plot(x, y, label='Combined Shape', color='blue')
        
        xh, yh = self.hull.exterior.xy
        plt.plot(xh, yh, label='Convex Hull', color='orange')
        
        plt.scatter(self.hull_points[:, 0], self.hull_points[:, 1], label='Hull Points', color='green', s=10)
        
        if self.bestcoords is not None:
            plt.plot([self.bestcoords[0][0], self.bestcoords[1][0]], 
                     [self.bestcoords[0][1], self.bestcoords[1][1]], 
                     label='Max Distance Pair', color='red', linewidth=2)
        
        plt.xlabel('RA Offset (arcseconds)')
        plt.ylabel('DEC Offset (arcseconds)')
        plt.title('Combined Shape and Convex Hull of Source')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def length(self):
        """
        A method to calculate the angular size of the source based on the maximum distance between points on the convex hull of the union of ellipses representing the components of the source.    

        Returns:
            float: The estimated angular size of the source in arcseconds.
        """
        return np.sqrt(self.mdist2)

class SubdirData:
    pass;


class ImageDataArrays:
    """
    A class to collect unscaled (physical units) image data arrays for images in subdir from the original files and from pybdsf analysis,
    which will be useful for calculating the completeness corrections. All units from the image data arrays are in mJy,
    though the input images are expected to be normalized 0-1 and the scaled peak flux values in Jy. All arrays reference
    the same image at the same index and have the same length in the first dimension, though the order is nonstandard.

    Parameters
    ----------
    subdir : str
        The subdirectory to generate the image data arrays for. It is assumed when running the program that the data, pybdsf
        log files, and pybdsf gaus_resid images are all prepared for the data to generate the arrays of, though if any
        of the three are not present for an image it will not be included in the arrays with no error

    load_from_files : bool = True
        Attempt to load from file instead of going through and opening each fits file. Can save time on loading if running
        frequently. Default True. If any arrays cannot be loaded, all are read from the fits files.
    """
    def __init__( self, config_name: str, load_from_files: bool = True ):
        self.logger = get_logger( __name__, logging.DEBUG )
        self.du = DistributedUtils()
        self.config = pth.config[ config_name ]

        for subdir in [ self.config[ 'generated_subdir' ], self.config[ 'dataset_subdir' ] ]:
            self.logger.debug( f'Entering image data arrays for subdir {subdir}' )
            subdir_data = SubdirData()
            if load_from_files:
                self.logger.debug( 'Attempting to load from files' )
                parent = utils.paths.NP_ARRAY_PARENT
                for array_name in [ 'images', 'residual_images', 'model_images', 'model_fluxes', 'peak_fluxes', 'sigma_clipped_means', 'sigma_clipped_rmsds', 'image_scale_factors', 'las_values' ]:
                    try:
                        val = np.load( parent / subdir / ( array_name + '.npy' ) )
                        self.logger.debug( f'{subdir}/{array_name} exists!' )
                        setattr( subdir_data, array_name, val )
                    except OSError:
                        load_from_files = False
                        self.logger.debug( f'{subdir}/{array_name} does not exist' )
            
            if not load_from_files:
                self.logger.debug( 'Not loading from files, either clearing cache or failed to load cache' )
                # Log analyzer arrays
                log_analyzer = LogAnalyzer( subdir )
                normalized_model_fluxes, log_analyzer_inds = log_analyzer.for_each( la.get_model_flux, return_nums=True )
                normalized_model_fluxes = np.array( normalized_model_fluxes )
                sigma_clipped_means = np.array( log_analyzer.for_each( la.get_sigma_clipped_mean ) ) / 1000 #normalized Jy units
                sigma_clipped_rmsds = np.array( log_analyzer.for_each( la.get_sigma_clipped_rms ) ) / 1000 #normalized Jy units
                unclipped_rmsds = np.array( log_analyzer.for_each( la.get_rms ) )
                log_analyzer_values = [ normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds ]
                self.logger.debug( 'Log analyzer length: %i', len( log_analyzer_inds ) )

                use_dataset_h5 = subdir == self.config[ 'dataset_subdir' ] and self.config[ 'train_data_path' ] != 'None'
                if use_dataset_h5:
                    self.logger.debug( f'Using h5 dataset {self.config[ "train_data_path" ]}' )
                else:
                    self.logger.debug( f'Not using dataset h5 for {subdir}' )
    
                # Data arrays
                if use_dataset_h5:
                    with h5py.File( self.config[ 'train_data_path' ], 'r' ) as train_data:
                        images = train_data[ 'images' ][ : ]
                        data_inds = train_data[ 'indices' ][ : ]
                        las_values = train_data[ 'cat_info' ][ 'LAS' ][ : ]
                    peak_fluxes_mjy = np.max( images, axis=(1,2) ) * 1000
                    data_values = [ images, peak_fluxes_mjy ]
                else:
                    data_files = RecursiveFileAnalyzer( pth.FITS_PARENT / subdir )
                    images, data_inds = data_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                    images = np.array( images )

                    peak_fluxes_transformed = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?\D+(\d+)\.fits$', kwargs=dict( key='FXSCLD' ) ) )
                    data_values = [ images, peak_fluxes_transformed ]

                self.logger.debug( 'Data files length: %i', len( data_inds ) )
    
                # Residual folder
                residual_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' )
                residual_images, residual_indexes = residual_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                residual_images = np.array( residual_images )
                residual_values = [ residual_images ]
                self.logger.debug( 'Gaussian residual files length: %i', len( residual_indexes ) )
    
                # Model folder
                model_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_model' )
                model_images, model_indexes = model_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                model_images = np.array( model_images )
                model_values = [ model_images ]
                self.logger.debug( 'Gaussian model files length: %i', len( model_images ) )

                # Catalog folder, only run if not getting LAS values from dataset
                if not use_dataset_h5:
                    catalog_files = RecursiveFileAnalyzer( pth.PYBDSF_CATALOG_PARENT / subdir )
                    las_values, catalog_indexes = catalog_files.for_each( get_las, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                    las_values = np.array( las_values )
                    catalog_values = [ las_values ]
                    self.logger.debug( 'Catalog files length: %i', len( las_values ) )
                
                # Wrap everything and match indices/values for all different folders so everything aligns properly
                inds_array = [ log_analyzer_inds, data_inds, residual_indexes, model_indexes ]
                values_array = [ log_analyzer_values, data_values, residual_values, model_values ]
                if not use_dataset_h5:
                    inds_array.append( catalog_indexes )
                    values_array.append( catalog_values )

                intersect = reduce( lambda x, y : np.intersect1d( x, y, assume_unique=True ), inds_array )
                for i in range( len( inds_array ) ):
                    for j in range( len( values_array[ i ] ) ):
                        values = values_array[ i ][ j ]
    
                        # Get index of indices in inds_array[ i ] that are in the intersection ordered by the intersection
                        # Source - https://stackoverflow.com/a/32191125
                        # Posted by Alex Riley, modified by community. See post 'Timeline' for change history
                        # Retrieved 2025-12-02, License - CC BY-SA 3.0
                        sorter = np.argsort( inds_array[ i ] )
                        index_indices = sorter[ np.searchsorted( inds_array[ i ], intersect, sorter=sorter ) ]
    
                        values_array[ i ][ j ] = values[ index_indices ]
    
                # Unwrap everything into its original values
                if use_dataset_h5:
                    log_analyzer_values, data_values, residual_values, model_values = values_array

                    images, peak_fluxes_mjy = data_values
                else:
                    log_analyzer_values, data_values, residual_values, model_values, catalog_values = values_array
                    las_values, = catalog_values

                    images, peak_fluxes_transformed = data_values

                    # Get the unscaled fluxes and unscale everything accordingly
                    pt = PeakFluxPowerTransformer( subdir, maxvals=np.max( images, axis=(1,2) ) )
                    peak_fluxes_mjy = pt.inverse_transform( peak_fluxes_transformed ) * 1000

                normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds = log_analyzer_values
                residual_images, = residual_values
                model_images, = model_values

    
                if self.config[ 'do_unscaling' ] == 'True':
                    image_scale_factors = peak_fluxes_mjy / np.max( images, axis=(1,2) ) #Scale from current image maxes (~1) to what the values should be as per peak fluxes
                else:
                    image_scale_factors = np.ones( images.shape[ 0 ] )
                unscaled_sigma_clipped_rmsds = sigma_clipped_rmsds * image_scale_factors
                unscaled_sigma_clipped_means = sigma_clipped_means * image_scale_factors
                model_fluxes = normalized_model_fluxes * image_scale_factors
                unscaled_images = images * image_scale_factors[ :, np.newaxis, np.newaxis ]
                unscaled_residual_images = np.array( residual_images ) * image_scale_factors[ :, np.newaxis, np.newaxis ]
                unscaled_model_images = np.array( model_images ) * image_scale_factors[ :, np.newaxis, np.newaxis ]
                
    
                # Save unscaled variables to class
                subdir_data.images = unscaled_images
                subdir_data.residual_images = unscaled_residual_images
                subdir_data.model_images = unscaled_model_images
                subdir_data.model_fluxes = model_fluxes
                subdir_data.peak_fluxes = peak_fluxes_mjy
                subdir_data.las_values = las_values
                subdir_data.sigma_clipped_means = unscaled_sigma_clipped_means
                subdir_data.sigma_clipped_rmsds = unscaled_sigma_clipped_rmsds
                subdir_data.image_scale_factors = image_scale_factors

                self.logger.debug( 'saved all parameters to subdir_data' )

                if subdir == self.config[ 'generated_subdir' ]:
                    self.generated_data = subdir_data
                    self.logger.debug( 'Saving subdir_data to generated_data' )
                else:
                    self.dataset_data = subdir_data
                    self.logger.debug( 'Saving subdir_data to dataset_data' )

        self.logger.debug( 'Done! Saving image data arrays...' ) 
        self.save_all_arrays()
    
    def save_all_arrays( self ):
        """
        Save all numpy arrays to a file for ease of loading
        """
        parent = utils.paths.NP_ARRAY_PARENT
        dataset_dict = vars( self.dataset_data )
        generated_dict = vars( self.generated_data )
        for subdir_dict, subdir in zip( [ dataset_dict, generated_dict ], [ self.config[ 'dataset_subdir' ], self.config[ 'generated_subdir' ] ] ):
            for key, val in subdir_dict.items():
                if isinstance( val, np.ndarray ):
                    np.save( parent / subdir / ( key + '.npy' ), val )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", help=f"Which config to use for image data arrays, as defined in {pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()


    # constructing the object saves the numpy arrays if they don't exist
    ImageDataArrays( args.config )
    print( "done" )
