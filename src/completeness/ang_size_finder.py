from typing import Any
from pathlib import Path

from astropy.io import fits
from tqdm import tqdm
import logging
import utils.paths as paths
import utils.logging
from utils.recursive_file_analyzer import RecursiveFileAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import pandas as pd
import re 

class AngularSizeFinder:
    """
    A class to estimate the angular size of a radio galaxy image on a 80x80 grid.
    """
    def __init__(self,
                 root_dir: Path = paths.STORAGE_PARENT / "src/completeness/retrained_loguniform_catalogs",
                 flux_threshold: float = 0.95):
        """ This class processes PyBDSF FITS files containing component data for radio sources, filters the components based on flux, and estimates the angular size of the sources by creating a shape from the components and calculating the maximum distance between points on the convex hull of this shape.

        Args:
            root_dir (Path, optional): The root directory containing the FITS files. Defaults to paths.STORAGE_PARENT/"src/completeness/retrained_loguniform_catalogs".
            flux_threshold (float, optional): The threshold for filtering components based on their fractional total flux. Defaults to 0.95.
        """
        self.logger = utils.logging.get_logger("AngularSizeFinder", logging.DEBUG)
        self.root_dir = root_dir

        # Decide a flux threshold for filtering components. PyBDSF can sometimes fit islands to noise,
        # and so we sort and then filter islands based on their fractional total flux.
        # The threshold below represents the fraction of total flux to keep, so dimmer islands are
        # removed while keeping above this threshold.
        self.flux_threshold = flux_threshold

    def extract_component_data(self, file_path: str) -> tuple[str, list[np.ndarray]]:
        """
        Process a single FITS file to extract the component data necessary for estimating the angular size of the source.
        
        Args:
            file_path (Path): The path to the FITS file to be processed.
        """
        # index = int(re.search(r"\D*(\d+)", file_path).group(1)) - this doesn't work, the re pattern isn't write for the full path. ask luna
        index = Path(file_path).stem
        print(f"Processing file {file_path} with index {index}...")
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            return index, [data["Isl_id"], data["Total_flux"], data["RA"], data["DEC"], data["DC_Maj"], data["DC_Min"], data["PA"]]
    
    def main(self):
        rfa = RecursiveFileAnalyzer(self.root_dir)
        pths = rfa.get_unwrapped_list()[:100]
        
        results = rfa.run_pipeline(function=self.extract_component_data,
                                   file_paths_override=pths)
        # Save the results to a text file
        with open("angular_size_results.txt", "w") as f:
            for result in results:
                f.write(f"{result}\n")

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
        self.logger.info("Calculating mean RA and DEC for the source...")
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
        
        self.logger.info("Calculating maximum distance on convex hull...")
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
            
            # Add a small buffer (0.1 arcseconds) to the major and minor axes to ensure that the ellipses overlap and form a single connected shape, even if the components are very close together
            new_polygon = self.ellipse(x, y, component[1]['DC_Maj'] + 0.1, component[1]['DC_Min'] + 0.1, component[1]['PA'])
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


if __name__ == "__main__":
    
    root = paths.STORAGE_PARENT / "src/completeness/retrained_loguniform_catalogs"
    # # root = paths.DATASET_PARENT / "dr2_cutouts_download"
    ang_size_finder = AngularSizeFinder(root)
    ang_size_finder.main()
    
    # print("Components DataFrame:", components.head())
    
    # final_shape = MakeShape(components)
    # angular_size = final_shape.length()
    # print(f"Estimated angular size of the source: {angular_size:.2f} arcseconds")
    # final_shape.plot()
