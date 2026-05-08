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
import os 
import h5py
from matplotlib import transforms

class AngularSizeFinder:
    """
    A class to estimate the angular size of a radio galaxy image on a 80x80 grid.
    """
    def __init__(self,
                 root_dir: Path = paths.STORAGE_PARENT / "src/completeness/retrained_loguniform_catalogs",
                 flux_threshold: float = 0.95):
        """
        This class processes PyBDSF FITS files containing Gaussian component data for radio sources, filters the components based on total flux, and estimates the angular size of the sources by creating a shape from the components and calculating the maximum distance between points on the convex hull of this shape.

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
        
        # Get a list of FITS files to limit for testing purposes
        self.rfa = RecursiveFileAnalyzer(self.root_dir)

    # ---------- ASSEMBLING SIZE ESTIMATES ----------

    def extract_component_data(self, file_path: Path) -> list[tuple]:
        """
        Process a single FITS file to extract the component data necessary for estimating the angular size of the source.
        
        Args:
            file_path (Path): The path to the FITS file to be processed.
        """
        components = []
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            for row in data:
                components.append((row["Total_flux"], row["RA"], row["DEC"], row["DC_Maj"], row["DC_Min"], row["PA"]))
        
        return self.filter_components(components)

    def filter_components(self, components: list[tuple]) -> list[tuple]:
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
            if cumulative_flux / sum_flux >= self.flux_threshold:
                break

        return filtered_components

    def fit_shape_and_estimate_size(self, components: list[tuple]) -> float:
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

    def estimate_angular_sizes(self,
            dir : str | None = None,
            pattern : str = r'.*?\D+(\d+)\.fits$',
            output_file : str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to estimate the angular sizes of sources from the FITS files in the root directory, and optionally save the results to a CSV file.

        Args:
            dir (str | None, optional): The root directory containing the FITS files. Defaults to None.
            pattern (str, optional): The regex pattern to match FITS files. Defaults to r'.*?\D+(\d+)\.fits$'.
            output_file (str | None, optional): The name of the CSV file to save the estimated angular sizes to. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the indices of the sources and their estimated angular sizes.
        """
        # Run the pipeline to extract component data from each FITS file,
        if dir:
            components_list, indices = self.rfa.run_pipeline(function=self.extract_component_data,
                                                             root_dir=dir, pattern=pattern, return_nums=True, mode="file")
        else:
            components_list, indices = self.rfa.run_pipeline(function=self.extract_component_data,
                                                             pattern=pattern, return_nums=True, mode="file")

        # Estimate the angular size of each image based on the component data
        sizes = []
        for components in components_list:
            # If there's only one component, either originaly or after filtering, we return an angular size based on DC_Maj
            if len(components) == 1:
                sizes.append(2 * components[0][3] * 3600) # DC_Maj is the 4th element in the component tuple
            else:
                angular_size = self.fit_shape_and_estimate_size(components)
                sizes.append(angular_size)
        
        # Save the estimated angular sizes to a CSV file if an output file name is provided
        if output_file:
            df = pd.DataFrame(sizes, columns=['Estimated Angular Size (arcseconds)'])
            df.to_csv(output_file, index=False)

        return np.array(indices), np.array(sizes)

    def run(self, output_file: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to run the entire pipeline for estimating the angular sizes of sources from the FITS files in the root directory, and optionally save the results to a CSV file.

        Args:
            output_file (str | None, optional): The name of the CSV file to save/load the estimated angular sizes to. If None, the results will not be saved to a file. Defaults to None.
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays: the first array contains the indices of the sources corresponding to the FITS files processed, and the second array contains the estimated angular sizes of these sources in arcseconds.
        """
        # If no output file, actually run the program
        if output_file is None or not os.path.exists(output_file):
            indices, sizes = self.estimate_angular_sizes(output_file=output_file)
        else:
            sizes = np.genfromtxt(output_file, delimiter=',', skip_header=1)
            _, indices = np.array(self.rfa.get_unwrapped_list(pattern=r'.*?\D+(\d+)\.fits$', return_nums=True))
        
        return indices, sizes


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

    def plot(self, rotate=False):
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
        
        xh_points, yh_points = self.hull_points[:, 0], self.hull_points[:, 1]
        plt.scatter(xh_points, yh_points, label='Hull Points', color='green', s=10)
        
        if self.bestcoords is not None:
            bestcoords_x = [self.bestcoords[0][0], self.bestcoords[1][0]]
            bestcoords_y = [self.bestcoords[0][1], self.bestcoords[1][1]]
            plt.plot(bestcoords_x, bestcoords_y,
                     label='Max Distance Pair', color='red', linewidth=2)
        
        plt.xlabel('DEC Offset (arcseconds)')
        plt.ylabel('RA Offset (arcseconds)')
        
        tr = transforms.Affine2D().rotate_deg(90) + transforms.Affine2D().translate(0, 0) + plt.gca().transData
        for line in plt.gca().get_lines():
            line.set_transform(tr)
        
        # Ensuret the axes are equal to avoid distortion of the shape
        max_x = max(abs(xh_points)+1)
        max_y = max(abs(yh_points)+1)
        plt.xlim(-max_x, max_x)
        plt.ylim(-max_y, max_y)

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
    output_file = 'estimated_angular_sizes.csv'

    ang_size_finder = AngularSizeFinder(root)
    indices, sizes = ang_size_finder.run(output_file=output_file)
    
    # idx, res = ang_size_finder.extract_component_data(root / "10000-19999/cutout15275.fits")
    # shape = MakeShape(pd.DataFrame(res, columns=['Total_flux', 'RA', 'DEC', 'DC_Maj', 'DC_Min', 'PA']))
    # shape.plot(True)
        
    # Check for estimated angular sizes that are above 250 arcseconds; some weird 10^6 results
    outliers = np.where(sizes > 250)[0]
    # print("Outliers with estimated angular size above 250 arcseconds:")
    # print(outliers)
    # print("Indices of outliers:")
    # print(indices[outliers])
    indices = np.delete(indices, outliers)
    sizes = np.delete(sizes, outliers)

    # Get Hardcastle LAS values for comparison
    with h5py.File("hardcastle_catalogue/clean_hardcastle_catalogue.h5", "r") as f:
        cat_indexes = f["indices"][:]
        las_values = f["cat_info"][:]["LAS"]
        
        # filter the df and indices to only include sources that are in the Hardcastle catalogue, and get the corresponding LAS values
        mask = np.isin(indices, cat_indexes)
        mask2 = np.isin(cat_indexes, indices)
        indices = indices[mask]
        sizes = sizes[mask]
        las_values = las_values[mask2]
    
    # Plot a histogram of the estimated angular sizes
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Estimated Angular Sizes of Radio Sources')
    plt.xlabel('Estimated Angular Size (arcseconds)')
    plt.ylabel('Number of Sources')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('angular_size_distribution_cutouts.png')
    plt.show()
    
    
    # max = np.max(sizes)
    # max_idx = np.argmax(sizes)
    # print(f"Maximum difference at index {max_idx}, source index {indices[max_idx]}")
    # print(f"Maximum angular size: {sizes[max_idx]} arcseconds")
    # print(f"LAS value from Hardcastle catalogue: {las_values[max_idx]} arcseconds")
    
    # Plot the difference between the estimated angular sizes and the LAS values from the Hardcastle catalogue
    diff = sizes - las_values
    # min = np.min(diff)
    # min_idx = np.argmin(diff)
    # print(f"Minimum difference: {min} arcseconds, at index {min_idx}, source index {indices[min_idx]}")
    # print(f"Estimated angular size: {sizes[min_idx]} arcseconds")
    # print(f"LAS value from Hardcastle catalogue: {las_values[min_idx]} arcseconds")
    
    # max = np.max(diff)
    # max_idx = np.argmax(diff)
    # print(f"Maximum difference: {max} arcseconds, at index {max_idx}, source index {indices[max_idx]}")
    # print(f"Estimated angular size: {sizes[max_idx]} arcseconds")
    # print(f"LAS value from Hardcastle catalogue: {las_values[max_idx]} arcseconds")
    
    plt.figure(figsize=(10, 6))
    plt.hist(diff, bins=50, color='lightcoral', edgecolor='black')
    plt.title('Difference Between Estimated Angular Sizes and LAS Values from the Hardcastle Catalogue')
    plt.xlabel('Estimated Angular Size - LAS Value (arcseconds)')
    plt.ylabel('Number of Sources')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('angular_size_difference.png')
    plt.show()