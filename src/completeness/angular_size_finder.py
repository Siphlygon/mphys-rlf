import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import transforms
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

import utils.logging
import utils.paths as paths
from utils.recursive_file_analyzer import RecursiveFileAnalyzer


class MakeShape():
    """
    Code in this class is adapted from the LoTSS-Catalogue GitHub, which contains the code to create the
    optically-identified LoTSS catalogues. The exact file is found here:
    https://github.com/mhardcastle/lotss-catalogue/blob/master/dr2_catalogue/make_catalogue.py
    """
    def __init__(self,
                 clist: pd.DataFrame,
                 show_progress: bool = False):
        """
        A class to create a shape representing a radio source given a list of its components, created by taking the
        union of ellipses representing each component.
        
        Parameters
        ----------
        clist : pd.DataFrame
            A DataFrame containing the component information for the source, with columns 'RA', 'DEC', 'DC_Maj',
            'DC_Min', and 'PA' representing the right ascension, declination, major axis, minor axis, and position
            angle of each component, respectively.
        show_progress : bool, optional
            Whether to show a progress bar for the processing of components, by default False.
        """
        self.logger = utils.logging.get_logger("MakeShape", logging.DEBUG)
        self.show_progress = show_progress

        # Set the RA and DEC of the source to the mean RA and DEC of its components
        ra = np.mean(clist['RA'])
        dec = np.mean(clist['DEC'])
        self.ra = ra
        self.dec = dec

        # Create an ellipse for each component in the list, and take the union of these ellipses to form a shape
        # representing the source
        ellist = self.create_ellipse_list(clist)
        self.combined_polygon = unary_union(ellist)

        # Calculate the convex hull of the combined shape, and find the maximum distance between any two points on the
        # convex hull, which will be used as an estimate of the angular size of the source
        self.hull = self.combined_polygon.convex_hull
        hull_points = np.asarray(self.hull.exterior.coords)  # type: ignore

        self.best_coords, self.mdist2 = self.find_furthest_points(hull_points)
        self.hull_points = hull_points


    def ellipse(self,
                x0 : float,
                y0 : float,
                a : float,
                b : float,
                pa : float,
                n : int = 200) -> Polygon:
        """
        A function to create an ellipse given its center (x0, y0), semi-major axis a, semi-minor axis b, and position
        angle pa. The function returns a Polygon object representing the ellipse.
        
        Parameters
        ----------
        x0 : float
            The x-coordinate of the center of the ellipse.
        y0 : float
            The y-coordinate of the center of the ellipse.
        a : float
            The semi-major axis of the ellipse.
        b : float
            The semi-minor axis of the ellipse.
        pa : float
            The position angle of the ellipse in degrees, measured counter-clockwise from the positive x-axis.
        n : int, optional
            The number of points to use to approximate the ellipse, by default 200.
        
        Returns
        -------
        Polygon
            A Shapely Polygon object representing the ellipse.
        """
        # Generate n points evenly spaced around a unit circle
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        st = np.sin(theta)
        ct = np.cos(theta)

        # Convert the position angle from degrees to radians and adjust it by 90 degrees to align with the standard
        # mathematical convention for ellipses
        pa = np.deg2rad(pa + 90)
        sa = np.sin(pa)
        ca = np.cos(pa)

        # Calculate the coordinates of points using the parametric equations of an ellipse
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        return Polygon(p)


    def create_ellipse_list(self, clist : pd.DataFrame) -> list[Polygon]:
        """
        A function to create a list of Polygon objects representing the ellipses for each component in the component
        list.

        Parameters
        ----------
        clist : pd.DataFrame
            A DataFrame containing the component information for the source, with columns 'RA', 'DEC', 'DC_Maj',
            'DC_Min', and 'PA' representing the right ascension, declination, major axis, minor axis, and position
            angle of each component, respectively.
        
        Returns
        -------
        list[Polygon]
            A list of Shapely Polygon objects representing the ellipses for each component.
        """
        ellist = []
        for component in tqdm(clist.iterrows(),
                              desc="Creating ellipses for components...", disable=not self.show_progress):
            ra_n = component[1]['RA']
            dec_n = component[1]['DEC']

            # Convert the RA and DEC differences to arcseconds, accounting for the cosine of the declination for the RA
            # component
            x = 3600 * np.cos(self.dec * np.pi/180.0) * (self.ra - ra_n)
            y = 3600 * (dec_n - self.dec)

            # Get the major and minor axes of the ellipse representing the component, and convert to arcseconds.
            dc_maj_n = component[1]['DC_Maj'] * 3600
            dc_min_n = component[1]['DC_Min'] * 3600

            # Add a small buffer (0.1 arcseconds) to the major and minor axes to ensure that the ellipses overlap and
            # form a single connected shape, even if the components are very close together
            new_polygon = self.ellipse(x, y, dc_maj_n + 0.1, dc_min_n + 0.1, component[1]['PA'])
            ellist.append(new_polygon)

        return ellist


    def find_furthest_points(self, points: np.ndarray) -> tuple[tuple[tuple[float, float], tuple[float, float]], float]:
        """
        A function to find the pair of points in a given set of points that are furthest apart, and return these points
        as a tuple.

        Parameters
        ----------
        points : np.ndarray
            An array of shape (n, 2) containing the coordinates of the points.

        Returns
        -------
        tuple[tuple[tuple[float, float], tuple[float, float]], float]
            A tuple containing the pair of points that are furthest apart, where each point is represented as a tuple of
            (x, y) coordinates, along with the maximum distance.
        """
        mdist2 = 0
        best_coords = None
        for point in tqdm(points, desc="Finding furthest points...", disable=not self.show_progress):
            # Calculate the squared distance from the current point to all other points in the set, and find the maximum
            # distance
            dist2 = (points[:, 0] - point[0])**2.0 + (points[:, 1] - point[1])**2.0
            idist = np.argmax(dist2)
            mdist = dist2[idist]

            # Update the maximum distance and the corresponding pair of points
            if mdist > mdist2:
                mdist2 = mdist
                best_coords = (point, points[idist]) # (point, furthest_point)

        if best_coords is None:
            self.logger.error("No furthest points found. Check the input points array.")
            return ((0, 0), (0, 0)), 0

        return best_coords, mdist2


    def plot(self):
        """
        A method to plot the combined shape of the source and its convex hull, along with the points on the convex hull
        and the pair of points that are furthest apart, which are used to estimate the angular size of the source.
        """
        plt.figure(figsize=(8, 8))

        # Plot the combined shape of the source, which is formed by taking the union of ellipses representing each
        # Some sources are combined together, into a MultiPolygon
        if isinstance(self.combined_polygon, MultiPolygon):
            for geom in self.combined_polygon.geoms:
                x, y = geom.exterior.xy
                plt.plot(x, y, label='Combined Shape', color='blue')
        else:
            x, y = self.combined_polygon.exterior.xy  # type: ignore
            plt.plot(x, y, label='Combined Shape', color='blue')

        xh, yh = self.hull.exterior.xy  # type: ignore
        plt.plot(xh, yh, label='Convex Hull', color='orange')

        xh_points, yh_points = self.hull_points[:, 0], self.hull_points[:, 1]
        plt.scatter(xh_points, yh_points, label='Hull Points', color='green', s=10)

        if self.best_coords is not None:
            bestcoords_x = [self.best_coords[0][0], self.best_coords[1][0]]
            bestcoords_y = [self.best_coords[0][1], self.best_coords[1][1]]
            plt.plot(bestcoords_x, bestcoords_y,
                     label='Max Distance Pair', color='red', linewidth=2)

        plt.xlabel('DEC Offset (arcseconds)')
        plt.ylabel('RA Offset (arcseconds)')

        # Rotate the plot by 90 degrees to align with the standard astronomical convention, where RA increases to the
        # left and DEC increases upwards. This is done by applying an affine transformation to the plot.
        tr = transforms.Affine2D().rotate_deg(90) + transforms.Affine2D().translate(0, 0) + plt.gca().transData
        for line in plt.gca().get_lines():
            line.set_transform(tr)

        # Ensure the axes are equal to avoid distortion of the shape
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
        A method to calculate the angular size of the source based on the maximum distance between points on the convex
        hull of the union of ellipses representing the components of the source.    

        Returns
        -------
        float
            The estimated angular size of the source in arcseconds.
        """
        return np.sqrt(self.mdist2)



class AngularSizeFinder:
    """
    A class to estimate the angular size of a set of radio galaxy images on a 80x80 grid based on the component data
    extracted from PyBDSF catalogue FITS files.
    
    The class processes the FITS files, filters the components based on total flux, and estimates the angular size of
    the sources by creating a shape from the components and calculating the maximum distance between points on the
    convex hull of this shape.
    """
    def __init__(self,
                 root_dir: Path = paths.STORAGE_PARENT / "src/completeness/retrained_loguniform_catalogs",
                 flux_threshold: float = 0.95):
        """
        This class processes PyBDSF catalogue FITS files containing Gaussian component data for radio sources, filters
        the components based on total flux, and estimates the angular size of the sources by creating a shape from the
        components and calculating the maximum distance between points on the convex hull of this shape.

        Parameters
        ----------
        root_dir : Path, optional
            The root directory containing the FITS files to be processed, by default
            paths.STORAGE_PARENT / "src/completeness/retrained_loguniform_catalogs"
        flux_threshold : float, optional
            The fraction of total flux to keep when filtering components, by default 0.95. Components contributing to
            the dimmest flux are removed while keeping total flux above this threshold.
        """
        self.logger = utils.logging.get_logger("AngularSizeFinder", logging.DEBUG)
        self.root_dir = root_dir

        # Decide a flux threshold for filtering components. PyBDSF can sometimes fit islands to noise and so we sort and
        # then filter islands based on their fractional total flux. The threshold below represents the fraction of total
        # flux to keep, so the dimmest islands are removed while keeping total flux above this fractional threshold.
        self.flux_threshold = flux_threshold

        self.rfa = RecursiveFileAnalyzer(self.root_dir)


    # ---------- ASSEMBLING SIZE ESTIMATES ----------
    def extract_component_data(self, file_path: Path) -> list[tuple]:
        """
        Process a single FITS file to extract the component data for estimating the angular size of the source.
        
        Parameters
        ----------
        file_path : Path
            The path to the FITS file containing the component data for a single source.
        
        Returns
        -------
        list[tuple]
            A list of tuples, where each tuple contains the total flux, RA, DEC, major axis, minor axis, and position
            angle of a component. The components are filtered based on their fractional total flux.
        """
        components = []
        with fits.open(file_path) as hdul:
            data = hdul[1].data  # type: ignore
            for row in data:
                components.append((row["Total_flux"], row["RA"], row["DEC"], row["DC_Maj"], row["DC_Min"], row["PA"]))

        return self.filter_components(components)


    def filter_components(self, components: list[tuple]) -> list[tuple]:
        """
        Filter the components based on their fractional total flux, keeping only those components that contribute to a
        specified fraction of the total flux of the source.

        Parameters
        ----------
        components : list[tuple]
            A list of tuples representing the components, where each tuple contains the component's island ID, total
            flux, RA, DEC, major axis, minor axis, and position angle.

        Returns
        -------
        list[tuple]
            A list of tuples representing the filtered components.
        """
        assert components, "No components found in the data. Check the FITS file and the expected column names."

        # Sort components by total flux in descending order
        components.sort(key=lambda x: x[0], reverse=True)

        # Calculate the total flux of the source by summing the total flux of all components
        sum_flux = sum(component[0] for component in components)
        if sum_flux == 0:
            return []

        # Filter components based on their contribution to the total flux, removing the dimmest components while keeping
        # total flux above the specified threshold
        filtered_components = []
        cumulative_flux = 0
        for component in components:
            cumulative_flux += component[0]
            filtered_components.append(component)
            if cumulative_flux / sum_flux >= self.flux_threshold:
                break

        return filtered_components


    def fit_shape_and_estimate_size(self, components: list[tuple]) -> float:
        """
        Create a shape representing the source from the filtered components and estimate the angular size of the source
        based on this shape.

        Parameters
        ----------
        components : list[tuple]
            A list of tuples representing the filtered components, where each tuple contains the total flux, RA, DEC,
            major axis, minor axis, and position angle of a component.
        
        Returns
        -------
        float
            The estimated angular size of the source in arcseconds, calculated as the maximum distance between any two
            points on the convex hull of the combined shape formed by the components.
        """
        assert components, "No components to create shape from. Check the filtering step and the input data."

        # Create a shape representing the source from the filtered components by taking the union of ellipses
        # representing each component, where the ellipses are defined by the major and minor axes and position angle of
        # the components. The angular size is estimated as the maximum distance between any two points on the convex
        # hull of the combined shape.
        shape = MakeShape(pd.DataFrame(components, columns=['Total_flux', 'RA', 'DEC', 'DC_Maj', 'DC_Min', 'PA']))
        return shape.length()


    # ---------- RUNNING THE PIPELINE ----------
    def estimate_angular_sizes(self,
            fits_dir : str | Path | None = None,
            pattern : str = r'.*?\D+(\d+)\.fits$',
            output_file : str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to estimate the angular sizes of sources from the FITS files in the root directory, and optionally save
        the results to a CSV file.

        Parameters
        ----------
        fits_dir : str | Path | None, optional
            The root directory containing the FITS files, by default None.
        pattern : str, optional
            The regex pattern to match FITS files, by default r'.*?\D+(\d+)\.fits$'.
        output_file : str | Path | None, optional
            The name of the CSV file to save the estimated angular sizes to, by default None.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the indices of the sources and their estimated angular sizes.
        
        
        """
        # If the output file already exists, read the sizes from the file and return them along with the corresponding
        # indices
        if output_file is not None and os.path.exists(output_file):
            try:
                ang_sizes = np.genfromtxt(output_file, delimiter=',', skip_header=1)
            except Exception as e:
                raise Exception(f"Failed to read {output_file}. Please check the file and try again: {e}") from e
            _, fits_indices = self.rfa.get_unwrapped_list(path=fits_dir, pattern=pattern, return_nums=True)

            return np.array(fits_indices), ang_sizes

        # Run the pipeline to extract component data from each FITS file,
        if fits_dir:
            components_list, fits_indices = self.rfa.run_pipeline(function=self.extract_component_data,
                                                                  root_dir=fits_dir,
                                                                  pattern=pattern, return_nums=True, mode="file")
        else:
            components_list, fits_indices = self.rfa.run_pipeline(function=self.extract_component_data,
                                                                  pattern=pattern, return_nums=True, mode="file")

        # Estimate the angular size of each image based on the component data
        ang_sizes = []
        for components in components_list:
            # If there's only one component, either originally or after filtering, return a size based on DC_Maj
            if len(components) == 1:
                ang_sizes.append(2 * components[0][3] * 3600)
            else:
                angular_size = self.fit_shape_and_estimate_size(components)
                ang_sizes.append(angular_size)

        # Save the estimated angular sizes to a CSV file if an output file name is provided
        if output_file:
            df = pd.DataFrame(ang_sizes, columns=['Estimated Angular Size (arcseconds)'])
            df.to_csv(output_file, index=False)

        return np.array(fits_indices), np.array(ang_sizes)



if __name__ == "__main__":
    root = paths.STORAGE_PARENT / "src/completeness/dr2_cutouts_download_catalogs"
    SAVE_FILE = 'estimated_angular_sizes.csv'

    ang_size_finder = AngularSizeFinder(root)
    indices, sizes = ang_size_finder.estimate_angular_sizes(output_file=SAVE_FILE)

    # Check for estimated angular sizes that are above 250 arcseconds - "outliers"
    outliers = np.where(sizes > 250)[0]
    indices = np.delete(indices, outliers)
    sizes = np.delete(sizes, outliers)

    for i in range(0, round(max(sizes)), 5):
        print(f"Size bin: {i} - {i+5} arcseconds")
        print(f"Number of sources in this size bin: {len(sizes[(sizes >= i) & (sizes < i+5)])}")

    # Plot a histogram of the estimated angular sizes
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Estimated Angular Sizes of Radio Sources')
    plt.xlabel('Estimated Angular Size (arcseconds)')
    plt.ylabel('Number of Sources')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(SAVE_FILE.replace('.csv', '_distribution.png'))
    plt.show()
