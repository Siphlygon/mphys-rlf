import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import simple_sample
import plotting.image_plots


class A50Finder:
    """
    A class to manually find the quantity A50, which is an area containing 50% of the total flux intensity of the source. It does this through two functions create_A50_list, which creates and returns the list containing the data of all pixels in the A50, and plot_A50_contour, which plots the contour of the A50 region on top of the image.
    """

    def __init__(self):
        pass

    def create_A50_list(self, image):
        """
        Creates a list of all pixels in the A50 region.

        Args:
            image (np.ndarray): The input image from which to extract the A50 region.

        Returns:
            np.ndarray: A structured array containing the data of all pixels in the A50 region.
        """
        # Create a structured array to hold pixel positions and brightness
        A50_array = np.array([])

        # Add positional information to every pixel
        positions = np.array(np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))).T.reshape(-1, 2)
        brightness = image.flatten()
        pixels = np.hstack((positions, brightness[:, None]))

        # Sort pixels by brightness in descending order
        sorted_pixels = pixels[np.argsort(-pixels[:, 2])]

        # Calculate total brightness
        total_brightness = np.sum(image)

        # Add pixels to A_50 until total brightness exceeds 50%
        cumulative_brightness = 0
        for pixel in sorted_pixels:
            A50_array = np.append(A50_array, pixel)
            cumulative_brightness += pixel[2]

            # A50 is
            if cumulative_brightness > 0.5 * total_brightness:
                break

        return A50_array

    def plot_A50_contour(self, image, A50_array):
        """
        Plots a contour level highlighting the A50 area on a given image.

        Args:
            image (np.ndarray): The input image from which to extract the A50 region.
            A50_array (np.ndarray): A structured array containing the data of all pixels in the A50 region.

        Returns:
            None
        """
        # Create a50 contour level
        A50_level = [np.min(A50_array)]

        x_mesh, y_mesh = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        contour = plt.contour(x_mesh, y_mesh, image, levels=A50_level, colors=['white'])

        # note: return is unlikely to be used
        return contour


if __name__ == "__main__":
    # Create an instance of A50finder
    A50_finder = A50Finder()

    # Run the quick sample to get an image
    simple_sampler = simple_sample.SimpleSampler()
    samples = simple_sampler.run(
        model_name="LOFAR_model",
        distribute_model=False,
        return_steps=False  # this returns only the final image
    )

    # Grab the image array from the samples
    final_image = samples[0][0]

    # Plot the final image
    fig, ax = plotting.image_plots.plot_image_grid(samples[0])

    # Create the A50 list
    A50 = A50_finder.create_A50_list(final_image)

    # Plot the A50 contour
    A50_finder.plot_A50_contour(final_image, A50)

    # Save the contour plot
    fig.savefig("a50contour.png")
