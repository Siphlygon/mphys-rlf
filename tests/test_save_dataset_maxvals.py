"""Unit tests for diffracc/dataset_prep/save_dataset_maxvals.py."""
import h5py
import numpy as np

from diffracc.dataset_prep.save_dataset_maxvals import write_maxvals_of_h5_to_file


def _write_images_h5(path, images: np.ndarray):
    """Write a numpy array of images to an HDF5 file at the given path, under the key 'images'."""
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images)


class TestWriteMaxvalsOfH5ToFile:
    """Tests for the write_maxvals_of_h5_to_file() function in save_dataset_maxvals.py."""

    def test_saves_per_image_max_values(self, tmp_path):
        """Test that write_maxvals_of_h5_to_file() correctly saves the maximum pixel values of each image."""
        images = np.zeros((3, 4, 4), dtype=np.float32)
        images[0, 1, 1] = 5.0
        images[1, 2, 2] = -1.0  # max of an otherwise-zero image is 0.0
        images[2, 0, 0] = 9.0
        infile = tmp_path / "dataset.h5"
        outfile = tmp_path / "maxvals.npy"
        _write_images_h5(infile, images)

        write_maxvals_of_h5_to_file(outfile, infile)

        np.testing.assert_allclose(np.load(outfile), [5.0, 0.0, 9.0])

    def test_output_length_matches_number_of_images(self, tmp_path):
        """Test that the output array length matches the number of images in the input HDF5 file."""
        images = np.random.default_rng(0).normal(size=(10, 8, 8)).astype(np.float32)
        infile = tmp_path / "dataset.h5"
        outfile = tmp_path / "maxvals.npy"
        _write_images_h5(infile, images)

        write_maxvals_of_h5_to_file(outfile, infile)

        assert np.load(outfile).shape == (10,)
