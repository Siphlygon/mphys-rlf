"""Unit tests for diffracc/plotting/image_plots.py."""
import numpy as np
import pytest

from diffracc.plotting import image_plots as ip


class TestPlotImageGrid:
    """Unit tests for the plot_image_grid function."""

    def test_auto_computes_square_grid_for_perfect_square_count(self):
        """Test that the function automatically computes a square grid when the number of images is a perfect square."""
        imgs = np.zeros((9, 4, 4))
        fig, axs = ip.plot_image_grid(imgs)
        assert axs.shape == (3, 3)

    def test_auto_computes_grid_with_extra_row_for_non_square_count(self):
        """
        Test that the function automatically computes a grid with an extra row when the number of images is not a
        perfect square.
        """
        # sqrt(10) truncates to 3 -> 3 cols, plus enough extra rows to fit all 10 images (3x3=9, needs 1 more row)
        imgs = np.zeros((10, 4, 4))
        fig, axs = ip.plot_image_grid(imgs)
        assert axs.shape[1] == 3
        assert axs.shape[0] * axs.shape[1] >= 10

    def test_respects_explicit_n_rows_and_n_cols(self):
        """Test that the function respects explicitly provided n_rows and n_cols arguments."""
        imgs = np.zeros((6, 4, 4))
        fig, axs = ip.plot_image_grid(imgs, n_rows=2, n_cols=3)
        assert axs.shape == (2, 3)

    def test_infers_n_cols_from_n_rows(self):
        """Test that the function infers n_cols from n_rows when only n_rows is provided."""
        imgs = np.zeros((6, 4, 4))
        fig, axs = ip.plot_image_grid(imgs, n_rows=2)
        assert axs.shape[0] == 2
        assert axs.shape[0] * axs.shape[1] >= 6

    def test_infers_n_rows_from_n_cols(self):
        """Test that the function infers n_rows from n_cols when only n_cols is provided."""
        imgs = np.zeros((6, 4, 4))
        fig, axs = ip.plot_image_grid(imgs, n_cols=3)
        assert axs.shape[1] == 3
        assert axs.shape[0] * axs.shape[1] >= 6

    def test_accepts_list_of_images(self):
        """Test that the function accepts a list of images as input."""
        imgs = [np.zeros((4, 4)) for _ in range(4)]
        fig, axs = ip.plot_image_grid(imgs)
        assert axs.shape == (2, 2)

    def test_mismatched_titles_length_raises_assertion_error(self):
        """Test that an AssertionError is raised when the number of titles does not match the number of images."""
        imgs = np.zeros((4, 4, 4))
        with pytest.raises(AssertionError):
            ip.plot_image_grid(imgs, titles=["only", "two"])

    def test_float_title_is_formatted_in_scientific_notation(self):
        """Test that a float title is formatted in scientific notation."""
        imgs = np.zeros((1, 4, 4))
        fig, axs = ip.plot_image_grid(imgs, titles=[1234.5678], n_rows=1, n_cols=1)
        ax = axs if not isinstance(axs, np.ndarray) else axs.flat[0]
        assert ax.get_title() == f"{1234.5678:.2e}"

    def test_int_title_is_formatted_as_plain_string(self):
        """Test that an int title is formatted as a plain string."""
        imgs = np.zeros((1, 4, 4))
        fig, axs = ip.plot_image_grid(imgs, titles=[42], n_rows=1, n_cols=1)
        ax = axs if not isinstance(axs, np.ndarray) else axs.flat[0]
        assert ax.get_title() == "42"

    def test_string_title_is_used_as_is(self):
        """Test that a string title is used as-is."""
        imgs = np.zeros((1, 4, 4))
        fig, axs = ip.plot_image_grid(imgs, titles=["my label"], n_rows=1, n_cols=1)
        ax = axs if not isinstance(axs, np.ndarray) else axs.flat[0]
        assert ax.get_title() == "my label"

    def test_suptitle_is_set_on_figure(self):
        """Test that the suptitle is set on the figure when provided."""
        imgs = np.zeros((1, 4, 4))
        fig, axs = ip.plot_image_grid(imgs, suptitle="My Grid", n_rows=1, n_cols=1)
        assert fig._suptitle.get_text() == "My Grid"

    def test_savefig_writes_a_file(self, tmp_path):
        """Test that the savefig argument writes a file to the specified path."""
        imgs = np.zeros((1, 4, 4))
        out_path = tmp_path / "grid.png"
        ip.plot_image_grid(imgs, savefig=out_path, n_rows=1, n_cols=1)
        assert out_path.exists()

    def test_reuses_provided_fig_axs(self):
        """Test that the function reuses provided figure and axes when fig_axs is given."""
        imgs = np.zeros((1, 4, 4))
        fig, axs = ip.plot_image_grid(np.zeros((1, 4, 4)), n_rows=1, n_cols=1)
        fig2, axs2 = ip.plot_image_grid(imgs, fig_axs=(fig, axs))
        assert fig2 is fig
        assert axs2 is axs


class TestRandomImageGrid:
    """Unit tests for the random_image_grid function."""

    def test_selects_n_img_random_images_from_dataset(self):
        """Test that the function selects n_img random images from the provided dataset."""
        dset = [np.full((4, 4), i) for i in range(20)]
        fig, axs = ip.random_image_grid(dset, n_img=4)
        assert axs.shape == (2, 2)

    def test_unpacks_context_tuples(self):
        """Test that the function correctly unpacks (image, context) tuples in the dataset."""
        # each dataset item is (image, context) - random_image_grid should plot the image, not the tuple
        dset = [(np.full((4, 4), float(i)), {"label": i}) for i in range(5)]
        fig, axs = ip.random_image_grid(dset, n_img=1, n_rows=1, n_cols=1)
        # no error means the (image, context) tuple was correctly unpacked to just the image

    def test_idx_titles_sets_titles_to_dataset_indices(self):
        """Test that the idx_titles argument sets the titles of the axes to the indices of the images in the dataset."""
        dset = [np.full((4, 4), float(i)) for i in range(5)]
        fig, axs = ip.random_image_grid(dset, n_img=5, idx_titles=True, n_rows=1, n_cols=5)
        titles = {ax.get_title() for ax in axs.flat}
        assert titles == {"0", "1", "2", "3", "4"}


class TestPlotImageGridFromFile:
    """Tests for the plot_image_grid_from_file function."""

    def test_loads_tensor_and_plots_last_timestep_first_channel(self, tmp_path, monkeypatch):
        """Test that the function loads a tensor from a file and plots the last timestep of the first channel."""
        import torch

        fake_tensor = torch.zeros((6, 3, 1, 4, 4))  # (n_img, T, C, H, W)
        fake_tensor[:, -1, 0] = torch.arange(6).reshape(6, 1, 1).float()

        monkeypatch.setattr(ip.torch, "load", lambda path, map_location=None: fake_tensor)

        path = tmp_path / "samples.pt"
        path.touch()
        fig, axs = ip.plot_image_grid_from_file(path, n_img=6)
        # sqrt(6) truncates to 2 -> 2 cols, plus enough extra rows to fit all 6 images (2x2=4, needs 2 more -> 1 more row)
        assert axs.shape == (3, 2)

    def test_save_true_writes_grid_png_next_to_source_file(self, tmp_path, monkeypatch):
        """Test that the save argument writes a grid.png file next to the source .pt file."""
        import torch

        fake_tensor = torch.zeros((1, 2, 1, 4, 4))
        monkeypatch.setattr(ip.torch, "load", lambda path, map_location=None: fake_tensor)

        path = tmp_path / "samples.pt"
        path.touch()
        ip.plot_image_grid_from_file(path, save=True, n_img=1, n_rows=1, n_cols=1)
        assert (tmp_path / "samples_grid.png").exists()

    def test_save_false_does_not_write_a_file(self, tmp_path, monkeypatch):
        """Test that the save argument set to False does not write a file."""
        import torch

        fake_tensor = torch.zeros((1, 2, 1, 4, 4))
        monkeypatch.setattr(ip.torch, "load", lambda path, map_location=None: fake_tensor)

        path = tmp_path / "samples.pt"
        path.touch()
        ip.plot_image_grid_from_file(path, save=False, n_img=1, n_rows=1, n_cols=1)
        assert not (tmp_path / "samples_grid.png").exists()
