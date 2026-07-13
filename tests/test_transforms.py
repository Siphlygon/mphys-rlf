"""Unit tests for diffracc/data/transforms.py's image transform helpers."""
import numpy as np
import pytest
import torch

from diffracc.data import transforms as tr


class TestSafeToTensor:
    """Tests that safe_to_tensor correctly converts various input types to torch.Tensor."""

    def test_passes_through_existing_tensor_unchanged(self):
        """Test that safe_to_tensor returns the same tensor instance if the input is already a torch.Tensor."""
        t = torch.rand(3, 4, 4)
        assert tr.safe_to_tensor(t) is t

    def test_converts_numpy_array_to_tensor(self):
        """Test that safe_to_tensor converts a numpy array to a torch.Tensor with the correct shape."""
        arr = np.random.rand(4, 4, 3).astype(np.float32)  # HWC, as torchvision's to_tensor expects
        out = tr.safe_to_tensor(arr)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 4, 4)  # to_tensor converts HWC -> CHW


class TestToTensor:
    """Tests that ToTensor transform correctly wraps safe_to_tensor."""

    def test_returns_a_lambda_transform_that_wraps_safe_to_tensor(self):
        """Test that ToTensor returns a callable that applies safe_to_tensor to its input."""
        transform = tr.ToTensor()
        t = torch.rand(2, 5, 5)
        assert transform(t) is t


class TestSingleChannel:
    """Tests that single_channel correctly processes multi-channel and 2D inputs."""

    def test_multi_channel_tensor_keeps_only_first_channel(self):
        """Test that single_channel returns a tensor with only the first channel of a multi-channel input."""
        img = torch.rand(3, 8, 8)
        out = tr.single_channel(img)
        assert out.shape == (1, 8, 8)
        torch.testing.assert_close(out[0], img[0])

    def test_already_single_channel_tensor_unchanged(self):
        """Test that single_channel returns the same tensor if it already has a single channel."""
        img = torch.rand(1, 8, 8)
        out = tr.single_channel(img)
        torch.testing.assert_close(out, img)

    def test_2d_tensor_gets_channel_dim_added(self):
        """Test that single_channel adds a channel dimension to a 2D tensor."""
        img = torch.rand(8, 8)
        out = tr.single_channel(img)
        assert out.shape == (1, 8, 8)

    def test_2d_numpy_array_gets_channel_dim_added(self):
        """Test that single_channel adds a channel dimension to a 2D numpy array."""
        img = np.random.rand(8, 8)
        out = tr.single_channel(img)
        assert out.shape == (1, 8, 8)
        assert isinstance(out, np.ndarray)


class TestRandomRotate90Tensor:
    """Tests that random_rotate_90_tensor correctly rotates tensors by multiples of 90 degrees."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_matches_torch_rot90_for_given_k(self, monkeypatch, k):
        """Test that random_rotate_90_tensor produces the same result as torch.rot90 for a given k."""
        monkeypatch.setattr(tr.random, "choice", lambda seq: k)
        img = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        out = tr.random_rotate_90_tensor(img)
        expected = torch.rot90(img, k=k, dims=(-2, -1))
        torch.testing.assert_close(out, expected)

    def test_preserves_shape(self, monkeypatch):
        """Test that random_rotate_90_tensor preserves the shape of the input tensor regardless of rotation."""
        monkeypatch.setattr(tr.random, "choice", lambda seq: 1)
        img = torch.rand(1, 80, 80)
        assert tr.random_rotate_90_tensor(img).shape == img.shape

    def test_preserves_the_set_of_pixel_values(self, monkeypatch):
        """Test that random_rotate_90_tensor preserves the set of pixel values in the input tensor."""
        monkeypatch.setattr(tr.random, "choice", lambda seq: 2)
        img = torch.rand(1, 10, 10)
        out = tr.random_rotate_90_tensor(img)
        np.testing.assert_allclose(np.sort(out.flatten().numpy()), np.sort(img.flatten().numpy()))

    def test_2d_input_gets_unsqueezed(self, monkeypatch):
        """Test that random_rotate_90_tensor adds a channel dimension to 2D input arrays."""
        monkeypatch.setattr(tr.random, "choice", lambda seq: 0)
        img = torch.rand(10, 10)
        out = tr.random_rotate_90_tensor(img)
        assert out.shape == (1, 10, 10)

    def test_accepts_numpy_input(self, monkeypatch):
        """Test that random_rotate_90_tensor can accept a numpy array as input and returns a torch.Tensor."""
        monkeypatch.setattr(tr.random, "choice", lambda seq: 0)
        img = np.random.rand(10, 10).astype(np.float32)
        out = tr.random_rotate_90_tensor(img)
        assert isinstance(out, torch.Tensor)


class TestTrainTransformNoScale:
    """Tests that TrainTransformNoScale correctly applies the transformation pipeline to input images."""

    def test_pipeline_runs_and_produces_single_channel_tensor(self):
        """Test that TrainTransformNoScale processes an input image and returns a single-channel torch.Tensor."""
        transform = tr.TrainTransformNoScale(image_size=40)
        img = np.random.rand(40, 40, 1).astype(np.float32)
        out = transform(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 40, 40)
