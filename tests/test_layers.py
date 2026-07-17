"""
Unit tests for diffracc/model/layers.py.

ResidualBlock (and its DownsampleBlock/UpsampleBlock subclasses) end their out_layers in a zero_module-wrapped conv - a
standard "zero-init residual" trick. That means a freshly-initialized block's residual branch contributes exactly zero,
so the block's forward pass is exactly the identity on x (or exactly x_upd(x)/ res_conv(x) when resizing/changing
channels) regardless of its other (randomly-initialized) weights or the time embedding - verified empirically before
writing these tests, and used throughout as a hand-derivable invariant.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from diffracc.model import layers


class TestClampTensor:
    """Unit tests for layers.clamp_tensor, which is used to prevent NaNs from propagating through the model."""

    def test_values_within_range_are_unchanged(self):
        """Test that values within the clamping range are unchanged."""
        x = torch.tensor([1.0, -2.0, 3.5])
        torch.testing.assert_close(layers.clamp_tensor(x), x)

    def test_infinite_values_are_clamped_to_a_finite_value(self):
        """Test that infinite values are clamped to a finite value, and that the result is finite."""
        x = torch.tensor([torch.inf, -torch.inf])
        clamped = layers.clamp_tensor(x)
        assert torch.isfinite(clamped).all()

    def test_clamp_value_matches_dtype_max_minus_1000(self):
        """Test that the clamping value is equal to the maximum finite value for the tensor's dtype minus 1000."""
        clamp_value = torch.finfo(torch.float32).max - 1000
        x = torch.tensor([torch.inf, -torch.inf])
        clamped = layers.clamp_tensor(x)
        torch.testing.assert_close(clamped, torch.tensor([clamp_value, -clamp_value]))


class TestZeroModule:
    """Unit tests for layers.zero_module, which is used to zero-initialise the weights of a module."""

    def test_all_parameters_become_exactly_zero(self):
        """Test that all parameters of the module become exactly zero after applying zero_module."""
        module = nn.Linear(4, 4)
        layers.zero_module(module)
        for param in module.parameters():
            assert torch.all(param == 0)

    def test_returns_the_same_module_object(self):
        """Test that zero_module returns the same module object that was passed in, rather than creating a new one."""
        module = nn.Linear(4, 4)
        assert layers.zero_module(module) is module


class TestUpsampleDownsample:
    """Unit tests for layers.upsample and layers.downsample, which are used to resize feature maps."""

    def test_upsample_doubles_spatial_size(self):
        """Test that upsample doubles the spatial size of the input tensor."""
        up = layers.upsample(4)
        x = torch.randn(2, 4, 8, 8)
        out = up(x)
        assert out.shape == (2, 4, 16, 16)

    def test_upsample_use_conv_false_keeps_channels_and_is_conv_free(self):
        """
        Test that upsample with use_conv=False keeps the number of channels the same and does not use a convolution.
        """
        up = layers.upsample(4, out_channels=99, use_conv=False)  # out_channels ignored when use_conv=False
        assert isinstance(up[1], nn.Identity)
        x = torch.randn(1, 4, 4, 4)
        assert up(x).shape == (1, 4, 8, 8)

    def test_upsample_use_conv_true_changes_channels(self):
        """Test that upsample with use_conv=True changes the number of channels."""
        up = layers.upsample(4, out_channels=6, use_conv=True)
        x = torch.randn(1, 4, 4, 4)
        assert up(x).shape == (1, 6, 8, 8)

    def test_downsample_halves_spatial_size(self):
        """Test that downsample halves the spatial size of the input tensor."""
        down = layers.downsample(4)
        x = torch.randn(2, 4, 8, 8)
        out = down(x)
        assert out.shape == (2, 4, 4, 4)

    def test_downsample_changes_channels_when_requested(self):
        """Test that downsample changes the number of channels when requested."""
        down = layers.downsample(4, out_channels=10)
        x = torch.randn(1, 4, 8, 8)
        assert down(x).shape == (1, 10, 4, 4)


class TestTimestepEmbedSequential:
    """
    Unit tests for layers.TimestepEmbedSequential, which is used to sequentially apply layers that may or may not
    require a timestep embedding.
    """

    class _TimestepAdd(layers.TimestepBlock):
        """A simple TimestepBlock that adds the timestep embedding to the input tensor."""
        def forward(self, x, emb):
            return x + emb

    def test_dispatches_emb_only_to_timestep_blocks(self):
        """
        Test that TimestepEmbedSequential dispatches the timestep embedding only to layers that are TimestepBlocks.
        """
        seq = layers.TimestepEmbedSequential(self._TimestepAdd(), nn.ReLU())
        x = torch.tensor([-1.0, 2.0])
        emb = torch.tensor([1.0, 1.0])
        out = seq(x, emb)
        # (x + emb) = [0, 3], then ReLU -> [0, 3]
        torch.testing.assert_close(out, torch.tensor([0.0, 3.0]))


class TestSinusoidalEmbedding:
    """Unit tests for layers.SinusoidalEmbedding, which is used to embed timesteps into a higher-dimensional space."""

    def test_output_shape(self):
        """Test that the output shape of SinusoidalEmbedding is correct given an input tensor of timesteps."""
        emb = layers.SinusoidalEmbedding(dim=8)
        out = emb(torch.tensor([1.0, 2.0, 3.0]))
        assert out.shape == (3, 8)

    def test_matches_hand_computed_formula(self):
        """Test that SinusoidalEmbedding matches the hand-computed formula for sinusoidal embeddings."""
        dim = 4
        emb = layers.SinusoidalEmbedding(dim=dim)
        time = torch.tensor([0.0, 2.0])
        out = emb(time)

        half_dim = dim // 2
        freqs = math.log(1e5) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim) * -freqs)
        embeddings = time[:, None] * freqs[None, :]
        expected = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        torch.testing.assert_close(out, expected)

    def test_zero_time_gives_sin_zero_cos_one(self):
        """Test that a timestep of zero gives a sinusoidal embedding of [0, 0, ..., 1, 1]."""
        emb = layers.SinusoidalEmbedding(dim=6)
        out = emb(torch.tensor([0.0]))
        half = 3
        torch.testing.assert_close(out[0, :half], torch.zeros(half))
        torch.testing.assert_close(out[0, half:], torch.ones(half))


class TestFourierEmbedding:
    """Unit tests for layers.FourierEmbedding, which is used to embed timesteps into a higher-dimensional space."""

    def test_output_shape(self):
        """Test that the output shape of FourierEmbedding is correct given an input tensor of timesteps."""
        emb = layers.FourierEmbedding(dim=8)
        out = emb(torch.tensor([1.0, 2.0, 3.0]))
        assert out.shape == (3, 8)

    def test_matches_hand_computed_formula(self):
        """Test that FourierEmbedding matches the hand-computed formula for Fourier embeddings."""
        emb = layers.FourierEmbedding(dim=4, scale=1.0)
        x = torch.tensor([0.5, -1.0])
        out = emb(x)
        expected = x.outer(2 * np.pi * emb.freqs)
        expected = torch.cat([expected.cos(), expected.sin()], dim=-1)
        torch.testing.assert_close(out, expected)


class TestLinearFeatureEmbedding:
    """
    Unit tests for layers.LinearFeatureEmbedding, which is used to embed features into a higher-dimensional space using
    a linear layer.
    """

    def test_output_shape(self):
        """Test that the output shape of LinearFeatureEmbedding is correct given an input tensor of features."""
        emb = layers.LinearFeatureEmbedding(dim_in=3, dim_out=8)
        out = emb(torch.randn(5, 3))
        assert out.shape == (5, 8)

    def test_gradients_flow_to_all_parameters(self):
        """Test that gradients flow to all parameters of LinearFeatureEmbedding during backpropagation."""
        emb = layers.LinearFeatureEmbedding(dim_in=3, dim_out=8)
        out = emb(torch.randn(5, 3, requires_grad=False))
        out.sum().backward()
        for name, param in emb.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all()


class TestAdditiveContextEmbedding:
    """
    Unit tests for layers.AdditiveContextEmbedding, which is used to embed features into a higher-dimensional space
    using independent sub-embeddings for each feature.
    """

    def test_embed_each_shape(self):
        """Test that the output shape of embed_each is correct given an input tensor of features."""
        emb = layers.AdditiveContextEmbedding(3, layers.LinearFeatureEmbedding, dim_in=1, dim_out=8)
        out = emb.embed_each(torch.randn(5, 3))
        assert out.shape == (5, 3, 8)

    def test_forward_is_sum_of_embed_each(self):
        """
        Test that the forward pass of AdditiveContextEmbedding is equal to the sum of the embeddings from embed_each.
        """
        emb = layers.AdditiveContextEmbedding(3, layers.LinearFeatureEmbedding, dim_in=1, dim_out=8)
        x = torch.randn(5, 3)
        per_feature = emb.embed_each(x)
        torch.testing.assert_close(emb(x), per_feature.sum(dim=1))

    def test_each_feature_uses_an_independent_sub_embedding(self):
        """
        Test that each feature uses an independent sub-embedding, so different feature values produce different
        embeddings.
        """
        emb = layers.AdditiveContextEmbedding(2, layers.LinearFeatureEmbedding, dim_in=1, dim_out=4)
        # Different feature values in column 0 vs column 1 should generally produce different per-feature
        # embeddings, since each feature has its own (independently initialised) embedding layer.
        x = torch.tensor([[1.0, 1.0]])
        per_feature = emb.embed_each(x)
        assert not torch.allclose(per_feature[0, 0], per_feature[0, 1])


class TestWeightStandardizedConv2d:
    """Unit tests for layers.WeightStandardizedConv2d, which is a convolutional layer with weight standardisation."""

    def test_output_shape(self):
        """Test that the output shape of WeightStandardizedConv2d is correct given an input tensor."""
        conv = layers.WeightStandardizedConv2d(3, 5, 3, padding=1)
        out = conv(torch.randn(2, 3, 8, 8))
        assert out.shape == (2, 5, 8, 8)

    def test_matches_hand_computed_weight_standardization(self):
        """Test that WeightStandardizedConv2d matches the hand-computed weight standardisation."""
        conv = layers.WeightStandardizedConv2d(1, 2, 2, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[0.0, 0.0], [0.0, 10.0]]]]))
        x = torch.randn(1, 1, 4, 4)

        out = conv(x)

        eps = 1e-5
        weight = conv.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        normalized = (weight - mean) * (var + eps).rsqrt()
        expected = torch.nn.functional.conv2d(x, normalized, None)

        torch.testing.assert_close(out, expected)


class TestResidualLinearAttention:
    """Unit tests for layers.ResidualLinearAttention, which is a linear attention layer with a residual connection."""

    def test_output_shape_matches_input(self):
        """Test that the output shape of ResidualLinearAttention matches the input shape."""
        attn = layers.ResidualLinearAttention(dim=32, heads=2, head_channels=4)
        x = torch.randn(2, 32, 5, 5)
        assert attn(x).shape == x.shape

    def test_is_identity_when_output_projection_is_zeroed(self):
        """Test that ResidualLinearAttention is the identity function when the output projection is zeroed."""
        # The residual connection means zeroing to_out's weights (which zero_module already does for the real
        # ResidualBlock's own final layer, though not for this class itself) makes the block the identity.
        attn = layers.ResidualLinearAttention(dim=32, heads=2, head_channels=4)
        layers.zero_module(attn.to_out[0])
        attn.to_out[1] = nn.Identity()  # remove the GroupNorm on the (now always-zero) branch too
        x = torch.randn(2, 32, 5, 5)
        torch.testing.assert_close(attn(x), x)

    def test_gradients_flow_to_all_parameters(self):
        """Test that gradients flow to all parameters of ResidualLinearAttention during backpropagation."""
        attn = layers.ResidualLinearAttention(dim=32, heads=2, head_channels=4)
        out = attn(torch.randn(2, 32, 5, 5))
        out.sum().backward()
        for name, param in attn.named_parameters():
            assert param.grad is not None, name


class TestResidualBlock:
    """Unit tests for layers.ResidualBlock, which is a residual block with optional downsampling/upsampling."""

    def test_output_shape_same_channels(self):
        """Test that the output shape of ResidualBlock is correct when the input and output channels are the same."""
        block = layers.ResidualBlock(4, 4, 16, norm_groups=1)
        out = block(torch.randn(2, 4, 8, 8), torch.randn(2, 16))
        assert out.shape == (2, 4, 8, 8)

    def test_output_shape_changed_channels(self):
        """Test that the output shape of ResidualBlock is correct when the input and output channels are different."""
        block = layers.ResidualBlock(4, 8, 16, norm_groups=1)
        out = block(torch.randn(2, 4, 8, 8), torch.randn(2, 16))
        assert out.shape == (2, 8, 8, 8)

    def test_at_init_same_channel_block_is_exactly_the_identity(self):
        """
        Test that a freshly-initialised ResidualBlock with the same input and output channels is exactly the
        identity.
        """
        block = layers.ResidualBlock(4, 4, 16, norm_groups=1).eval()
        x = torch.randn(2, 4, 8, 8)
        out = block(x, torch.randn(2, 16))
        torch.testing.assert_close(out, x)

    def test_identity_holds_regardless_of_time_embedding(self):
        """
        Test that the identity property of a freshly-initialised ResidualBlock holds regardless of the time embedding.
        """
        block = layers.ResidualBlock(4, 4, 16, norm_groups=1).eval()
        x = torch.randn(2, 4, 8, 8)
        out_a = block(x, torch.zeros(2, 16))
        out_b = block(x, torch.randn(2, 16) * 100)
        torch.testing.assert_close(out_a, x)
        torch.testing.assert_close(out_b, x)

    def test_at_init_changed_channel_block_equals_res_conv(self):
        """
        Test that a freshly-initialised ResidualBlock with different input and output channels equals the residual
        convolution.
        """
        block = layers.ResidualBlock(4, 8, 16, norm_groups=1).eval()
        x = torch.randn(2, 4, 8, 8)
        out = block(x, torch.randn(2, 16))
        expected = block.res_conv(x)
        torch.testing.assert_close(out, expected)

    def test_gradients_flow_to_all_parameters(self):
        """Test that gradients flow to all parameters of ResidualBlock during backpropagation."""
        block = layers.ResidualBlock(4, 4, 16, norm_groups=1, dropout=0.0)
        out = block(torch.randn(2, 4, 8, 8), torch.randn(2, 16))
        out.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all()


class TestDownsampleUpsampleBlock:
    """
    Unit tests for layers.DownsampleBlock and layers.UpsampleBlock, which are residual blocks that downsample or
    upsample the input.
    """

    def test_downsample_block_halves_spatial_size(self):
        """Test that DownsampleBlock halves the spatial size of the input tensor."""
        block = layers.DownsampleBlock(4, 16, norm_groups=1)
        out = block(torch.randn(2, 4, 8, 8), torch.randn(2, 16))
        assert out.shape == (2, 4, 4, 4)

    def test_upsample_block_doubles_spatial_size(self):
        """Test that UpsampleBlock doubles the spatial size of the input tensor."""
        block = layers.UpsampleBlock(4, 16, norm_groups=1)
        out = block(torch.randn(2, 4, 8, 8), torch.randn(2, 16))
        assert out.shape == (2, 4, 16, 16)

    def test_at_init_downsample_block_equals_x_upd(self):
        """Test that a freshly-initialised DownsampleBlock equals the x_upd function, which is the identity on x."""
        block = layers.DownsampleBlock(4, 16, norm_groups=1).eval()
        x = torch.randn(2, 4, 8, 8)
        out = block(x, torch.randn(2, 16))
        torch.testing.assert_close(out, block.x_upd(x))

    def test_at_init_upsample_block_equals_x_upd(self):
        """Test that a freshly-initialised UpsampleBlock equals the x_upd function, which is the identity on x."""
        block = layers.UpsampleBlock(4, 16, norm_groups=1).eval()
        x = torch.randn(2, 4, 8, 8)
        out = block(x, torch.randn(2, 16))
        torch.testing.assert_close(out, block.x_upd(x))


class TestResidualBlockAttention:
    """Unit tests for layers.ResidualBlockAttention, which combines a residual block and an attention layer."""

    def test_applies_res_block_then_attention(self):
        """Test that ResidualBlockAttention applies the residual block first, then the attention layer."""
        res_block = layers.ResidualBlock(32, 32, 16, norm_groups=1).eval()
        attn = layers.ResidualLinearAttention(32, heads=2, head_channels=4)
        combined = layers.ResidualBlockAttention(res_block, attn)

        x = torch.randn(2, 32, 8, 8)
        emb = torch.randn(2, 16)
        out = combined(x, emb)

        expected = attn(res_block(x, emb))
        torch.testing.assert_close(out, expected)
