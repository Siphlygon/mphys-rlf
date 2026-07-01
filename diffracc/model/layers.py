import math
from functools import partial
from abc import abstractmethod

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def clamp_tensor(x):
    """
    Clamp the tensor values to avoid overflow or underflow.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Clamped tensor.
    """
    dtype = torch.half if torch.is_autocast_enabled() else x.dtype
    clamp_value = torch.finfo(dtype).max - 1000
    x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def zero_module(module):
    """
    Sets all parameters of a module to zero. Used for initializing the
    optimizers.

    Parameters
    ----------
    module : nn.Module
        Module to be zeroed.

    Returns
    -------
    nn.Module
        Zeroed module.
    """
    for param in module.parameters():
        param.detach().zero_()
    return module


def upsample(in_channels, out_channels=None, use_conv=True):
    """
    Upsampling layer, NxN --> 2Nx2N, using nearest neighbor algorithm.
    Basically, every pixel is quadrupled, and the new pixels are filled with
    the value of the original pixel.

    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int, optional
        Output channels, if None (default) the number of channels is conserved.
    use_conv : bool, optional
        Whether to apply convolution layer after upsampling. True by default.

    Returns
    -------
    nn.Sequential
        Upsampling layer.
    """
    return nn.Sequential(
        # Upsampling quadruples each pixel.
        nn.Upsample(scale_factor=2, mode="nearest"),
        # Convolution leaves image size unchanged.
        (
            nn.Conv2d(in_channels, (out_channels or in_channels), 3, padding=1)
            if use_conv
            else nn.Identity()
        ),
    )


def downsample(in_channels, out_channels=None):
    """
    Downsampling layer, NxN -> N/2 x N/2. Works by splitting image into 4, then
    doing 1x1 convolution using the 4 sub-images as input channels.
    In the original U-Net, this is done by max pooling.

    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int, optional
        Output channels, if None (default) the number of channels is conserved.

    Returns
    -------
    nn.Sequential
        Downsampling layer.
    """
    return nn.Sequential(
        # Rearrange: Split each image into 4 smaller and concatenate along
        # channel dimension.
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # Convolution leaves image sizes unchanged, changes channel dimensions
        # back to original (or specified dim_out).
        nn.Conv2d(in_channels * 4, (out_channels or in_channels), 1),
    )


class TimestepBlock(nn.Module):
    """
    Abstract base class for any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        emb : torch.Tensor
            Timestep embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential module where forward() takes timestep embeddings as a second
    argument.
    """

    def forward(self, x, emb):
        """
        Apply the sequential module to `x` given `emb` timestep embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        emb : torch.Tensor
            Timestep embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SinusoidalEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, 1) corresponding to the time values of
    the noised images, and returns embedding of shape (batch_size, dim).
    For a good explanation of the embedding formula, see:
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

    """

    def __init__(self, dim):
        """
        Initialize the SinusoidalEmbedding layer.

        Parameters
        ----------
        dim : int
            Number of dimensions of the embedding vector.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Forward pass of the SinusoidalEmbedding layer.

        Parameters
        ----------
        time : torch.Tensor
            Input tensor of shape (batch_size, 1) representing the time values.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        freqs = math.log(1e5) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freqs)
        embeddings = time[:, None] * freqs[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FourierEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, 1) corresponding to values of a
    continuous context feature, and returns
    embedding of shape (batch_size, dim).
    For a good explanation of the embedding formula, see:
    https://bmild.github.io/fourfeat/
    """

    def __init__(self, dim, scale=16):
        """
        Initialize the FourierEmbedding layer.

        Parameters
        ----------
        dim : int
            The dimension of the layer.
        scale : int, optional
            Scaling factor for the frequencies, by default 16.
        """
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        """
        Forward pass of the FourierEmbedding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Embedding tensor.
        """
        x = x.outer((2 * np.pi * self.freqs.to(x.device)).to(x.dtype))
        embeddings = torch.cat([x.cos(), x.sin()], dim=-1)
        return embeddings


class LinearFeatureEmbedding(nn.Module):
    """
    Linear embedding module, which is used to inject context information into the
    model.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize the LinearFeatureEmbedding layer.

        Parameters
        ----------
        dim_in : int
            Input dimension.
        dim_out : int
            Output dimension.
        """
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        """
        Forward pass of the LinearFeatureEmbedding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        return x


class AdditiveContextEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, context_dim) corresponding to the
    values of the entire context, and returns embedding
    of shape (batch_size, dim), which is the sum of embeddings
    of all single features. Works for different embedding layers.
    """

    def __init__(self, context_dim, emb_cls, **cls_kwargs):
        """
        Initialize the AdditiveContextEmbedding layer.

        Parameters
        ----------
        context_dim : int
            Dimension of the context.
        emb_cls : nn.Module
            Embedding layer class.
        **cls_kwargs : dict
            Additional keyword arguments to be passed to the embedding class.
        """
        super().__init__()
        self.emb_layers = [emb_cls(**cls_kwargs) for _ in range(context_dim)]
        for i, layer in enumerate(self.emb_layers):
            self.add_module(f"emb_{i}", layer)

    def forward(self, x):
        """
        Forward pass of the AdditiveContextEmbedding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Embedding tensor.
        """
        print(f"Context on {x.device}")
        return sum(
            layer(x[:, i].view(-1, 1)) for i, layer in enumerate(self.emb_layers)
        )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight-standardized 2d convolutional layer, built from a standard
    conv2d layer. Works better with group normalization.
    https://arxiv.org/abs/1903.10520
    https://kushaj.medium.com/weight-standardization-a-new-normalization-in-town-54b2088ce355 #noqa
    """

    def forward(self, x):
        """
        Forward pass of the WeightStandardizedConv2d layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3  # Epsilon
        weight = self.weight

        # Tensors with mean and variance, same shape as weight tensors
        # for subsequent operations.
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")  # o = outp. channels
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        out = F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Clamp inf values to avoid Infs/NaNs
        if torch.isinf(out).any():
            out = clamp_tensor(out)

        return out


class ResidualLinearAttention(nn.Module):
    """
    Basically the same as regular multi-head attention, but this implementation
    is more efficient, (linear vs quadratic).
    To be exact, when using softmax this is not precisely mathematically
    equivalent, but a very good approximation.
    https://arxiv.org/abs/1812.01243

    Parameters
    ----------
    dim : int
        Dimension of the input.
    heads : int, optional
        Number of attention heads, by default 4.
    dim_head : int, optional
        Dimension of each attention head, by default 32.
    """

    def __init__(self, dim, heads=4, head_channels=32):
        """
        Initialize the ResidualLinearAttention layer.

        Parameters
        ----------
        dim : int
            Dimension of the input.
        heads : int, optional
            Number of attention heads, by default 4.
        head_channels : int, optional
            Number of channels of each attention head, by default 32.
        """
        super().__init__()
        self.scale = head_channels**-0.5
        self.heads = heads
        hidden_dim = head_channels * heads
        self.pre_norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=True)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, bias=True), nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        """
        Forward pass of the ResidualLinearAttention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        res = x
        _, _, h, w = x.shape

        # qkv: Tuple of 3 tensors of shape [b, dim_head*heads, h, w]:
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # Reshape to three tensors of shape [b, heads, dim_head, h*w]:
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q * self.scale

        # Trick to prevent overflow in softmax
        q_shift = q - q.amax(dim=-2, keepdim=True).detach()
        k_shift = k - k.amax(dim=-1, keepdim=True).detach()

        q_norm = q_shift.softmax(dim=-2)
        k_norm = k_shift.softmax(dim=-1)

        context = torch.einsum("b h d n, b h e n -> b h d e", k_norm, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q_norm)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out) + res


class ResidualBlock(TimestepBlock):
    """
    Basic U-Net Block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
        up=False,
        down=False,
        use_conv=False,
    ):
        """
        Initialize the Layer class.

        Parameters
        ----------
        in_channels : int
            The input channels of the layer.
        out_channels : int
            The output channels of the layer.
        time_emb_dim : int
            The dimensions of the time embedding vector.
        dropout : float, optional
            The dropout probability, by default 0.0.
        norm_groups : int, optional
            The number of groups for group normalization, by default 32.
        up : bool, optional
            Whether to perform upsampling, by default False.
        down : bool, optional
            Whether to perform downsampling, by default False.
        use_conv : bool, optional
            Whether to use convolutional layers, for upsampling by default False.
            If true, a 3x3 convolution is also applied to the residual layer in
            the case of resampling. If false, this convolution is 1x1, i.e. FCN.
        """
        super().__init__()

        self.resize = up or down
        self.do_res_conv = in_channels != out_channels

        # Input layers
        self.in_layers = nn.Sequential(
            nn.GroupNorm(norm_groups, in_channels),
            nn.SiLU(),
            WeightStandardizedConv2d(
                in_channels, out_channels, 3, padding=1, bias=True
            ),
        )

        # Resampling layers
        if up:
            self.h_upd = upsample(in_channels, use_conv=False)
            self.x_upd = upsample(in_channels, use_conv=False)

        elif down:
            self.h_upd = downsample(in_channels)
            self.x_upd = downsample(in_channels)

        # Time embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )

        # Output layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                WeightStandardizedConv2d(
                    out_channels, out_channels, 3, padding=1, bias=True
                )
            ),
        )

        # Residual layers
        if self.do_res_conv:
            self.res_conv = (
                WeightStandardizedConv2d(in_channels, out_channels, 3)
                if use_conv
                else nn.Conv2d(in_channels, out_channels, 1)
            )

    def forward(self, x, time_emb):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        time_emb : torch.Tensor
            The time embedding tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the forward pass.
        """
        # Input Layers
        if self.resize:
            # Split input layers into Norm+SiLU and Conv
            # (up/downsample will happen in between)
            in_pre, in_conv = self.in_layers[:-1], self.in_layers[-1]

            # Hidden state
            h = in_pre(x)  # Norm+SiLU
            h = self.h_upd(h)  # Up/downsample
            h = in_conv(h)  # Conv

            # Residual
            x = self.x_upd(x)  # Up/downsample residual

        else:
            h = self.in_layers(x)

        # Time embedding
        time_emb = self.emb_layers(time_emb)  # SiLU and Linear
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale, shift = time_emb.chunk(2, dim=1)

        # Output layers
        # Split output layers into Norm and SiLU+Dropout+Conv
        # (time embedding will be applied in between)
        out_norm, out_post = self.out_layers[0], self.out_layers[1:]
        # Apply Norm and time embedding
        h = out_norm(h) * (scale + 1) + shift
        # Apply SiLU+Dropout+Conv
        h = out_post(h)
        # Residual layer
        if self.do_res_conv:
            x = self.res_conv(x)

        return clamp_tensor(h + x)


class ResidualBlockAttention(nn.Module):
    """
    Sequential module that applies a residual block and an attention block.
    """

    def __init__(self, resBlock, attnBlock):
        """
        Initialize the class.

        Parameters
        ----------
        resBlock : ResBlock
            The ResBlock object.
        attnBlock : AttnBlock
            The AttnBlock object.
        """
        super().__init__()
        self.resBlock = resBlock
        self.attnBlock = attnBlock

    def forward(self, x, time_emb):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        time_emb : torch.Tensor
            The time embedding tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the layer.
        """
        x = self.resBlock(x, time_emb)
        x = self.attnBlock(x)
        return x


class DownsampleBlock(ResidualBlock):
    """
    Basic U-Net block with downsampling.
    """

    def __init__(
        self,
        channels,
        time_emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
    ):
        """
        Initialize the Layer class.

        Parameters
        ----------
        channels : int
            The number of input and output channels.
        time_emb_dim : int
            The dimension of the time embedding.
        dropout : float, optional
            The dropout rate, by default 0.0.
        norm_groups : int, optional
            The number of groups to normalize the input channels, by default 32.
        """
        super().__init__(
            channels,
            channels,
            time_emb_dim,
            dropout=dropout,
            norm_groups=norm_groups,
            up=False,
            down=True,
            use_conv=False,
        )


class UpsampleBlock(ResidualBlock):
    """
    Basic U-Net block with upsampling.
    """

    def __init__(
        self,
        channels,
        time_emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
    ):
        """
        Initialize the Layer class.

        Parameters
        ----------
        channels : int
            The number of input and output channels.
        time_emb_dim : int
            The dimension of the time embedding.
        dropout : float, optional
            The dropout rate, by default 0.0.
        norm_groups : int, optional
            The number of groups to normalize the channels, by default 32.
        """
        super().__init__(
            channels,
            channels,
            time_emb_dim,
            dropout=dropout,
            norm_groups=norm_groups,
            up=True,
            down=False,
            use_conv=False,
        )
