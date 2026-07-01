"""
Created following the tutorial on 
https://huggingface.co/blog/annotated-diffusion

"""

from inspect import signature
from typing import Any

import torch.nn as nn

from model.layers import *
from model.config import ModelConfig


class configModuleBase(nn.Module):
    """
    A base class for modules that use a configuration object.

    This class provides a class method to construct an instance of the class
    using a configuration object.
    """

    @classmethod
    def from_config(
        cls: type, config: ModelConfig, *args: Any, **kwargs: Any
    ) -> nn.Module:
        """
        Construct a neural network module from a given configuration.

        Parameters
        ----------
        cls : type
            The class of the neural network module.
        config : modelConfig
            The configuration object containing the necessary parameters for constructing the module.
        *args : Any
            Additional positional arguments to be passed to the constructor of the module.
        **kwargs : Any
            Additional keyword arguments to be passed to the constructor of the module.

        Returns
        -------
        nn.Module
            The constructed neural network module.

        Notes
        -----
        This method is used to create a neural network module from a given
        configuration object. It is a class method, meaning it can be called
        on the class itself without the need for an instance.

        The `config` parameter should be an instance of the `modelConfig`
        class, which contains the necessary parameters for constructing the
        module. The `*args` and `**kwargs` parameters are used to pass
        additional arguments to the constructor of the module, if needed.

        As a special case for the Unet class:
        If the `config` object has a `context` attribute and the `cls` class
        has a `context_dim` parameter, the `context_dim` parameter of the
        config object will be set to the length of the `context` attribute.

        The method returns the constructed neural network module.

        Examples
        --------
        >>> config = modelConfig(...)
        >>> module = Unet.from_config(config, ...)
        """
        # Special case for Unet
        if (
            hasattr(config, "context")
            and "context_dim" in signature(cls).parameters.keys()
        ):
            config.context_dim = len(config.context)
        return config.construct(cls, *args, **kwargs)


class Unet(configModuleBase):
    """
    U-Net model implementation.

    Attributes
    ----------
    init_channels : int
        Number of initial channels.
    out_channels : int, optional
        Number of output channels. If not provided, it defaults to the number of input channels.
    channel_mults : tuple of int, optional
        Multipliers for the number of channels at each resolution level. Default is (1, 2, 4, 8).
        This also determines the number of resolution levels.
    image_channels : int, optional
        Number of input image channels. Default is 1.
    n_labels : int, optional
        Number of class labels. Default is 0, i.e. unlabeled.
    context_dim : int, optional
        Dimension of the context vector. Default is 0, i.e. no context.
    label_dropout : float, optional
        Dropout rate for the class label embedding. Default is 0.
    context_dropout : float, optional
        Dropout rate for the context embedding. Default is 0.
    norm_groups : int, optional
        Number of groups for group normalization. Default is 32.
    dropout : float, optional
        Dropout rate for the residual blocks. Default is 0.
    num_res_blocks : int, optional
        Number of residual blocks at each resolution level. Default is 2.
    attention_levels : int, optional
        Number of resolution levels with attention blocks. Default is 3.
        Counting starts at lowest resolution levels.
    attention_heads : int, optional
        Number of heads for attention layers. Default is 4.
    attention_head_channels : int, optional
        Number of channels in each attention head. Default is 32.
    feature_emb : nn.Module
        Feature embedding layer.
    time_emb : nn.Module
        Time embedding layer.
    label_emb : nn.Module
        Label embedding layer.
    context_emb : nn.Module
        Context embedding layer.
    init_conv : nn.Module
        Initial convolution layer.
    down_blocks : nn.ModuleList
        List of down blocks.
    up_blocks : nn.ModuleList
        List of up blocks.
    middle_block : nn.Module
        Middle block.
    out : nn.Module
        Output layer.
    """

    def __init__(
        self,
        init_channels,
        out_channels=None,
        channel_mults=(1, 2, 4, 8),
        image_channels=1,
        n_labels=0,
        context_dim=0,
        label_dropout=0,
        context_dropout=0,
        norm_groups=32,
        dropout=0,
        num_res_blocks=2,
        attention_levels=3,
        attention_heads=4,
        attention_head_channels=32,
    ):
        super().__init__()

        # Determine channels
        self.input_channels = image_channels
        self.out_channels = out_channels or image_channels
        self.init_channels = init_channels

        # Time and label embeddings
        emb_dim = init_channels * 4
        self.time_emb = SinusoidalEmbedding(emb_dim)
        self.label_emb = nn.Linear(n_labels, emb_dim, bias=False) if n_labels else None
        self.label_dropout = label_dropout
        self.n_labels = n_labels

        # Context embedding
        self.context_emb = (
            LinearFeatureEmbedding(context_dim, emb_dim) if context_dim else None
        )
        self.context_dim = context_dim
        self.context_dropout = context_dropout

        # Feature embedding
        self.feature_emb = LinearFeatureEmbedding(emb_dim, emb_dim)

        # Initial convolution layer
        self.init_conv = WeightStandardizedConv2d(
            self.input_channels, self.init_channels, 3, padding=1
        )
        input_block_chans = [self.init_channels]

        # Create lists of down- and up-blocks
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        ch = self.init_channels
        n_levels = len(channel_mults)

        # Fill down-block list
        for res_level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                # Create residual block
                block = ResidualBlock(
                    ch,
                    int(mult * init_channels),
                    emb_dim,
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                ch = int(mult * init_channels)

                # Add attention layer to block if necessary
                if n_levels - res_level <= attention_levels:
                    attn = ResidualLinearAttention(
                        ch, heads=attention_heads, head_channels=attention_head_channels
                    )
                    block = ResidualBlockAttention(block, attn)

                # Add residual(-attention) block to down-list
                self.down_blocks.append(block)
                input_block_chans.append(ch)

            # Add downsample block if not last resolution level
            if res_level != n_levels - 1:
                block = DownsampleBlock(
                    ch, emb_dim, dropout=dropout, norm_groups=norm_groups
                )
                self.down_blocks.append(block)
                input_block_chans.append(ch)

        # Create middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, emb_dim, dropout=dropout, norm_groups=norm_groups),
            ResidualLinearAttention(
                ch, heads=attention_heads, head_channels=attention_head_channels
            ),
            ResidualBlock(ch, ch, emb_dim, dropout=dropout, norm_groups=norm_groups),
        )

        # Fill up-blocks (loop in reverse, hence [::-1])
        for res_level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                # Get residual input channels from list
                ich = input_block_chans.pop()

                # Create residual block
                block = ResidualBlock(
                    ch + ich,
                    int(mult * init_channels),
                    emb_dim,
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                ch = int(mult * init_channels)

                # Add attention layer to block if necessary
                if n_levels - res_level <= attention_levels:
                    attn = ResidualLinearAttention(
                        ch, heads=attention_heads, head_channels=attention_head_channels
                    )
                    block = ResidualBlockAttention(block, attn)

                # Add residual(-attention) block to up-list
                self.up_blocks.append(block)

                # Add upsample block if not last resolution level
                if res_level and i == num_res_blocks:
                    block = UpsampleBlock(
                        ch, emb_dim, dropout=dropout, norm_groups=norm_groups
                    )
                    self.up_blocks.append(block)

        # Final residual block
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, time, context=None, class_labels=None):
        """
        Forward pass of the U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        time : torch.Tensor
            Time parameter tensor of shape (batch_size, 1).
        context : torch.Tensor, optional
            Context tensor of shape (batch_size, context_dim). Default is None.
        class_labels : torch.Tensor, optional
            Class label tensor of shape (batch_size,). Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Time embedding
        emb = self.time_emb(time)

        # Class label embedding
        if self.label_emb is not None and class_labels is not None:
            labels_enc = F.one_hot(class_labels, num_classes=self.n_labels).to(x.dtype)

            # Apply label dropout
            if self.training and self.label_dropout:
                mask = torch.rand([x.shape[0], 1], device=x.device)
                mask = mask >= self.label_dropout
                labels_enc = labels_enc * mask.to(labels_enc.dtype)

            # Add label embedding to time embedding & apply activation
            emb = emb + self.label_emb(labels_enc)

        # Context embedding
        if self.context_emb is not None and context is not None:
            context_emb = self.context_emb(context.to(x.dtype))

            if self.training and self.context_dropout:
                mask = torch.rand([x.shape[0], 1], device=x.device)
                mask = mask >= self.context_dropout
                context_emb = context_emb * mask.to(context_emb.dtype)

            emb = emb + context_emb

        # Feature embedding
        emb = self.feature_emb(emb)

        # Initial convolution
        x = self.init_conv(x)
        h = [x]

        # Encoder
        for module in self.down_blocks:
            x = module(x, emb)
            h.append(x)

        # Middle block
        x = self.middle_block(x, emb)

        # Decoder
        for module in self.up_blocks:
            if not isinstance(module, UpsampleBlock):
                x = torch.cat((x, h.pop()), dim=1)
            x = module(x, emb)
        return self.out(x)


class EDMPrecond(configModuleBase):
    """
    Wrapper for Unet to apply preconditioning as introduced in EDM paper.

    Attributes
    ----------
    model : Unet
        The inner model used for applying preconditioning.
    sigma_min : numeric, optional
        The minimum value for the sigma parameter.
    sigma_max : numeric, optional
        The maximum value for the sigma parameter.
    sigma_data : numeric, optional
        The sigma_data parameter.
    """

    def __init__(
        self,
        model,
        sigma_min=0,
        sigma_max=torch.inf,
        sigma_data=0.5,
    ):
        """
        Initialize the wrapper model.

        Parameters
        ----------
        model : Unet
            The inner model used for applying preconditioning.
        sigma_min : numeric, optional
            The minimum value for the sigma parameter. Default is 0.
        sigma_max : numeric, optional
            The maximum value for the sigma parameter. Default is infinity.
        sigma_data : numeric, optional
            The sigma_data parameter. Default is 0.5.
        """
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    @classmethod
    def from_config(cls, config):
        model = Unet.from_config(config)
        return config.construct(cls, model=model)

    def forward(self, x, sigma, context=None, class_labels=None):
        """
        Forward pass of the EDMPrecond model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        sigma : torch.Tensor
            The noise level, i.e. time parameter value.
        context : torch.Tensor, optional
            Context tensor. Default is None.
        class_labels : torch.Tensor, optional
            Class label tensor. Default is None.

        Returns
        -------
        torch.Tensor
            The denoised output tensor.
        """

        # Expand sigma to shape [batch_size, 1, 1, 1]
        sigma = sigma.view([-1, 1, 1, 1])

        # Weight coefficients for each term
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        # Apply inner model
        F_x = self.model(
            c_in * x, c_noise.flatten(), context=context, class_labels=class_labels
        )

        # Generate denoiser output
        D_x = c_skip * x + c_out * F_x

        return D_x
