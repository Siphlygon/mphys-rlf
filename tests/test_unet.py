"""
Unit tests for diffracc/model/unet.py.

Unet's own forward-pass tests use attention_levels=0 and norm_groups=1 to keep test models tiny and fast -
ResidualLinearAttention hardcodes nn.GroupNorm(32, dim), so any resolution level with attention needs channel counts
divisible by 32. One dedicated test builds a (still small) attention-enabled Unet to cover that path too.
"""
import pytest
import torch

from diffracc.model import layers
from diffracc.model import unet as unet_module
from diffracc.model.config import ModelConfig
from diffracc.model.unet import EDMPrecond, Unet, configModuleBase


def _tiny_unet_no_zero_init(monkeypatch, **overrides):
    """
    Every ResidualBlock (there are many throughout the Unet, not just the outermost output layer) ends its "computed"
    branch in a zero_module-wrapped conv. A freshly-built Unet therefore outputs exactly zero for any input - at each
    block, the skip/residual branch (identity or a plain conv) is the only one carrying gradient, since a zero-weight
    conv has a zero Jacobian and so backpropagates exactly zero gradient regardless of the loss used. 
    
    This is confirmed empirically, and this suppresses the time/label/context embeddings' effect on the output at every
    level they're used, not just the last one. That's real, intentional architecture behaviour (covered directly by
    TestZeroInitOutputLayer below), but it means tests that need real end-to-end signal (dropout-gating comparisons,
    embedding-gradient checks) must build the model with zero_module disabled.

    Two separate patches are needed: layers.py's own ResidualBlock.__init__ calls resolve `zero_module` via layers.py's
    module globals (so patching layers.zero_module fixes those), but unet.py did `from .layers import zero_module`,
    binding its own separate name into unet.py's namespace at import time - patching layers.zero_module alone does not
    change what unet.zero_module already points to.
    """
    monkeypatch.setattr(layers, "zero_module", lambda module: module)
    monkeypatch.setattr(unet_module, "zero_module", lambda module: module)
    return _tiny_unet(**overrides)


def _tiny_unet(**overrides):
    """
    Helper to build a tiny Unet for testing, with default parameters that keep the model small and fast while still
    covering all code paths (attention, context, label embeddings, etc.) at least once.
    """
    # init_channels=32 (with channel_mults up to x2 => bottleneck channels 64) so the middle block's attention -
    # which unet.py always includes regardless of attention_levels - satisfies ResidualLinearAttention's
    # hardcoded GroupNorm(32, dim) even when attention_levels=0 disables attention on the down/up path.
    defaults = dict(init_channels=32, channel_mults=(1, 2), image_channels=1, norm_groups=1,
                    attention_levels=0, num_res_blocks=1)
    defaults.update(overrides)
    return Unet(**defaults)


class TestConfigModuleBase:
    """
    Unit tests for the configModuleBase class, which provides the from_config() method to build a module from a
    ModelConfig.
    """
    
    def test_from_config_passes_all_config_params_to_init(self):
        """Test that from_config() passes all parameters in the ModelConfig to the target class's __init__ method."""

        class DummyModule(configModuleBase):
            """Class that takes some parameters in its __init__ to test from_config() passing them through."""
            def __init__(self, a, b=2, c=3):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

        config = ModelConfig(a=10, b=20)
        module = DummyModule.from_config(config)
        assert isinstance(module, DummyModule)
        assert module.a == 10
        assert module.b == 20
        assert module.c == 3  # default value preserved

    class _NeedsContextDim(configModuleBase):
        """
        A dummy target class for testing that from_config() sets context_dim based on the length of the context list.
        """
        def __init__(self, context_dim=0):
            super().__init__()
            self.context_dim = context_dim

    class _NoContextDim(configModuleBase):
        """
        A dummy target class for testing that from_config() does not fail when the target class has no context_dim.
        """
        def __init__(self, other=1):
            super().__init__()
            self.other = other

    def test_sets_context_dim_from_config_context_length(self):
        """Test that from_config() sets context_dim based on the length of the context list."""
        config = ModelConfig(context=["peak_flux", "las"])
        module = self._NeedsContextDim.from_config(config)
        assert module.context_dim == 2

    def test_no_context_attr_leaves_context_dim_untouched(self):
        """Test that from_config() does not set context_dim if the target class has no context_dim attribute."""
        config = ModelConfig(context_dim=5)
        module = self._NeedsContextDim.from_config(config)
        assert module.context_dim == 5

    def test_class_without_context_dim_param_is_unaffected(self):
        """Test that from_config() does not fail when the target class has no context_dim parameter."""
        config = ModelConfig(context=["a", "b"], other=3)
        module = self._NoContextDim.from_config(config)
        assert module.other == 3


class TestUnetForward:
    """
    Unit tests for the Unet's forward pass, covering different input/output channel counts, label and context
    embeddings, and attention.
    """

    def test_forward_pass_runs_without_errors(self):
        """
        Test that a forward pass through the Unet runs without errors and produces an output of the expected shape.
        """
        model = _tiny_unet()
        x = torch.randn(2, 1, 8, 8)
        time = torch.rand(2)
        out = model(x, time)
        assert out.shape == (2, 1, 8, 8)

    def test_default_output_shape_matches_input_channels(self):
        """Test that the default output shape matches the input channels."""
        model = _tiny_unet()
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2))
        assert out.shape == (2, 1, 8, 8)

    def test_custom_out_channels(self):
        """Test that specifying a custom number of output channels produces the expected output shape."""
        model = _tiny_unet(out_channels=3)
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2))
        assert out.shape == (2, 3, 8, 8)

    def test_multi_channel_input(self):
        """Test that specifying a multi-channel input produces the expected output shape."""
        model = _tiny_unet(image_channels=2)
        out = model(torch.randn(2, 2, 8, 8), torch.rand(2))
        assert out.shape == (2, 2, 8, 8)

    def test_class_labels_do_not_crash_and_affect_output(self):
        """Test that providing class labels does not crash the forward pass and affects the output."""
        # Regression test for the missing `torch.nn.functional` import that made F.one_hot raise NameError
        # any time class_labels were actually used.
        model = _tiny_unet(n_labels=3).eval()
        x = torch.randn(2, 1, 8, 8)
        time = torch.rand(2)
        labels = torch.tensor([0, 2])
        out = model(x, time, class_labels=labels)
        assert out.shape == (2, 1, 8, 8)
        assert torch.isfinite(out).all()

    def test_single_condition_context_path(self):
        """Test that providing a single-dimensional context does not crash the forward pass and affects the output."""
        model = _tiny_unet(context_dim=1).eval()
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2), context=torch.randn(2, 1))
        assert out.shape == (2, 1, 8, 8)

    def test_multi_condition_context_path_uses_additive_embedding(self):
        """Test that providing a multi-dimensional context does not crash the forward pass and affects the output."""
        model = _tiny_unet(context_dim=3).eval()
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2), context=torch.randn(2, 3))
        assert out.shape == (2, 1, 8, 8)

    def test_attention_enabled_path(self):
        """Test that enabling attention does not crash the forward pass and affects the output."""
        # Channel counts divisible by 32, needed by ResidualLinearAttention's hardcoded GroupNorm(32, ...).
        model = Unet(init_channels=32, channel_mults=(1, 2), image_channels=1, norm_groups=1,
                    attention_levels=1, num_res_blocks=1)
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2))
        assert out.shape == (2, 1, 8, 8)


class TestZeroInitOutputLayer:
    """
    Unit tests for the Unet's zero-initialized output layer, which should produce exactly zero output for any input.
    """

    def test_fresh_model_output_is_exactly_zero(self):
        """
        Test that a freshly built Unet with zero-initialized output layer produces exactly zero output for any input.
        """
        model = _tiny_unet()
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2))
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_holds_regardless_of_conditioning(self):
        """Test that the zero-output property holds regardless of class labels or context provided."""
        model = _tiny_unet(n_labels=3, context_dim=1)
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2), class_labels=torch.tensor([0, 1]),
                    context=torch.randn(2, 1))
        torch.testing.assert_close(out, torch.zeros_like(out))


class TestUnetDropoutGating:
    """Unit tests for the Unet's dropout gating functionality."""

    def test_label_dropout_only_applied_in_training_mode(self, monkeypatch):
        """Test that label_dropout is only applied in training mode, and not in evaluation mode."""
        model = _tiny_unet_no_zero_init(monkeypatch, n_labels=3, label_dropout=1.0)  # 100% drop rate
        x = torch.randn(2, 1, 8, 8)
        time = torch.rand(2)
        labels = torch.tensor([0, 1])

        model.eval()
        with_labels_eval = model(x, time, class_labels=labels)
        without_labels_eval = model(x, time, class_labels=None)
        # Eval mode never applies dropout, so a real (nonzero) label embedding is used - should differ from
        # not passing labels at all.
        assert not torch.allclose(with_labels_eval, without_labels_eval)

        model.train()
        with torch.no_grad():
            with_labels_train = model(x, time, class_labels=labels)
            without_labels_train = model(x, time, class_labels=None)
        # Train mode with label_dropout=1.0 always drops the label -> identical to not passing labels.
        torch.testing.assert_close(with_labels_train, without_labels_train)

    def test_context_dropout_only_applied_in_training_mode(self, monkeypatch):
        """Test that context_dropout is only applied in training mode, and not in evaluation mode."""
        model = _tiny_unet_no_zero_init(monkeypatch, context_dim=1, context_dropout=1.0)
        x = torch.randn(2, 1, 8, 8)
        time = torch.rand(2)
        context = torch.randn(2, 1)

        model.eval()
        with_context_eval = model(x, time, context=context)
        without_context_eval = model(x, time, context=None)
        assert not torch.allclose(with_context_eval, without_context_eval)

        model.train()
        with torch.no_grad():
            with_context_train = model(x, time, context=context)
            without_context_train = model(x, time, context=None)
        torch.testing.assert_close(with_context_train, without_context_train)

    def test_multi_condition_context_dropout_only_applied_in_training_mode(self, monkeypatch):
        """Test that multi-condition context_dropout is only applied in training mode, and not in evaluation mode."""
        model = _tiny_unet_no_zero_init(monkeypatch, context_dim=3, context_dropout=1.0)
        x = torch.randn(2, 1, 8, 8)
        time = torch.rand(2)
        context = torch.randn(2, 3)

        model.eval()
        with_context_eval = model(x, time, context=context)
        without_context_eval = model(x, time, context=None)
        assert not torch.allclose(with_context_eval, without_context_eval)

        model.train()
        with torch.no_grad():
            with_context_train = model(x, time, context=context)
            without_context_train = model(x, time, context=None)
        torch.testing.assert_close(with_context_train, without_context_train)


class TestUnetGradientFlow:
    """Unit tests to ensure that gradients flow through the Unet's parameters during backpropagation."""

    def test_base_parameters_receive_gradients(self, monkeypatch):
        """Test that the base parameters of the Unet receive gradients during backpropagation."""
        model = _tiny_unet_no_zero_init(monkeypatch)
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2))
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            assert torch.any(param.grad != 0), name

    def test_label_embedding_receives_gradient_when_labels_used(self, monkeypatch):
        """Test that the label embedding layer receives gradients when class labels are provided."""
        model = _tiny_unet_no_zero_init(monkeypatch, n_labels=3)
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2), class_labels=torch.tensor([0, 1]))
        out.sum().backward()
        assert model.label_emb.weight.grad is not None
        assert torch.any(model.label_emb.weight.grad != 0)

    def test_context_embedding_receives_gradient_when_context_used(self, monkeypatch):
        """Test that the context embedding layer receives gradients when context is provided."""
        model = _tiny_unet_no_zero_init(monkeypatch, context_dim=1)
        out = model(torch.randn(2, 1, 8, 8), torch.rand(2), context=torch.randn(2, 1))
        out.sum().backward()
        grads = [p.grad for p in model.context_emb.parameters()]
        assert all(g is not None for g in grads)
        assert any(torch.any(g != 0) for g in grads)


class _IdentityInnerModel(torch.nn.Module):
    """
    A fake inner model for EDMPrecond that returns its (preconditioned) input unchanged, i.e. F_x = c_in * x,
    isolating EDMPrecond's own preconditioning formula from the real Unet's behaviour.
    """
    def forward(self, x, time, context=None, class_labels=None):
        return x


class TestEDMPrecond:
    """Unit tests for the EDMPrecond wrapper, which implements the preconditioning formula around an inner model."""

    def test_matches_hand_computed_preconditioning_coefficients(self):
        """Test that EDMPrecond's output matches the hand-computed preconditioning formula for a simple inner model."""
        sigma_data = 0.5
        precond = EDMPrecond(_IdentityInnerModel(), sigma_data=sigma_data)
        x = torch.randn(2, 1, 4, 4)
        sigma = torch.tensor([1.0, 4.0])

        out = precond(x, sigma)

        sigma_v = sigma.view([-1, 1, 1, 1]).float()
        c_skip = sigma_data**2 / (sigma_v**2 + sigma_data**2)
        c_out = sigma_v * sigma_data / (sigma_v**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigma_data**2 + sigma_v**2).sqrt()
        expected = c_skip * x + c_out * (c_in * x)  # F_x = c_in*x since the inner model is the identity

        torch.testing.assert_close(out, expected)

    def test_output_shape_matches_input(self):
        """
        Test that EDMPrecond's output shape matches the input shape, regardless of batch size or spatial dimensions.
        """
        precond = EDMPrecond(_IdentityInnerModel())
        out = precond(torch.randn(3, 1, 8, 8), torch.tensor([1.0, 2.0, 3.0]))
        assert out.shape == (3, 1, 8, 8)

    def test_scalar_sigma_is_expanded_to_batch(self):
        """Test that a scalar sigma is correctly expanded to match the batch size of the input."""
        precond = EDMPrecond(_IdentityInnerModel())
        out = precond(torch.randn(3, 1, 4, 4), torch.tensor(2.0))
        assert out.shape == (3, 1, 4, 4)

    def test_from_config_builds_unet_and_wraps_it(self):
        """Test that EDMPrecond.from_config() correctly builds a Unet from the given ModelConfig and wraps it."""
        config = ModelConfig(init_channels=32, channel_mults=(1, 2), image_channels=1, norm_groups=1,
                             attention_levels=0, num_res_blocks=1, sigma_min=0.1, sigma_max=50.0, sigma_data=0.3)
        precond = EDMPrecond.from_config(config)
        assert isinstance(precond, EDMPrecond)
        assert isinstance(precond.model, Unet)
        assert precond.sigma_min == pytest.approx(0.1)
        assert precond.sigma_max == pytest.approx(50.0)
        assert precond.sigma_data == pytest.approx(0.3)

    def test_from_config_produces_a_working_forward_pass(self):
        """Test that EDMPrecond.from_config() produces a working forward pass with a Unet inside."""
        config = ModelConfig(init_channels=32, channel_mults=(1, 2), image_channels=1, norm_groups=1,
                             attention_levels=0, num_res_blocks=1)
        precond = EDMPrecond.from_config(config)
        out = precond(torch.randn(2, 1, 8, 8), torch.tensor([1.0, 2.0]))
        assert out.shape == (2, 1, 8, 8)
        assert torch.isfinite(out).all()
