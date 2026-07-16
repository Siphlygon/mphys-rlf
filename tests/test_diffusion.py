"""
Unit tests for diffracc/model/diffusion.py.

edm_sampling is tested against a tiny fake denoiser (not a real Unet/EDMPrecond - unet.py has its own test file)
that always denoises to exactly zero. With that denoiser, both the Euler step and the 2nd-order trapezoidal
correction reduce to the same closed form x_next = x_cur * sigma_next / sigma_cur (worked out by hand below), which
makes the whole sampling trajectory exactly predictable without needing any real network.
"""
import numpy as np
import pytest
import torch

from diffracc.model import diffusion


class TestGetSamplingNoiseLevels:
    """
    Tests for the get_sampling_noise_levels() function in diffusion.py, which generates a sequence of noise levels for
    the EDM sampler.
    """

    def test_length_is_timesteps_plus_one(self):
        """Test that the returned noise levels tensor has length timesteps + 1 (the last one is always zero)."""
        sigmas = diffusion.get_sampling_noise_levels(10)
        assert sigmas.shape == (11,)

    def test_first_step_equals_sigma_max(self):
        """Test that the first noise level returned is equal to sigma_max (the starting noise level)."""
        sigmas = diffusion.get_sampling_noise_levels(10, sigma_min=2e-3, sigma_max=80)
        assert sigmas[0].item() == pytest.approx(80.0)

    def test_last_real_step_equals_sigma_min(self):
        """Test that the last real noise level returned is equal to sigma_min."""
        sigmas = diffusion.get_sampling_noise_levels(10, sigma_min=2e-3, sigma_max=80)
        assert sigmas[-2].item() == pytest.approx(2e-3, rel=1e-5)

    def test_appended_final_value_is_exactly_zero(self):
        """Test that the final appended noise level is exactly zero, regardless of sigma_min."""
        sigmas = diffusion.get_sampling_noise_levels(10)
        assert sigmas[-1].item() == 0.0

    def test_monotonically_decreasing(self):
        """Test that the noise levels are monotonically decreasing (strictly) from sigma_max to sigma_min."""
        sigmas = diffusion.get_sampling_noise_levels(10)
        assert torch.all(sigmas[:-1] > sigmas[1:])

    def test_matches_hand_computed_formula(self):
        """Test that the noise levels match the hand-derived formula for the EDM noise schedule."""
        timesteps, sigma_min, sigma_max, rho = 4, 1.0, 16.0, 2.0
        sigmas = diffusion.get_sampling_noise_levels(timesteps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        step_inds = np.arange(timesteps)
        rho_inv = 1 / rho
        expected = (sigma_max**rho_inv + step_inds / (timesteps - 1) * (sigma_min**rho_inv - sigma_max**rho_inv)) ** rho
        expected = np.append(expected, 0.0)
        np.testing.assert_allclose(sigmas.numpy(), expected, rtol=1e-6)


class TestStochasticChurn:
    """Tests for the stochastic_churn() function in diffusion.py, which implements the stochastic churn mechanism."""

    def test_zero_churn_leaves_sigma_and_image_unchanged(self):
        """Test that when S_churn=0, the output sigma and image are unchanged from the input."""
        sigma_cur = torch.tensor(5.0)
        x_cur = torch.randn(2, 1, 4, 4)
        sigma_hat, x_hat = diffusion.stochastic_churn(10, S_churn=0, S_min=0, S_max=torch.inf,
                                                       S_noise=1, sigma_cur=sigma_cur, x_cur=x_cur)
        assert sigma_hat.item() == pytest.approx(5.0)
        torch.testing.assert_close(x_hat, x_cur)

    def test_sigma_outside_smin_smax_range_is_a_no_op(self):
        """Test that when sigma_cur is outside the [S_min, S_max] range, the output sigma and image are unchanged."""
        sigma_cur = torch.tensor(100.0)  # outside [S_min, S_max] below
        x_cur = torch.randn(2, 1, 4, 4)
        sigma_hat, x_hat = diffusion.stochastic_churn(10, S_churn=5, S_min=0, S_max=50,
                                                       S_noise=1, sigma_cur=sigma_cur, x_cur=x_cur)
        assert sigma_hat.item() == pytest.approx(100.0)
        torch.testing.assert_close(x_hat, x_cur)

    def test_sigma_hat_matches_gamma_formula(self):
        """
        Test that the output sigma_hat matches the formula sigma_hat = (1 + gamma) * sigma_cur, where gamma is defined
        in the EDM paper.
        """
        timesteps, S_churn = 20, 2.0
        sigma_cur = torch.tensor(5.0)
        x_cur = torch.zeros(1, 1, 2, 2)
        sigma_hat, _ = diffusion.stochastic_churn(timesteps, S_churn=S_churn, S_min=0, S_max=torch.inf,
                                                   S_noise=1, sigma_cur=sigma_cur, x_cur=x_cur)
        gamma = min(S_churn / timesteps, np.sqrt(2) - 1)
        assert sigma_hat.item() == pytest.approx((1 + gamma) * 5.0)

    def test_gamma_is_capped_at_sqrt2_minus_1(self):
        """Test that when S_churn/timesteps > sqrt(2)-1, gamma is capped at sqrt(2)-1."""
        # S_churn/timesteps = 100 >> sqrt(2)-1, so gamma should clip to sqrt(2)-1
        sigma_cur = torch.tensor(1.0)
        x_cur = torch.zeros(1, 1, 2, 2)
        sigma_hat, _ = diffusion.stochastic_churn(1, S_churn=100.0, S_min=0, S_max=torch.inf,
                                                   S_noise=1, sigma_cur=sigma_cur, x_cur=x_cur)
        assert sigma_hat.item() == pytest.approx((1 + (np.sqrt(2) - 1)) * 1.0)

    def test_injected_noise_std_matches_formula(self):
        """
        Test that the standard deviation of the injected noise matches the formula
        S_noise * sqrt(sigma_hat^2 - sigma_cur^2).
        """
        torch.manual_seed(0)
        timesteps, S_churn, S_noise = 10, 5.0, 2.0
        sigma_cur = torch.tensor(3.0)
        x_cur = torch.zeros(1, 1, 500, 500)  # large tensor for a statistically reliable std estimate
        sigma_hat, x_hat = diffusion.stochastic_churn(timesteps, S_churn=S_churn, S_min=0, S_max=torch.inf,
                                                       S_noise=S_noise, sigma_cur=sigma_cur, x_cur=x_cur)
        expected_std = (torch.sqrt(sigma_hat**2 - sigma_cur**2) * S_noise).item()
        actual_std = (x_hat - x_cur).std().item()
        assert actual_std == pytest.approx(expected_std, rel=0.05)


class _FakeConditionalModel:
    """Records calls; returns 1.0 when unconditioned, 2.0 when context or class_labels are supplied."""
    def __init__(self):
        self.calls = []

    def __call__(self, img, sigma, context=None, class_labels=None):
        self.calls.append({"sigma": sigma.clone(), "context": context, "class_labels": class_labels})
        value = 2.0 if (context is not None or class_labels is not None) else 1.0
        return torch.full_like(img, value)


class TestDenoisedGuided:
    """Tests for the denoised_guided() function in diffusion.py, which implements classifier-free guidance."""

    def test_no_context_or_labels_calls_model_once_unconditioned(self):
        """
        Test that when no context or class_labels are provided, the model is called once unconditioned and the output is
        correct.
        """
        model = _FakeConditionalModel()
        img = torch.zeros(3, 1, 4, 4)
        out = diffusion.denoised_guided(model, img, torch.tensor(1.0))
        assert len(model.calls) == 1
        torch.testing.assert_close(out, torch.full_like(img, 1.0))

    def test_zero_guidance_strength_skips_conditioned_pass_even_with_context(self):
        """Test that when guidance_strength is 0, the conditioned pass is skipped even if context is provided."""
        model = _FakeConditionalModel()
        img = torch.zeros(2, 1, 4, 4)
        context = torch.ones(2, 3)
        out = diffusion.denoised_guided(model, img, torch.tensor(1.0), context=context, guidance_strength=0)
        assert len(model.calls) == 1
        torch.testing.assert_close(out, torch.full_like(img, 1.0))

    def test_context_with_nonzero_guidance_combines_conditioned_and_unconditioned(self):
        """
        Test that when context is provided and guidance_strength > 0, the output is a linear combination of the
        conditioned and unconditioned outputs.
        """
        model = _FakeConditionalModel()
        img = torch.zeros(2, 1, 4, 4)
        context = torch.ones(2, 3)
        guidance_strength = 0.5
        out = diffusion.denoised_guided(model, img, torch.tensor(1.0), context=context,
                                        guidance_strength=guidance_strength)
        assert len(model.calls) == 2
        # (1+g)*cond - g*uncond = 1.5*2.0 - 0.5*1.0 = 2.5
        expected = (1 + guidance_strength) * 2.0 - guidance_strength * 1.0
        torch.testing.assert_close(out, torch.full_like(img, expected))

    def test_sigma_is_expanded_to_batch_size(self):
        """
        Test that when sigma is a scalar, it is expanded to match the batch size of the input image before being passed
        to the model.
        """
        model = _FakeConditionalModel()
        img = torch.zeros(4, 1, 4, 4)
        diffusion.denoised_guided(model, img, torch.tensor(2.5))
        assert model.calls[0]["sigma"].shape == (4,)

    def test_class_labels_are_cast_to_long(self):
        """Test that when class_labels are provided, they are cast to torch.long before being passed to the model."""
        model = _FakeConditionalModel()
        img = torch.zeros(2, 1, 4, 4)
        labels = torch.tensor([0.0, 1.0])
        diffusion.denoised_guided(model, img, torch.tensor(1.0), class_labels=labels, guidance_strength=0.1)
        assert model.calls[1]["class_labels"].dtype == torch.long


class _ZeroDenoiser(torch.nn.Module):
    """A minimal fake EDM-style model: always denoises to exactly zero, and exposes sigma_min/sigma_max."""
    def __init__(self, sigma_min=0.0, sigma_max=float("inf")):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, img, sigma, context=None, class_labels=None):
        self.calls.append({"img_shape": img.shape, "sigma": sigma.clone(), "context": context,
                           "class_labels": class_labels})
        return torch.zeros_like(img)


class TestEdmSampling:
    """Tests for the edm_sampling() function in diffusion.py, which implements the EDM sampling algorithm."""

    def test_first_returned_image_is_latents_scaled_by_sigma_max(self):
        """Test that the first image returned by edm_sampling() is equal to the input latents scaled by sigma_max."""
        model = _ZeroDenoiser(sigma_min=2e-3, sigma_max=80)
        latents = torch.randn(2, 1, 4, 4)
        imgs = diffusion.edm_sampling(model, latents=latents, image_size=4, timesteps=5)
        sigmas = diffusion.get_sampling_noise_levels(5, sigma_min=2e-3, sigma_max=80)
        torch.testing.assert_close(imgs[0], latents * sigmas[0])

    def test_final_image_is_exactly_zero_regardless_of_churn(self):
        """Test that the final image returned by edm_sampling() is exactly zero, regardless of the stochastic churn."""
        # With a denoiser that always outputs 0, the last Euler step (sigma_next=0) lands exactly on the
        # denoised value, i.e. exactly zero - true whether or not stochastic churn perturbed sigma along the way.
        latents = torch.randn(2, 1, 4, 4)
        for s_churn in (0, 5.0):
            model = _ZeroDenoiser(sigma_min=2e-3, sigma_max=80)
            imgs = diffusion.edm_sampling(model, latents=latents, image_size=4, timesteps=5, S_churn=s_churn)
            torch.testing.assert_close(imgs[-1], torch.zeros_like(latents))

    def test_intermediate_trajectory_matches_hand_derived_closed_form(self):
        """
        Test that the intermediate images returned by edm_sampling() match the hand-derived closed form for a zero
        denoiser.
        """
        # With denoised==0, both the Euler step and the 2nd-order correction reduce to
        # x_next = x_cur * sigma_next / sigma_cur, so imgs[k] == latents * sigmas[0] * sigmas[k]/sigmas[0]
        # == latents * sigmas[k] for every step before the last (deterministic, S_churn=0).
        model = _ZeroDenoiser(sigma_min=2e-3, sigma_max=80)
        latents = torch.randn(1, 1, 4, 4)
        timesteps = 6
        imgs = diffusion.edm_sampling(model, latents=latents, image_size=4, timesteps=timesteps)
        sigmas = diffusion.get_sampling_noise_levels(timesteps, sigma_min=2e-3, sigma_max=80)
        for k in range(timesteps):  # all but the final appended-zero step
            torch.testing.assert_close(imgs[k], latents * sigmas[k], rtol=1e-4, atol=1e-6)

    def test_returns_timesteps_plus_one_images(self):
        """Test that edm_sampling() returns a list of length timesteps + 1 (the last one is always zero)."""
        model = _ZeroDenoiser()
        latents = torch.randn(1, 1, 4, 4)
        imgs = diffusion.edm_sampling(model, latents=latents, image_size=4, timesteps=7)
        assert len(imgs) == 8

    def test_sigma_bounds_are_clamped_to_model_limits(self):
        """
        Test that when sigma_min/sigma_max are requested outside the model's own range, they are clamped to the model's
        range.
        """
        model = _ZeroDenoiser(sigma_min=1.0, sigma_max=10.0)
        latents = torch.randn(1, 1, 4, 4)
        # Request far wider bounds than the model supports - should be clamped to the model's own range.
        imgs = diffusion.edm_sampling(model, latents=latents, image_size=4, timesteps=3,
                                      sigma_min=1e-5, sigma_max=1000.0)
        assert imgs[0].abs().max().item() == pytest.approx((latents * 10.0).abs().max().item())

    def test_latents_wrong_channel_count_raises_assertion_error(self):
        """Test that edm_sampling() raises an AssertionError when latents have the wrong number of channels."""
        model = _ZeroDenoiser()
        latents = torch.randn(1, 2, 4, 4)  # 2 channels, should be 1
        with pytest.raises(AssertionError):
            diffusion.edm_sampling(model, latents=latents, image_size=4, timesteps=3)

    def test_latents_wrong_spatial_size_raises_assertion_error(self):
        """Test that edm_sampling() raises an AssertionError when latents have the wrong spatial size."""
        model = _ZeroDenoiser()
        latents = torch.randn(1, 1, 8, 8)
        with pytest.raises(AssertionError):
            diffusion.edm_sampling(model, latents=latents, image_size=4, timesteps=3)

    def test_context_batch_size_mismatch_raises_assertion_error(self):
        """
        Test that edm_sampling() raises an AssertionError when context batch size doesn't match latents batch size.
        """
        model = _ZeroDenoiser()
        latents = torch.randn(2, 1, 4, 4)
        context = torch.zeros(3, 5)  # batch size 3, should match latents' batch size 2
        with pytest.raises(AssertionError):
            diffusion.edm_sampling(model, context_batch=context, latents=latents, image_size=4, timesteps=3)

    def test_random_latents_generated_when_not_provided(self):
        """Test that edm_sampling() generates random latents when latents are not provided."""
        torch.manual_seed(0)
        model = _ZeroDenoiser(sigma_min=2e-3, sigma_max=80)
        imgs = diffusion.edm_sampling(model, image_size=4, batch_size=3, timesteps=3)
        assert imgs[0].shape == (3, 1, 4, 4)

    def test_accepts_numpy_latents(self):
        """Test that edm_sampling() accepts latents as a numpy array and returns torch tensors."""
        model = _ZeroDenoiser(sigma_min=2e-3, sigma_max=80)
        latents_np = np.random.default_rng(0).normal(size=(2, 1, 4, 4)).astype(np.float32)
        imgs = diffusion.edm_sampling(model, latents=latents_np, image_size=4, timesteps=3)
        assert isinstance(imgs[0], torch.Tensor)
        assert imgs[0].shape == (2, 1, 4, 4)

    def test_label_batch_size_mismatch_raises_assertion_error(self):
        """
        Test that edm_sampling() raises an AssertionError when label batch size doesn't match latents batch size.
        """
        model = _ZeroDenoiser()
        latents = torch.randn(2, 1, 4, 4)
        labels = torch.zeros(3)  # batch size 3, should match latents' batch size 2
        with pytest.raises(AssertionError):
            diffusion.edm_sampling(model, label_batch=labels, latents=latents, image_size=4, timesteps=3)

    def test_label_batch_is_forwarded_to_the_model(self):
        """Test that when label_batch is provided, it is forwarded to the model during sampling."""
        model = _ZeroDenoiser()
        latents = torch.randn(2, 1, 4, 4)
        labels = torch.tensor([0, 1])
        diffusion.edm_sampling(model, label_batch=labels, latents=latents, image_size=4, timesteps=2)
        conditioned_calls = [c for c in model.calls if c["class_labels"] is not None]
        assert len(conditioned_calls) > 0
        for c in conditioned_calls:
            torch.testing.assert_close(c["class_labels"], labels.long())

    def test_context_is_forwarded_to_the_model(self):
        """Test that when context_batch is provided, it is forwarded to the model during sampling."""
        model = _ZeroDenoiser()
        latents = torch.randn(2, 1, 4, 4)
        context = torch.arange(6.0).reshape(2, 3)
        diffusion.edm_sampling(model, context_batch=context, latents=latents, image_size=4, timesteps=2)
        # denoised_guided always makes an unconditioned call with context=None first; the conditioned call (with
        # guidance_strength>0 default) should have received the real context.
        conditioned_calls = [c for c in model.calls if c["context"] is not None]
        assert len(conditioned_calls) > 0
        for c in conditioned_calls:
            torch.testing.assert_close(c["context"], context)
