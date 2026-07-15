"""
Unit tests for diffracc/training/train_utils.py.

Everything here runs on CPU with tiny stand-in modules - no GPU, no real diffusion model. The EDM loss/preconditioning
maths, the EMA update rules, the training-time EMA weight swap, and the infinite DataLoader helper (including its
DistributedSampler set_epoch driving and the shuffle/sampler mutual-exclusion) are all exercised directly.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Sampler, TensorDataset

from diffracc.training import train_utils


class _ConstModel(nn.Module):
    """
    A denoiser stand-in that ignores its input and returns a constant image, so edm_loss's output is fully determined
    and its weighting maths can be checked against a hand-computed value.
    """
    def __init__(self, value: float = 0.3):
        super().__init__()
        self.value = value

    def forward(self, x, sigmas, context=None, class_labels=None):
        return torch.full_like(x, self.value)


class TestSampleSigmas:
    """Tests for sample_sigmas()'s log-normal noise-level sampling."""

    def test_shape_is_per_image_broadcastable(self):
        """Testing sigmas come out shaped (batch, 1, 1, 1) so they broadcast over (batch, C, H, W) images."""
        imgs = torch.zeros(7, 1, 8, 8)
        sigmas = train_utils.sample_sigmas(imgs)
        assert sigmas.shape == (7, 1, 1, 1)

    def test_all_positive(self):
        """Testing sigmas are an exp() of a normal, so strictly positive."""
        sigmas = train_utils.sample_sigmas(torch.zeros(256, 1, 4, 4))
        assert bool((sigmas > 0).all())

    def test_zero_std_collapses_to_exp_pmean(self):
        """Testing with p_std=0 the randomness drops out and every sigma equals exp(p_mean) exactly."""
        sigmas = train_utils.sample_sigmas(torch.zeros(16, 1, 4, 4), p_mean=-1.4, p_std=0.0)
        assert torch.allclose(sigmas, torch.full_like(sigmas, float(np.exp(-1.4))))

    def test_pmean_controls_log_mean(self):
        """Testing the mean of log(sigma) should track p_mean (up to sampling noise)."""
        torch.manual_seed(0)
        sigmas = train_utils.sample_sigmas(torch.zeros(20000, 1, 1, 1), p_mean=-2.5, p_std=1.0)
        assert sigmas.log().mean().item() == pytest.approx(-2.5, abs=0.05)


class TestEdmLoss:
    """Tests for edm_loss()'s weighted MSE and its options."""

    def _weight(self, sigmas, sigma_data):
        """Helper to compute the EDM weight for a given sigma and sigma_data, for comparison with edm_loss's output."""
        return (sigmas**2 + sigma_data**2) / (sigmas * sigma_data) ** 2

    def test_matches_hand_computed_weighted_mse(self):
        """Testing the loss equals mean(weight * (D - img)^2) with the EDM weight, for a known model output."""
        torch.manual_seed(0)
        img = torch.randn(4, 1, 8, 8)
        sigmas = torch.full((4, 1, 1, 1), 0.5)
        noise = torch.zeros_like(img)  # value irrelevant (model ignores input), but lets us pass sigmas explicitly
        model = _ConstModel(0.3)

        loss = train_utils.edm_loss(model, img, sigma_data=0.5, sigmas=sigmas, noise=noise)

        expected = (self._weight(sigmas, 0.5) * (0.3 - img) ** 2).mean()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_return_output_returns_denoised(self):
        """Testing with return_output=True the model's denoised image is returned alongside the loss."""
        img = torch.zeros(2, 1, 4, 4)
        sigmas = torch.full((2, 1, 1, 1), 0.5)
        loss, d = train_utils.edm_loss(
            _ConstModel(0.7), img, sigmas=sigmas, noise=torch.zeros_like(img), return_output=True)
        assert d.shape == img.shape
        assert torch.allclose(d, torch.full_like(d, 0.7))

    def test_mean_false_returns_per_element(self):
        """Testing with mean=False the loss keeps the image shape instead of being reduced to a scalar."""
        img = torch.zeros(3, 1, 4, 4)
        sigmas = torch.full((3, 1, 1, 1), 0.5)
        loss = train_utils.edm_loss(
            _ConstModel(), img, sigmas=sigmas, noise=torch.zeros_like(img), mean=False)
        assert loss.shape == img.shape

    def test_noise_without_sigmas_raises(self):
        """Testing passing noise but no sigmas is ambiguous (no noise level for the noise), and must raise."""
        img = torch.zeros(2, 1, 4, 4)
        with pytest.raises(AssertionError):
            train_utils.edm_loss(_ConstModel(), img, noise=torch.zeros_like(img))


class TestUseEMA:
    """Tests for the UseEMA context manager, which temporarily loads EMA weights into the live model."""

    @staticmethod
    def _filled_linear(value: float) -> nn.Linear:
        m = nn.Linear(3, 3)
        with torch.no_grad():
            m.weight.fill_(value)
            m.bias.fill_(value)
        return m

    def test_swaps_in_and_restores(self):
        """Testing inside the context the model holds the EMA weights; on exit the original weights are restored."""
        model = self._filled_linear(1.0)

        class _FakeEMA:  # AveragedModel exposes the averaged network at .module, which UseEMA reads
            module = self._filled_linear(2.0)
        fake_ema = _FakeEMA()

        with train_utils.UseEMA(model, fake_ema):
            assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))
        assert torch.allclose(model.weight, torch.full_like(model.weight, 1.0))

    def test_restores_even_on_exception(self):
        """Testing the original weights are restored even if the body raises (context manager __exit__ still runs)."""
        model = self._filled_linear(1.0)

        class _FakeEMA:
            module = self._filled_linear(9.0)

        with pytest.raises(RuntimeError):
            with train_utils.UseEMA(model, _FakeEMA()):
                raise RuntimeError("boom")
        assert torch.allclose(model.weight, torch.full_like(model.weight, 1.0))


class TestPowerEmaAvgFn:
    """Tests for get_power_ema_avg_fn()'s Karras power-EMA update."""

    def test_matches_closed_form(self):
        """Testing beta = (1 - 1/num_averaged)^(gamma+1), then update = beta*ema + (1-beta)*current."""
        gamma = 6.94
        fn = train_utils.get_power_ema_avg_fn(gamma)
        ema, cur, n = torch.tensor(1.0), torch.tensor(0.0), 10
        beta = (1 - 1 / n) ** (gamma + 1)
        assert fn(ema, cur, n).item() == pytest.approx(beta, rel=1e-6)

    def test_larger_gamma_weights_current_more(self):
        """Testing a larger gamma -> smaller beta -> the update leans further toward the current parameter."""
        ema, cur, n = torch.tensor(1.0), torch.tensor(0.0), 10
        low = train_utils.get_power_ema_avg_fn(1.0)(ema, cur, n).item()
        high = train_utils.get_power_ema_avg_fn(16.97)(ema, cur, n).item()
        assert high < low  # both are the retained-EMA fraction; higher gamma retains less


class _RangeSampler(Sampler):
    """A trivial sequential sampler that records the epochs it is told about, to verify set_epoch is driven."""
    def __init__(self, n):
        self.n = n
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


class TestLoadData:
    """Tests for load_data()'s infinite DataLoader wrapper."""

    def test_yields_batches_of_requested_size(self):
        """Testing batches come out at the requested batch size (drop_last means only full batches)."""
        ds = TensorDataset(torch.arange(20).float().view(20, 1))
        gen = train_utils.load_data(ds, batch_size=4, num_workers=0)
        batch = next(gen)
        assert batch[0].shape == (4, 1)

    def test_is_infinite_across_epoch_boundary(self):
        """Testing the generator keeps yielding past one epoch's worth of batches (20/4 = 5 per epoch)."""
        ds = TensorDataset(torch.arange(20).float().view(20, 1))
        gen = train_utils.load_data(ds, batch_size=4, num_workers=0)
        assert sum(1 for _ in (next(gen) for _ in range(12))) == 12  # would StopIteration if finite

    def test_distributed_sampler_set_epoch_is_driven(self):
        """Testing with a sampler, set_epoch is called with an incrementing epoch each time the loader is re-iterated."""
        ds = TensorDataset(torch.arange(20).float().view(20, 1))
        sampler = _RangeSampler(20)
        gen = train_utils.load_data(ds, batch_size=4, num_workers=0, sampler=sampler)
        for _ in range(7):  # cross into the second epoch (5 batches per epoch)
            next(gen)
        assert sampler.epochs[:2] == [0, 1]

    def test_sampler_and_shuffle_do_not_conflict(self):
        """Testing a sampler is passed with shuffle suppressed, so DataLoader does not raise the 'sampler + shuffle' error."""
        ds = TensorDataset(torch.arange(20).float().view(20, 1))
        gen = train_utils.load_data(ds, batch_size=4, shuffle=True, num_workers=0, sampler=_RangeSampler(20))
        assert next(gen)[0].shape == (4, 1)  # constructed and yielded without error
