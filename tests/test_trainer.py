"""
Unit tests for the pure, CPU-testable logic of diffracc/training/trainer.py.

A full DiffusionTrainer.__init__ builds a UNet, picks CUDA devices and (under DDP) joins a process group, none of
which is available on a CPU-only box. So these tests target the self-contained methods - effective-batch-size
arithmetic, batch unpacking, the primary-process flag, and the module-level EMA update function - by instantiating a
bare trainer via __new__ and setting only the attributes each method reads. The training loop itself (training_step,
validation_loss, batch_loss) needs a GPU/DDP and is verified on the cluster, not here.
"""
import types

import pytest
import torch

from diffracc.model.config import ModelConfig
from diffracc.training.trainer import DiffusionTrainer, get_ema_avg_fn


def _bare_trainer(**attrs) -> DiffusionTrainer:
    """Create a DiffusionTrainer without running __init__, then set the attributes a given method needs."""
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    for key, value in attrs.items():
        setattr(trainer, key, value)
    return trainer


class TestComputeEffectiveBatchSize:
    """Tests for compute_effective_batch_size() = batch_size * n_gpus * accumulation_steps."""

    def test_product_of_all_three_factors(self):
        """Testing 64 per-GPU x 2 GPUs x 2 accumulation steps = 256 (the Martínez-comparable effective batch)."""
        trainer = _bare_trainer(n_gpus=2, config=ModelConfig(batch_size=64, accumulation_steps=2))
        assert trainer.compute_effective_batch_size() == 256

    def test_defaults_accumulation_to_one_when_absent(self):
        """Testing with no accumulation_steps in the config it defaults to 1, so effective = batch_size * n_gpus."""
        trainer = _bare_trainer(n_gpus=2, config=ModelConfig(batch_size=64))
        assert trainer.compute_effective_batch_size() == 128

    def test_single_gpu_no_accumulation(self):
        """Testing one GPU, no accumulation: effective batch equals the per-GPU batch."""
        trainer = _bare_trainer(n_gpus=1, config=ModelConfig(batch_size=32))
        assert trainer.compute_effective_batch_size() == 32

    def test_string_batch_size_is_coerced(self):
        """Testing batch_size read from a config file can be a string; it is coerced to int before multiplying."""
        trainer = _bare_trainer(n_gpus=2, config=ModelConfig(batch_size="64", accumulation_steps=2))
        assert trainer.compute_effective_batch_size() == 256


class TestUnpackBatch:
    """Tests for unpack_batch()'s (img, context, labels) disambiguation by batch shape."""

    @staticmethod
    def _trainer_with_context_dim(context_dim):
        """Create a bare trainer with a model that has the given context_dim, so unpack_batch can read it."""
        # unpack_batch reads self.inner_model.model.context_dim to decide whether a length-2 batch is (img, context)
        # or (img, labels).
        inner_model = types.SimpleNamespace(model=types.SimpleNamespace(context_dim=context_dim))
        return _bare_trainer(inner_model=inner_model)

    def test_bare_tensor_is_image_only(self):
        """Testing a plain tensor batch is the image, with no context or labels."""
        trainer = self._trainer_with_context_dim(1)
        img = torch.zeros(4, 1, 8, 8)
        assert trainer.unpack_batch(img) == (img, None, None)

    def test_length_two_with_context_dim_is_context(self):
        """Testing when the model has context, a length-2 batch is (image, context)."""
        trainer = self._trainer_with_context_dim(2)
        img, ctx = torch.zeros(4, 1, 8, 8), torch.zeros(4, 2)
        assert trainer.unpack_batch([img, ctx]) == (img, ctx, None)

    def test_length_two_without_context_dim_is_labels(self):
        """Testing with no context configured, a length-2 batch is (image, labels)."""
        trainer = self._trainer_with_context_dim(0)
        img, labels = torch.zeros(4, 1, 8, 8), torch.zeros(4)
        assert trainer.unpack_batch([img, labels]) == (img, None, labels)

    def test_length_three_is_image_context_labels(self):
        """Testing a length-3 batch is unpacked as (image, context, labels) regardless of context_dim."""
        trainer = self._trainer_with_context_dim(1)
        img, ctx, labels = torch.zeros(2, 1, 4, 4), torch.zeros(2, 1), torch.zeros(2)
        assert trainer.unpack_batch([img, ctx, labels]) == (img, ctx, labels)

    def test_bad_length_raises(self):
        """Testing a list that is neither length 2 nor 3 is malformed and raises."""
        trainer = self._trainer_with_context_dim(1)
        with pytest.raises(ValueError):
            trainer.unpack_batch([torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)])


class TestIsPrimary:
    """Tests for the is_primary() flag used to gate logging/output to one rank."""

    def test_reflects_primary_attribute(self):
        """Testing the bare trainer's primary attribute is reflected in is_primary()."""
        assert _bare_trainer(primary=True).is_primary() is True
        assert _bare_trainer(primary=False).is_primary() is False


class TestGetEmaAvgFn:
    """Tests for the module-level get_ema_avg_fn() EMA update (decay*ema + (1-decay)*current)."""

    def test_update_matches_closed_form(self):
        """Testing the update is a straight convex blend of the EMA and current parameters at the given decay."""
        fn = get_ema_avg_fn(0.9)
        ema, cur = torch.tensor(2.0), torch.tensor(1.0)
        # 0.9*2 + 0.1*1 = 1.9
        assert fn(ema, cur, 5).item() == pytest.approx(1.9)

    def test_decay_out_of_range_raises(self):
        """Testing a decay outside [0, 1] is invalid and rejected at construction."""
        with pytest.raises(ValueError):
            get_ema_avg_fn(1.5)
        with pytest.raises(ValueError):
            get_ema_avg_fn(-0.1)
