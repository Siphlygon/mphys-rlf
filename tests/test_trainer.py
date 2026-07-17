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


def _fake_om(**attrs):
    """A stand-in for OutputManager exposing just the attributes the wandb-resume helpers read."""
    defaults = dict(pickup=False, ready_to_write=True)
    defaults.update(attrs)
    return types.SimpleNamespace(**defaults)


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


class TestWandbRunIdFile:
    """Tests for _wandb_run_id_file()'s path, which deliberately does not embed the model name."""

    def test_path_is_fixed_name_inside_results_folder(self, tmp_path):
        """Testing the marker file lives at <results_folder>/wandb_run_id.txt, not templated on the model name."""
        trainer = _bare_trainer(OM=_fake_om(results_folder=tmp_path))
        assert trainer._wandb_run_id_file() == tmp_path / "wandb_run_id.txt"


class TestWandbJobMetadata:
    """Tests for _wandb_job_metadata()'s SLURM/file-path collection."""

    def test_collects_slurm_env_vars(self, monkeypatch, tmp_path):
        """Testing SLURM_JOB_ID/SLURMD_NODENAME/SLURM_JOB_NODELIST are read straight from the environment."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURMD_NODENAME", "compute-0-9")
        monkeypatch.setenv("SLURM_JOB_NODELIST", "compute-0-9")
        trainer = _bare_trainer(OM=_fake_om(results_folder=tmp_path, config_file=tmp_path / "config_x.json"))

        metadata = trainer._wandb_job_metadata()

        assert metadata["slurm_job_id"] == "12345"
        assert metadata["slurm_node"] == "compute-0-9"
        assert metadata["slurm_nodelist"] == "compute-0-9"

    def test_missing_slurm_env_vars_are_none(self, monkeypatch, tmp_path):
        """Testing that outside SLURM (e.g. local runs), the SLURM fields are None rather than raising."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURMD_NODENAME", raising=False)
        monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
        trainer = _bare_trainer(OM=_fake_om(results_folder=tmp_path, config_file=tmp_path / "config_x.json"))

        metadata = trainer._wandb_job_metadata()

        assert metadata["slurm_job_id"] is None
        assert metadata["slurm_node"] is None
        assert metadata["slurm_nodelist"] is None

    def test_includes_paths_when_ready_to_write(self, tmp_path):
        """Testing config_file/results_folder are included (as strings) when output writing is set up."""
        config_file = tmp_path / "config_x.json"
        trainer = _bare_trainer(OM=_fake_om(ready_to_write=True, results_folder=tmp_path, config_file=config_file))

        metadata = trainer._wandb_job_metadata()

        assert metadata["config_file"] == str(config_file)
        assert metadata["results_folder"] == str(tmp_path)

    def test_paths_are_none_when_not_ready_to_write(self, tmp_path):
        """Testing config_file/results_folder are None when output writing was never set up (e.g. non-primary rank)."""
        trainer = _bare_trainer(
            OM=_fake_om(ready_to_write=False, results_folder=tmp_path, config_file=tmp_path / "config_x.json"))

        metadata = trainer._wandb_job_metadata()

        assert metadata["config_file"] is None
        assert metadata["results_folder"] is None


class TestResolveWandbResume:
    """Tests for _resolve_wandb_resume()'s decision between a fresh wandb run and reattaching to a saved one."""

    def test_fresh_run_when_not_pickup(self, tmp_path):
        """Testing a non-pickup run always starts fresh, even if a stale run-id file happens to exist."""
        (tmp_path / "wandb_run_id.txt").write_text("stale-id", encoding="utf-8")
        trainer = _bare_trainer(OM=_fake_om(pickup=False, ready_to_write=True, results_folder=tmp_path))
        assert trainer._resolve_wandb_resume(write_output=True) == (None, None)

    def test_fresh_run_when_write_output_false(self, tmp_path):
        """Testing that without output writing, there's nowhere to read a saved id from, so it's always fresh."""
        (tmp_path / "wandb_run_id.txt").write_text("some-id", encoding="utf-8")
        trainer = _bare_trainer(OM=_fake_om(pickup=True, ready_to_write=True, results_folder=tmp_path))
        assert trainer._resolve_wandb_resume(write_output=False) == (None, None)

    def test_fresh_run_when_pickup_but_no_saved_id_yet(self, tmp_path):
        """Testing a pickup of a model trained before this feature existed (no run-id file) starts fresh, not error."""
        trainer = _bare_trainer(OM=_fake_om(pickup=True, ready_to_write=True, results_folder=tmp_path))
        assert trainer._resolve_wandb_resume(write_output=True) == (None, None)

    def test_resumes_saved_run_id_on_pickup(self, tmp_path):
        """Testing a pickup with a saved run-id file reattaches to that exact run with resume='allow'."""
        (tmp_path / "wandb_run_id.txt").write_text("saved-run-id\n", encoding="utf-8")
        trainer = _bare_trainer(OM=_fake_om(pickup=True, ready_to_write=True, results_folder=tmp_path))
        assert trainer._resolve_wandb_resume(write_output=True) == ("saved-run-id", "allow")


class TestLoadState:
    """
    Tests for load_state(), which reads model/EMA/optimizer weights back from a checkpoint on pickup.
    """

    def test_loads_model_ema_and_optimizer_from_checkpoint(self, tmp_path):
        """
        Testing load_state() restores model/ema_model/optimizer state from the checkpoint expected_parameters_file
        points at, without ever touching self.OM.parameters_file.
        """
        model = torch.nn.Linear(2, 2)
        ema_model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with torch.no_grad():
            ema_model.weight.fill_(9.0)  # distinct from model's random init, so we can confirm the right dict loaded

        # Populate optimizer.state (step counters, running moment estimates) - an unstepped optimizer's state dict
        # is just empty tensors/lists, which doesn't exercise the same checkpoint shape a real crash left behind.
        loss = model(torch.randn(1, 2)).sum()
        loss.backward()
        optimizer.step()

        checkpoint_path = tmp_path / "parameters_mymodel.pt"
        torch.save(
            {"model": model.state_dict(), "ema_model": ema_model.state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint_path,
        )

        fresh_model = torch.nn.Linear(2, 2)
        fresh_ema = torch.nn.Linear(2, 2)
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        trainer = _bare_trainer(
            OM=types.SimpleNamespace(expected_parameters_file=lambda: checkpoint_path),
            inner_model=fresh_model, ema_model=fresh_ema, optimizer=fresh_optimizer, power_ema=False,
        )

        trainer.load_state()

        assert torch.allclose(fresh_model.weight, model.weight)
        assert torch.allclose(fresh_ema.weight, torch.full_like(fresh_ema.weight, 9.0))
        assert fresh_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
