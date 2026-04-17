import math, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_save
from tinygrad.helpers import getenv

import examples.mlperf.flux as flux_helpers
import examples.mlperf.initializers as initializers
import examples.mlperf.model_eval as model_eval
from examples.mlperf.model_train import train_flux


def _identity_jit(fn): return fn


class _FakeDevice:
  DEFAULT = "CPU"
  def __getitem__(self, key): return key


def _fake_flux_batch(batch_size:int, txt_tokens:int=8, timestep_offset:int=0) -> dict[str, Tensor]:
  return {
    "mean": Tensor.zeros(batch_size, 16, 32, 32, dtype=dtypes.default_float, device="CPU").contiguous().realize(),
    "logvar": Tensor.zeros(batch_size, 16, 32, 32, dtype=dtypes.default_float, device="CPU").contiguous().realize(),
    "t5_encodings": Tensor.zeros(batch_size, txt_tokens, flux_helpers.FLUX_T5_EMBED_DIM,
                                  dtype=dtypes.default_float, device="CPU").contiguous().realize(),
    "clip_encodings": Tensor.zeros(batch_size, flux_helpers.FLUX_CLIP_EMBED_DIM,
                                    dtype=dtypes.default_float, device="CPU").contiguous().realize(),
    "timestep": Tensor([(timestep_offset + i) % len(flux_helpers.FLUX_FIXED_EVAL_TIMESTEPS) for i in range(batch_size)],
                       dtype=dtypes.int32, device="CPU").contiguous().realize(),
  }


class TestMLPerfFluxEval(unittest.TestCase):
  def test_eval_flux_aggregates_validation_loss_across_batches(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_dir = Path(tmpdir)
      (ckpt_dir / "flux_step0004.safetensors").write_bytes(b"")

      original_env = {k: os.environ.get(k) for k in ("MODEL", "GPUS", "BS", "DATADIR", "VAL_DATASET", "EVAL_CKPT_DIR", "STOP_IF_CONVERGED")}
      os.environ.update({
        "MODEL": "flux",
        "GPUS": "1",
        "BS": "2",
        "DATADIR": str(ckpt_dir / "dataset-root"),
        "EVAL_CKPT_DIR": str(ckpt_dir),
        "STOP_IF_CONVERGED": "0",
      })
      getenv.cache_clear()

      calls = {"load": [], "loader": []}
      batches = [
        {
          "mean": Tensor.zeros(2, 1, 1, 1, dtype=dtypes.bfloat16, device="CPU"),
          "logvar": Tensor.zeros(2, 1, 1, 1, dtype=dtypes.bfloat16, device="CPU"),
          "t5_encodings": Tensor.zeros(2, 1, 1, dtype=dtypes.bfloat16, device="CPU"),
          "clip_encodings": Tensor.zeros(2, 1, dtype=dtypes.bfloat16, device="CPU"),
          "timestep": Tensor([0, 1], dtype=dtypes.int32, device="CPU"),
        },
        {
          "mean": Tensor.zeros(1, 1, 1, 1, dtype=dtypes.bfloat16, device="CPU"),
          "logvar": Tensor.zeros(1, 1, 1, 1, dtype=dtypes.bfloat16, device="CPU"),
          "t5_encodings": Tensor.zeros(1, 1, 1, dtype=dtypes.bfloat16, device="CPU"),
          "clip_encodings": Tensor.zeros(1, 1, dtype=dtypes.bfloat16, device="CPU"),
          "timestep": Tensor([1], dtype=dtypes.int32, device="CPU"),
        },
      ]

      def fake_loader(dataset_path, bs):
        calls["loader"].append((dataset_path, bs))
        self.assertEqual(dataset_path, str(Path(os.environ["DATADIR"]) / "val-*"))
        self.assertEqual(bs, 2)
        yield from batches

      loss_calls = {"count": 0}
      def fake_validation_losses(model, mean, logvar, txt, vec, timestep_ids, guidance=None, latent_noise=None, flow_noise=None):
        loss_calls["count"] += 1
        return Tensor([1.0, 3.0], device="CPU") if loss_calls["count"] == 1 else Tensor([5.0], device="CPU")

      fake_flux_module = types.ModuleType("extra.models.flux")
      class FakeFluxParams:
        def __init__(self, **kwargs): self.kwargs = kwargs
      class FakeFlux:
        def __init__(self, params): self.params = params
      fake_flux_module.Flux = FakeFlux
      fake_flux_module.FluxParams = FakeFluxParams

      try:
        with patch.dict(sys.modules, {"extra.models.flux": fake_flux_module}), \
             patch.object(model_eval, "Device", _FakeDevice()), \
             patch.object(model_eval, "TinyJit", _identity_jit), \
             patch.object(Tensor, "shard_", lambda self, devices, axis=0: self), \
             patch.object(initializers, "init_flux", side_effect=lambda model, pretrained, devices, strict=True: model), \
             patch.object(flux_helpers, "flux_validation_losses", side_effect=fake_validation_losses), \
             patch.object(model_eval, "safe_load_metadata", return_value=(None, 0, {"__metadata__": {}})), \
             patch.object(model_eval, "safe_load", side_effect=lambda path: {"model.weight": Tensor([1.0], device="CPU")}), \
             patch.object(model_eval, "load_state_dict", side_effect=lambda model, state, strict=True: calls["load"].append((sorted(state.keys()), strict))), \
             patch("examples.mlperf.dataloader.batch_load_val_flux_preprocessed", side_effect=fake_loader):
          validation_loss, ckpt_iteration = model_eval.eval_flux()
      finally:
        for key, value in original_env.items():
          if value is None: os.environ.pop(key, None)
          else: os.environ[key] = value
        getenv.cache_clear()

    self.assertEqual(ckpt_iteration, 4)
    self.assertEqual(calls["loader"], [(str(Path(ckpt_dir / "dataset-root") / "val-*"), 2)])
    self.assertEqual(calls["load"], [(["weight"], True)])
    self.assertAlmostEqual(validation_loss, 2.5)

  def test_train_then_eval_flux_uses_same_checkpoint_dir(self):
    with tempfile.TemporaryDirectory(prefix="flux-train-eval-") as tmpdir:
      env = {
        "FAKEDATA": "1",
        "GPUS": "1",
        "BS": "1",
        "EVAL_BS": "1",
        "TRAIN_STEPS": "1",
        "EVAL_STEPS": "1",
        "EVAL_INTERVAL": "1",
        "CKPT": "1",
        "CKPT_INTERVAL": "1",
        "TARGET": "10.0",
        "SAVE_CKPT_DIR": tmpdir,
        "EVAL_CKPT_DIR": tmpdir,
        "VAL_DATASET": "fake-val",
        "STOP_IF_CONVERGED": "0",
        "FLUX_HIDDEN_SIZE": "128",
        "FLUX_DOUBLE_STREAM_BLOCKS": "0",
        "FLUX_SINGLE_STREAM_BLOCKS": "0",
        "FLUX_MLP_RATIO": "1.0",
        "FLUX_T5_TOKENS": "8",
      }
      loader_calls = []

      def fake_loader(dataset_path, bs):
        loader_calls.append((dataset_path, bs))
        yield _fake_flux_batch(bs, txt_tokens=8)

      with patch.dict(os.environ, env, clear=False):
        try:
          getenv.cache_clear()
          with Tensor.train():
            train_loss = train_flux()
          with patch("examples.mlperf.dataloader.batch_load_val_flux_preprocessed", side_effect=fake_loader):
            validation_loss, ckpt_iteration = model_eval.eval_flux()
        finally:
          getenv.cache_clear()

      self.assertTrue(math.isfinite(train_loss))
      self.assertTrue(math.isfinite(validation_loss))
      self.assertEqual(ckpt_iteration, 1)
      self.assertEqual(loader_calls, [("fake-val", 1)])
      self.assertTrue((Path(tmpdir) / "flux_step1.safetensors").exists())
      self.assertTrue((Path(tmpdir) / "flux.safetensors").exists())

  def test_eval_flux_reports_model_shape_override_mismatch(self):
    with tempfile.TemporaryDirectory(prefix="flux-eval-mismatch-") as tmpdir:
      ckpt_path = Path(tmpdir) / "flux_step1.safetensors"
      saved_model_config = dict(flux_helpers.FLUX_MLPERF_MODEL_CONFIG)
      saved_model_config.update({"hidden_size": 128, "depth": 0, "depth_single_blocks": 0, "mlp_ratio": 1.0, "num_heads": 1})
      safe_save({"model.weight": Tensor([1.0], device="CPU")}, str(ckpt_path), metadata={"flux_model_config": saved_model_config})

      original_env = {k: os.environ.get(k) for k in ("GPUS", "BS", "DATADIR", "VAL_DATASET", "EVAL_CKPT_DIR", "STOP_IF_CONVERGED",
                                                     "FLUX_HIDDEN_SIZE", "FLUX_DOUBLE_STREAM_BLOCKS", "FLUX_SINGLE_STREAM_BLOCKS",
                                                     "FLUX_MLP_RATIO", "FLUX_ATTENTION_HEADS")}
      os.environ.update({
        "GPUS": "1",
        "BS": "1",
        "DATADIR": str(Path(tmpdir) / "dataset-root"),
        "EVAL_CKPT_DIR": tmpdir,
        "STOP_IF_CONVERGED": "0",
      })
      for key in ("FLUX_HIDDEN_SIZE", "FLUX_DOUBLE_STREAM_BLOCKS", "FLUX_SINGLE_STREAM_BLOCKS", "FLUX_MLP_RATIO", "FLUX_ATTENTION_HEADS"):
        os.environ.pop(key, None)
      getenv.cache_clear()

      fake_flux_module = types.ModuleType("extra.models.flux")
      class FakeFluxParams:
        def __init__(self, **kwargs): self.kwargs = kwargs
      class FakeFlux:
        def __init__(self, params): self.params = params
      fake_flux_module.Flux = FakeFlux
      fake_flux_module.FluxParams = FakeFluxParams

      try:
        with patch.dict(sys.modules, {"extra.models.flux": fake_flux_module}), \
             patch.object(model_eval, "Device", _FakeDevice()), \
             patch.object(model_eval, "TinyJit", _identity_jit), \
             patch.object(Tensor, "shard_", lambda self, devices, axis=0: self), \
             patch.object(initializers, "init_flux", side_effect=lambda model, pretrained, devices, strict=True: model):
          with self.assertRaises(ValueError) as cm:
            model_eval.eval_flux()
      finally:
        for key, value in original_env.items():
          if value is None: os.environ.pop(key, None)
          else: os.environ[key] = value
        getenv.cache_clear()

    self.assertIn("Set the same FLUX_* debug model-shape overrides used during training.", str(cm.exception))
    self.assertIn("hidden_size", str(cm.exception))


if __name__ == "__main__":
  unittest.main()
