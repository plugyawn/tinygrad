import os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

import examples.mlperf.flux as flux_helpers
import examples.mlperf.initializers as initializers
import examples.mlperf.model_eval as model_eval


def _identity_jit(fn): return fn


class _FakeDevice:
  DEFAULT = "CPU"
  def __getitem__(self, key): return key


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


if __name__ == "__main__":
  unittest.main()
