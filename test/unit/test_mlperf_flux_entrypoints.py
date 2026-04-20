import os, shutil, unittest
from pathlib import Path
from unittest.mock import patch

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_state_dict, safe_save


def _flux_test_env(**extra:str) -> dict[str, str]:
  env = {
    "BS": "1",
    "CKPT": "0",
    "EVAL_BS": "1",
    "EVAL_INTERVAL": "1",
    "EVAL_STEPS": "8",
    "FAKEDATA": "1",
    "FLUX_DOUBLE_STREAM_BLOCKS": "1",
    "FLUX_HIDDEN_SIZE": "128",
    "FLUX_SINGLE_STREAM_BLOCKS": "1",
    "FLUX_T5_TOKENS": "1",
    "GPUS": "1",
    "INITMLPERF": "0",
    "LOGMLPERF": "0",
    "PRETRAINED": "",
    "RUNMLPERF": "0",
    "TARGET": "1000000",
    "TRAIN_BEAM": "0",
    "TRAIN_STEPS": "1",
    "WANDB": "",
  }
  env.update(extra)
  return env


def _fake_val_batches(_dataset_path:str, _bs:int):
  for timestep in range(8):
    yield {
      "mean": Tensor.zeros(1, 16, 32, 32, dtype=dtypes.float32, device="CPU").contiguous(),
      "logvar": Tensor.zeros(1, 16, 32, 32, dtype=dtypes.float32, device="CPU").contiguous(),
      "t5_encodings": Tensor.zeros(1, 1, 4096, dtype=dtypes.float32, device="CPU").contiguous(),
      "clip_encodings": Tensor.zeros(1, 768, dtype=dtypes.float32, device="CPU").contiguous(),
      "timestep": Tensor([timestep], dtype=dtypes.int32, device="CPU").contiguous(),
    }

def _test_ckpt_dir(name:str) -> Path:
  path = Path.cwd() / name
  shutil.rmtree(path, ignore_errors=True)
  path.mkdir()
  return path


class TestMLPerfFluxEntrypoints(unittest.TestCase):
  def test_train_flux_runs_default_entrypoint(self):
    from examples.mlperf.model_train import train_flux

    tmpdir = _test_ckpt_dir(".flux-train-entrypoint")
    try:
      with patch.dict(os.environ, _flux_test_env(SAVE_CKPT_DIR=tmpdir.as_posix()), clear=False):
        getenv.cache_clear()
        ret = train_flux()

        self.assertIsInstance(ret, float)
        self.assertTrue((Path(tmpdir) / "flux.safetensors").exists())
    finally:
      shutil.rmtree(tmpdir)

  def test_eval_flux_runs_default_entrypoint(self):
    from extra.models.flux import Flux, FluxParams
    from examples.mlperf import dataloader
    from examples.mlperf.flux import flux_model_config_from_env
    from examples.mlperf.initializers import init_flux
    from examples.mlperf.model_eval import eval_flux

    tmpdir = _test_ckpt_dir(".flux-eval-entrypoint")
    try:
      with patch.dict(os.environ, _flux_test_env(EVAL_CKPT_DIR=tmpdir.as_posix(), VAL_DATASET="ignored"), clear=False):
        getenv.cache_clear()
        model_config = flux_model_config_from_env()
        devices = [f"{Device.DEFAULT}:0"]
        model = init_flux(Flux(FluxParams(**model_config)), None, devices, strict=True)
        safe_save(get_state_dict(model), str(Path(tmpdir) / "flux_step1.safetensors"),
                  metadata={"flux_model_config": model_config, "flux_global_batch_size": 1})

        with patch.object(dataloader, "batch_load_val_flux_preprocessed", _fake_val_batches):
          getenv.cache_clear()
          validation_loss, ckpt_iteration = eval_flux()

        self.assertIsInstance(validation_loss, float)
        self.assertEqual(ckpt_iteration, 1)
    finally:
      shutil.rmtree(tmpdir)

  def test_eval_flux_accepts_final_model_checkpoint(self):
    from extra.models.flux import Flux, FluxParams
    from examples.mlperf import dataloader
    from examples.mlperf.flux import flux_model_config_from_env
    from examples.mlperf.initializers import init_flux
    from examples.mlperf.model_eval import eval_flux

    tmpdir = _test_ckpt_dir(".flux-final-eval-entrypoint")
    try:
      with patch.dict(os.environ, _flux_test_env(EVAL_CKPT_DIR=tmpdir.as_posix(), VAL_DATASET="ignored"), clear=False):
        getenv.cache_clear()
        model_config = flux_model_config_from_env()
        devices = [f"{Device.DEFAULT}:0"]
        model = init_flux(Flux(FluxParams(**model_config)), None, devices, strict=True)
        safe_save(get_state_dict(model), str(Path(tmpdir) / "flux.safetensors"),
                  metadata={"flux_model_config": model_config, "flux_global_batch_size": 1, "flux_step": 7})

        with patch.object(dataloader, "batch_load_val_flux_preprocessed", _fake_val_batches):
          getenv.cache_clear()
          validation_loss, ckpt_iteration = eval_flux()

        self.assertIsInstance(validation_loss, float)
        self.assertEqual(ckpt_iteration, 7)
    finally:
      shutil.rmtree(tmpdir)


if __name__ == "__main__":
  unittest.main()
