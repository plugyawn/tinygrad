import math, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

from tinygrad import Tensor
from tinygrad.nn.state import safe_load_metadata
from tinygrad.helpers import getenv

from examples.mlperf.model_train import train_flux


class _FakeMLLogger:
  def __init__(self):
    self.calls = []
    self.logger = types.SimpleNamespace(propagate=True)

  def event(self, key, value=None, metadata=None, **kwargs): self.calls.append(("event", key, value, metadata))
  def start(self, key, value=None, metadata=None, **kwargs): self.calls.append(("start", key, value, metadata))
  def end(self, key, value=None, metadata=None, **kwargs): self.calls.append(("end", key, value, metadata))


def _fake_mlperf_logging():
  logger = _FakeMLLogger()
  constants = types.ModuleType("mlperf_logging.mllog.constants")
  for key, value in {
    "SUBMISSION_ORG": "submission_org",
    "SUBMISSION_PLATFORM": "submission_platform",
    "SUBMISSION_DIVISION": "submission_division",
    "SUBMISSION_STATUS": "submission_status",
    "SUBMISSION_BENCHMARK": "submission_benchmark",
    "CACHE_CLEAR": "cache_clear",
    "INIT_START": "init_start",
    "INIT_STOP": "init_stop",
    "CLOSED": "closed",
    "ONPREM": "onprem",
    "FLUX1": "flux1",
  }.items():
    setattr(constants, key, value)
  mllog = types.ModuleType("mlperf_logging.mllog")
  mllog.config = lambda **kwargs: None
  mllog.get_mllogger = lambda: logger
  package = types.ModuleType("mlperf_logging")
  package.mllog = mllog
  return {
    "mlperf_logging": package,
    "mlperf_logging.mllog": mllog,
    "mlperf_logging.mllog.constants": constants,
  }, logger


class TestMLPerfFluxTrain(unittest.TestCase):
  @staticmethod
  def tiny_flux_env(tmpdir:str) -> dict[str, str]:
    return {
      "FAKEDATA": "1",
      "GPUS": "1",
      "BS": "1",
      "EVAL_BS": "1",
      "TRAIN_STEPS": "1",
      "EVAL_STEPS": "1",
      "EVAL_INTERVAL": "1",
      "CKPT_INTERVAL": "1",
      "SAVE_CKPT_DIR": tmpdir,
      "FLUX_HIDDEN_SIZE": "128",
      "FLUX_DOUBLE_STREAM_BLOCKS": "0",
      "FLUX_SINGLE_STREAM_BLOCKS": "0",
      "FLUX_MLP_RATIO": "1.0",
      "FLUX_T5_TOKENS": "8",
    }

  def test_train_flux_fake_smoke(self):
    with tempfile.TemporaryDirectory(prefix="flux-train-smoke-") as tmpdir:
      env = self.tiny_flux_env(tmpdir) | {"CKPT": "1", "TARGET": "10.0"}
      with patch.dict(os.environ, env, clear=False):
        try:
          getenv.cache_clear()
          with Tensor.train():
            eval_loss = train_flux()
        finally:
          getenv.cache_clear()

      self.assertTrue(math.isfinite(eval_loss))
      self.assertTrue((Path(tmpdir) / "flux_step1.safetensors").exists())
      self.assertTrue((Path(tmpdir) / "flux.safetensors").exists())
      _, _, metadata = safe_load_metadata(Path(tmpdir) / "flux_step1.safetensors")
      self.assertEqual(metadata["__metadata__"]["flux_global_batch_size"], 1)

  def test_train_flux_init_mlperf_logs(self):
    with tempfile.TemporaryDirectory(prefix="flux-init-mlperf-") as tmpdir:
      env = self.tiny_flux_env(tmpdir) | {"LOGMLPERF": "1", "INITMLPERF": "1", "BENCHMARK": "1", "CKPT": "0"}
      fake_modules, logger = _fake_mlperf_logging()
      with patch.dict(os.environ, env, clear=False), patch.dict(sys.modules, fake_modules), \
           patch("examples.mlperf.model_train.diskcache_clear", lambda: None):
        try:
          getenv.cache_clear()
          with Tensor.train():
            train_flux()
        finally:
          getenv.cache_clear()

      self.assertIn(("event", "submission_benchmark", "flux1", None), logger.calls)
      self.assertIn(("start", "init_start", None, None), logger.calls)
      self.assertIn(("end", "init_stop", None, None), logger.calls)


if __name__ == "__main__":
  unittest.main()
