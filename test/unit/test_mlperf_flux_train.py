import math, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

from tinygrad import Tensor
from tinygrad.helpers import getenv

from examples.mlperf.model_train import train_flux


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


if __name__ == "__main__":
  unittest.main()
