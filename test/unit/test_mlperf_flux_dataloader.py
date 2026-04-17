import io, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from tinygrad import dtypes
from examples.mlperf.dataloader import batch_load_flux_preprocessed, load_flux_empty_encodings


def _bf16_bits(values) -> np.ndarray:
  return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _npy_bytes(values) -> bytes:
  buf = io.BytesIO()
  np.save(buf, np.asarray(values), allow_pickle=False)
  return buf.getvalue()


class TestMLPerfFluxDataloader(unittest.TestCase):
  def test_batch_load_flux_preprocessed(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      shard0 = root / "train-00000"
      shard1 = root / "train-00001"
      shard0.mkdir()
      shard1.mkdir()
      (shard0 / "state.json").write_text("{}")
      (shard1 / "state.json").write_text("{}")

      datasets = {
        str(shard0): [{
          "__key__": "sample-0",
          "t5_encodings": _npy_bytes(_bf16_bits([[1.0, 2.0], [3.0, 4.0]])),
          "clip_encodings": _npy_bytes(_bf16_bits([5.0, 6.0])),
          "mean": _npy_bytes(_bf16_bits([[7.0, 8.0]])),
          "logvar": _npy_bytes(_bf16_bits([[9.0, 10.0]])),
          "timestep": 1,
        }],
        str(shard1): [{
          "__key__": "sample-1",
          "t5_encodings": _npy_bytes(_bf16_bits([[11.0, 12.0], [13.0, 14.0]])),
          "clip_encodings": _npy_bytes(_bf16_bits([15.0, 16.0])),
          "mean": _npy_bytes(_bf16_bits([[17.0, 18.0]])),
          "logvar": _npy_bytes(_bf16_bits([[19.0, 20.0]])),
          "timestep": 3,
        }],
      }
      fake_datasets = types.ModuleType("datasets")
      fake_datasets.load_from_disk = lambda path: datasets[path]

      with patch.dict(sys.modules, {"datasets": fake_datasets}):
        batches = list(batch_load_flux_preprocessed(root / "train-*", BS=2))

    self.assertEqual(len(batches), 1)
    batch = batches[0]
    self.assertEqual(batch["__key__"], ["sample-0", "sample-1"])
    self.assertEqual(batch["timestep"].dtype, dtypes.int32)
    np.testing.assert_array_equal(batch["timestep"].numpy(), [1, 3])
    self.assertEqual(batch["t5_encodings"].dtype, dtypes.bfloat16)
    np.testing.assert_allclose(batch["t5_encodings"].float().numpy(), [[[1.0, 2.0], [3.0, 4.0]], [[11.0, 12.0], [13.0, 14.0]]], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(batch["clip_encodings"].float().numpy(), [[5.0, 6.0], [15.0, 16.0]], atol=0.0, rtol=0.0)

  def test_load_flux_empty_encodings(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      np.save(root / "t5_empty.npy", np.array([[1.5, 2.5]], dtype=np.float16), allow_pickle=False)
      np.save(root / "clip_empty.npy", np.array([[3.5, 4.5]], dtype=np.float16), allow_pickle=False)

      empty = load_flux_empty_encodings(root)

    self.assertEqual(empty["t5_encodings"].dtype, dtypes.float16)
    self.assertEqual(empty["clip_encodings"].dtype, dtypes.float16)
    np.testing.assert_allclose(empty["t5_encodings"].numpy(), [[1.5, 2.5]], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(empty["clip_encodings"].numpy(), [[3.5, 4.5]], atol=0.0, rtol=0.0)


if __name__ == "__main__":
  unittest.main()
