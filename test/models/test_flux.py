#!/usr/bin/env python
from __future__ import annotations

import unittest
from dataclasses import dataclass
import math
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from extra.models.flux import Flux, FluxParams
from examples.mlperf.flux import FLUX_FIXED_EVAL_TIMESTEPS, flux_aggregate_validation_loss, flux_eval_timesteps, flux_validation_target_met

try:
  import torch
  import torch.nn.functional as F
  TORCH_AVAILABLE = True
except ModuleNotFoundError:
  TORCH_AVAILABLE = False
  class _TorchNNStub:
    class Module: ...
  class _TorchStub:
    Tensor = object
    dtype = object
    float32 = object()
    nn = _TorchNNStub()
  torch = _TorchStub()
  F = None


def torchify(x, dtype:torch.dtype=torch.float32):
  return torch.tensor(x.tolist(), dtype=dtype)


def torch_rope(pos:torch.Tensor, dim:int, theta:int) -> torch.Tensor:
  assert dim % 2 == 0
  pos = pos.float()
  scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
  omega = torch.exp(-math.log(theta) * scale)
  out = pos.unsqueeze(-1) * omega
  rot = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
  return rot.reshape(*out.shape, 2, 2)


def torch_apply_rope(xq:torch.Tensor, xk:torch.Tensor, freqs_cis:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape_as(xq).type_as(xq), xk_out.reshape_as(xk).type_as(xk)


def torch_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, pe:torch.Tensor) -> torch.Tensor:
  q, k = torch_apply_rope(q, k, pe)
  return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def torch_timestep_embedding(t:torch.Tensor, dim:int, max_period:int=10000, time_factor:float=1000.0) -> torch.Tensor:
  t = time_factor * t
  half = dim // 2
  freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
  args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
  embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
  if dim % 2:
    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
  return embedding.type_as(t) if torch.is_floating_point(t) else embedding


def _split_qkv(qkv:torch.Tensor, num_heads:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, num_heads, -1).permute(2, 0, 3, 1, 4)
  return qkv[0], qkv[1], qkv[2]


class TorchEmbedND(torch.nn.Module):
  def __init__(self, dim:int, theta:int, axes_dim:list[int]):
    super().__init__()
    self.dim = dim
    self.theta = theta
    self.axes_dim = axes_dim

  def forward(self, ids:torch.Tensor) -> torch.Tensor:
    emb = torch.cat([torch_rope(ids[..., i], dim, self.theta) for i, dim in enumerate(self.axes_dim)], dim=2)
    return emb.unsqueeze(1)


class TorchMLPEmbedder(torch.nn.Module):
  def __init__(self, in_dim:int, hidden_dim:int):
    super().__init__()
    self.in_layer = torch.nn.Linear(in_dim, hidden_dim, bias=True)
    self.out_layer = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.out_layer(F.silu(self.in_layer(x)))


class TorchRMSNorm(torch.nn.Module):
  def __init__(self, dim:int):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(dim))

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    x_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
    return x.to(dtype=x_dtype) * self.weight


class TorchQKNorm(torch.nn.Module):
  def __init__(self, dim:int):
    super().__init__()
    self.query_norm = TorchRMSNorm(dim)
    self.key_norm = TorchRMSNorm(dim)

  def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.query_norm(q).to(v), self.key_norm(k).to(v)


class TorchSelfAttention(torch.nn.Module):
  def __init__(self, dim:int, num_heads:int=8, qkv_bias:bool=False):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.norm = TorchQKNorm(head_dim)
    self.proj = torch.nn.Linear(dim, dim)

  def forward(self, x:torch.Tensor, pe:torch.Tensor) -> torch.Tensor:
    q, k, v = _split_qkv(self.qkv(x), self.num_heads)
    q, k = self.norm(q, k, v)
    return self.proj(torch_attention(q, k, v, pe))


@dataclass
class TorchModulationOut:
  shift: torch.Tensor
  scale: torch.Tensor
  gate: torch.Tensor


class TorchModulation(torch.nn.Module):
  def __init__(self, dim:int, double:bool):
    super().__init__()
    self.is_double = double
    self.multiplier = 6 if double else 3
    self.lin = torch.nn.Linear(dim, self.multiplier * dim, bias=True)

  def forward(self, vec:torch.Tensor) -> tuple[TorchModulationOut, TorchModulationOut|None]:
    out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
    return TorchModulationOut(*out[:3]), TorchModulationOut(*out[3:]) if self.is_double else None


class TorchDoubleStreamBlock(torch.nn.Module):
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float, qkv_bias:bool=False):
    super().__init__()
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.num_heads = num_heads
    self.hidden_size = hidden_size

    self.img_mod = TorchModulation(hidden_size, double=True)
    self.img_norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn = TorchSelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)
    self.img_norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_mlp = torch.nn.Sequential(
      torch.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
      torch.nn.GELU(approximate="tanh"),
      torch.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
    )

    self.txt_mod = TorchModulation(hidden_size, double=True)
    self.txt_norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_attn = TorchSelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)
    self.txt_norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_mlp = torch.nn.Sequential(
      torch.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
      torch.nn.GELU(approximate="tanh"),
      torch.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
    )

  def forward(self, img:torch.Tensor, txt:torch.Tensor, vec:torch.Tensor, pe:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    assert img_mod2 is not None and txt_mod2 is not None

    img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
    img_q, img_k, img_v = _split_qkv(self.img_attn.qkv(img_modulated), self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
    txt_q, txt_k, txt_v = _split_qkv(self.txt_attn.qkv(txt_modulated), self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    q = torch.cat([txt_q, img_q], dim=2)
    k = torch.cat([txt_k, img_k], dim=2)
    v = torch.cat([txt_v, img_v], dim=2)
    txt_attn, img_attn = torch_attention(q, k, v, pe).split([txt.shape[1], img.shape[1]], dim=1)

    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
    return img, txt


class TorchSingleStreamBlock(torch.nn.Module):
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float=4.0):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

    self.linear1 = torch.nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
    self.norm = TorchQKNorm(hidden_size // num_heads)
    self.pre_norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.modulation = TorchModulation(hidden_size, double=False)
    self.mlp_act = torch.nn.GELU(approximate="tanh")

  def forward(self, x:torch.Tensor, vec:torch.Tensor, pe:torch.Tensor) -> torch.Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = self.linear1(x_mod).split([3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    q, k, v = _split_qkv(qkv, self.num_heads)
    q, k = self.norm(q, k, v)
    output = self.linear2(torch.cat([torch_attention(q, k, v, pe), self.mlp_act(mlp)], dim=2))
    return x + mod.gate * output


class TorchLastLayer(torch.nn.Module):
  def __init__(self, hidden_size:int, patch_size:int, out_channels:int):
    super().__init__()
    self.norm_final = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = torch.nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation = torch.nn.Sequential(
      torch.nn.SiLU(),
      torch.nn.Linear(hidden_size, 2 * hidden_size, bias=True),
    )

  def forward(self, x:torch.Tensor, vec:torch.Tensor) -> torch.Tensor:
    shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
    x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
    return self.linear(x)


class TorchFlux(torch.nn.Module):
  def __init__(self, params:FluxParams):
    super().__init__()
    pe_dim = params.hidden_size // params.num_heads
    self.params = params
    self.in_channels = params.in_channels
    self.out_channels = params.out_channels
    self.hidden_size = params.hidden_size
    self.num_heads = params.num_heads
    self.pe_embedder = TorchEmbedND(pe_dim, params.theta, params.axes_dim)
    self.img_in = torch.nn.Linear(self.in_channels, self.hidden_size, bias=True)
    self.time_in = TorchMLPEmbedder(256, self.hidden_size)
    self.vector_in = TorchMLPEmbedder(params.vec_in_dim, self.hidden_size)
    self.guidance_in = TorchMLPEmbedder(256, self.hidden_size) if params.guidance_embed else torch.nn.Identity()
    self.txt_in = torch.nn.Linear(params.context_in_dim, self.hidden_size, bias=True)
    self.double_blocks = torch.nn.ModuleList([
      TorchDoubleStreamBlock(self.hidden_size, self.num_heads, params.mlp_ratio, qkv_bias=params.qkv_bias) for _ in range(params.depth)
    ])
    self.single_blocks = torch.nn.ModuleList([
      TorchSingleStreamBlock(self.hidden_size, self.num_heads, params.mlp_ratio) for _ in range(params.depth_single_blocks)
    ])
    self.final_layer = TorchLastLayer(self.hidden_size, 1, self.out_channels)

  def forward(self, img:torch.Tensor, img_ids:torch.Tensor, txt:torch.Tensor, txt_ids:torch.Tensor, timesteps:torch.Tensor, y:torch.Tensor,
              guidance:torch.Tensor|None=None) -> torch.Tensor:
    img = self.img_in(img)
    vec = self.time_in(torch_timestep_embedding(timesteps, 256))
    if self.params.guidance_embed:
      if guidance is None:
        raise ValueError("Didn't get guidance strength for guidance distilled model.")
      vec = vec + self.guidance_in(torch_timestep_embedding(guidance, 256))
    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    pe = self.pe_embedder(torch.cat([txt_ids, img_ids], dim=1))
    for block in self.double_blocks:
      img, txt = block(img, txt, vec, pe)

    img = torch.cat([txt, img], dim=1)
    for block in self.single_blocks:
      img = block(img, vec, pe)

    img = img[:, txt.shape[1]:]
    return self.final_layer(img, vec)


def set_equal_weights(model, torch_model):
  state, torch_state = get_state_dict(model), torch_model.state_dict()
  assert set(state.keys()) == set(torch_state.keys())
  for k, v in state.items():
    torch_state[k].copy_(torch.tensor(v.numpy().tolist(), dtype=torch_state[k].dtype))


class TestFlux(unittest.TestCase):
  def test_flux_eval_timesteps_accept_fixed_eval_constants(self):
    timesteps = flux_eval_timesteps(FLUX_FIXED_EVAL_TIMESTEPS)
    np.testing.assert_allclose(timesteps.numpy(), FLUX_FIXED_EVAL_TIMESTEPS, atol=0.0, rtol=0.0)

  def test_flux_eval_timesteps_accept_scalar_bucket_id(self):
    timesteps = flux_eval_timesteps(3)
    self.assertEqual(timesteps.shape, (1,))
    np.testing.assert_allclose(timesteps.numpy(), [3 / 8], atol=0.0, rtol=0.0)

  def test_flux_aggregate_validation_loss_rejects_invalid_buckets(self):
    with self.assertRaisesRegex(ValueError, "out of range"):
      flux_aggregate_validation_loss(Tensor([1.0]), [-1])

  def test_flux_validation_target_met_accepts_tensor(self):
    self.assertTrue(flux_validation_target_met(Tensor([0.5]).reshape(()), target=0.6))
    self.assertFalse(flux_validation_target_met(Tensor([0.7]).reshape(()), target=0.6))

  def test_flux_rejects_odd_rope_axes(self):
    with self.assertRaisesRegex(ValueError, "RoPE axes must be even"):
      Flux(FluxParams(
        in_channels=4,
        out_channels=4,
        vec_in_dim=6,
        context_in_dim=8,
        hidden_size=24,
        mlp_ratio=2.0,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 1, 3],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
      ))

  @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
  def test_flux_torch_parity(self):
    params = FluxParams(
      in_channels=5,
      out_channels=7,
      vec_in_dim=9,
      context_in_dim=11,
      hidden_size=32,
      mlp_ratio=2.0,
      num_heads=4,
      depth=2,
      depth_single_blocks=3,
      axes_dim=[2, 2, 4],
      theta=10_000,
      qkv_bias=True,
      guidance_embed=True,
    )

    Tensor.manual_seed(0)
    model = Flux(params)
    with torch.no_grad():
      torch_model = TorchFlux(params).eval()
    set_equal_weights(model, torch_model)

    rng = np.random.default_rng(0)
    bsz, img_len, txt_len = 2, 6, 5
    img = rng.standard_normal((bsz, img_len, params.in_channels), dtype=np.float32)
    txt = rng.standard_normal((bsz, txt_len, params.context_in_dim), dtype=np.float32)
    img_ids = rng.integers(0, 8, size=(bsz, img_len, len(params.axes_dim)), dtype=np.int32).astype(np.float32)
    txt_ids = rng.integers(0, 8, size=(bsz, txt_len, len(params.axes_dim)), dtype=np.int32).astype(np.float32)
    timesteps = rng.random(bsz, dtype=np.float32)
    guidance = rng.random(bsz, dtype=np.float32)
    y = rng.standard_normal((bsz, params.vec_in_dim), dtype=np.float32)

    out = model(Tensor(img), Tensor(img_ids), Tensor(txt), Tensor(txt_ids), Tensor(timesteps), Tensor(y), Tensor(guidance))
    torch_out = torch_model(
      torchify(img),
      torchify(img_ids),
      torchify(txt),
      torchify(txt_ids),
      torchify(timesteps),
      torchify(y),
      torchify(guidance),
    )

    self.assertEqual(out.shape, (bsz, img_len, params.out_channels))
    torch_out_np = np.asarray(torch_out.detach().float().tolist(), dtype=np.float32)
    np.testing.assert_allclose(out.numpy(), torch_out_np, atol=3e-5, rtol=3e-5)

  def test_flux_requires_guidance_for_guidance_models(self):
    params = FluxParams(
      in_channels=4,
      out_channels=4,
      vec_in_dim=6,
      context_in_dim=8,
      hidden_size=24,
      mlp_ratio=2.0,
      num_heads=4,
      depth=1,
      depth_single_blocks=1,
      axes_dim=[2, 2, 2],
      theta=10_000,
      qkv_bias=True,
      guidance_embed=True,
    )
    model = Flux(params)

    with self.assertRaisesRegex(ValueError, "guidance strength"):
      model(
        Tensor.zeros(1, 3, params.in_channels),
        Tensor.zeros(1, 3, len(params.axes_dim)),
        Tensor.zeros(1, 2, params.context_in_dim),
        Tensor.zeros(1, 2, len(params.axes_dim)),
        Tensor.zeros(1),
        Tensor.zeros(1, params.vec_in_dim),
      )


if __name__ == '__main__':
  unittest.main()
