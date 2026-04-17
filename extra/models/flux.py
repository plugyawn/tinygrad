from dataclasses import dataclass
import math
from tinygrad import Tensor, dtypes, nn


# Reference: https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
# Reference: https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py


def rope(pos:Tensor, dim:int, theta:int) -> Tensor:
  assert dim % 2 == 0, f"rope dim must be even, got {dim}"
  pos = pos.float()
  scale = Tensor.arange(0, dim, 2, device=pos.device).float() / dim
  omega = (-math.log(theta) * scale).exp()
  out = pos.unsqueeze(-1) * omega
  rot = Tensor.stack(out.cos(), -out.sin(), out.sin(), out.cos(), dim=-1)
  return rot.reshape(*out.shape, 2, 2)


def apply_rope(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
  xq_dtype, xk_dtype = xq.dtype, xk.dtype
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape(*xq.shape).cast(xq_dtype), xk_out.reshape(*xk.shape).cast(xk_dtype)


def attention(q:Tensor, k:Tensor, v:Tensor, pe:Tensor) -> Tensor:
  q, k = apply_rope(q, k, pe)
  return q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def _split_qkv(qkv:Tensor, num_heads:int) -> tuple[Tensor, Tensor, Tensor]:
  qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, num_heads, -1).permute(2, 0, 3, 1, 4)
  return qkv[0], qkv[1], qkv[2]


class EmbedND:
  def __init__(self, dim:int, theta:int, axes_dim:list[int]):
    self.dim = dim
    self.theta = theta
    self.axes_dim = axes_dim

  def __call__(self, ids:Tensor) -> Tensor:
    assert ids.shape[-1] == len(self.axes_dim), f"expected {len(self.axes_dim)} rope axes, got {ids.shape[-1]}"
    emb = Tensor.cat(*[rope(ids[..., i], dim, self.theta) for i, dim in enumerate(self.axes_dim)], dim=2)
    return emb.unsqueeze(1)


def timestep_embedding(t:Tensor, dim:int, max_period:int=10000, time_factor:float=1000.0) -> Tensor:
  t = time_factor * t
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half, device=t.device).float() / half).exp()
  args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
  embedding = args.cos().cat(args.sin(), dim=-1)
  if dim % 2:
    embedding = embedding.cat(Tensor.zeros(embedding.shape[0], 1, device=embedding.device, dtype=embedding.dtype), dim=-1)
  return embedding.cast(t.dtype) if dtypes.is_float(t.dtype) else embedding


class MLPEmbedder:
  def __init__(self, in_dim:int, hidden_dim:int):
    self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
    self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.out_layer(self.in_layer(x).silu())


class QKNorm:
  def __init__(self, dim:int):
    self.query_norm = nn.RMSNorm(dim)
    self.key_norm = nn.RMSNorm(dim)

  def __call__(self, q:Tensor, k:Tensor, v:Tensor) -> tuple[Tensor, Tensor]:
    return self.query_norm(q).cast(v.dtype), self.key_norm(k).cast(v.dtype)


class SelfAttention:
  def __init__(self, dim:int, num_heads:int=8, qkv_bias:bool=False):
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.norm = QKNorm(head_dim)
    self.proj = nn.Linear(dim, dim)

  def __call__(self, x:Tensor, pe:Tensor) -> Tensor:
    q, k, v = _split_qkv(self.qkv(x), self.num_heads)
    q, k = self.norm(q, k, v)
    return self.proj(attention(q, k, v, pe))


@dataclass
class ModulationOut:
  shift: Tensor
  scale: Tensor
  gate: Tensor


class Modulation:
  def __init__(self, dim:int, double:bool):
    self.is_double = double
    self.multiplier = 6 if double else 3
    self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

  def __call__(self, vec:Tensor) -> tuple[ModulationOut, ModulationOut|None]:
    out = self.lin(vec.silu()).unsqueeze(1).chunk(self.multiplier, dim=-1)
    return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None


class DoubleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float, qkv_bias:bool=False):
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.num_heads = num_heads
    self.hidden_size = hidden_size

    self.img_mod = Modulation(hidden_size, double=True)
    self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
    self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_mlp = [
      nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
      Tensor.gelu,
      nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
    ]

    self.txt_mod = Modulation(hidden_size, double=True)
    self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
    self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_mlp = [
      nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
      Tensor.gelu,
      nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
    ]

  def __call__(self, img:Tensor, txt:Tensor, vec:Tensor, pe:Tensor) -> tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    assert img_mod2 is not None and txt_mod2 is not None

    img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
    img_q, img_k, img_v = _split_qkv(self.img_attn.qkv(img_modulated), self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
    txt_q, txt_k, txt_v = _split_qkv(self.txt_attn.qkv(txt_modulated), self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    q = txt_q.cat(img_q, dim=2)
    k = txt_k.cat(img_k, dim=2)
    v = txt_v.cat(img_v, dim=2)
    txt_attn, img_attn = attention(q, k, v, pe).split([txt.shape[1], img.shape[1]], dim=1)

    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * (((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift).sequential(self.img_mlp))

    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * (((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift).sequential(self.txt_mlp))
    return img, txt


class SingleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float=4.0):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

    self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
    self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
    self.norm = QKNorm(hidden_size // num_heads)
    self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.modulation = Modulation(hidden_size, double=False)

  def __call__(self, x:Tensor, vec:Tensor, pe:Tensor) -> Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = self.linear1(x_mod).split([3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    q, k, v = _split_qkv(qkv, self.num_heads)
    q, k = self.norm(q, k, v)
    output = attention(q, k, v, pe).cat(mlp.gelu(), dim=2)
    return x + mod.gate * self.linear2(output)


class LastLayer:
  def __init__(self, hidden_size:int, patch_size:int, out_channels:int):
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation = [Tensor.silu, nn.Linear(hidden_size, 2 * hidden_size, bias=True)]

  def __call__(self, x:Tensor, vec:Tensor) -> Tensor:
    shift, scale = vec.sequential(self.adaLN_modulation).chunk(2, dim=1)
    x = (1 + scale.unsqueeze(1)) * self.norm_final(x) + shift.unsqueeze(1)
    return self.linear(x)


@dataclass
class FluxParams:
  in_channels: int
  out_channels: int
  vec_in_dim: int
  context_in_dim: int
  hidden_size: int
  mlp_ratio: float
  num_heads: int
  depth: int
  depth_single_blocks: int
  axes_dim: list[int]
  theta: int
  qkv_bias: bool
  guidance_embed: bool


class Flux:
  def __init__(self, params:FluxParams):
    self.params = params
    self.in_channels = params.in_channels
    self.out_channels = params.out_channels
    if params.hidden_size % params.num_heads != 0:
      raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
    if any(dim % 2 for dim in params.axes_dim):
      raise ValueError(f"RoPE axes must be even, got {params.axes_dim}")
    pe_dim = params.hidden_size // params.num_heads
    if sum(params.axes_dim) != pe_dim:
      raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

    self.hidden_size = params.hidden_size
    self.num_heads = params.num_heads
    self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
    self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
    self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
    self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
    self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else None
    self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=True)

    self.double_blocks = [
      DoubleStreamBlock(
        self.hidden_size,
        self.num_heads,
        mlp_ratio=params.mlp_ratio,
        qkv_bias=params.qkv_bias,
      ) for _ in range(params.depth)
    ]
    self.single_blocks = [
      SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio) for _ in range(params.depth_single_blocks)
    ]
    self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

  def __call__(self, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, timesteps:Tensor, y:Tensor,
               guidance:Tensor|None=None) -> Tensor:
    if img.ndim != 3 or txt.ndim != 3:
      raise ValueError("Input img and txt tensors must have 3 dimensions.")

    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256))
    if self.params.guidance_embed:
      if guidance is None:
        raise ValueError("Didn't get guidance strength for guidance distilled model.")
      assert self.guidance_in is not None
      vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    pe = self.pe_embedder(txt_ids.cat(img_ids, dim=1))
    for block in self.double_blocks:
      img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

    img = txt.cat(img, dim=1)
    for block in self.single_blocks:
      img = block(img, vec=vec, pe=pe)

    img = img[:, txt.shape[1]:]
    return self.final_layer(img, vec)

  forward = __call__
