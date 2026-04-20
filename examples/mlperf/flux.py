import math
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

# MLPerf Training v6.0 / text_to_image reference constants.
FLUX_IMAGE_SIZE = 256
FLUX_VAE_DOWNSCALE_FACTOR = 8
FLUX_LATENT_SPATIAL_SIZE = FLUX_IMAGE_SIZE // FLUX_VAE_DOWNSCALE_FACTOR
FLUX_LATENT_CHANNELS = 16
FLUX_AUTOENCODER_SHIFT = 0.1159
FLUX_AUTOENCODER_SCALE = 0.3611
FLUX_T5_MAX_TOKENS = 256
FLUX_T5_EMBED_DIM = 4096
FLUX_CLIP_MAX_TOKENS = 77
FLUX_CLIP_EMBED_DIM = 768

FLUX_HIDDEN_SIZE = 3072
FLUX_ATTENTION_HEADS = 24
FLUX_DOUBLE_STREAM_BLOCKS = 19
FLUX_SINGLE_STREAM_BLOCKS = 38
FLUX_PATCH_SIZE = 2
FLUX_PACKED_CHANNELS = FLUX_LATENT_CHANNELS * FLUX_PATCH_SIZE * FLUX_PATCH_SIZE
FLUX_PACKED_SPATIAL_SIZE = FLUX_LATENT_SPATIAL_SIZE // FLUX_PATCH_SIZE
FLUX_MLP_RATIO = 4.0

FLUX_TRAIN_SAMPLES = 1_099_776
FLUX_EVAL_SAMPLES = 29_696
FLUX_EVAL_TIMESTEP_BUCKETS = 8
FLUX_EVAL_TIMESTEP_DIVISOR = 8.0
FLUX_EVAL_FREQUENCY_SAMPLES = 262_144
FLUX_CHECKPOINT_FREQUENCY_SAMPLES = 512_000
FLUX_QUALITY_TARGET = 0.586

FLUX_LOGVAR_MIN = -30.0
FLUX_LOGVAR_MAX = 20.0
FLUX_LR_WARMUP_STEPS = 1000
FLUX_ADAMW_BETA1 = 0.9
FLUX_ADAMW_BETA2 = 0.999
FLUX_ADAMW_EPS = 1e-8
FLUX_ADAMW_WEIGHT_DECAY = 0.0

FLUX_FIXED_EVAL_TIMESTEPS = tuple(i / FLUX_EVAL_TIMESTEP_DIVISOR for i in range(FLUX_EVAL_TIMESTEP_BUCKETS))

FLUX_MLPERF_MODEL_CONFIG = {
  "in_channels": FLUX_PACKED_CHANNELS,
  "out_channels": FLUX_PACKED_CHANNELS,
  "vec_in_dim": FLUX_CLIP_EMBED_DIM,
  "context_in_dim": FLUX_T5_EMBED_DIM,
  "hidden_size": FLUX_HIDDEN_SIZE,
  "mlp_ratio": FLUX_MLP_RATIO,
  "num_heads": FLUX_ATTENTION_HEADS,
  "depth": FLUX_DOUBLE_STREAM_BLOCKS,
  "depth_single_blocks": FLUX_SINGLE_STREAM_BLOCKS,
  "axes_dim": [16, 56, 56],
  "theta": 10_000,
  "qkv_bias": True,
  "guidance_embed": False,
}


def flux_model_config_from_env() -> dict[str, int|float|bool|list[int]]:
  model_config = dict(FLUX_MLPERF_MODEL_CONFIG)
  model_config["hidden_size"] = getenv("FLUX_HIDDEN_SIZE", model_config["hidden_size"])
  model_config["depth"] = getenv("FLUX_DOUBLE_STREAM_BLOCKS", model_config["depth"])
  model_config["depth_single_blocks"] = getenv("FLUX_SINGLE_STREAM_BLOCKS", model_config["depth_single_blocks"])
  model_config["mlp_ratio"] = getenv("FLUX_MLP_RATIO", model_config["mlp_ratio"])
  if (num_heads:=getenv("FLUX_ATTENTION_HEADS", 0)):
    model_config["num_heads"] = num_heads
  elif model_config["hidden_size"] != FLUX_MLPERF_MODEL_CONFIG["hidden_size"]:
    default_head_dim = FLUX_HIDDEN_SIZE // FLUX_ATTENTION_HEADS
    assert model_config["hidden_size"] % default_head_dim == 0, (
      f"FLUX_HIDDEN_SIZE={model_config['hidden_size']} must be divisible by {default_head_dim} when FLUX_ATTENTION_HEADS is unset"
    )
    model_config["num_heads"] = model_config["hidden_size"] // default_head_dim
  assert model_config["hidden_size"] % model_config["num_heads"] == 0, (
    f"hidden_size={model_config['hidden_size']} must be divisible by num_heads={model_config['num_heads']}"
  )
  assert model_config["hidden_size"] // model_config["num_heads"] == sum(model_config["axes_dim"]), (
    f"hidden_size/num_heads must equal {sum(model_config['axes_dim'])} to keep Flux positional encoding dimensions aligned"
  )
  return model_config


def flux_checkpoint_step_interval(global_batch_size:int) -> int:
  assert global_batch_size > 0, f"invalid global_batch_size={global_batch_size}"
  return math.ceil(FLUX_CHECKPOINT_FREQUENCY_SAMPLES / global_batch_size)


def flux_eval_step_interval(global_batch_size:int) -> int:
  assert global_batch_size > 0, f"invalid global_batch_size={global_batch_size}"
  return math.ceil(FLUX_EVAL_FREQUENCY_SAMPLES / global_batch_size)


def flux_validation_target_met(validation_loss:float|Tensor, target:float=FLUX_QUALITY_TARGET) -> bool:
  if isinstance(validation_loss, Tensor): validation_loss = float(validation_loss.item())
  return validation_loss <= target


def flux_sample_latents(mean:Tensor, logvar:Tensor, latent_noise:Tensor|None=None) -> Tensor:
  if latent_noise is None: latent_noise = Tensor.randn(*mean.shape, device=mean.device, dtype=mean.dtype)
  std = (0.5 * logvar.clip(FLUX_LOGVAR_MIN, FLUX_LOGVAR_MAX)).exp()
  return (mean + std * latent_noise - FLUX_AUTOENCODER_SHIFT) * FLUX_AUTOENCODER_SCALE


def flux_sample_training_timesteps(batch_size:int, device=None, dtype=dtypes.float32, timestep_noise:Tensor|None=None) -> Tensor:
  if timestep_noise is None: timestep_noise = Tensor.randn(batch_size, device=device, dtype=dtype)
  return timestep_noise.cast(dtype).sigmoid()


def flux_eval_timestep_ids(timestep_ids:Tensor|int|float|list[int]|list[float]|tuple[int, ...]|tuple[float, ...], device=None) -> Tensor:
  if isinstance(timestep_ids, Tensor):
    if dtypes.is_int(timestep_ids.dtype):
      from tinygrad.engine.jit import JitError
      try: values = timestep_ids.to("CPU").numpy().reshape(-1).tolist()
      except JitError: return timestep_ids.cast(dtypes.int32)
    else:
      values = timestep_ids.numpy().reshape(-1).tolist()
  else:
    values = timestep_ids
  if isinstance(values, (int, float)): values = [values]
  else: values = list(values)

  bucket_ids = []
  for value in values:
    if isinstance(value, int):
      bucket_id = value
    else:
      bucket_id = round(float(value) * FLUX_EVAL_TIMESTEP_DIVISOR)
      if not math.isclose(bucket_id / FLUX_EVAL_TIMESTEP_DIVISOR, float(value), abs_tol=1e-6):
        raise ValueError(f"eval timestep {value} does not align to MLPerf buckets")
    if not 0 <= bucket_id < FLUX_EVAL_TIMESTEP_BUCKETS:
      raise ValueError(f"eval timestep bucket {bucket_id} out of range")
    bucket_ids.append(bucket_id)
  return Tensor(bucket_ids, device=device, dtype=dtypes.int32)


def flux_eval_timesteps(timestep_ids:Tensor|int|float|list[int]|list[float]|tuple[int, ...]|tuple[float, ...], device=None,
                        dtype=dtypes.float32) -> Tensor:
  return flux_eval_timestep_ids(timestep_ids, device=device).cast(dtype) / FLUX_EVAL_TIMESTEP_DIVISOR

def flux_validation_noises(shape:tuple[int, ...], seed:int, batch_index:int, device=None, dtype=dtypes.float32) -> tuple[Tensor, Tensor]:
  latent_seed, flow_seed = seed + batch_index * 2, seed + batch_index * 2 + 1
  latent_noise = Tensor(np.random.RandomState(latent_seed).standard_normal(size=shape).astype(np.float32), device=device, dtype=dtypes.float32)
  flow_noise = Tensor(np.random.RandomState(flow_seed).standard_normal(size=shape).astype(np.float32), device=device, dtype=dtypes.float32)
  return latent_noise.cast(dtype), flow_noise.cast(dtype)


def flux_pack_latents(latents:Tensor) -> Tensor:
  bsz, channels, height, width = latents.shape
  assert channels == FLUX_LATENT_CHANNELS, f"expected {FLUX_LATENT_CHANNELS} latent channels, got {channels}"
  assert height == width == FLUX_LATENT_SPATIAL_SIZE, f"expected {FLUX_LATENT_SPATIAL_SIZE}x{FLUX_LATENT_SPATIAL_SIZE} latents, got {height}x{width}"
  assert height % FLUX_PATCH_SIZE == 0 and width % FLUX_PATCH_SIZE == 0
  return latents.reshape(
    bsz, channels, FLUX_PACKED_SPATIAL_SIZE, FLUX_PATCH_SIZE, FLUX_PACKED_SPATIAL_SIZE, FLUX_PATCH_SIZE,
  ).permute(0, 2, 4, 1, 3, 5).reshape(
    bsz, FLUX_PACKED_SPATIAL_SIZE * FLUX_PACKED_SPATIAL_SIZE, FLUX_PACKED_CHANNELS,
  )


def flux_text_ids(batch_size:int, txt_len:int, device=None) -> Tensor:
  return Tensor.zeros(batch_size, txt_len, 3, device=device, dtype=dtypes.float32)


def flux_image_ids(batch_size:int, device=None) -> Tensor:
  rows = Tensor.arange(FLUX_PACKED_SPATIAL_SIZE, device=device, dtype=dtypes.float32).reshape(
    FLUX_PACKED_SPATIAL_SIZE, 1,
  ).expand(FLUX_PACKED_SPATIAL_SIZE, FLUX_PACKED_SPATIAL_SIZE)
  cols = Tensor.arange(FLUX_PACKED_SPATIAL_SIZE, device=device, dtype=dtypes.float32).reshape(
    1, FLUX_PACKED_SPATIAL_SIZE,
  ).expand(FLUX_PACKED_SPATIAL_SIZE, FLUX_PACKED_SPATIAL_SIZE)
  ids = Tensor.stack(Tensor.zeros_like(rows), rows, cols, dim=-1).reshape(1, FLUX_PACKED_SPATIAL_SIZE * FLUX_PACKED_SPATIAL_SIZE, 3)
  return ids.expand(batch_size, *ids.shape[1:])


def flux_rectified_flow_inputs(latents:Tensor, timesteps:Tensor, flow_noise:Tensor|None=None) -> tuple[Tensor, Tensor]:
  assert timesteps.shape[0] == latents.shape[0], f"expected {latents.shape[0]} timesteps, got {timesteps.shape[0]}"
  if flow_noise is None: flow_noise = Tensor.randn(*latents.shape, device=latents.device, dtype=latents.dtype)
  t = timesteps.cast(latents.dtype).reshape(timesteps.shape[0], *([1] * (len(latents.shape) - 1)))
  return (1 - t) * flow_noise + t * latents, latents - flow_noise


def flux_rectified_flow_losses(model, latents:Tensor, txt:Tensor, vec:Tensor, timesteps:Tensor,
                               guidance:Tensor|None=None, flow_noise:Tensor|None=None) -> Tensor:
  assert txt.shape[0] == vec.shape[0] == latents.shape[0], (
    f"batch mismatch latents={latents.shape[0]}, txt={txt.shape[0]}, vec={vec.shape[0]}"
  )
  noisy_latents, velocity_target = flux_rectified_flow_inputs(latents, timesteps, flow_noise=flow_noise)
  noisy_tokens = flux_pack_latents(noisy_latents)
  target_tokens = flux_pack_latents(velocity_target)
  img_ids = flux_image_ids(noisy_tokens.shape[0], device=noisy_tokens.device)
  txt_ids = flux_text_ids(txt.shape[0], txt.shape[1], device=txt.device)
  velocity_pred = model(noisy_tokens, img_ids, txt, txt_ids, timesteps.cast(dtypes.float32), vec, guidance)
  return ((velocity_pred - target_tokens) ** 2).reshape(velocity_pred.shape[0], -1).mean(axis=-1)


def flux_train_loss(model, mean:Tensor, logvar:Tensor, txt:Tensor, vec:Tensor, guidance:Tensor|None=None,
                    latent_noise:Tensor|None=None, flow_noise:Tensor|None=None, timestep_noise:Tensor|None=None) -> Tensor:
  latents = flux_sample_latents(mean, logvar, latent_noise=latent_noise)
  timesteps = flux_sample_training_timesteps(latents.shape[0], device=latents.device, timestep_noise=timestep_noise)
  return flux_rectified_flow_losses(model, latents, txt, vec, timesteps, guidance=guidance, flow_noise=flow_noise).mean()


def flux_validation_losses(model, mean:Tensor, logvar:Tensor, txt:Tensor, vec:Tensor,
                           timestep_ids:Tensor|int|float|list[int]|list[float]|tuple[int, ...]|tuple[float, ...],
                           guidance:Tensor|None=None, latent_noise:Tensor|None=None, flow_noise:Tensor|None=None) -> Tensor:
  latents = flux_sample_latents(mean, logvar, latent_noise=latent_noise)
  timesteps = flux_eval_timesteps(timestep_ids, device=latents.device)
  return flux_rectified_flow_losses(model, latents, txt, vec, timesteps, guidance=guidance, flow_noise=flow_noise)


def flux_aggregate_validation_loss(losses:Tensor, timestep_ids:Tensor|int|float|list[int]|list[float]|tuple[int, ...]|tuple[float, ...]) -> Tensor:
  losses = losses.reshape(-1).cast(dtypes.float32)
  timestep_ids = flux_eval_timestep_ids(timestep_ids, device=losses.device)
  missing_buckets = sorted(set(range(FLUX_EVAL_TIMESTEP_BUCKETS)).difference(map(int, timestep_ids.to("CPU").numpy().reshape(-1).tolist())))
  if missing_buckets: raise ValueError(f"missing eval timestep buckets: {missing_buckets}")
  bucket_ids = Tensor.arange(FLUX_EVAL_TIMESTEP_BUCKETS, device=timestep_ids.device, dtype=dtypes.int32).reshape(FLUX_EVAL_TIMESTEP_BUCKETS, 1)
  matches = (bucket_ids == timestep_ids.reshape(1, -1)).cast(dtypes.float32)
  count_per_bucket = matches.sum(axis=1)
  mean_per_bucket = (matches * losses.reshape(1, -1)).sum(axis=1) / count_per_bucket.maximum(1.0)
  present = (count_per_bucket > 0).cast(dtypes.float32)
  return (mean_per_bucket * present).sum() / present.sum()


def flux_validation_loss(model, mean:Tensor, logvar:Tensor, txt:Tensor, vec:Tensor,
                         timestep_ids:Tensor|int|float|list[int]|list[float]|tuple[int, ...]|tuple[float, ...],
                         guidance:Tensor|None=None, latent_noise:Tensor|None=None, flow_noise:Tensor|None=None) -> Tensor:
  losses = flux_validation_losses(model, mean, logvar, txt, vec, timestep_ids, guidance=guidance,
                                  latent_noise=latent_noise, flow_noise=flow_noise)
  return flux_aggregate_validation_loss(losses, timestep_ids)


__all__ = [
  "FLUX_ADAMW_BETA1", "FLUX_ADAMW_BETA2", "FLUX_ADAMW_EPS", "FLUX_ADAMW_WEIGHT_DECAY",
  "FLUX_ATTENTION_HEADS", "FLUX_CHECKPOINT_FREQUENCY_SAMPLES", "FLUX_CLIP_EMBED_DIM", "FLUX_CLIP_MAX_TOKENS",
  "FLUX_DOUBLE_STREAM_BLOCKS", "FLUX_EVAL_FREQUENCY_SAMPLES", "FLUX_EVAL_SAMPLES", "FLUX_EVAL_TIMESTEP_BUCKETS",
  "FLUX_EVAL_TIMESTEP_DIVISOR", "FLUX_FIXED_EVAL_TIMESTEPS", "FLUX_HIDDEN_SIZE", "FLUX_IMAGE_SIZE",
  "FLUX_LATENT_CHANNELS", "FLUX_LATENT_SPATIAL_SIZE", "FLUX_LOGVAR_MAX", "FLUX_LOGVAR_MIN",
  "FLUX_LR_WARMUP_STEPS", "FLUX_MLPERF_MODEL_CONFIG", "FLUX_MLP_RATIO", "FLUX_PACKED_CHANNELS",
  "FLUX_PACKED_SPATIAL_SIZE", "FLUX_PATCH_SIZE", "FLUX_AUTOENCODER_SCALE", "FLUX_AUTOENCODER_SHIFT",
  "FLUX_QUALITY_TARGET", "FLUX_SINGLE_STREAM_BLOCKS", "FLUX_T5_EMBED_DIM", "FLUX_T5_MAX_TOKENS",
  "FLUX_TRAIN_SAMPLES", "FLUX_VAE_DOWNSCALE_FACTOR", "flux_aggregate_validation_loss",
  "flux_model_config_from_env",
  "flux_checkpoint_step_interval", "flux_eval_step_interval", "flux_eval_timestep_ids", "flux_eval_timesteps",
  "flux_image_ids", "flux_pack_latents", "flux_rectified_flow_inputs", "flux_rectified_flow_losses", "flux_sample_latents",
  "flux_sample_training_timesteps", "flux_train_loss", "flux_validation_loss",
  "flux_validation_losses", "flux_validation_noises", "flux_validation_target_met", "flux_text_ids",
]
