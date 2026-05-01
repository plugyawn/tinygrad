#!/usr/bin/env python3

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import csv, json, math, os, random, subprocess, sys, time
import numpy as np
from pathlib import Path
from typing import Optional
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from extra.lr_scheduler import OneCycleLR
from tinygrad import nn, dtypes, Tensor, Device, GlobalCounters, TinyJit, Variable
from tinygrad.nn.state import get_state_dict
from tinygrad.nn import optim
from tinygrad.helpers import Context, BEAM, WINO, getenv, colored, prod
from extra.bench_log import BenchEvent, WallTimeEvent

NUM_GPUS = getenv("GPUS", 1)
BS = getenv("BS", 1024)
EVAL_BS = getenv("EVAL_BS", 10000 if NUM_GPUS == 1 else BS)
GPUS = [f'{Device.DEFAULT}:{i}' for i in range(NUM_GPUS)]
assert BS % len(GPUS) == 0, f"{BS=} is not a multiple of {len(GPUS)=}"
assert EVAL_BS % len(GPUS) == 0, f"{EVAL_BS=} is not a multiple of {len(GPUS)=}"
if getenv("FUSE_OPTIM", 1): Context(FUSE_OPTIM=1).__enter__()

bias_scaler = 64
hyp = {
  'seed' : 201,
  'opt': {
    'bias_lr':            1.525 * bias_scaler/512,
    'non_bias_lr':        1.525 / 512,
    'bias_decay':         6.687e-4 * BS/bias_scaler,
    'non_bias_decay':     6.687e-4 * BS,
    'final_lr_ratio':     0.07,
    'initial_div_factor': 1e16,
    'label_smoothing':    0.20,
    'momentum':           0.85,
    'percent_start':      0.23,
    'loss_scale_scaler':  1./32,
    'scaling_factor':     1./9,
  },
  'net': {
    'kernel_size':         2,
    'cutmix_size':         3,
    'cutmix_epochs':       6,
    'pad_amount':          2,
    'batch_norm_momentum': 0.4,
    'base_depth':          64,
    'whitening_examples':  50000,
  },
  'ema': {
    'epochs':       10,
    'decay_base':   .95,
    'decay_pow':    3.0,
    'every_n_steps': 5,
  },
  'misc': {
    'train_epochs': 12.1,
  }
}

class UnsyncedBatchNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1, num_devices=len(GPUS)):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
    self.num_devices = num_devices
    self.updates:list[Tensor] = []

    if affine: self.weight, self.bias = Tensor.ones(sz, dtype=dtypes.float32), Tensor.zeros(sz, dtype=dtypes.float32)
    else: self.weight, self.bias = None, None

    self.running_mean = Tensor.zeros(num_devices, sz, dtype=dtypes.float32, requires_grad=False)
    self.running_var = Tensor.ones(num_devices, sz, dtype=dtypes.float32, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, dtype=dtypes.int, requires_grad=False)

  def __call__(self, x:Tensor):
    xr = x.reshape(self.num_devices, -1, *x.shape[1:]).cast(dtypes.float32)
    batch_mean, batch_invstd = self.calc_stats(xr)
    ret = xr.batchnorm(
      self.weight.reshape(1, -1).expand((self.num_devices, -1)),
      self.bias.reshape(1, -1).expand((self.num_devices, -1)),
      batch_mean, batch_invstd, axis=(0, 2))
    return ret.reshape(x.shape).cast(x.dtype)

  def calc_stats(self, x:Tensor):
    self.updates = []
    if Tensor.training:
      # This requires two full memory accesses to x
      # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
      # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
      batch_mean = x.mean(axis=(1,3,4))
      y = (x - batch_mean.detach().reshape(shape=[batch_mean.shape[0], 1, -1, 1, 1]))  # d(var)/d(mean) = 0
      batch_var = (y*y).mean(axis=(1,3,4))
      batch_invstd = batch_var.add(self.eps).pow(-0.5)

      # NOTE: wow, this is done all throughout training in most PyTorch models
      if self.track_running_stats:
        self.updates.append(self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(self.running_mean.dtype)))
        batch_var_adjust = prod(y.shape[1:])/(prod(y.shape[1:])-y.shape[2])
        self.updates.append(self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * batch_var_adjust * batch_var.detach().cast(self.running_var.dtype)))
        self.updates.append(self.num_batches_tracked.assign(self.num_batches_tracked + 1))
    else:
      batch_mean = self.running_mean
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
      batch_invstd = self.running_var.reshape(self.running_var.shape[0], 1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
    return batch_mean, batch_invstd

class BatchNorm(nn.BatchNorm2d if getenv("SYNCBN") else UnsyncedBatchNorm):
  def __init__(self, num_features):
    super().__init__(num_features, track_running_stats=True, eps=1e-12, momentum=hyp['net']['batch_norm_momentum'], affine=True)
    self.weight.requires_grad = False
    self.bias.requires_grad = True

class ConvGroup:
  def __init__(self, channels_in, channels_out):
    self.conv1 = nn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)

    self.norm1 = BatchNorm(channels_out)
    self.norm2 = BatchNorm(channels_out)

  def __call__(self, x):
    x = self.conv1(x)
    x = x.max_pool2d(2)
    x = x.float()
    x = self.norm1(x)
    x = x.cast(dtypes.default_float)
    x = x.gelu()
    x = self.conv2(x)
    x = x.float()
    x = self.norm2(x)
    x = x.cast(dtypes.default_float)
    x = x.gelu()

    return x

class SpeedyResNet:
  def __init__(self, W):
    self.whitening = W
    self.net = [
      ConvGroup(W.shape[0], 64),
      ConvGroup(64, 256),
      ConvGroup(256, 512),
      lambda x: x.max((2,3)),
      nn.Linear(512, 10, bias=False),
      lambda x: x * hyp['opt']['scaling_factor'],
    ]

  def _forward(self, x):
    return x.conv2d(self.whitening).gelu().sequential(self.net)

  def __call__(self, x, training=True):
    if training: return self._forward(x)
    out = self._forward(x.cat(x[..., ::-1], dim=0))
    return (out[:x.shape[0]] + out[x.shape[0]:]) * 0.5

def _dirac_like(w:np.ndarray) -> np.ndarray:
  ret = np.zeros_like(w)
  oc, ic, kh, kw = ret.shape
  for i in range(min(oc, ic)): ret[i, i, kh//2, kw//2] = 1
  return ret

def _assign_weight(t:Tensor, w:np.ndarray):
  t.assign(Tensor(w.astype(np.float32), device=t.device, requires_grad=False).cast(t.dtype)).realize()

def init_hlb_weights(model:SpeedyResNet):
  for layer in model.net:
    if not isinstance(layer, ConvGroup): continue

    conv1 = layer.conv1.weight.float().numpy()
    std_pre, mean_pre = conv1.std(), conv1.mean()
    conv1 = conv1 + _dirac_like(conv1)
    std_post, mean_post = conv1.std(), conv1.mean()
    _assign_weight(layer.conv1.weight, (conv1 - mean_post) / std_post * std_pre + mean_pre)
    _assign_weight(layer.conv2.weight, _dirac_like(layer.conv2.weight.float().numpy()))

def train_cifar():
  artifact_dir = getenv("ARTIFACT_DIR", "")
  if not artifact_dir and getenv("BENCHMARK_LOG", ""):
    artifact_dir = f"artifacts/hlb_cifar10/a100_{time.strftime('%Y%m%d_%H%M%S')}"
  artifact_path = Path(artifact_dir) if artifact_dir else None
  if artifact_path: artifact_path.mkdir(parents=True, exist_ok=True)
  artifact_env_keys = {"DEV", "DEFAULT_FLOAT", "GPUS", "BS", "EVAL_BS", "BEAM", "JITBEAM", "WINO", "TC_OPT", "TARGET_EVAL_ACC_PCT",
                       "ASSERT_MAX_WALL_TIME", "ASSERT_MIN_STEP_TIME", "BENCHMARK_LOG", "ARTIFACT_DIR", "RUN_PHASE",
                       "TRAIN_EPOCHS", "STEPS", "EVAL_STEPS", "SEED", "WHITEN_EXAMPLES", "WHITEN_SPLITS", "CUTMIX", "RANDOM_CROP", "RANDOM_FLIP",
                       "SYNCBN", "FUSE_OPTIM", "LATEBEAM", "LATEWINO", "DISABLE_BACKWARD", "LOG_EPOCHS", "LOG_STEPS", "JIT_EVAL", "SYNC_STEPS"}
  artifact_env = {k: os.environ[k] for k in sorted(artifact_env_keys) if k in os.environ}
  artifact_log = open(artifact_path/"run.log", "w", buffering=1) if artifact_path else None

  def log(*args, **kwargs):
    print(*args, **kwargs)
    if artifact_log is not None:
      log_kwargs = {k:v for k,v in kwargs.items() if k in {"sep", "end", "flush"}}
      print(*args, file=artifact_log, **log_kwargs)

  def _run(cmd):
    try: return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, cwd=REPO_ROOT).strip()
    except Exception as e: return str(e)

  def _write_plot(steps, accs):
    if artifact_path is None: return
    try:
      from PIL import Image, ImageDraw
      img = Image.new("RGB", (900, 520), "white")
      draw = ImageDraw.Draw(img)
      pad = 54
      draw.rectangle((pad, pad, 850, 460), outline="black")
      def line(vals, color, lo, hi):
        pts = [(pad + i*796/max(1, len(vals)-1), 460 - (v-lo)*406/max(1e-12, hi-lo)) for i,v in enumerate(vals)]
        if len(pts) > 1: draw.line(pts, fill=color, width=2)
      if steps: line(steps, "blue", 0, max(steps))
      if accs: line(accs, "green", 0, 100)
      draw.text((pad, 20), "blue: step wall ms, green: eval acc pct", fill="black")
      img.save(artifact_path/"loss_acc_time.png")
    except Exception as e:
      (artifact_path/"loss_acc_time.png.err").write_text(str(e))

  def set_seed(seed):
    Tensor.manual_seed(seed)
    random.seed(seed)

  # ========== Model ==========
  def whitening(X, kernel_size=hyp['net']['kernel_size']):
    X_np = X.float().numpy()
    X_np = X_np[np.random.default_rng(getenv("SEED", hyp['seed'])).permutation(X_np.shape[0])]
    X_np = X_np[:getenv("WHITEN_EXAMPLES", hyp['net']['whitening_examples'])]

    def _cov(X):
      X = X - X.mean(axis=0, keepdims=True)
      return (X.T @ X) / (X.shape[0] - 1)

    def _patches(data, patch_size=(kernel_size,kernel_size)):
      h, w = patch_size
      c = data.shape[1]
      axis = (2, 3)
      return np.lib.stride_tricks.sliding_window_view(data, window_shape=(h,w), axis=axis).transpose((0,3,2,1,4,5)).reshape((-1,c,h,w))

    def _eigens(patches):
      n,c,h,w = patches.shape
      Σ = _cov(patches.reshape(n, c*h*w))
      Λ, V = np.linalg.eigh(Σ, UPLO='U')
      return np.flip(Λ, 0).reshape(-1,1,1,1), np.flip(V.T.reshape(c*h*w, c, h, w), 0)

    # NOTE: np.linalg.eigh only supports float32 so the whitening layer weights need to be converted to float16 manually
    split_size = getenv("WHITEN_SPLITS", 5000)
    split_data = [X_np] if split_size <= 0 else [X_np[i:i+split_size] for i in range(0, X_np.shape[0], split_size)]
    eigens = [_eigens(_patches(x)) for x in split_data if x.shape[0]]
    Λ = np.stack([x[0] for x in eigens], axis=0).mean(axis=0)
    V = np.stack([x[1] for x in eigens], axis=0).mean(axis=0)
    W = V/np.sqrt(Λ+1e-2)
    W = np.concatenate((W, -W), axis=0)

    return Tensor(W.astype(np.float32), requires_grad=False).cast(dtypes.default_float)

  # ========== Loss ==========
  def cross_entropy(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
    divisor = y.shape[1]
    assert isinstance(divisor, int), "only supported int divisor"
    y = (1 - label_smoothing)*y + label_smoothing / divisor
    ret = -x.log_softmax(axis=1).mul(y).sum(axis=1)
    if reduction=='none': return ret
    if reduction=='sum': return ret.sum()
    if reduction=='mean': return ret.mean()
    raise NotImplementedError(reduction)

  # ========== Preprocessing ==========
  # NOTE: this only works for RGB in format of NxCxHxW and pads the HxW
  def pad_reflect(X, size=2) -> Tensor:
    X = X[...,:,1:size+1].flip(-1).cat(X, X[...,:,-(size+1):-1].flip(-1), dim=-1)
    X = X[...,1:size+1,:].flip(-2).cat(X, X[...,-(size+1):-1,:].flip(-2), dim=-2)
    return X

  # return a binary mask in the format of BS x C x H x W where H x W contains a random square mask
  def make_square_mask(shape, mask_size) -> Tensor:
    BS, _, H, W = shape
    low_x = Tensor.randint(BS, low=0, high=W-mask_size+1).reshape(BS,1,1,1)
    low_y = Tensor.randint(BS, low=0, high=H-mask_size+1).reshape(BS,1,1,1)
    idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,1,W))
    idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,1,H,1))
    return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

  # Similar, but different enough.
  def make_random_crop_indices(shape, mask_size) -> Tensor:
    BS, _, H, W = shape
    low_x = Tensor.randint(BS, low=0, high=W-mask_size+1).reshape(BS,1,1,1)
    low_y = Tensor.randint(BS, low=0, high=H-mask_size+1).reshape(BS,1,1,1)
    idx_x = Tensor.arange(mask_size, dtype=dtypes.int32).reshape((1,1,1,mask_size))
    idx_y = Tensor.arange(mask_size, dtype=dtypes.int32).reshape((1,1,mask_size,1))
    return low_x, low_y, idx_x, idx_y

  def random_crop(X:Tensor, crop_size=32):
    Xs, Ys, Xi, Yi = make_random_crop_indices(X.shape, crop_size)
    return X.gather(-1, (Xs + Xi).expand(-1, 3, X.shape[2], -1)).gather(-2, ((Ys+Yi).expand(-1, 3, crop_size, crop_size)))

  def cutmix(X, Y, order, mask_size=3):
    mask = make_square_mask(X.shape, mask_size)
    X_patch, Y_patch = X[order], Y[order]
    X_cutmix = mask.where(X_patch, X)
    mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
    return X_cutmix, Y_cutmix

  @TinyJit
  def augmentations(X:Tensor, Y:Tensor):
    perms = Tensor.randperm(X.shape[0], device=X.device)
    if getenv("RANDOM_CROP", 1):
      X = random_crop(X, crop_size=32)
    if getenv("RANDOM_FLIP", 1):
      # NOTE: RANGEIFY=1 needs this contiguous or the X[perms] is very slow
      X = (Tensor.rand(X.shape[0],1,1,1) < 0.5).where(X.flip(-1), X).contiguous() # flip LR
    X, Y = X[perms], Y[perms]
    return X, Y

  @TinyJit
  def augmentations_cutmix(X:Tensor, Y:Tensor):
    perms = Tensor.randperm(X.shape[0], device=X.device)
    mix_perms = Tensor.randperm(X.shape[0], device=X.device)
    if getenv("RANDOM_CROP", 1):
      X = random_crop(X, crop_size=32)
    if getenv("RANDOM_FLIP", 1):
      X = (Tensor.rand(X.shape[0],1,1,1) < 0.5).where(X.flip(-1), X).contiguous()
    X, Y = X[perms], Y[perms]
    return cutmix(X, Y, mix_perms, mask_size=hyp['net']['cutmix_size'])

  # the operations that remain inside batch fetcher is the ones that involves random operations
  def fetch_batches(X_in:Tensor, Y_in:Tensor, BS:int, is_train:bool, epoch:int=0, epoch_fraction:float=1.0, do_cutmix:bool=False):
    st = time.monotonic()
    X, Y = X_in, Y_in
    if is_train:
      X, Y = (augmentations_cutmix if do_cutmix and getenv("CUTMIX", 1) else augmentations)(X, Y)
    et = time.monotonic()
    if getenv("LOG_EPOCHS", 0): log(f"shuffling {'training' if is_train else 'test'} dataset in {(et-st)*1e3:.2f} ms ({epoch=})")

    full_batch_count = X.shape[0] // BS if epoch_fraction >= 1 else round(epoch_fraction * X.shape[0] / BS)
    full_batches = full_batch_count * BS
    if full_batches == 0: return
    if not is_train:
      for i in range(0, full_batches, BS): yield X[i:i+BS], Y[i:i+BS]
      return
    vi = Variable("i", 0, full_batches - BS)
    for i in range(0, full_batches, BS):
      vib = vi.bind(i)
      yield X[vib:vib+BS], Y[vib:vib+BS]

  class modelEMA():
    def __init__(self, w, net):
      # self.model_ema = copy.deepcopy(net) # won't work for opencl due to unpickeable pyopencl._cl.Buffer
      self.net_ema = SpeedyResNet(w)
      for net_ema_param, net_param in zip(get_state_dict(self.net_ema).values(), get_state_dict(net).values()):
        net_ema_param.requires_grad = False
        net_ema_param.assign(net_param.detach()).realize()

    @TinyJit
    def update(self, net, decay):
      updates = []
      for net_ema_param, (param_name, net_param) in zip(get_state_dict(self.net_ema).values(), get_state_dict(net).items()):
        if dtypes.is_float(net_param.dtype):
          ema = net_ema_param.detach()*decay + net_param.detach()*(1.-decay)
          updates.append(net_ema_param.assign(ema))
          if not (("norm" in param_name and "weight" in param_name) or "whitening" in param_name):
            updates.append(net_param.assign(ema.detach()))
      Tensor.realize(*updates)

  set_seed(getenv('SEED', hyp['seed']))

  X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
  # one-hot encode labels
  Y_train, Y_test = Y_train.one_hot(10), Y_test.one_hot(10)
  # preprocess data
  X_train, X_test = X_train.float() / 255.0, X_test.float() / 255.0
  cifar_std, cifar_mean = X_train.std_mean(axis=(0, 2, 3))
  def preprocess(X:Tensor): return (X - cifar_mean.reshape(1,3,1,1)) / cifar_std.reshape(1,3,1,1)
  X_train, X_test = preprocess(X_train), preprocess(X_test)

  # precompute whitening patches
  W = whitening(X_train)

  # initialize model weights
  model = SpeedyResNet(W)
  init_hlb_weights(model)

  # padding is not timed in the original repo since it can be done all at once
  X_train = pad_reflect(X_train, size=hyp['net']['pad_amount'])

  # Convert data and labels to the default dtype
  X_train, Y_train = X_train.cast(dtypes.default_float), Y_train.cast(dtypes.default_float)
  X_test, Y_test = X_test.cast(dtypes.default_float), Y_test.cast(dtypes.default_float)

  if len(GPUS) > 1:
    for k, x in get_state_dict(model).items():
      if not getenv('SYNCBN') and ('running_mean' in k or 'running_var' in k):
        x.shard_(GPUS, axis=0)
      else:
        x.to_(GPUS)

  # parse the training params into bias and non-bias
  params_dict = get_state_dict(model)
  params_bias = []
  params_non_bias = []
  for params in params_dict:
    if params_dict[params].requires_grad is not False:
      if 'bias' in params:
        params_bias.append(params_dict[params])
      else:
        params_non_bias.append(params_dict[params])

  opt_bias     = optim.SGD(params_bias,     lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
  opt_non_bias = optim.SGD(params_non_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['non_bias_decay'])

  num_steps_per_epoch = X_train.shape[0] // BS
  train_epochs = getenv("TRAIN_EPOCHS", hyp['misc']['train_epochs'])
  default_steps = math.ceil(num_steps_per_epoch * train_epochs)
  steps = getenv("STEPS", default_steps)

  # NOTE taken from the hlb_CIFAR repository, might need to be tuned
  initial_div_factor = hyp['opt']['initial_div_factor']
  final_lr_ratio = hyp['opt']['final_lr_ratio']
  pct_start = hyp['opt']['percent_start']
  lr_sched_bias     = OneCycleLR(opt_bias,     max_lr=hyp['opt']['bias_lr'],     pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=steps)
  lr_sched_non_bias = OneCycleLR(opt_non_bias, max_lr=hyp['opt']['non_bias_lr'], pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=steps)

  def train_step(model, optimizer, lr_scheduler, X, Y):
    out = model(X)
    loss_batchsize_scaler = 512/BS
    loss = cross_entropy(out, Y, reduction='none', label_smoothing=hyp['opt']['label_smoothing']).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler'])
    state_updates = [u for layer in model.net if isinstance(layer, ConvGroup) for norm in (layer.norm1, layer.norm2) for u in getattr(norm, "updates", [])]

    if not getenv("DISABLE_BACKWARD"):
      # index 0 for bias and 1 for non-bias
      optimizer.zero_grad()
      loss.backward()
      state_updates += optimizer.schedule_step() + lr_scheduler[0].schedule_step() + lr_scheduler[1].schedule_step()
    return loss.realize(*state_updates)

  train_step_jitted = TinyJit(train_step)

  def eval_step(model, X, Y):
    out = model(X, training=False)
    loss = cross_entropy(out, Y, reduction='mean')
    correct = out.argmax(axis=1) == Y.argmax(axis=1)
    return correct.sum().realize(), loss.mul(Y.shape[0]).realize()
  eval_step_jitted     = TinyJit(eval_step) if getenv("JIT_EVAL", 1) else eval_step
  eval_step_ema_jitted = TinyJit(eval_step) if getenv("JIT_EVAL", 1) else eval_step

  step_times = []
  timing_rows = [] if artifact_path else None
  accuracy_rows = [] if artifact_path else None
  model_ema: Optional[modelEMA] = None
  projected_ema_decay_val = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']
  i = 0
  eval_acc_pct = 0.0
  eval_loss = 0.0
  timed_wall = 0.0
  eval_time = 0.0
  ema_epoch_start = math.floor(train_epochs) - hyp['ema']['epochs']
  cutmix_epoch_start = train_epochs - hyp['net']['cutmix_epochs']
  eval_steps = getenv("EVAL_STEPS", steps)

  def run_eval(step:int):
    nonlocal eval_time
    correct_sum = correct_len = 0
    loss_sum = 0.0
    correct_sum_ema = correct_len_ema = 0
    loss_sum_ema = 0.0
    use_jit_eval = getenv("JIT_EVAL", 1) and X_test.shape[0] == EVAL_BS
    eval_model = eval_step_jitted if use_jit_eval else eval_step
    eval_ema = eval_step_ema_jitted if use_jit_eval else eval_step
    eval_st = time.monotonic()
    for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, is_train=False):
      if len(GPUS) > 1:
        Xt.shard_(GPUS, axis=0)
        Yt.shard_(GPUS, axis=0)

      with Tensor.train(False): correct, loss = eval_model(model, Xt, Yt)
      correct_sum += int(correct.numpy().item())
      correct_len += Yt.shape[0]
      loss_sum += float(loss.numpy().item())
      if model_ema:
        with Tensor.train(False): correct_ema, loss_ema = eval_ema(model_ema.net_ema, Xt, Yt)
        correct_sum_ema += int(correct_ema.numpy().item())
        correct_len_ema += Yt.shape[0]
        loss_sum_ema += float(loss_ema.numpy().item())

    eval_time += time.monotonic() - eval_st
    acc, loss = correct_sum/correct_len*100.0, loss_sum/correct_len
    log(f"eval     {correct_sum}/{correct_len} {acc:.2f}%, {loss:7.2f} val_loss STEP={step}")
    if accuracy_rows is not None: accuracy_rows.append({"step": step, "split": "eval", "accuracy_pct": acc, "loss": loss})
    if not model_ema: return acc, loss

    acc_ema, loss_ema = correct_sum_ema/correct_len_ema*100.0, loss_sum_ema/correct_len_ema
    log(f"eval ema {correct_sum_ema}/{correct_len_ema} {acc_ema:.2f}%, {loss_ema:7.2f} val_loss STEP={step}")
    if accuracy_rows is not None: accuracy_rows.append({"step": step, "split": "eval_ema", "accuracy_pct": acc_ema, "loss": loss_ema})
    return acc_ema, loss_ema

  with Tensor.train():
    timed_st = time.monotonic()
    total_epochs = math.ceil(train_epochs) if steps == default_steps else math.ceil(steps / num_steps_per_epoch)
    for epoch in range(total_epochs):
      if i >= steps: break
      epoch_fraction = 1 if epoch + 1 < train_epochs else train_epochs % 1
      if steps != default_steps: epoch_fraction = 1
      do_cutmix = epoch >= cutmix_epoch_start
      batcher = fetch_batches(X_train, Y_train, BS=BS, is_train=True, epoch=epoch, epoch_fraction=epoch_fraction, do_cutmix=do_cutmix)
      for X, Y in batcher:
        if i >= steps: break
        GlobalCounters.reset()

        with WallTimeEvent(BenchEvent.STEP):
          st = time.monotonic()
          if len(GPUS) > 1:
            X.shard_(GPUS, axis=0)
            Y.shard_(GPUS, axis=0)

          with Context(BEAM=getenv("LATEBEAM", BEAM.value), WINO=getenv("LATEWINO", WINO.value)):
            loss = train_step_jitted(model, optim.OptimizerGroup(opt_bias, opt_non_bias), [lr_sched_bias, lr_sched_non_bias], X, Y)
          if epoch >= ema_epoch_start and (i+1) % hyp['ema']['every_n_steps'] == 0:
            if model_ema is None:
              model_ema = modelEMA(W, model)
            else:
              model_ema.update(model, Tensor([projected_ema_decay_val*((i+1)/steps)**hyp['ema']['decay_pow']]))

          if getenv("SYNC_STEPS", 0):
            for d in list(Device._opened_devices): Device[d].synchronize()
        cl = time.monotonic()
        step_times.append((cl-st)*1000.0)
        if timing_rows is not None:
          timing_rows.append({"step": i, "epoch": epoch, "wall_ms": (cl-st)*1000.0, "mem_gb": GlobalCounters.mem_used/1e9,
                              "global_ops": GlobalCounters.global_ops, "gflops": GlobalCounters.global_ops*1e-9/(cl-st)})
        if (log_steps:=getenv("LOG_STEPS", 0)) and (i % log_steps == 0 or i+1 == steps):
          loss_cpu = loss.numpy()
          device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
          log(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {device_str}, {loss_cpu:7.2f} loss, {opt_non_bias.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS, {GlobalCounters.global_ops*1e-9:9.2f} GOPS")
        i += 1

        if eval_steps and i % eval_steps == 0 and not getenv("DISABLE_BACKWARD"):
          eval_acc_pct, eval_loss = run_eval(i)

    for d in list(Device._opened_devices): Device[d].synchronize()
    timed_wall = time.monotonic() - timed_st
    if eval_steps and eval_acc_pct == 0.0 and not getenv("DISABLE_BACKWARD"):
      eval_acc_pct, eval_loss = run_eval(i)
    log(f"timed_wall {timed_wall:.3f}s, steps {i}, eval_acc_pct {eval_acc_pct:.2f}, eval_loss {eval_loss:.4f}")

  min_step_time = min(step_times) if step_times else None
  assert_time = getenv("ASSERT_MIN_STEP_TIME", 0.0)
  assert_wall = getenv("ASSERT_MAX_WALL_TIME", 0.0)
  target = getenv("TARGET_EVAL_ACC_PCT", 0.0)
  checks = {
    "min_step_time_ms": {"target": assert_time, "actual": min_step_time, "passed": (not assert_time) or (min_step_time is not None and min_step_time < assert_time)},
    "max_wall_time_s": {"target": assert_wall, "actual": timed_wall, "passed": (not assert_wall) or timed_wall < assert_wall},
    "target_eval_acc_pct": {"target": target, "actual": eval_acc_pct, "passed": (not target) or eval_acc_pct >= target},
  }

  if artifact_path:
    assert timing_rows is not None and accuracy_rows is not None
    with open(artifact_path/"timings.csv", "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=["step", "epoch", "wall_ms", "mem_gb", "global_ops", "gflops"])
      w.writeheader()
      w.writerows(timing_rows)
    with open(artifact_path/"accuracy.csv", "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=["step", "split", "accuracy_pct", "loss"])
      w.writeheader()
      w.writerows(accuracy_rows)
    (artifact_path/"env.txt").write_text("\n".join(f"{k}={v}" for k,v in artifact_env.items())+"\n")
    (artifact_path/"git.txt").write_text(_run(["git", "rev-parse", "HEAD"])+"\n"+_run(["git", "status", "--short"])+"\n")
    (artifact_path/"nvidia-smi.txt").write_text(_run(["nvidia-smi"])+"\n")
    gpu_model = Device.DEFAULT
    if "NV" in Device.DEFAULT or "CUDA" in Device.DEFAULT:
      gpu_model = (gpu_lines[0] if (gpu_lines:=_run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()) else Device.DEFAULT)
    run_phase = getenv("RUN_PHASE", "single")
    summary = {
      "git_commit": _run(["git", "rev-parse", "HEAD"]),
      "gpu_model": gpu_model,
      "command": " ".join(sys.argv),
      "env_vars": artifact_env,
      "final_accuracy": eval_acc_pct, "wall_time": timed_wall, "run_phase": run_phase,
      "cold_wall_time": timed_wall if run_phase == "cold" else None,
      "warm_wall_time": timed_wall if run_phase == "warm" else None,
      "train_time": timed_wall, "eval_time": eval_time, "best_step_time": min_step_time,
      "mean_step_time": sum(step_times)/len(step_times) if step_times else None,
      "kernel_search_settings": {"BEAM": getenv("BEAM", 0), "JITBEAM": getenv("JITBEAM", 0), "WINO": getenv("WINO", 0), "TC_OPT": getenv("TC_OPT", 0)},
      "checks": checks,
    }
    (artifact_path/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    _write_plot([r["wall_ms"] for r in timing_rows], [r["accuracy_pct"] for r in accuracy_rows])

  failures = []
  if assert_time and not checks["min_step_time_ms"]["passed"]:
    failures.append(f"Speed regression, expected min step time of < {assert_time} ms but took: {min_step_time} ms")
  if assert_wall and not checks["max_wall_time_s"]["passed"]:
    failures.append(f"Speed regression, expected wall time of < {assert_wall} s but took: {timed_wall:.3f} s")

  if target:
    if eval_acc_pct >= target:
      log(colored(f"{eval_acc_pct=} >= {target}", "green"))
    else:
      failures.append(colored(f"{eval_acc_pct=} < {target}", "red"))

  for failure in failures: log(failure)
  if artifact_log is not None: artifact_log.close()
  if failures: raise AssertionError("; ".join(failures))

if __name__ == "__main__":
  with WallTimeEvent(BenchEvent.FULL):
    train_cifar()
