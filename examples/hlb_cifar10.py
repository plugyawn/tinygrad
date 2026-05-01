#!/usr/bin/env python3

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import math, random, time
import numpy as np
from typing import Optional
from extra.lr_scheduler import OneCycleLR
from tinygrad import nn, dtypes, Tensor, Device, GlobalCounters, TinyJit, Variable
from tinygrad.nn.state import get_state_dict
from tinygrad.nn import optim
from tinygrad.helpers import Context, BEAM, WINO, getenv, colored, prod
from extra.bench_log import BenchEvent, WallTimeEvent

BS = getenv("BS", 1024)
EVAL_BS = getenv("EVAL_BS", 2500)
GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1))]
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
        self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(self.running_mean.dtype)).realize()
        batch_var_adjust = prod(y.shape[1:])/(prod(y.shape[1:])-y.shape[2])
        self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * batch_var_adjust * batch_var.detach().cast(self.running_var.dtype)).realize()
        self.num_batches_tracked.assign(self.num_batches_tracked + 1).realize()
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
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to compute with
    # TODO: remove the pad but instead let the kernel optimize itself
    return x.conv2d(self.whitening).gelu().pad((1,0,0,1)).sequential(self.net)

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

  def set_seed(seed):
    Tensor.manual_seed(seed)
    random.seed(seed)

  # ========== Model ==========
  def whitening(X, kernel_size=hyp['net']['kernel_size']):
    X = X[:getenv("WHITEN_EXAMPLES", hyp['net']['whitening_examples'])]

    def _cov(X):
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
      return np.flip(Λ, 0), np.flip(V.T.reshape(c*h*w, c, h, w), 0)

    # NOTE: np.linalg.eigh only supports float32 so the whitening layer weights need to be converted to float16 manually
    Λ, V = _eigens(_patches(X.float().numpy()))
    W = V/np.sqrt(Λ+1e-2)[:,None,None,None]
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
    if getenv("LOG_EPOCHS", 1): print(f"shuffling {'training' if is_train else 'test'} dataset in {(et-st)*1e3:.2f} ms ({epoch=})")

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
      for net_ema_param, (param_name, net_param) in zip(get_state_dict(self.net_ema).values(), get_state_dict(net).items()):
        if dtypes.is_float(net_param.dtype):
          ema = net_ema_param.detach()*decay + net_param.detach()*(1.-decay)
          net_ema_param.assign(ema).realize()
          if not (("norm" in param_name and "weight" in param_name) or "whitening" in param_name):
            net_param.assign(ema.detach()).realize()

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

    if not getenv("DISABLE_BACKWARD"):
      # index 0 for bias and 1 for non-bias
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler[0].step()
      lr_scheduler[1].step()
    return loss.realize()

  train_step_jitted = TinyJit(train_step)

  def eval_step(model, X, Y):
    out = model(X, training=False)
    loss = cross_entropy(out, Y, reduction='mean')
    correct = out.argmax(axis=1) == Y.argmax(axis=1)
    return correct.realize(), loss.realize()
  eval_step_jitted     = eval_step
  eval_step_ema_jitted = eval_step

  step_times = []
  model_ema: Optional[modelEMA] = None
  projected_ema_decay_val = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']
  i = 0
  eval_acc_pct = 0.0
  eval_loss = 0.0
  timed_wall = 0.0
  ema_epoch_start = math.floor(train_epochs) - hyp['ema']['epochs']
  cutmix_epoch_start = train_epochs - hyp['net']['cutmix_epochs']
  eval_steps = getenv("EVAL_STEPS", steps)

  def run_eval(step:int):
    corrects, losses = [], []
    corrects_ema, losses_ema = [], []
    for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, is_train=False):
      if len(GPUS) > 1:
        Xt.shard_(GPUS, axis=0)
        Yt.shard_(GPUS, axis=0)

      with Tensor.train(False): correct, loss = eval_step_jitted(model, Xt, Yt)
      losses.append(loss.numpy().tolist())
      corrects.extend(correct.numpy().tolist())
      if model_ema:
        with Tensor.train(False): correct_ema, loss_ema = eval_step_ema_jitted(model_ema.net_ema, Xt, Yt)
        losses_ema.append(loss_ema.numpy().tolist())
        corrects_ema.extend(correct_ema.numpy().tolist())

    correct_sum, correct_len = sum(corrects), len(corrects)
    acc, loss = correct_sum/correct_len*100.0, sum(losses)/len(losses)
    print(f"eval     {correct_sum}/{correct_len} {acc:.2f}%, {loss:7.2f} val_loss STEP={step}")
    if not model_ema: return acc, loss

    correct_sum_ema, correct_len_ema = sum(corrects_ema), len(corrects_ema)
    acc_ema, loss_ema = correct_sum_ema/correct_len_ema*100.0, sum(losses_ema)/len(losses_ema)
    print(f"eval ema {correct_sum_ema}/{correct_len_ema} {acc_ema:.2f}%, {loss_ema:7.2f} val_loss STEP={step}")
    return acc_ema, loss_ema

  with Tensor.train():
    timed_st = time.monotonic()
    for epoch in range(math.ceil(train_epochs)):
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
            model_ema.update(model, Tensor([projected_ema_decay_val*((i+1)/steps)**hyp['ema']['decay_pow']]))

        cl = time.monotonic()
        step_times.append((cl-st)*1000.0)
        if getenv("LOG_STEPS", 50) and (i % getenv("LOG_STEPS", 50) == 0 or i+1 == steps):
          loss_cpu = loss.numpy()
          device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
          print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {device_str}, {loss_cpu:7.2f} loss, {opt_non_bias.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS, {GlobalCounters.global_ops*1e-9:9.2f} GOPS")
        i += 1

        if eval_steps and i % eval_steps == 0 and not getenv("DISABLE_BACKWARD"):
          eval_acc_pct, eval_loss = run_eval(i)

    if eval_acc_pct == 0.0 and not getenv("DISABLE_BACKWARD"):
      eval_acc_pct, eval_loss = run_eval(i)
    timed_wall = time.monotonic() - timed_st
    print(f"timed_wall {timed_wall:.3f}s, steps {i}, eval_acc_pct {eval_acc_pct:.2f}, eval_loss {eval_loss:.4f}")

  if (assert_time:=getenv("ASSERT_MIN_STEP_TIME")):
    min_time = min(step_times)
    assert min_time < assert_time, f"Speed regression, expected min step time of < {assert_time} ms but took: {min_time} ms"

  if (assert_wall:=getenv("ASSERT_MAX_WALL_TIME", 0.0)):
    assert timed_wall < assert_wall, f"Speed regression, expected wall time of < {assert_wall} s but took: {timed_wall:.3f} s"

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if eval_acc_pct >= target:
      print(colored(f"{eval_acc_pct=} >= {target}", "green"))
    else:
      raise ValueError(colored(f"{eval_acc_pct=} < {target}", "red"))

if __name__ == "__main__":
  with WallTimeEvent(BenchEvent.FULL):
    train_cifar()
