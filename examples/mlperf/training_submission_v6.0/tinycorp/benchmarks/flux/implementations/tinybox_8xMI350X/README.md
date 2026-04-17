# 1. Problem

This benchmark uses Flux for MLPerf Training v6.0 text-to-image.

## Requirements

From the repository root, install tinygrad, MLPerf logging, and the Hugging Face datasets loader used by the preprocessed Flux dataloader.
```
python3 -m pip install -e .
git clone https://github.com/mlcommons/logging.git mlperf-logging
python3 -m pip install -e mlperf-logging
python3 -m pip install datasets numpy tqdm
```

# 2. Directions

## Dataset layout

The submission wrapper expects preprocessed datasets on disk and points at them with:
```
DATADIR=/raid/datasets/flux
TRAIN_DATASET=${DATADIR}/cc12m_preprocessed/*
VAL_DATASET=${DATADIR}/coco_preprocessed/*
```

Each resolved dataset path must be readable by `datasets.load_from_disk(...)`. Training samples must include `t5_encodings`, `clip_encodings`, `mean`, and `logvar`. Validation samples must additionally include `timestep`.

## Benchmark flow

`run_and_time.sh` performs two phases:
1. `INITMLPERF=1` on `FAKEDATA=1` with a tiny Flux shape:
```
TRAIN_STEPS=1
EVAL_INTERVAL=1
EVAL_STEPS=1
CKPT=0
FLUX_HIDDEN_SIZE=128
FLUX_ATTENTION_HEADS=1
FLUX_DOUBLE_STREAM_BLOCKS=0
FLUX_SINGLE_STREAM_BLOCKS=0
FLUX_MLP_RATIO=1.0
FLUX_T5_TOKENS=8
```
   This warms beam search and emits init logging without loading the submission-sized model.
2. `RUNMLPERF=1` in offline-eval mode. `dev_run.sh` defaults to:
```
GPUS=8
BS=128
EVAL_BS=128
LR=1e-4
WARMUP_STEPS=1000
MAX_NORM=1.0
CKPT=1
EVAL_INTERVAL=0
STOP_IF_CONVERGED=1
TOTAL_CKPTS=7
DEFAULT_FLOAT=bfloat16
```

It derives `TRAIN_STEPS` from `TOTAL_CKPTS * ceil(512000 / BS)` unless `TRAIN_STEPS` is set explicitly, writes `flux_step<step>.safetensors` into `SAVE_CKPT_DIR`, then runs `eval_flux` over `EVAL_CKPT_DIR` with `BS` overridden to `EVAL_BS`. Checkpoints are written under:
```
/raid/weights/flux/training_checkpoints/${RUN_NAME}
```

## Running

### tinybox_8xMI350X

#### Steps to run benchmark
```
examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/flux/implementations/tinybox_8xMI350X/run_and_time.sh
```

#### Direct development run
```
BS=128 TOTAL_CKPTS=7 examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/flux/implementations/tinybox_8xMI350X/dev_run.sh
```

For `dev_run.sh`, override `BS`, `EVAL_BS`, `TOTAL_CKPTS`, `TRAIN_STEPS`, `SAVE_CKPT_DIR`, `EVAL_CKPT_DIR`, `TRAIN_DATASET`, or `VAL_DATASET` as needed for the host and dataset layout. The `FLUX_*` model-shape overrides in `run_and_time.sh` are init-only and are unset before the real benchmark run.
For the submission wrapper, override only dataset/checkpoint locations, `PRETRAINED`, `SEED`, `RUN_NAME`, or `LOGFILE`. Use `dev_run.sh` for experiments that intentionally change benchmark batch size, checkpoint cadence, or debug model shape.
