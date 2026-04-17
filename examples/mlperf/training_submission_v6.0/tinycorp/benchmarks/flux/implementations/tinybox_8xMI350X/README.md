# 1. Problem

This benchmark uses Flux for MLPerf Training v6.0 text-to-image.

## Requirements

Install tinygrad with the MLPerf extras and the Hugging Face datasets loader used by the preprocessed Flux dataloader.
```
git clone https://github.com/tinygrad/tinygrad.git
python3 -m pip install -e ".[mlperf]"
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

## Offline train to eval flow

`dev_run.sh` keeps Flux training in offline-eval mode by default:
```
CKPT=1
EVAL_INTERVAL=0
```

It derives `TRAIN_STEPS` from `TOTAL_CKPTS * ceil(512000 / BS)` unless `TRAIN_STEPS` is set explicitly, writes `flux_step<step>.safetensors` into `SAVE_CKPT_DIR`, then runs `eval_flux` over `EVAL_CKPT_DIR` with `BS` overridden to `EVAL_BS`.

Default checkpoint location:
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

Override `BS`, `EVAL_BS`, `TOTAL_CKPTS`, `TRAIN_STEPS`, `SAVE_CKPT_DIR`, `EVAL_CKPT_DIR`, `TRAIN_DATASET`, or `VAL_DATASET` as needed for the host and dataset layout.
