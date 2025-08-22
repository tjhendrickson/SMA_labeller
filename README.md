# SMALabeller

> **Slice-wise U‑Net training for 3D MRI with MONAI + PyTorch**
>
> 2D training over 3D volumes, AMP, OneCycleLR, early stopping, BatchNorm calibration, TensorBoard logging, and robust checkpointing. Optional Ray Tune reporting.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Repository Layout](#repository-layout)
* [Installation](#installation)
* [Data Expectations](#data-expectations)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
* [TensorBoard](#tensorboard)
* [Checkpoints](#checkpoints)
* [BatchNorm Calibration (Why/How)](#batchnorm-calibration-whyhow)
* [Ray Tune Integration (Optional)](#ray-tune-integration-optional)
* [API Reference](#api-reference)
* [Troubleshooting](#troubleshooting)
* [Performance Tips](#performance-tips)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Overview

`SMALabellerApp` is a training harness for U‑Net models on 3D MRI data. It converts 3D volumes shaped `[B, C, H, W, D]` into 2D slices `[B×D, C, H, W]` for memory‑efficient training while preserving a MONAI‑friendly dataset pipeline.

The app supports:

* **2D U‑Net** (slice‑wise over 3D volumes)
* (Scaffolded) **3D U‑Net**

It includes conveniences such as **mixed precision (AMP)**, **OneCycleLR**, **TensorBoard logging** (scalars, histograms, sample images), **gradient‑norm logging**, and **stateful checkpointing** (best/last/periodic) with **early stopping** and **atomic saves**.

---

## Features

* **Slice‑wise training from 3D volumes**: memory‑friendly reshaping `[B, C, H, W, D] → [B×D, C, H, W]`.
* **AMP** via `torch.cuda.amp.GradScaler` and autocast for fast & stable training.
* **OneCycleLR** scheduler with AdamW optimizer.
* **Early Stopping** with automatic **best‑model restore**.
* **BatchNorm calibration** every epoch using forward‑only micro‑batches to refresh running stats.
* **TensorBoard**: batch/epoch scalars, parameter/gradient histograms, and **validation image triplets** (img/label/pred).
* **Robust checkpointing**: `best.pth`, `last.pth`, and `periodic_####.pth` with atomic writes and retention policy.
* **Reproducible split**: `sklearn.model_selection.train_test_split(..., random_state=42)`.
* **Plug‑and‑play losses**: Dice, BCEWithLogits, or Combo (Dice + CE).
* **Optional Ray Tune** metric reporting.

---

## Repository Layout

```
.
├── app.py                      # Entrypoint containing SMALabellerApp (example name)
├── model/
│   └── model.py                # UNet2D, UNet3D, Custom3DMRIDatasetMONAI, losses, EarlyStopping
├── utils/
│   └── utils.py                # enumerateWithEstimate, benchmark_loss_step
├── data/
│   ├── imagesTr_resampled/     # training images (*.nii.gz with _0000 suffix)
│   └── labelsTr_resampled/     # training labels (*.nii.gz)
├── requirements.txt
└── README.md
```

> **Note:** File names are illustrative—match your actual script/module names.

---

## Installation

### 1) Create an environment

```bash
conda create -n smalabeller python=3.10 -y
conda activate smalabeller
```

### 2) Install dependencies

Install **PyTorch** matching your CUDA/OS from [https://pytorch.org](https://pytorch.org) . Then:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` could be:

```text
monai>=1.3
scikit-learn>=1.2
tensorboard>=2.12
nibabel>=5.1
# Optional (for HPO/metrics reporting)
ray[tune]>=2.7
```

> PyTorch is intentionally omitted—please install per your platform/CUDA.

---

## Data Expectations

The app builds pairs of image/label paths by scanning the **labels** directory and looking for a matching image file in the **images** directory.

**Naming convention**

* **Labels**: `labelsTr_resampled/<base>.nii.gz`
* **Images**: `imagesTr_resampled/<base>_0000.nii.gz`

**Assumptions**

* Volumes are single‑channel (`C=1`) NIfTI (`.nii.gz`).
* Dataset returns MONAI MetaTensors with keys `image` and `label`, each shaped `[B, C, H, W, D]` after collation.
* Variable depth `D` is supported via `pad_list_data_collate`.

Example tree:

```
data/
├── imagesTr_resampled/
│   ├── subj01_0000.nii.gz
│   ├── subj02_0000.nii.gz
│   └── ...
└── labelsTr_resampled/
    ├── subj01.nii.gz
    ├── subj02.nii.gz
    └── ...
```

---

## Quickstart

### CLI (recommended)

```bash
python app.py \
  --unet_dimensions 2D \
  --num_workers 8 \
  --batch_size 2 \
  --epochs 100 \
  --training_images /path/to/imagesTr_resampled \
  --training_labels /path/to/labelsTr_resampled \
  --tb_prefix SMALabeller \
  --val_set_size 0.2 \
  cconelea
```

This runs 2D U‑Net training with AMP, OneCycleLR, BN calibration, early stopping, TensorBoard logging, and checkpointing. Logs land under `~/runs/<tb_prefix>/<timestamp>-{trn,val}_cls-<comment>`.

### Programmatic use

```python
from app import SMALabellerApp

app = SMALabellerApp(args={
    "unet_dimensions": "2D",
    "num_workers": 8,
    "batch_size": 2,
    "epochs": 100,
    "training_images": "/path/to/imagesTr_resampled",
    "training_labels": "/path/to/labelsTr_resampled",
    "tb_prefix": "SMALabeller",
    "val_set_size": 0.2,
    "comment": "exp01"
})

config = {
    "lr": 2.3173e-5,
    "max_lr": 5.1206e-3,
    "loss_function": "cross_entropy",   # one of: dice | cross_entropy | combination
    "augmentations": True,
    "dropout_rate": 0.45012              # logged; your model may consume this
}

app.do2dTraining(config, ray_tune=False)
```

---

## Configuration

Training behavior is controlled by both **CLI arguments** and a **`config` dict**.

### CLI arguments

| Arg                 | Type  | Default       | Description                                        |
| ------------------- | ----- | ------------- | -------------------------------------------------- |
| `--unet_dimensions` | str   | `3D`          | `2D` or `3D` (2D is the actively implemented path) |
| `--num_workers`     | int   | `8`           | DataLoader workers                                 |
| `--batch_size`      | int   | `2`           | Batch size per step                                |
| `--epochs`          | int   | `1`           | Training epochs                                    |
| `--training_images` | str   | path          | Directory of `*_0000.nii.gz` images                |
| `--training_labels` | str   | path          | Directory of label NIfTIs                          |
| `--tb_prefix`       | str   | `SMALabeller` | Root under `~/runs/` for TensorBoard               |
| `--val_set_size`    | float | `0.2`         | Fraction of data for validation                    |
| `comment`           | str   | `cconelea`    | Suffix for TB logdir names                         |

### Config dictionary

| Key             | Type  | Example         | Notes                                                     |
| --------------- | ----- | --------------- | --------------------------------------------------------- |
| `lr`            | float | `2.3e-5`        | AdamW base LR                                             |
| `max_lr`        | float | `5.1e-3`        | OneCycleLR peak LR                                        |
| `loss_function` | str   | `cross_entropy` | One of: `dice`, `cross_entropy`, `combination`            |
| `augmentations` | bool  | `True`          | Toggles dataset augmentations (dataset‑specific)          |
| `dropout_rate`  | float | `0.45`          | Logged to TB; consumption depends on model implementation |

---

## TensorBoard

Logs are written under:

```
~/runs/<tb_prefix>/<YYYY-mm-dd_HH.MM.SS>-trn_cls-<comment>
~/runs/<tb_prefix>/<YYYY-mm-dd_HH.MM.SS>-val_cls-<comment>
```

Start TensorBoard:

```bash
tensorboard --logdir ~/runs/SMALabeller
```

What you’ll see:

* **Scalars**: training/validation loss (batch & epoch), LR, iteration time, best/bad epochs, etc.
* **Histograms**: parameters & gradients per epoch.
* **Images** (validation only): small panels of `img`, `lbl`, `pred` slices near the volume midpoint.

---

## Checkpoints

Checkpoints live in:

```
~/runs/<tb_prefix>/<timestamp>/checkpoints/
```

Files:

* `best.pth` – model with lowest validation loss so far.
* `last.pth` – state after the most recent epoch.
* `periodic_####.pth` – retained every `K_PERIOD` epochs (default: 5), newest 3 kept.

Each checkpoint stores:

* `epoch`, `global_step`, `monitor_value` (validation loss)
* `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`
* `config`

Saves use an **atomic write** (`.tmp` → rename) to avoid partial files.

At the end of training, the **best model** is automatically **restored** via the `EarlyStopping` helper and announced in logs.

---

## BatchNorm Calibration (Why/How)

Small or highly augmented batches can make BN running stats noisy. This app:

1. **Resets** BN running stats each epoch before validation.
2. Runs **forward‑only** passes (`calibrate_bn`) on a handful of **training** batches to refresh stats (kept in FP32 for stability).
3. Switches to `eval()` for validation and **disables tracking** on BN to “freeze” stats.

Calibration uses a micro‑batching helper (`forward_slices_only`) that processes 2D slices in chunks to avoid OOM. Tune `slice_batch` and `warmup_batches` as needed.

---

## Ray Tune Integration (Optional)

If you call `do2dTraining(config, ray_tune=True)`, the app reports metrics using `ray.train.report` with keys:

* `training_loss`
* `validation_loss`
* `epoch`

**Example Tuner snippet** (sketch):

```python
from ray import tune
from app import SMALabellerApp

def trainable(config):
    app = SMALabellerApp(args={
        "unet_dimensions": "2D",
        "num_workers": 8,
        "batch_size": 2,
        "epochs": 40,
        "training_images": "/data/imagesTr_resampled",
        "training_labels": "/data/labelsTr_resampled",
        "tb_prefix": "SMALabeller",
        "val_set_size": 0.2,
        "comment": "ray"
    })
    app.do2dTraining(config, ray_tune=True)

search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "max_lr": tune.loguniform(1e-3, 1e-2),
    "loss_function": tune.choice(["dice", "cross_entropy", "combination"]),
    "augmentations": tune.choice([True, False]),
    "dropout_rate": tune.uniform(0.0, 0.6),
}

tuner = tune.Tuner(trainable, param_space=search_space)
results = tuner.fit()
```

---

## API Reference

### `class SMALabellerApp`

**Constructor**

* Accepts either CLI args (when run as a script) or a Python `dict`/`list` via `args`.

**Key methods**

* `do2dTraining(config, ray_tune=False)`: full training loop for 2D U‑Net.
* `compute2dBatchLoss(...)`: computes batch loss; supports `dice`, `cross_entropy`, `combination`.
* `init2dModel() / init3dModel()`: instantiate U‑Net models.
* `initOptimizer(config)`: AdamW + OneCycleLR.
* `initTensorboardWriters(hparams)`: creates TB writers and logs hyperparams.
* `build_data_dicts()`: scans `training_images`/`training_labels` into MONAI‑style dicts.
* `calibrate_bn(forward_fn, warmup_batches)`: refresh BN stats with forward‑only passes.
* `save_checkpoint(path, epoch, global_step, monitor_value, config)`: atomic checkpoint save.
* `cleanup_old(pattern, keep)`: rotate older periodic checkpoints.

**Utilities**

* `_tb_log_val_images(...)`: logs `(img, lbl, pred)` slices for quick sanity checks.
* `_grad_global_norm()`: gradient norm logging for stability monitoring.
* `_current_lr()`: grab current LR from optimizer.

---

## Troubleshooting

**NaNs detected in outputs or labels**

* The loop prints warnings if NaNs are found. Check data normalization, label range (0/1), and loss scale. Consider reducing `max_lr`.

**Out‑of‑memory errors**

* Lower `batch_size`, reduce `slice_batch` in BN calibration, or crop volumes. Keep AMP enabled. Ensure `num_workers` fits RAM.

**TensorBoard graph logging failed**

* The graph trace uses a dummy `torch.randn(1, 1, 128, 128)`. If your model expects a different spatial size/channels, this is only a log‑time warning and training still proceeds.

**"Inconsistent batched metadata" with MetaTensors**

* The MONAI collate can pad metadata across samples. Access tensors via `.as_tensor()` where needed (the code already handles this in key spots).

**No images in TB**

* Validation images log only for the **first** batch of each epoch and only in the **2D** path. Ensure `val_writer` exists and dataset keys are `image`/`label`.

**3D training path**

* The 3D loop is scaffolded but commented. Implement/enable as needed.

---

## Performance Tips

* **OneCycleLR**: choose `max_lr` 10–50× higher than base `lr` as a starting point.
* **num\_workers**: 4–12 is typical; benchmark your storage.
* **pin\_memory=True** and `.to(device, non_blocking=True)` already enabled.
* **Grad‑norm**: spikes may indicate instability—lower LR or check augmentations.
* **Warmup BN**: `warmup_batches=4–16` is common; keep calibration in FP32 for stable stats.

---

## License

Add a `LICENSE` file. Example: [MIT License](https://opensource.org/license/mit/).

---

## Acknowledgments

* Built with **PyTorch** and **MONAI**.
* Optional experiment tracking with **Ray Tune**.
