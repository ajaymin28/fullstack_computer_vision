# fullstack\_computer\_vision

**End‑to‑end, production‑minded computer vision pipelines — from data to trained models to deployable APIs.**

This repository hosts small, focused examples that you can compose into a full development cycle. Each stage is intentionally minimal, easy to read, and ready to extend.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Stage 1 — Image Classification (ResNet)](#stage-1--image-classification-resnet)

  * [Setup](#setup)
  * [Data](#data)
  * [Train](#train)
  * [Evaluate & Test](#evaluate--test)
  * [Configuration](#configuration)
  * [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)

---

## Project Structure

```
fullstack_computer_vision/
├── config/
│   └── resnet_train_cfg.py          # Training hyperparameters & paths
├── dataset_loader/
│   └── plant_disease.py             # Torch Dataset for plant disease images
├── training/
│   └── train_plant_disease_resnet.py# ResNet training script
├── tests/
│   └── test_plant_ds.py             # Dataset/unit tests
└── README.md
```

---

## Stage 1 — Image Classification (ResNet)

Build and train a baseline image classifier on a plant‑disease dataset (or any folder‑structured dataset). This stage focuses on clean data loading, sane defaults, and a reproducible training loop.

### Setup

**Python** ≥ 3.10 and **PyTorch** with CUDA (optional).

```bash
# (Recommended) Create a fresh environment
conda create -n fcv python=3.10 -y
conda activate fcv

# Install dependencies
pip install -r requirements.txt  # if present
# Or minimal set:
pip install torch torchvision tqdm wandb opencv-python pytest

# Make repo importable when running modules
# Linux/macOS
export PYTHONPATH="$(pwd)"
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path
```

### Data

Point the config to your dataset root. Expected layout (class‑per‑folder):

```
<DATA_ROOT>/
  ├─ class_a/
  │    ├─ img_001.jpg
  │    └─ ...
  ├─ class_b/
  │    ├─ img_042.jpg
  │    └─ ...
  └─ ...
```

If you use PlantVillage or a similar dataset, ensure train/val splits are configured in `resnet_train_cfg.py`.

### Train

From the repository root:

```bash
python -m training.train_plant_disease_resnet
```

* Logs: stdout/TQDM, optionally to [Weights & Biases](https://wandb.ai/) if enabled in the config (`WANDB_PROJECT` env var recommended).
* Artifacts: checkpoints and metrics are saved to the output directory defined in the config.

### Evaluate & Test

Run the dataset/unit tests:

```bash
pytest -q tests/test_plant_ds.py
```

(Extend with evaluation scripts as you add them; a validation loop is included in the training script.)

### Configuration

`config/resnet_train_cfg.py` centralizes paths and hyperparameters. A typical configuration includes fields like:

```python
# Example shape — adapt to your actual config structure
DATA_ROOT = "data/plant_disease"
OUTPUT_DIR = "runs/resnet"

MODEL = {
    "name": "resnet50",
    "pretrained": True,
    "num_classes": 38,  # set to your dataset
}

TRAINING = {
    "epochs": 30,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "mixed_precision": True,  # amp autocast
    "seed": 42,
}

AUGMENTATION = {
    "img_size": 224,
    "random_resized_crop": True,
    "hflip_prob": 0.5,
    # normalization uses ImageNet mean/std in code
}

LOGGING = {
    "use_wandb": True,
    "project": "fullstack_cv",
}
```

> **Tip:** Keep your config the single source of truth. CLI flags can override config values later (see Roadmap).

### Troubleshooting

* **ImportError: attempted relative import with no known parent package**
  Run modules from the repo root (as shown above) **or** set `PYTHONPATH` to the project root.
* **CUDA issues**
  Verify your PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`.
* **Non‑determinism**
  Fix seeds in your config and avoid stochastic augmentations during validation.

---

## Roadmap

* **Stage 2 – Training Quality**

  * TorchMetrics, better LR schedulers, label smoothing, class‑imbalance handling
  * Deterministic training switches & rich evaluation reports
* **Stage 3 – Experiment Management**

  * Hydra/OMEGACONF configs, model registry, checkpoint/versioning
* **Stage 4 – API & Packaging**

  * FastAPI inference service with batch & streaming endpoints
  * Dockerfile + lightweight GPU/CPU images
  * Minimal CI (lint, tests) via GitHub Actions

> Have ideas or requests? Open an issue with a short design note.

---

## Contributing

Contributions are welcome! Please open an issue or PR. Keep examples minimal and well‑documented.