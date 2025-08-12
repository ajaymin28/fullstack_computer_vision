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
  * [Evaluate & Test]
  * [Model/Experiment Tracking]
  * [API Development]
  * [Docker Deployment]
  * [Troubleshooting](#troubleshooting)
  * [Video Tutorials](#video-tutorial)

* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)

---

## Project Structure

```
fullstack_computer_vision/
├── data/
│   └── plant_disease_recognition    # Dataset
│       └── train
│           └── Healthy (class 0)
│           └── Powdery (class 1)
│           └── Rust    (class 2)
│       └── test
│       └── validation
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

### Setup

**Python** ≥ 3.10 and **PyTorch** with CUDA (optional).

```bash
# (Recommended) Create a fresh environment
conda create -n fcv python=3.10 -y
conda activate fcv

# minimal set:
pip install torch torchvision tqdm wandb opencv-python pytest

# Make repo importable when running modules
# Linux/macOS
export PYTHONPATH="$(pwd)"
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path
```

---

## Stage 1 — Image Classification (ResNet)

Build and train a baseline image classifier on a plant‑disease dataset (or any folder‑structured dataset). This stage focuses on clean data loading, sane defaults, and a reproducible training loop.

### Video Tutorial

To be added


### Data

- [Plant disease dataset (Kaggle)](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset/data)

Follow the video tutorial series to setup the data

### Train

From the repository root:

```bash
python -m training.train_plant_disease_resnet
```

* Artifacts: checkpoints and metrics are saved to the checkpoint directory defined in the config.


### Troubleshooting

* **ImportError: attempted relative import with no known parent package**
  Run modules from the repo root (as shown above) **or** set `PYTHONPATH` to the project root.
* **CUDA issues**
  Verify your PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`.

---

<!-- ## Roadmap

* **Stage 2 – Training Quality**
  * TorchMetrics, better LR schedulers, label smoothing, class‑imbalance handling
  * Deterministic training switches & rich evaluation reports

* **Stage 3 – Experiment Management**
  * Hydra/OMEGACONF configs, model registry, checkpoint/versioning

* **Stage 4 – API & Packaging**
  * FastAPI inference service with batch & streaming endpoints
  * Dockerfile + lightweight GPU/CPU images
  * Minimal CI (lint, tests) via GitHub Actions

> Have ideas or requests? Open an issue with a short design note. -->

---


## Stage 2 — Object Detection
TBA


## Stage 3 — Segmentation
TBA


## Stage 4 — Vision Language Models
TBA


## Contributing

Contributions are welcome! Please open an issue or PR. Keep examples minimal and well‑documented.