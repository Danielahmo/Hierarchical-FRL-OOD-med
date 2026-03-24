# Hierarchical Frequency Regularization for OOD Detection in Medical Images

> **ISBI 2026** | Hierarchical frequency regularization for out-of-distribution detection in generative models for medical images

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the official implementation of **Hierarchical Frequency Regularization Learning (FRL)** for out-of-distribution (OOD) detection in generative models applied to medical imaging. The method incorporates both **high** and **low frequency** components into a VAE–DCGAN architecture, enabling robust OOD detection across 2D histological images and 3D volumetric medical scans (MRI, CT). Article published on ISBI, 2026.

![model_architecture](https://github.com/Danielahmo/Hierarchical-FRL-OOD-med/blob/main/images_readme/Model_architecture.png)

### Key Contributions

- **Hierarchical frequency regularization**: Systematically integrates high-frequency, low-frequency, and combined frequency representations into a generative OOD detection framework.
- **3D adaptation**: Extends the VAE–DCGAN architecture to 3D medical volumes (MRI, CT) using MSE reconstruction loss for computational efficiency.
- **Robustness evaluation**: Demonstrated on histological (MIDOG++) and volumetric (MOOD Challenge) datasets, including robustness to noise, motion, elastic deformation, ghosting, and intensity shifts.
- **State-of-the-art results**: FRL (high+low) achieves **AUC = 0.989** on MIDOG++ and competitive performance on 3D medical data.

---

## Method

Given an input image $x$, the model computes low- and high-frequency components via Gaussian filtering:

$$x_L = K_\sigma * x, \quad x_H = \text{rgb2gray}(x) - \text{rgb2gray}(x_L)$$

The enriched input $x_F = [x,\, x_L,\, x_H]$ is passed through a VAE–DCGAN encoder-decoder. OOD scoring uses the **Negative Log-Likelihood (NLL)** estimated via importance sampling, and an optional **frequency-based score** $S_F$:

$$S_F(\mathbf{x}) = -\log p_\theta(\mathbf{x}_F) - L(\mathbf{x})$$

where $L(\mathbf{x})$ is image complexity estimated through lossless compression.

### Architecture

```
Input image x
    │
    ├─── Gaussian blur ──► x_L (low frequency)
    └─── Subtraction   ──► x_H (high frequency)
    
x_F = [x, x_L, x_H]  ──► Encoder ──► z ~ N(μ, σ²) ──► Decoder ──► Reconstruction
                                                                         │
                                                               NLL / S_F OOD score
```

---

## Results

| Model | cs-ID | Near OOD | Far OOD | SNR 30 | SNR 20 | SNR 10 | Gamma | Motion | Average |
|---|---|---|---|---|---|---|---|---|---|
| No FRL | 0.650 | 0.627 | 0.742 | 0.525 | 0.627 | 0.904 | 0.556 | 0.557 | 0.649 |
| FRL (high) | 0.877 | 0.892 | 1.000 | 1.000 | 1.000 | 1.000 | 0.546 | 0.899 | 0.902 |
| FRL (low) | 0.889 | 0.883 | 1.000 | 0.822 | 1.000 | 1.000 | 0.562 | 0.876 | 0.879 |
| FRL (high+low) | 0.861 | 0.893 | 1.000 | 1.000 | 1.000 | 1.000 | **0.997** | **1.000** | **0.969** |
| **FRL (HiLo)** | **0.899** | 0.884 | 1.000 | 0.647 | 0.998 | 1.000 | 0.553 | 0.873 | 0.857 |

*AUC performance on MIDOG++ dataset. Each model trained with 5 seeds.*

---

## Repository Structure

```
.
├── DCGAN_VAE_freq.py               # 2D VAE–DCGAN encoder/decoder
├── VAE3D.py                        # 3D VAE–DCGAN encoder/decoder
├── train_VAE_freq_2gauss.py        # Training script for 2D (MIDOG)
├── train_3DVAE_freq_2gauss_adjustB.py  # Training script for 3D (MOOD)
├── OOD_scores.py                   # 2D OOD evaluation (NLL, S_F)
├── OOD_3D_scores.py                # 3D OOD evaluation
├── utils_2D.py                     # 2D frequency utilities
├── utils.py                        # 3D frequency utilities
├── train_2D.sh                     # Shell script: train 2D model
├── train_3D.sh                     # Shell script: train 3D model
├── run_OOD_scores.sh               # Shell script: run 2D OOD evaluation
├── run_3D_OOD_scores.sh            # Shell script: run 3D OOD evaluation
└── requirements.txt                # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/Danielahmo/Hierarchical-FRL-OOD-med.git
cd Hierarchical-FRL-OOD-med
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
pip install -r requirements.txt

```

**Requirements:** `numpy`, `pandas`, `pillow`, `tqdm`, `torchio`, `opencv-python`

---

## Datasets

### MIDOG++ (2D histology)
The [MIDOG++](https://midog2022.grand-challenge.org/) dataset contains histological whole-slide images of mitotic cells across 7 tumor types. Preprocessing (50×50 px crops) and OOD splits follow [OpeMIBOOD](https://github.com/remic-othr/OpenMIBOOD).
For details of dowloadinf and preprocessing consult their repository.

### MOOD Challenge (3D MRI/CT)
The [MOOD Challenge](https://www.synapse.org/Synapse:syn21343101/wiki/599515) dataset includes:
- **Brain MRI**: 800 scans, 256×256×256 voxels
- **Abdomen CT**: 550 scans, 512×512×512 voxels

Scans are preprocessed to 64×64×64 voxels with Z-normalization. For preprocessing the data run:

```bash
python preprocessing_MRI_CT_scans.py --input_dir --output_dir
```

Input_dir if the path with the MRI or CT scans and output_dir the path where the preprocessed volumens are going to be saved.
---

## Training

### 2D (Histology – MIDOG++)

```bash
bash train_2D.sh
```

Or directly:
```bash
python train_VAE_freq_2gauss.py \
    --experiment ./experiments/seed_0 \
    --data_path /path/to/images \
    --train_txt /path/to/train_list.txt \
    --num_epoch 300 \
    --seed_val 0 \
    --batchSize 32
```

### 3D (MRI/CT – MOOD)

```bash
bash train_3D.sh
```

Or directly:
```bash
python train_3DVAE_freq_2gauss_adjustB.py \
    --experiment ./experiments/seed_0 \
    --data_path /path/to/volumes \
    --train_txt /path/to/train_list.txt \
    --num_epoch 10 \
    --seed_val 0 \
    --batchSize 16
```

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--imageSize` | 64 | Input spatial resolution |
| `--nz` | 100 | Latent dimension |
| `--ngf` | 32 | Number of feature maps |
| `--gauss_size` | 5 | Gaussian kernel size for frequency decomposition |
| `--lr` | 3e-4 | Learning rate |

---

## OOD Evaluation

### 2D Evaluation

Computes NLL and $S_F$ scores across ID, cs-ID, near-OOD, far-OOD, SNR levels, gamma, and motion corruptions.

```bash
bash run_OOD_scores.sh
```

Or directly:
```bash
python OOD_scores.py \
    --state_dict /path/to/checkpoint.pth \
    --data_path /path/to/images \
    --txt_path /path/to/lists \
    --save_path ./results/seed_0 \
    --repeat 20
```

### 3D Evaluation

Evaluates robustness under elastic deformation, ghosting, noise, and swap artifacts (via [TorchIO](https://torchio.readthedocs.io/)).

```bash
bash run_3D_OOD_scores.sh
```

Or directly:
```bash
python OOD_3D_scores.py \
    --state_dict /path/to/checkpoint.pth \
    --data_path /path/to/volumes \
    --test_path /path/to/val_list.txt \
    --save_path ./results/seed_0 \
    --repeat 10
```

Results are saved as CSV files per OOD type (e.g., `in_scores.csv`, `elastic_scores.csv`).

---

## Frequency Decomposition

Two utility modules handle frequency decomposition:

**2D** (`utils_2D.py`): Gaussian blur applied to grayscale image; high-frequency extracted by subtraction.

**3D** (`utils.py`): Separable 3D Gaussian kernel applied to volumetric data.

```python
from utils import process_freq

x_high, x_low = process_freq(x, gauss_size=5)
x_input = torch.cat([x, x_high, x_low], dim=1)
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{hierarchical_frl_2026,
  title     = {Hierarchical frequency regularization for out-of-distribution detection in generative models for medical images},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026}
}
```

This work builds on:
```bibtex
@inproceedings{FRL,
  title  = {Frequency-Regularized Likelihood for OOD Detection},
  author = {Cai, Mu et al.},
  year   = {2022}
}
```

---

## Acknowledgements

This code is based on the [FRL repository](https://github.com/mu-cai/FRL). We thank the organizers of the [MIDOG++ Challenge](https://midog2022.grand-challenge.org/) and the [MOOD Challenge](http://medicalood.dkfz.de/) for making their datasets publicly available.
