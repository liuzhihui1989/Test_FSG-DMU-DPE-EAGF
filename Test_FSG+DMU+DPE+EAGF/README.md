# Attention Guided Filter and Multi-Level Pyramid Adaptive Feature Fusion U-Net for Retinal Vessel Segmentation

This repository contains the implementation associated with the research **"Attention Guided Filter and Multi-Level Pyramid Adaptive Feature Fusion U-Net Model for Retinal Vessel Segmentation."**

The proposed U-Net-based framework integrates **Dual Pyramid Extraction (DPE)**, **Edge Aware Attention Guided Filter (EAGF)**, and **Multi-branch Dynamic Merge Unit (DMU)** to improve fine-vessel representation, boundary preservation, and vascular continuity in retinal fundus images.

## Network Architecture

- **DPE:** Combines an input pyramid and a feature pyramid with Adaptive Convolution Selection (ACS) to extract multi-scale vascular features.
- **EAGF:** Introduces Triple Pooling Attention (TPA) and attention-guided filtering into skip connections to preserve vessel boundaries and suppress background interference.
- **DMU:** Learns pixel-wise spatial fusion weights for three decoder side outputs and combines them through Softmax-normalized adaptive fusion.

> To display the network diagram, add the corresponding figure to the repository and update its path here.

## Environment

The experiments reported in the paper used:

| Component | Configuration |
|---|---|
| OS | Ubuntu 20.04 |
| GPU | NVIDIA GeForce RTX 4090 (48 GB VRAM, as reported) |
| GPU driver | 535.230.02 |
| CPU | AMD EPYC 7542 32-Core (16 cores enabled) |
| RAM | 62.9 GB |
| Python | 3.8 |
| PyTorch | 2.4.1 |
| CUDA | 12.1 |
| cuDNN | 8.9.7 |

## Experimental Results

The following results are reported in the accompanying manuscript. **All values are percentages (%).**

| Dataset | IoU | F1-score | ACC | AUC | SEN | SPE |
|---|---:|---:|---:|---:|---:|---:|
| DRIVE | 71.28 | 83.23 | 97.69 | 98.89 | 84.26 | 98.69 |
| STARE | 74.54 | 85.41 | 97.80 | 98.69 | 86.74 | 98.79 |
| CHASE_DB1 | 68.10 | 81.02 | 97.52 | 99.38 | 86.00 | 98.26 |
| HRF | 68.88 | 81.57 | 97.11 | 98.74 | 83.62 | 98.23 |
| FIVES | 80.51 | 89.20 | 98.70 | 97.99 | 88.69 | 99.34 |

## Dataset Preparation

Five publicly available retinal vessel segmentation datasets were evaluated:

| Dataset | Training / testing protocol | Network input size |
|---|---|---|
| DRIVE | 20 / 20 | 608 × 608 |
| STARE | Leave-one-out cross-validation (19 / 1 per fold) | 704 × 704 |
| CHASE_DB1 | 20 / 8 | 1024 × 1024 |
| HRF | 30 / 15 | 1344 × 1344 |
| FIVES | 600 / 200 | 512 × 512 |

DRIVE, STARE, and CHASE_DB1 images were center-padded. HRF images were resized and center-padded, while FIVES images were resized. Training data augmentation included CutMix, cropping, flipping, color jittering, and perspective transformation. Augmentation was applied only to training images, expanding the training sets tenfold.

Place downloaded images and their corresponding vessel annotations under `dataset/`. Update dataset paths in the relevant configuration files. Ensure that image/annotation pairs match correctly. Follow the original datasets' usage and distribution terms.

## Training

The manuscript reports the following settings:

- Training from scratch (no pretrained weights)
- Epochs: **200**
- Batch size: **8**
- Optimizer: **Adam**
- Initial learning rate: **0.001**
- Weight decay: **0.05**
- Data loader workers: **6**
- Loss: Average cross-entropy across side classifiers

The repository includes `train.py`, `main.py`, and a `configs/` directory. Configure dataset paths and training parameters before running the training entry point. 

## Inference

The repository includes `inference.py`. 

## Project Structure

```text
Test_FSG-DMU-DPE-EAGF/
├── aug_steps/          # Data augmentation components
├── configs/            # Experiment configurations
├── dataset/            # Retinal image datasets (prepare locally)
├── models/             # Model implementations
├── visualization/      # Visualization utilities
├── README.md
└── ...                  # Additional scripts and files in the repository
```

## Evaluation

The manuscript reports six segmentation metrics: Intersection over Union (IoU), F1-score, Accuracy (ACC), Area Under the ROC Curve (AUC), Sensitivity (SEN), and Specificity (SPE).

## Citation

If you use this implementation in your research, please cite the associated manuscript once its publication details are available:

> Zhihui Liu et al., *Attention Guided Filter and Multi-Level Pyramid Adaptive Feature Fusion U-Net Model for Retinal Vessel Segmentation*. Publication details to be added.

## Notes

Pretrained weights and download links will be provided here if released. This code is intended for research use and is not validated for clinical diagnosis.
