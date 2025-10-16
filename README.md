# C3Net: Context-Contrast Network for Camouflaged Object Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **C3Net: Context-Contrast Network for Camouflaged Object Detection**
> [Baber Jan](mailto:baberjan008@gmail.com)<sup>1,2</sup>, [Aiman H. El-Maleh](mailto:aimane@kfupm.edu.sa)<sup>1</sup>, [Abdul Jabbar Siddiqui](mailto:abduljabbar.siddiqui@kfupm.edu.sa)<sup>1</sup>, [Abdul Bais](mailto:Abdul.Bais@uregina.ca)<sup>3</sup>, [Saeed Anwar](mailto:saeed.anwar@uwa.edu.au)<sup>4</sup>
> <sup>1</sup>King Fahd University of Petroleum and Minerals
> <sup>2</sup>SDAIA-KFUPM Joint Research Center for Artificial Intelligence
> <sup>3</sup>University of Regina
> <sup>4</sup>University of Western Australia
>
> _Submitted to IEEE Transactions on Artificial Intelligence (TAI)_

---

## Overview

Camouflaged objects blend with surroundings through similar colors, textures, and patterns. Detection is challenging because traditional methods and foundation models fail on these objects. We identify six challenges: **Intrinsic Similarity (IS)**, **Edge Disruption (ED)**, **Extreme Scale Variation (ESV)**, **Environmental Complexities (EC)**, **Contextual Dependencies (CD)**, and **Salient-Camouflaged Object Disambiguation (SCOD)**. C3Net uses dual-pathway architecture to address all challenges. The **Edge Refinement Pathway (ERP)** processes early features with gradient-initialized modules for boundaries. The **Contextual Localization Pathway (CLP)** processes deep features with **Image-based Context Guidance (ICG)** for intrinsic saliency suppression without external models. The **Attentive Fusion Module (AFM)** combines both pathways through spatial gating. C3Net achieves **0.898** S-measure on COD10K, **0.904** on CAMO, and **0.913** on NC4K while maintaining efficient real-time processing.

### Key Contributions

- Dual-pathway decoder separating edge refinement from contextual localization prevents signal dilution
- Image-based Context Guidance (ICG) for intrinsic saliency suppression without external models
- Gradient-initialized Edge Enhancement Modules preserving classical edge detection principles while enabling learning
- Attentive Fusion Module for synergistic pathway integration through spatial attention gating

---

## Architecture

<p align="center">
  <img src="assets/architecture.svg" width="100%">
  <br>
  <em>C3Net architecture with dual-pathway decoder: Edge Refinement Pathway (ERP) processes early features for boundaries, Contextual Localization Pathway (CLP) with ICG mechanism handles semantic understanding, and Attentive Fusion Module (AFM) combines both pathways</em>
</p>

C3Net employs a dual-pathway decoder with three main components:

### 1. Edge Refinement Pathway (ERP)

Processes early encoder features (Stages 1 & 2) to recover precise object boundaries through cascaded Edge Enhancement Modules (EEMs). Each EEM employs multi-path convolutions initialized from Sobel and Laplacian operators, maintaining classical edge detection principles while adapting to camouflage patterns through learning.

### 2. Contextual Localization Pathway (CLP)

Processes deep encoder features (Stages 23 & 24) through Semantic Enhancement Units (SEUs) and our novel Image-based Context Guidance (ICG) mechanism. The ICG performs:

- **Appearance Analysis**: Direct extraction from input image
- **Guided Contrast Module (GCM)**: Foreground-background differentiation using spatial aggregation and global context
- **Iterative Attention Gating**: Progressive saliency suppression through two-stage refinement

### 3. Attentive Fusion Module (AFM)

Synergistically combines ERP and CLP outputs through spatial attention gating, where contextual features spatially modulate edge information, emphasizing relevant boundaries while suppressing distractors.

---

## Performance

| Dataset    | S<sub>α</sub> ↑ | F<sub>β</sub><sup>w</sup> ↑ | F<sub>β</sub><sup>m</sup> ↑ | E<sub>φ</sub> ↑ | MAE ↓  |
| ---------- | --------------- | --------------------------- | --------------------------- | --------------- | ------ |
| **COD10K** | **0.898**       | **0.851**                   | **0.859**                   | **0.961**       | **0.0162** |
| **CAMO**   | **0.904**       | **0.889**                   | **0.896**                   | **0.951**       | **0.0311** |
| **NC4K**   | **0.913**       | **0.895**                   | **0.903**                   | **0.958**       | **0.0220** |

C3Net achieves state-of-the-art on COD10K while outperforming previous best by 2.6% in S<sub>α</sub> and 2.3% in E<sub>φ</sub>. The model leads on CAMO with best S<sub>α</sub> of 0.904 and MAE of 0.0311, and maintains top performance on NC4K with 0.913 S<sub>α</sub>. Training uses 200 epochs with batch size 128 on NVIDIA H100 GPUs with efficient real-time inference on 392×392 resolution.

<p align="center">
  <img src="assets/qualitative_comparison.svg" width="100%">
  <br>
  <em>Visual comparison of C3Net with state-of-the-art methods on challenging COD cases. Each row exemplifies a specific challenge: (i) Intrinsic Similarity (IS), (ii) Edge Disruption (ED), (iii) Contextual Dependencies (CD), (iv) Multiple Instances, (v) Environmental Complexities (EC), (vi) Small Objects (ESV aspect), (vii) Large Objects (ESV aspect), and (viii) Salient-Camouflaged Object Disambiguation (SCOD). For each row, columns are: (a) Input Image, (b) OCENet, (c) BGNet, (d) ZoomNet, (e) SINetV2, (f) FSPNet, (g) FEDER, (h) C3Net (Ours), and (i) Ground Truth.</em>
</p>

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Baber-Jan/C3Net.git
cd C3Net
```

2. Run the setup script which handles environment creation and dataset validation:

```bash
chmod +x setup/setup.sh
./setup/setup.sh
```

The setup script performs:

- Creates conda environment with required dependencies
- Validates dataset organization if present
- Generates edge maps for training datasets if needed

Alternatively, manually create the environment:

```bash
conda env create -f setup/environment.yml
conda activate c3net
```

---

## Project Structure

```
C3Net/
├── configs/
│   └── default.yaml              # Training configuration
├── checkpoints/                  # Model weights
│   └── model_best.pth            # Pre-trained C3Net model
├── datasets/                     # Dataset directory
│   ├── COD10K/
│   ├── CAMO/
│   └── NC4K/
├── models/
│   ├── c3net.py                  # Main C3Net architecture
│   ├── backbone.py               # DINOv2 encoder
│   ├── edge_branch.py            # Edge Refinement Pathway (ERP)
│   ├── localization_branch.py   # Contextual Localization Pathway (CLP)
│   ├── fusion_head.py            # Attentive Fusion Module (AFM)
│   └── utils.py                  # Model utilities
├── utils/
│   ├── data_loader.py            # Dataset loading
│   ├── distributed_data_loader.py # Multi-GPU data loading
│   ├── distributed_utils.py      # Distributed training utilities
│   ├── edge_generator.py         # Edge map generation
│   ├── image_processor.py        # Image preprocessing
│   ├── loss_functions.py         # Multi-scale loss functions
│   ├── metrics.py                # Evaluation metrics
│   ├── run_manager.py            # Result directory management
│   └── visualization.py          # Result visualization
├── engine/
│   ├── trainer.py                # Training engine
│   ├── evaluator.py              # Evaluation engine
│   ├── predictor.py              # Prediction engine
│   └── distributed_trainer.py   # Multi-GPU support
├── setup/
│   ├── environment.yml           # Conda environment
│   └── setup.sh                  # Setup script
├── slurm_scripts/
│   └── train_multi_gpu.slurm     # SLURM cluster training script
└── main.py                       # Entry point
```

---

## Dataset Preparation

Download our COD datasets package containing CAMO, COD10K, and NC4K:

- **[Download Datasets (Google Drive)](https://drive.google.com/file/d/1GsEmE8820oOnSa_efqAzRX_8wL3Ha8ll/view?usp=sharing)** (~2.19GB)

After downloading, extract the datasets to the `datasets/` folder in your C3Net root directory:

```bash
unzip datasets.zip -d C3Net/
```

Expected directory structure:

```
C3Net/
├── configs/
├── datasets/                    # Extract datasets here
│   ├── CAMO/
│   │   ├── train/
│   │   │   ├── Imgs/           # Training images
│   │   │   ├── GT/             # Ground truth masks
│   │   │   └── Edges/          # Edge maps (auto-generated)
│   │   └── test/
│   │       ├── Imgs/
│   │       └── GT/
│   ├── COD10K/
│   │   ├── train/
│   │   │   ├── Imgs/
│   │   │   ├── GT/
│   │   │   └── Edges/
│   │   └── test/
│   │       ├── Imgs/
│   │       └── GT/
│   └── NC4K/
│       └── test/
│           ├── Imgs/
│           └── GT/
├── checkpoints/
└── ...
```

---

## Pre-trained Models

- **C3Net Model Checkpoint**: [Download (Google Drive)](https://drive.google.com/file/d/1dWZB3CfRKaCl5ZuXvsf8BOkvOfcyxIRb/view?usp=sharing)
  - Trained C3Net model for evaluation/inference
  - Place in: `checkpoints/model_best.pth`

- **Test Predictions**

  Following are segmentation masks generated by C3Net (reported in the paper) for each test dataset:

  - **CAMO** (250 predictions): [Download (Google Drive)](https://drive.google.com/file/d/1BvX--v49rIyZ7O7SbLpQPkoWR1y3bQe5/view?usp=sharing)
  - **COD10K** (2,026 predictions): [Download (Google Drive)](https://drive.google.com/file/d/1cPHmrsoCNdZhjMaLay3-384f8-EFWAwj/view?usp=sharing)
  - **NC4K** (4,121 predictions): [Download (Google Drive)](https://drive.google.com/file/d/1aLfeI2nKbiG3ZIv2VGFAfgTMNPEBJqFn/view?usp=sharing)

---

## Usage

### Configuration

All training configurations are specified in `configs/default.yaml` - review and modify hyperparameters, loss weights, and model settings as needed.

### Training

Train C3Net on COD10K + CAMO:

```bash
python main.py train --config configs/default.yaml
```

Training features:

- Automatic mixed precision (AMP) for efficient GPU utilization
- Multi-scale supervision with pathway-specific objectives
- Regular checkpointing and validation
- Early stopping with configurable patience

### Evaluation

Evaluate on test datasets:

```bash
python main.py evaluate --model checkpoints/model_best.pth
```

Evaluation provides:

- Standard COD metrics (S<sub>α</sub>, F<sub>β</sub><sup>w</sup>, F<sub>β</sub><sup>m</sup>, E<sub>φ</sub>, MAE)
- Quality-based result categorization
- Per-dataset performance analysis

### Prediction

For inference on new images:

```bash
python main.py predict \
    --model checkpoints/model_best.pth \
    --input path/to/image
```

The prediction outputs include:

- Binary segmentation mask
- Confidence heatmap
- Edge map
- Original image overlay

### Distributed Training (Multi-GPU)

For multi-GPU training using PyTorch DDP:

```bash
# Single-node multi-GPU (e.g., 8 GPUs)
python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29500 \
    engine/distributed_trainer.py \
    --config configs/default.yaml
```

**SLURM Cluster:**

```bash
sbatch slurm_scripts/train_multi_gpu.slurm
```

Customize the SLURM script for your cluster (partition name, GPU type, paths).

---

## Hardware Requirements

C3Net has been tested on:

- NVIDIA H100 GPU (optimal setup)
- CUDA 11.8 with PyTorch 2.1.0
- Recommended: 16GB+ GPU memory for training

The model supports various GPU configurations through adjustable batch sizes and memory optimization settings.

---

## Citation

If you find C3Net useful in your research, please cite:

```bibtex
@article{jan2025c3net,
  title={C3Net: Context-Contrast Network for Camouflaged Object Detection},
  author={Jan, Baber and El-Maleh, Aiman H. and Siddiqui, Abdul Jabbar and Bais, Abdul and Anwar, Saeed},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2025},
  note={Submitted to IEEE Transactions on Artificial Intelligence}
}
```

---

## Acknowledgments

This research was conducted at the SDAIA-KFUPM Joint Research Center for Artificial Intelligence, King Fahd University of Petroleum and Minerals. We gratefully acknowledge:

- King Fahd University of Petroleum and Minerals (KFUPM) for institutional support
- SDAIA-KFUPM Joint Research Center for Artificial Intelligence (JRCAI) for computational resources
- [DINOv2](https://arxiv.org/abs/2304.07193) for robust vision transformer features
- [DySample](https://arxiv.org/abs/2308.15085) for content-adaptive upsampling
- [PySODMetrics](https://github.com/lartpang/PySODMetrics) for evaluation framework
- Authors of COD benchmark datasets (COD10K, CAMO, NC4K)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

### Usage Terms

- Open source for research and non-commercial use
- Commercial use requires explicit permission
- Attribution required when using or adapting the code

The benchmark datasets (COD10K, CAMO, NC4K) maintain their original licenses. Please refer to their respective papers for terms of use.

---

## Contact

For questions or issues:

- **GitHub Issues**: [Create an issue](https://github.com/Baber-Jan/C3Net/issues)
- **Email**: baberjan008@gmail.com
- **Research Collaborations**: Contact Baber Jan at baberjan008@gmail.com

---

**Note**: This repository contains the official implementation of C3Net submitted to IEEE Transactions on Artificial Intelligence (TAI).
