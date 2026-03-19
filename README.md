# SpineSegPlus

> **Note:** This project is based on [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) by [NeuroPoly Lab](https://neuro.polymtl.ca/). We have made significant architectural extensions on top of the original project, including a two-stage lumbar intervertebral disc degeneration (IVD degeneration) segmentation pipeline, a comprehensive clinical parameter computation module, and an end-to-end inference workflow. Please see the [Acknowledgments](#acknowledgments) section for the original citation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [Stage 1 — Coarse Segmentation & Landmarks (Dataset 101)](#stage-1--coarse-segmentation--landmarks-dataset-101)
  - [Stage 2 — Fine-Grained Labeling (Dataset 102)](#stage-2--fine-grained-labeling-dataset-102)
  - [Stage 3 — IVD Degeneration Two-Stage Detection & Segmentation (Dataset 105)](#stage-3--ivd-degeneration-two-stage-detection--segmentation-dataset-105)
  - [Clinical Parameter Computation](#clinical-parameter-computation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Inference](#inference)
  - [Spine Segmentation (Step 1 & Step 2)](#spine-segmentation-step-1--step-2)
  - [IVD Degeneration Inference (Two-Stage Pipeline)](#ivd-degeneration-inference-two-stage-pipeline)
  - [Clinical Report Generation](#clinical-report-generation)
- [Training](#training)
  - [Step 1 & Step 2 Training](#step-1--step-2-training)
  - [IVD Degeneration Two-Stage Training (Dataset 105)](#ivd-degeneration-two-stage-training-dataset-105)
- [Output Data Structure](#output-data-structure)
- [List of Classes](#list-of-classes)
- [Documentation](#documentation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

**SpineSegPlus** is a comprehensive tool for automatic analysis of spinal MRI images. Building upon the original [TotalSpineSeg](https://github.com/neuropoly/totalspineseg), this project extends the pipeline with:

1. **Full-spine instance segmentation** — Automatic segmentation and labeling of all vertebrae (C1–L5), intervertebral discs (IVDs), sacrum, spinal cord, and spinal canal, robust to various MRI contrasts, acquisition orientations, and resolutions.
2. **Two-stage IVD degeneration detection and segmentation** — A novel pipeline that detects and segments lumbar intervertebral disc degeneration using anatomical priors and surface-aware loss functions.
3. **Clinical measurement computation** — Automated calculation of clinically relevant spinal parameters (vertebral height, disc height, Cobb angles, lumbar lordosis, sacral slope, lumbosacral angle, disc inclination angle, IVD degeneration protrusion metrics, etc.) with visualization.

The model backbone is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Multi-stage Spine Segmentation** | Step 1 (9-class coarse segmentation + landmark detection) → Step 2 (11-class fine-grained labeling with odd/even vertebrae) |
| **IVD Degeneration Two-Stage Pipeline** | Stage A (disc-level binary detection using 3D ResNet with SE attention) → Stage B (ROI-based fine segmentation using lightweight 3D U-Net with SDM supervision) |
| **Anatomical Prior Injection** | Disc index maps, disc boundary attention, disc–canal interface attention, and inter-vertebral space attention |
| **Comprehensive Clinical Report** | JSON-based report with vertebral heights, disc heights, HDR/DHI, sagittal alignment angles (LL, SS, LSA, DIA), IVD degeneration parameters (PD, PA, PAR, PLR), and disc signal intensity |
| **Visualization** | Automatic generation of measurement preview images with annotated overlays |
| **Localizer Support** | Improved labeling for limited FOV images using localizer-based reference |
| **Multi-contrast Robustness** | Validated on T1w, T2w, STIR, MTS, T2*, and even CT images |

---

## Architecture

### Stage 1 — Coarse Segmentation & Landmarks (Dataset 101)

- **Input**: Single-channel MRI (resampled to 1mm isotropic, reoriented to LPI)
- **Output**: 9 classes — spinal cord, spinal canal, IVDs, vertebrae, and 5 landmark classes (C2–C3, C7–T1, T12–L1, L5–S key discs and C1 vertebra)
- **Post-processing**: Iterative labeling algorithm assigns anatomical indices; odd-numbered IVDs are extracted for Step 2

### Stage 2 — Fine-Grained Labeling (Dataset 102)

- **Input**: 2-channel — MRI image + odd IVD mask from Step 1
- **Output**: 11 classes — spinal cord, spinal canal, IVDs, odd/even vertebrae, sacrum, and 4 landmark discs
- **Post-processing**: Iterative labeling reconstructs C1–L5 + sacrum labels; canal filling ensures anatomical continuity

### Stage 3 — IVD Degeneration Two-Stage Detection & Segmentation (Dataset 105)

A novel two-stage pipeline designed specifically for the extreme class imbalance and tiny lesion volume of lumbar intervertebral disc degeneration:

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage A — Detection (Disc-level Binary Classification)          │
│  • Input: MRI patch (96³) + disc_mask + disc_index_map          │
│  • Model: StageADetectorV2 (Residual 3D CNN + SE attention)     │
│  • Loss: Focal Loss (γ=2)                                       │
│  • Output: has_IVD (0 or 1) per disc level                       │
└──────────────────────────────────────┬───────────────────────────┘
                                       ↓ (positive discs only)
┌──────────────────────────────────────────────────────────────────┐
│  Stage B — Fine Segmentation (ROI-based IVD Mask Prediction)     │
│  • Input: MRI ROI (48³) + disc_mask + disc_index_map            │
│  • Model: SmallUNet3D (3-level 3D U-Net, dual-head)             │
│  • Loss: FocalTversky + Boundary + L1(SDM)                      │
│  • Output: IVD degeneration binary mask + signed distance map (SDM) │
└──────────────────────────────────────────────────────────────────┘
```

**Key innovations:**
- **Disc Index Maps**: Normalized positional encoding (0–1) derived from Step 2 vertebra labels, providing spatial priors
- **Mandatory 4-Class Sampling**: Each disc generates 4 patch types (IVD degeneration center, IVD degeneration boundary, disc boundary negative, disc interior negative) to ensure balanced training
- **Surface-Aware Supervision**: Signed distance map (SDM) regression forces the network to learn boundary geometry
- **Anatomical Attention**: Multi-level attention maps (disc boundary, disc–canal interface, inter-vertebral space) constrain predictions to anatomically plausible regions

### Clinical Parameter Computation

The `calculate.py` module provides automated computation of clinically relevant spinal measurements:

| Category | Parameters |
|:---------|:-----------|
| **Vertebral Morphometry** | Anterior/posterior vertebral height (VH), vertebral body AP diameter |
| **Disc Morphometry** | Disc height (DH) at anterior/mid/posterior locations, height-to-disc ratio (HDR), disc height index (DHI) |
| **Sagittal Alignment** | Lumbar Lordosis (LL), Sacral Slope (SS), Lumbosacral Angle (LSA) |
| **Disc Angles** | Disc Inclination Angle (DIA) per level |
| **IVD Degeneration Parameters** | Protrusion Distance (PD), Protrusion Area (PA), PA Ratio (PAR), Protrusion-to-Length Ratio (PLR) |
| **Signal Analysis** | Average Gray Level (AGL) per disc |

All measurements are output as a structured JSON report with accompanying visualization images.

---

## Project Structure

```
SpineSegPlus/
├── totalspineseg/                  # Core package
│   ├── __init__.py                 # Package exports
│   ├── inference.py                # Main spine segmentation inference (Step 1 & Step 2)
│   ├── init_inference.py           # Model initialization & weight download
│   ├── ldh_twostage/               # ★ NEW: IVD degeneration two-stage pipeline module
│   │   ├── models.py               #   StageADetectorV2 + SmallUNet3D architectures
│   │   ├── losses.py               #   FocalTversky, Boundary, SDM losses
│   │   ├── sampling.py             #   Mandatory 4-class patch sampling
│   │   ├── disc_index.py           #   Disc index map generation from Step 2 labels
│   │   ├── distance_maps.py        #   Signed distance map computation
│   │   ├── data.py                 #   PyTorch Dataset classes
│   │   └── metrics.py              #   Evaluation metrics
│   ├── nnunet_extensions/          # nnU-Net trainer extensions
│   ├── resources/                  # Label maps & dataset configurations
│   │   ├── labels_maps/            #   tss_map.json, nnunet_step1/2/5_ldh.json, etc.
│   │   └── datasets/              #   Dataset configurations
│   └── utils/                      # Utility modules
│       ├── iterative_label.py      #   Iterative anatomical labeling algorithm
│       ├── predict_nnunet.py       #   nnU-Net prediction wrapper (with monkeypatch)
│       ├── extract_alternate.py    #   Odd/even disc extraction
│       ├── extract_levels.py       #   Disc level extraction
│       ├── fill_canal.py           #   Canal topology repair
│       ├── resample.py             #   Image resampling
│       └── ...                     #   Other utilities
│
├── calculate.py                    # ★ NEW: Clinical parameter computation & report generation
├── example_usage.py                # Usage examples for programmatic integration
│
├── scripts/                        # Training & inference scripts
│   ├── prepare_dataset_105.py      # ★ NEW: Dataset 105 (IVD degeneration) data preparation
│   ├── train_ldh_stage_a.py        # ★ NEW: IVD degeneration Stage A training
│   ├── train_ldh_stage_b.py        # ★ NEW: IVD degeneration Stage B training
│   ├── infer_ldh.py                # ★ NEW: End-to-end IVD degeneration inference pipeline
│   ├── eval_ldh.py                 # ★ NEW: IVD degeneration evaluation
│   ├── prepare_datasets.sh         #   Dataset 101/102/103 preparation
│   ├── download_datasets.sh        #   Dataset download
│   └── train.sh                    #   Unified training entry point
│
├── docs/                           # Documentation
│   ├── LDH_TwoStage_Pipeline.md   #   Detailed IVD degeneration pipeline documentation
│   ├── LDH_Quick_Reference_CN.md  #   IVD degeneration quick reference (Chinese)
│   ├── Step1_2_5_Technical_Report_bilingual.md  # Technical report
│   └── ...                         #   Other documentation
│
├── pyproject.toml                  # Package configuration
├── LICENSE                         # License file
└── README.md                       # This file
```

---

## Dependencies

- **Python** >= 3.10, with pip >= 23 and setuptools >= 67
- **PyTorch** < 2.6 (with CUDA support recommended)
- **nnU-Net v2** (backbone for Step 1 and Step 2 training/inference)
- Key libraries: `nibabel`, `SimpleITK`, `nilearn`, `scipy`, `torchio`, `gryds`, `tqdm`, `matplotlib`

---

## Installation

1. Create and activate a virtual environment:
   ```bash
   conda create -n tss python=3.10
   conda activate tss
   ```

2. Install SpineSegPlus:
   ```bash
   git clone <repository-url> SpineSegPlus
   cd SpineSegPlus
   python3 -m pip install -e .[nnunetv2]
   ```

3. Install PyTorch with CUDA support:
   ```bash
   python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
       --index-url https://download.pytorch.org/whl/cu118 --upgrade
   ```

4. (Optional) Set model data directory:
   ```bash
   export TOTALSPINESEG_DATA="/path/to/data"
   ```

---

## Inference

### Spine Segmentation (Step 1 & Step 2)

```bash
# Process a single NIfTI file or folder
totalspineseg INPUT OUTPUT_FOLDER [--step1] [--iso]

# Examples
totalspineseg input.nii.gz output_folder
totalspineseg input_folder output_folder --iso --device cuda

# With localizer for limited FOV images
totalspineseg localizers localizers_output --iso
totalspineseg images output --loc localizers_output/step2_output --suffix _T2w --loc-suffix _T1w
```

For full options, run `totalspineseg --help`.

### IVD Degeneration Inference (Two-Stage Pipeline)

```bash
python scripts/infer_ldh.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --device cuda
```

The IVD degeneration inference pipeline will:
1. Run Step 1 + Step 2 to obtain full anatomical segmentation
2. Generate disc index maps from Step 2 labels
3. For each lumbar disc level (L1/L2 to L5/S1):
   - **Stage A**: Detect whether IVD degeneration is present
   - **Stage B** (if positive): Segment the IVD degeneration region in a focused ROI
4. Aggregate all disc-level predictions into a final IVD degeneration mask

### Clinical Report Generation

```bash
python calculate.py \
    --input-dir /path/to/processed_cases \
    --output-dir /path/to/reports
```

This generates:
- **JSON report** with all clinical measurements (vertebral height, disc height, Cobb angles, IVD degeneration parameters, etc.)
- **Visualization images** with annotated measurement overlays for each parameter

---

## Training

### Step 1 & Step 2 Training

**Hardware requirements:**
- ~3.5 TB disk space (with data augmentation)
- ≥32 GB RAM
- CUDA GPU with ≥8 GB VRAM

```bash
# Set environment variables
export TOTALSPINESEG="$(realpath .)"
export TOTALSPINESEG_DATA="/path/to/data"

# Download datasets
bash scripts/download_datasets.sh

# Prepare nnUNet datasets
bash scripts/prepare_datasets.sh [101|102|103|all] [-noaug]

# Train
bash scripts/train.sh [DATASET_ID [FOLD]]
```

### IVD Degeneration Two-Stage Training (Dataset 105)

```bash
# 1. Prepare Dataset 105 (generates Stage A patches + Stage B ROIs)
python scripts/prepare_dataset_105.py \
    --stagea-patch 96 \
    --stageb-roi 48 \
    --device cuda

# 2. Train (one command runs both stages + evaluation)
bash scripts/train.sh 105 0

# Or run stages individually:
python scripts/train_ldh_stage_a.py --epochs 50 --batch-size 32 --lr 1e-3
python scripts/train_ldh_stage_b.py --epochs 200 --batch-size 16 --lr 1e-4

# 3. Evaluate
python scripts/eval_ldh.py \
    --ckpt-dir /path/to/checkpoints \
    --device cuda
```

**Training time estimates (single V100/A100):**
| Stage | Epochs | Time |
|:------|:-------|:-----|
| Stage A (Detection) | 50 | ~2–4 hours |
| Stage B (Segmentation) | 200 | ~8–12 hours |

---

## Output Data Structure

### Spine Segmentation Output
```
output_folder/
├── input/              # Preprocessed input images (1mm iso, LPI)
├── preview/            # Preview images (JPEG)
├── step1_raw/          # Step 1 raw model output
├── step1_output/       # Step 1 iterative labeling result
├── step1_cord/         # Spinal cord soft segmentation
├── step1_canal/        # Spinal canal soft segmentation
├── step1_levels/       # Single-voxel disc level markers in canal centerline
├── step2_raw/          # Step 2 raw model output
└── step2_output/       # Final labeled segmentation (vertebrae, discs, cord, canal)
```

### Clinical Report Output
```
report_output/
├── result/
│   ├── report.json                     # Complete JSON measurement report
│   └── previews/                       # Visualization images
│       ├── vertebrae/                  #   Vertebral height, width measurements
│       ├── discs/                      #   Disc height, DIA measurements
│       └── global/                     #   LL, SS, LSA angle visualizations
└── raw/                                # Intermediate computation data
```

---

## List of Classes

> The mapping is also available in `totalspineseg/resources/labels_maps/tss_map.json`

| Label | Name |
|:------|:-----|
| 1 | spinal_cord |
| 2 | spinal_canal |
| 11 | vertebrae_C1 |
| 12 | vertebrae_C2 |
| 13 | vertebrae_C3 |
| 14 | vertebrae_C4 |
| 15 | vertebrae_C5 |
| 16 | vertebrae_C6 |
| 17 | vertebrae_C7 |
| 21 | vertebrae_T1 |
| 22 | vertebrae_T2 |
| 23 | vertebrae_T3 |
| 24 | vertebrae_T4 |
| 25 | vertebrae_T5 |
| 26 | vertebrae_T6 |
| 27 | vertebrae_T7 |
| 28 | vertebrae_T8 |
| 29 | vertebrae_T9 |
| 30 | vertebrae_T10 |
| 31 | vertebrae_T11 |
| 32 | vertebrae_T12 |
| 41 | vertebrae_L1 |
| 42 | vertebrae_L2 |
| 43 | vertebrae_L3 |
| 44 | vertebrae_L4 |
| 45 | vertebrae_L5 |
| 50 | sacrum |
| 63 | disc_C2_C3 |
| 64 | disc_C3_C4 |
| 65 | disc_C4_C5 |
| 66 | disc_C5_C6 |
| 67 | disc_C6_C7 |
| 71 | disc_C7_T1 |
| 72 | disc_T1_T2 |
| 73 | disc_T2_T3 |
| 74 | disc_T3_T4 |
| 75 | disc_T4_T5 |
| 76 | disc_T5_T6 |
| 77 | disc_T6_T7 |
| 78 | disc_T7_T8 |
| 79 | disc_T8_T9 |
| 80 | disc_T9_T10 |
| 81 | disc_T10_T11 |
| 82 | disc_T11_T12 |
| 91 | disc_T12_L1 |
| 92 | disc_L1_L2 |
| 93 | disc_L2_L3 |
| 94 | disc_L3_L4 |
| 95 | disc_L4_L5 |
| 100 | disc_L5_S |

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[LDH_TwoStage_Pipeline.md](docs/LDH_TwoStage_Pipeline.md)** — Comprehensive IVD degeneration pipeline architecture, implementation details, and usage guide
- **[LDH_Quick_Reference_CN.md](docs/LDH_Quick_Reference_CN.md)** — Quick reference for IVD degeneration pipeline (Chinese)
- **[Step1_2_5_Technical_Report_bilingual.md](docs/Step1_2_5_Technical_Report_bilingual.md)** — Bilingual technical report for Step 1, Step 2, and Step 5

---

## Acknowledgments

This project is based on and extends the work of [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) by the [NeuroPoly Lab](https://neuro.polymtl.ca/) at Polytechnique Montréal. Please cite the original work if you use this project:

```bibtex
@article{warszawer2025totalspineseg,
   title={TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI},
   author={Warszawer, Yehuda and Molinier, Nathan and Valosek, Jan and Benveniste, Pierre-Louis and Bédard, Sandrine and Shirbint, Emanuel and Mohamed, Feroze and Tsagkas, Charidimos and Kolind, Shannon and Lynd, Larry and Oh, Jiwon and Prat, Alexandre and Tam, Roger and Traboulsee, Anthony and Patten, Scott and Lee, Lisa Eunyoung and Achiron, Anat and Cohen-Adad, Julien},
   year={2025},
   journal={ResearchGate preprint},
   url={https://www.researchgate.net/publication/389881289_TotalSpineSeg_Robust_Spine_Segmentation_with_Landmark-Based_Labeling_in_MRI}
}
```

Please also cite nnU-Net, as the segmentation backbone is heavily based on it:

```bibtex
@article{isensee2021nnunet,
   title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
   author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
   journal={Nature methods},
   volume={18},
   number={2},
   pages={203--211},
   year={2021}
}
```

---

## License

See the [LICENSE](LICENSE) file for details.
