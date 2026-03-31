# Contrastive Multi-Modal Learning for Early Alzheimer's Risk Detection

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


# Project Overview

This project implements a **GPU-accelerated multi-modal deep learning pipeline** for early Alzheimer's disease (AD) risk detection using the [ADNI (Alzheimer's Disease Neuroimaging Initiative)](https://adni.loni.usc.edu/) dataset.

## What it does

- Fuses **structural MRI brain scans** (3D volumes) with **clinical tabular data** (cognitive scores, biomarkers, demographics) into a unified representation using **NT-Xent contrastive learning** (SimCLR-style).
- Classifies subjects into diagnostic groups: **CN** (Cognitively Normal), **EMCI** (Early MCI), **LMCI** (Late MCI), and **AD** (Alzheimer's Disease).
- Produces an **ensemble of three models** — Fusion Model, XGBoost, and Shallow NN — for robust prediction.
- Generates **SHAP feature importance** plots and **Grad-CAM MRI heatmaps** for clinical interpretability.

# Architecture

```
MRI NIfTI (.nii/.nii.gz)         Clinical CSV Data
       │                                 │
  [MRI Preprocessing]            [Feature Engineering]
  N4 Bias Correction             KNN Imputation + Scaling
  Skull Stripping                       │
  MNI152 Registration                   │
  GPU Resize → 64×64×64                 │
       │                                 │
  [3D CNN Encoder]               [MLP + Self-Attention]
  4 Conv Blocks → GAP            256→128→Attention→128
  → L2 Projection (128-d)        → L2 Projection (128-d)
       │                                 │
       └──────────┬──────────────────────┘
                  │
        [Cross-Modal Attention]
         NT-Xent Contrastive Loss
                  │
          [Classifier Head]
          256 → 128 → N_CLASSES
                  │
         ┌────────┼────────┐
    [XGBoost]  [SNN]  [Fusion Model]
         └────────┼────────┘
              [Ensemble]
                  │
           Final Prediction
           + SHAP + Grad-CAM
```   

## Key Features

| Feature | Detail |
|---|---|
| **Hardware Target** | NVIDIA RTX 4060 (8GB VRAM) — works on any CUDA GPU |
| **Mixed Precision** | FP16 via PyTorch AMP — ~2× speedup |
| **MRI Preprocessing** | N4 Bias Correction → Skull Strip → MNI152 Registration → GPU Resize |
| **Contrastive Learning** | NT-Xent loss (SimCLR) for cross-modal alignment |
| **Class Imbalance** | SMOTE oversampling + weighted cross-entropy loss |
| **Explainability** | SHAP (tabular) + Grad-CAM (MRI heatmaps) |
| **Missing Modality** | Learned replacement tokens for MRI-only or tabular-only inference |

# Project Structure

```
ML/
├── v1/
│   ├── alzheimer_multimodal_GPU.ipynb   ← Main notebook (30 cells)
│   ├── models/                          ← Saved PyTorch & XGBoost models
│   │   ├── fusion_best.pt
│   │   ├── fusion_model_final.pt
│   │   ├── cnn_encoder_final.pt
│   │   ├── tab_encoder_final.pt
│   │   ├── shallow_nn_final.pt
│   │   ├── xgboost_embedding.json
│   │   ├── xgboost_tabular.json
│   │   ├── scaler.pkl
│   │   ├── imputer.pkl
│   │   └── label_encoder.pkl
│   ├── outputs/
│   │   ├── per_subject_predictions.csv
│   │   └── metrics_summary.json
│   ├── plots/
│   │   ├── training_curves.png
│   │   ├── evaluation_plots.png
│   │   ├── shap_importance.png
│   │   ├── shap_beeswarm.png
│   │   └── gradcam_mri.png
│   └── cache/
│       └── mri_volumes_64x64x64.npz    ← Preprocessed MRI cache
│
└── Dataset/
    ├── ADNIMERGE_09Mar2026.csv
    ├── DXSUM_09Mar2026.csv
    ├── FAQ_09Mar2026.csv
    ├── MEDHIST_09Mar2026.csv
    ├── MMSE_09Mar2026.csv
    ├── NPI_09Mar2026.csv
    ├── VITALS_09Mar2026.csv
    ├── MRI_metadata_with_VISCODE.csv
    └── ADNI_T1_Baseline_MRI/            ← NIfTI MRI files
        └── ADNI/
            └── <PTID>/
                └── .../<scan>.nii.gz
```

# Setup Instructions

## 1. Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.10 |
| CUDA Toolkit | 11.8 or 12.x |
| GPU VRAM | 6GB+ (8GB recommended) |
| RAM | 16GB+ |
| Disk Space | ~20GB (dataset + cache) |

> **Note:** The pipeline also runs on CPU but MRI preprocessing will be very slow (~10× slower).

## 2. Clone / Download

```bash
# If using Git
git clone <your-repo-url>
cd ML/v1

# Or simply place the notebook in:
C:\Users\<YourName>\Documents\ML\v1\
```

## 3. Install Dependencies

Run **Cell 1** in the notebook, or install manually:

```bash
# PyTorch with CUDA 11.8 (RTX 4060 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Neuroimaging
pip install nibabel nilearn SimpleITK

# Machine Learning
pip install scikit-learn xgboost imbalanced-learn

# Explainability
pip install shap grad-cam

# Utilities
pip install pandas numpy scipy matplotlib seaborn tqdm openpyxl

# Optional (best N4 bias correction — skip if install fails)
pip install antspyx
```

## 4. Configure Paths

Open the notebook and update **Cell 3** with your local paths:

```python
BASE_DIR  = Path(r'C:\Users\<YourName>\Documents\ML\v1')
DATA_DIR  = Path(r'C:\Users\<YourName>\Documents\ML\Dataset')
MRI_DIR   = DATA_DIR / 'ADNI_T1_Baseline_MRI'

CSV_FILES = {
    'merge'   : r'...\ADNIMERGE_09Mar2026.csv',
    'dx'      : r'...\DXSUM_09Mar2026.csv',
    # ... (update all 8 paths)
}
```
## 5. Dataset Access

Data is from [ADNI](https://adni.loni.usc.edu/) — requires free registration:

1. Register at https://adni.loni.usc.edu/
2. Apply for data access (approved within 1–2 days)
3. Download from **ADNI Data → Search & Download**:
   - **CSV tables:** ADNIMERGE, DXSUM, FAQ, MEDHIST, MMSE, NPI, VITALS, MRI metadata
   - **MRI scans:** T1-weighted baseline scans → convert to NIfTI format (`.nii.gz`)

> **MRI Format:** The pipeline expects NIfTI files (`.nii` or `.nii.gz`). If you have DICOM files, convert them first using [dcm2niix](https://github.com/rordenlab/dcm2niix).

## How to Run the Code

## Quick Start (if MRI cache already exists)

If `cache/mri_volumes_64x64x64.npz` is already available, you can skip MRI preprocessing entirely:

```
Run cells: 1 → 2 → 3 → 4 → 5 → 6 → 8* → 9 → 10 → 11 → 12 → 13 → 14 → 15 → ...
                                          ↑
                              Cell 8 loads from cache automatically
```

## Full Pipeline (first run)

Run cells **in order, top to bottom**:

| Cell | Name | What it does | Time |
|------|------|-------------|------|
| 1 | Install Dependencies | `pip install` all packages | ~5 min (once) |
| 2 | GPU Setup & Imports | Detects CUDA GPU, imports libraries | < 1s |
| 3 | Configuration | Sets paths, hyperparameters | < 1s |
| 4 | Load CSV Data | Reads 8 ADNI CSV files | ~10s |
| 5 | Feature Engineering | Label creation, merging, KNN imputation | ~2 min |
| 6 | MRI Discovery | Finds all `.nii.gz` files, maps PTID→RID | ~5s |
| 7 | MRI Pipeline Setup | Defines preprocessing functions | < 1s |
| **8** | **MRI Preprocessing** | N4 → Skull strip → Register → GPU resize | **~60-90 min** (first run only) |
| 9 | Align Data | Joins MRI + tabular by subject ID | < 1s |
| 10 | Train/Val/Test Split | Stratified split + SMOTE | ~30s |
| 11 | DataLoaders | PyTorch GPU-ready dataloaders | < 1s |
| 12 | 3D CNN Encoder | Defines CNN architecture | < 1s |
| 13 | Tabular MLP Encoder | Defines MLP + self-attention | < 1s |
| 14 | Fusion Model | Builds full model + loss functions | < 1s |
| **15** | **Training** | 60 epochs, mixed precision, early stopping | **~10-30 min** |
| 16 | Training Curves | Plots & saves loss curves | < 1s |
| 17 | Extract Embeddings | GPU batch embedding extraction | ~30s |
| **18** | **XGBoost (Embeddings)** | Trains XGBoost on fused embeddings | ~2 min |
| **19** | **XGBoost (Tabular)** | Baseline XGBoost on raw features | ~2 min |
| **20** | **Shallow NN** | 2-layer PyTorch classifier | ~30s |
| 21 | Fusion Evaluation | Fusion model test metrics | < 1s |
| 22 | Ensemble Metrics | Final combined metrics (Acc, AUC, F1) | < 1s |
| 23 | Confusion Matrix & ROC | Plots saved to `plots/` | < 1s |
| 24 | SHAP Analysis | Feature importance plots | ~2 min |
| 25 | Grad-CAM | MRI attention heatmap | ~30s |
| 26 | Robustness Test | Missing-modality scenarios | ~30s |
| 27 | Per-Subject Report | CSV with individual predictions | < 1s |
| 28 | Save Models | Saves all models & artifacts | < 1s |
| 29 | Inference Function | `predict_patient()` production function | < 1s |
| 30 | Summary Dashboard | Final metrics summary | < 1s |

## Running Inference on a New Patient

After training, use the `predict_patient()` function from **Cell 29**:

```python
result = predict_patient(
    mri_nifti_path = None,           # Path to .nii.gz, or None for tabular-only
    tabular_raw_values = {
        'AGE'      : 75,
        'APOE4'    : 1,              # 0, 1, or 2 alleles
        'MMSE'     : 24,             # 0–30 (lower = worse)
        'CDRSB'    : 2.5,            # Clinical Dementia Rating
        'HIPPOCAMPUS': 6500,         # mm³
        'ENTORHINAL' : 3000,
        'VENTRICLES' : 38000,
        # ... add any known features
    }
)

# Output:
# {
#   'Diagnosis'    : 'EMCI',
#   'AD_Risk_%'    : 34.2,
#   'MRI_used'     : False,
#   'Probabilities': {'CN': 0.41, 'EMCI': 0.35, 'AD': 0.24}
# }
```

# Expected Results

| Model | Accuracy | AUC (OvR) |
|-------|----------|-----------|
| Fusion Model (CNN + MLP) | ~60% | ~0.90 |
| XGBoost (Embeddings) | ~64-68% | ~0.87 |
| XGBoost (Raw Tabular) | ~65%+ | ~0.85+ |
| Shallow NN | ~48-52% | ~0.82 |
| **Ensemble** | **~56-68%** | **~0.89** |

> Results vary based on the MRI-matched subject subset (163 subjects) and random seed.

# Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Invalid classes ... got [0 1 3]` | XGBoost label gap | Cell 18's remap guard handles this automatically |
| `RuntimeError: size mismatch classifier.6` | Stale checkpoint from 4-class run | Delete `models/fusion_best.pt`, re-run Cell 15 |
| `PermissionError: per_subject_predictions.csv` | File open in Excel | Close the CSV file, re-run Cell 27 |
| `SyntaxError: invalid character '─'` | Unicode box-drawing char in f-string | Use ASCII `-` instead |
| `ITK ERROR: Zero-valued spacing` | Scout/localizer scan (not T1) | Automatically skipped with warning |
| `ERR_CONNECTION_REFUSED` on localhost | Jupyter not running | Start Jupyter: `jupyter notebook` |

---

# Citation

If you use this project in your research, please cite the ADNI dataset:

```
Data used in preparation of this article were obtained from the
Alzheimer's Disease Neuroimaging Initiative (ADNI) database
(adni.loni.usc.edu). The ADNI was launched in 2003 as a
public-private partnership, led by Principal Investigator
Michael W. Weiner, MD.
```

# License

This project is for academic and research purposes. The ADNI dataset requires
separate data access approval from [adni.loni.usc.edu](https://adni.loni.usc.edu/).
