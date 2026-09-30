# LNG Tank Cable Vision Inspection Benchmark

This repository contains reproducibility code for the manuscript:

**Annotation-Budget-Aware Vision-Based Inspection of LNG Tank Cables: A Duplicate-Aware and Calibration-Controlled Benchmark of Supervised and Normal-Only Anomaly Detection**

The code is intended to reproduce the benchmark protocol, not to redistribute the AI-Hub image data. Users must obtain the original dataset according to the dataset provider's license.

## What this code reproduces

- Cable-only dataset preparation from the source archives
- Duplicate and near-duplicate audit
- Group-safe train/calibration/test splits
- Calibration-only threshold selection and held-out test evaluation
- Supervised baselines
- Normal-only anomaly baselines
- Label-budget and fine label-count crossover experiments
- Final tables and figures used in the manuscript
- Grad-CAM shortcut sanity-check examples

## Repository structure

```text
lng_cable_inspection_reproducibility.ipynb  # Main reproducibility notebook
requirements.txt                            # Python dependencies for Colab/local use
README.md                                   # This file
.gitignore                                  # Excludes data, checkpoints, and generated outputs
```

## Data

Place the dataset archives under a local or Google Drive directory and update the paths in the first notebook cell:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/LNG_Abnormal/revision_experiments")
GDRIVE_DATA_ROOT = Path("/content/drive/MyDrive/LNG_Abnormal")
LOCAL_DATASET_PATH = Path("/content/cable_dataset")
```

The repository intentionally does not include raw images, labels, checkpoints, or generated results.

## Recommended execution order

1. Run **00. Environment and Paths**.
2. Run **01. Dataset Preparation** after setting the dataset archive paths.
3. Run the audit and split sections.
4. Run protocol-v2 supervised and anomaly baselines.
5. Run label-budget and fine label-count experiments as compute allows.
6. Generate final tables and figures.

Long-running cells are resume-safe when `overwrite=False` and the required artifacts already exist.

## Reproducibility notes

- Operating thresholds are selected only on the calibration split and then frozen before test evaluation.
- Duplicate/near-duplicate groups are kept within a single split.
- The label-budget experiments vary abnormal training labels while retaining all normal training samples.
- Normal-only anomaly detectors are reported separately for labeled calibration and normal-only threshold variants.

## Citation

If this code supports your work, please cite the corresponding manuscript after publication.
