# LNG Tank Cable Inspection Reproducibility Code

This repository provides public reproducibility code for the LNG tank cable vision inspection benchmark. The code is intended to reproduce the analysis pipeline; it does not redistribute the AI-Hub image dataset, labels, checkpoints, or generated results.

## Files

```text
lng_cable_inspection.ipynb  # Colab-oriented reproducibility notebook
lng_cable_inspection.py     # Python script exported from the notebook
requirements.txt                            # Package list
README.md                                   # This file
.gitignore                                  # Excludes data and generated artifacts
```

## Main analysis components

- Cable-only dataset preparation from source archives
- Dataset audit and duplicate/near-duplicate grouping
- Group-safe train/calibration/test splitting
- Calibration-only threshold selection and held-out test evaluation
- Supervised image classification baselines
- Normal-only anomaly detection baselines
- Abnormal training-label budget experiments
- Fine label-count crossover experiments
- Normal-only threshold variants
- Grad-CAM qualitative shortcut checks
- Result tables and figures from saved prediction artifacts

## Data setup

Update the path configuration near the top of the notebook or script:

```python
PROJECT_ROOT = Path('/content/drive/MyDrive/LNG_Abnormal/revision_experiments')
GDRIVE_DATA_ROOT = Path('/content/drive/MyDrive/LNG_Abnormal')
LOCAL_DATASET_PATH = Path('/content/cable_dataset')
```

The default paths assume Google Colab with Google Drive mounted. Users must obtain the source dataset independently and follow the dataset provider's license.

## Running the analysis

Start with the notebook in Colab. Load the function sections first, then run the execution examples selectively. Long-running experiments use `overwrite=False` by default and skip completed artifacts when the required files already exist.

The Python file is provided for review, version control, and advanced users who prefer script-based execution.
