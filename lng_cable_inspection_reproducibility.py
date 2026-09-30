# LNG tank cable inspection reproducibility code

# This file is exported from lng_cable_inspection_reproducibility.ipynb.


# LNG Tank Cable Inspection Reproducibility Notebook This public notebook reproduces the leakage-controlled and calibratio

# %% public_configuration
from pathlib import Path

# Update these paths for your Colab or local environment.
PROJECT_ROOT = Path('/content/drive/MyDrive/LNG_Abnormal/revision_experiments')
GDRIVE_DATA_ROOT = Path('/content/drive/MyDrive/LNG_Abnormal')
LOCAL_DATASET_PATH = Path('/content/cable_dataset')

# Keep False for reproducibility. Set True only when intentionally deleting and rebuilding LOCAL_DATASET_PATH.
FORCE_REBUILD_DATASET = False

# Long-running functions skip existing completed artifacts when overwrite=False.
OVERWRITE_EXISTING = False

import numpy, scipy, sklearn
print(numpy.__version__)
print(scipy.__version__)
print(sklearn.__version__)

from scipy.signal import upfirdn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
print("numeric stack OK")

# !pip install -q timm transformers datasets faiss-cpu albumentations opencv-python-headless grad-cam open-clip-torch
# !pip install -q anomalib==2.4.1

from google.colab import drive
drive.mount('/content/drive')

# !apt-get install tree


# 00_environment

# %% 00_environment
import os
import gc
import json
import time
import math
import random
import hashlib
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    matthews_corrcoef,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = LOCAL_DATASET_PATH
RUN_ROOT = PROJECT_ROOT / "runs"
TABLE_ROOT = PROJECT_ROOT / "tables"
FIG_ROOT = PROJECT_ROOT / "figures"
MANIFEST_ROOT = PROJECT_ROOT / "manifests"
for p in [PROJECT_ROOT, RUN_ROOT, TABLE_ROOT, FIG_ROOT, MANIFEST_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

def now_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_env_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": str(DEVICE),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    return info

save_json(get_env_info(), PROJECT_ROOT / "environment_snapshot.json")
print(get_env_info())

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# %% experiment tracker
class ExperimentTracker:
    def __init__(self, model_name, experiment_name, config=None, overwrite=False):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.run_id = f"{model_name}__{experiment_name}"
        self.run_dir = RUN_ROOT / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.run_dir / "config.json"
        self.metrics_path = self.run_dir / "metrics.json"
        self.pred_path = self.run_dir / "predictions.csv"
        self.overwrite = overwrite
        self.config = config or {}
        self.config.update({
            "model_name": model_name,
            "experiment_name": experiment_name,
            "run_id": self.run_id,
            "seed": SEED,
            "created_at": now_id(),
            "env": get_env_info(),
        })
        save_json(self.config, self.config_path)

    def done(self):
        return self.metrics_path.exists() and self.pred_path.exists() and not self.overwrite

    def save_predictions(self, df):
        df.to_csv(self.pred_path, index=False)

    def save_metrics(self, metrics):
        save_json(metrics, self.metrics_path)

    def save_checkpoint(self, model, name="model.pt"):
        path = self.run_dir / name
        torch.save(model.state_dict(), path)
        return str(path)

    def artifact(self, name):
        return self.run_dir / name


# Create a "Cable-Only" Local Dataset

## Create a "Cable-Only" Local Dataset
import os
import shutil
import zipfile
from google.colab import drive
from tqdm.auto import tqdm

# --- 1. Define the specific cable-related files to extract ---
TARGET_CABLE_FILES = [
    # Training
    "TS_Cable_Cable_Damage_Cable.zip",
    "TS_Cable_Cable_Good_Cable.zip",
    # Validation
    "VS_Cable_Cable_Damage_Cable.zip",
    "VS_Cable_Cable_Good_Cable (1).zip"
]

# --- 2. Mount Google Drive and set paths ---
# Mount Google Drive manually in Colab if needed.
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

GDRIVE_DATA_ROOT = str(GDRIVE_DATA_ROOT)
LOCAL_DATASET_PATH = str(LOCAL_DATASET_PATH)

# --- Create local dataset directories safely ---
from pathlib import Path
import os
import shutil

LOCAL_DATASET_PATH = Path(LOCAL_DATASET_PATH)

if FORCE_REBUILD_DATASET:
    resolved = LOCAL_DATASET_PATH.resolve()
    allowed_roots = [Path('/content').resolve()]
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        raise RuntimeError(f'Refusing to delete outside allowed roots: {resolved}')
    if resolved in allowed_roots:
        raise RuntimeError(f'Refusing to delete root directory: {resolved}')
    if resolved.exists():
        shutil.rmtree(resolved)

for subdir in [
    'training/good',
    'training/abnormal/Cable_Damage',
    'validation/good',
    'validation/abnormal/Cable_Damage',
]:
    (LOCAL_DATASET_PATH / subdir).mkdir(parents=True, exist_ok=True)

print('dataset directory:', LOCAL_DATASET_PATH)

# --- 4. Unzip only the target cable archives ---
from pathlib import Path
import os, zipfile
from tqdm.auto import tqdm

UNZIP_MARKER_DIR = Path(LOCAL_DATASET_PATH) / "_unzip_markers"
UNZIP_MARKER_DIR.mkdir(parents=True, exist_ok=True)

def unzip_target_archives(set_type, force=False):
    folder_prefix = "TS_" if set_type == "Training" else "VS_"
    source_folder = os.path.join(GDRIVE_DATA_ROOT, f"{set_type}_image")
    files_to_unzip = [f for f in TARGET_CABLE_FILES if f.startswith(folder_prefix)]

    print(f"\nChecking {len(files_to_unzip)} cable archives for the {set_type} set...")

    for filename in tqdm(files_to_unzip, desc=f"Unzipping {set_type}"):
        source_zip_path = os.path.join(source_folder, filename)

        if not os.path.exists(source_zip_path):
            print("missing:", filename)
            continue

        marker_path = UNZIP_MARKER_DIR / f"{set_type}_{filename}.done"

        if marker_path.exists() and not force:
            # Already extracted successfully before
            continue

        if "_Good" in filename:
            dest_folder = os.path.join(LOCAL_DATASET_PATH, set_type.lower(), "good")
        else:
            dest_folder = os.path.join(LOCAL_DATASET_PATH, set_type.lower(), "abnormal", "Cable_Damage")

        os.makedirs(dest_folder, exist_ok=True)

        try:
            with zipfile.ZipFile(source_zip_path, "r") as z:
                z.extractall(path=dest_folder)

            marker_path.write_text(
                f"unzipped_from={source_zip_path}\ndest={dest_folder}\n",
                encoding="utf-8"
            )

        except Exception as e:
            print(f"Error unzipping {filename}: {e}")

unzip_target_archives("Training")
unzip_target_archives("Validation")

print(f"\nCreated or reused local cable-only dataset at: {LOCAL_DATASET_PATH}")

import os
import shutil

src_dir = "/content/cable_dataset/training/abnormal/Cable_Damage"
dst_dir = "/content/cable_dataset/training/abnormal"

# Move each image from Cable_Damage → abnormal
for file_name in os.listdir(src_dir):
    src_file = os.path.join(src_dir, file_name)
    dst_file = os.path.join(dst_dir, file_name)
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)

# Optionally remove empty Cable_Damage folder
os.rmdir(src_dir)
print("Moved all Cable_Damage images to abnormal folder.")

import os
import shutil

src_dir = "/content/cable_dataset/validation/abnormal/Cable_Damage"
dst_dir = "/content/cable_dataset/validation/abnormal"

# Move each image from Cable_Damage → abnormal
for file_name in os.listdir(src_dir):
    src_file = os.path.join(src_dir, file_name)
    dst_file = os.path.join(dst_dir, file_name)
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)

# Optionally remove empty Cable_Damage folder
os.rmdir(src_dir)
print("Moved all Cable_Damage images to abnormal folder.")

import os
import zipfile
from tqdm.auto import tqdm

# Define source and destination mappings
LABEL_FILES = {
    # Training Labels
    "TL_Cable_Cable_Good_Cable.zip": {
        "src": str(Path(GDRIVE_DATA_ROOT) / "Training_label"),
        "dst": "/content/cable_dataset/training/good_label"
    },
    "TL_Cable_Cable_Damage_Cable.zip": {
        "src": str(Path(GDRIVE_DATA_ROOT) / "Training_label"),
        "dst": "/content/cable_dataset/training/abnormal_label"
    },
    # Validation Labels
    "VL_Cable_Cable_Good_Cable.zip": {
        "src": str(Path(GDRIVE_DATA_ROOT) / "Validation_label"),
        "dst": "/content/cable_dataset/validation/good_label"
    },
    "VL_Cable_Cable_Damage_Cable.zip": {
        "src": str(Path(GDRIVE_DATA_ROOT) / "Validation_label"),
        "dst": "/content/cable_dataset/validation/abnormal_label"
    },
}

# Unzip each label zip file
for zip_name, paths in tqdm(LABEL_FILES.items(), desc="Unzipping cable label files"):
    src_path = os.path.join(paths["src"], zip_name)
    dst_path = paths["dst"]

    os.makedirs(dst_path, exist_ok=True)

    if not os.path.exists(src_path):
        print(f"{zip_name} not found in {paths['src']}")
        continue

    try:
        with zipfile.ZipFile(src_path, 'r') as zip_ref:
            zip_ref.extractall(dst_path)
            print(f"Extracted: {zip_name} → {dst_path}")
    except Exception as e:
        print(f"Error unzipping {zip_name}: {e}")


# Data Description

import os

base_path = "/content/cable_dataset"

def count_files(path):
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

for split in ["training", "validation"]:
    print(f"\n- {split.capitalize()} Set")
    for cls in ["good", "abnormal"]:
        img_path = os.path.join(base_path, split, cls)
        label_path = os.path.join(base_path, split, f"{cls}_label")
        print(f"  - {cls.capitalize()} Images: {count_files(img_path)}")
        print(f"  - {cls.capitalize()} Labels: {count_files(label_path)}")

import json

sample_label_path = "/content/cable_dataset/training/good_label/4201_16_0013c1d9-d8ca-4fdc-a7c4-abbe49ffee59.json"

with open(sample_label_path, 'r') as f:
    data = json.load(f)

print(json.dumps(data, indent=2, ensure_ascii=False))

categories = data.get("categories", [])
for cat in categories:
    print(f"ID: {cat['id']}, Name: {cat['name']}")


# See the data

import json

with open("/content/cable_dataset/training/good_label/4201_16_0013c1d9-d8ca-4fdc-a7c4-abbe49ffee59.json", "r") as f:
    data = json.load(f)

import pprint
pprint.pprint(data)

import cv2
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the image
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# Load the label JSON
def load_label(label_path):
    with open(label_path, "r") as f:
        return json.load(f)

# Draw bounding boxes using 'bbox' and 'category_id'
def draw_boxes(image, label_data, box_color=(0, 255, 0), thickness=2):
    categories = {cat["id"]: cat["name"] for cat in label_data.get("categories", [])}
    annotated = image.copy()

    for ann in label_data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        label = categories.get(ann["category_id"], "Unknown")

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        # Put label text
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    return annotated

# Show image with matplotlib
def show_image(title, image_np):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.title(title)
    plt.axis("off")
    plt.show()


# 00_Preparation


# 01_threshold_calibration_protocol

## %% 01_threshold_calibration_protocol
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def make_calibration_test_from_validation(
    manifest_df,
    calib_size=0.5,
    seed=SEED,
    group_col=None,
):
    val = manifest_df[
        (manifest_df["split"].eq("validation")) & (manifest_df["label"].isin([0, 1]))
    ].copy()
    if group_col and group_col in val.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=1 - calib_size, random_state=seed)
        idx_cal, idx_test = next(splitter.split(val, val["label"], groups=val[group_col]))
        calib = val.iloc[idx_cal].copy()
        test = val.iloc[idx_test].copy()
    else:
        calib, test = train_test_split(
            val,
            train_size=calib_size,
            random_state=seed,
            stratify=val["label"] if val["label"].nunique() == 2 else None,
        )
    calib["split_calibrated"] = "calibration"
    test["split_calibrated"] = "test"
    out = pd.concat([calib, test], ignore_index=True)
    out_path = MANIFEST_ROOT / f"calibration_test_split_seed{seed}.csv"
    out.to_csv(out_path, index=False)
    display(out.groupby(["split_calibrated", "class_name"]).size().reset_index(name="n"))
    print("Saved:", out_path)
    return calib, test

def threshold_from_scores(y_true, y_score, method="youden", target_fpr=0.05, target_recall=0.95, normal_percentile=95):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if method == "youden":
        return choose_threshold_youden(y_true, y_score)
    if method == "max_f1":
        thresholds = np.unique(y_score)
        best_t, best_f1 = thresholds[0], -1
        for t in thresholds:
            pred = (y_score >= t).astype(int)
            val = f1_score(y_true, pred, zero_division=0)
            if val > best_f1:
                best_t, best_f1 = t, val
        return float(best_t)
    if method == "normal_percentile":
        normal_scores = y_score[y_true == 0]
        return float(np.percentile(normal_scores, normal_percentile))
    if method == "fixed_fpr":
        normal_scores = np.sort(y_score[y_true == 0])
        if len(normal_scores) == 0:
            return float(np.nan)
        return float(np.quantile(normal_scores, 1 - target_fpr))
    if method == "target_recall":
        abnormal_scores = np.sort(y_score[y_true == 1])
        if len(abnormal_scores) == 0:
            return float(np.nan)
        return float(np.quantile(abnormal_scores, 1 - target_recall))
    raise ValueError(f"Unknown threshold method: {method}")

def evaluate_with_frozen_threshold(
    tracker,
    calib_df,
    test_df,
    threshold_method="youden",
    score_col="y_score",
):
    threshold = threshold_from_scores(
        calib_df["y_true"],
        calib_df[score_col],
        method=threshold_method,
    )
    save_json(
        {
            "threshold": threshold,
            "threshold_method": threshold_method,
            "calibration_n": int(len(calib_df)),
            "test_n": int(len(test_df)),
        },
        tracker.artifact("threshold_protocol.json"),
    )
    metrics, pred_df = save_binary_evaluation(
        tracker,
        test_df["y_true"],
        test_df[score_col],
        paths=test_df.get("path"),
        threshold=threshold,
    )
    metrics["threshold_method"] = threshold_method
    metrics["threshold_source"] = "calibration"
    tracker.save_metrics(metrics)
    return metrics, pred_df


# 02_duplicate_aware_group_split

# %% 02_duplicate_aware_group_split
def build_duplicate_groups(dup_df=None):
    if dup_df is None:
        dup_path = MANIFEST_ROOT / "duplicate_audit.csv"
        if not dup_path.exists():
            raise FileNotFoundError("Run duplicate audit first.")
        dup_df = pd.read_csv(dup_path)
    df = dup_df.copy()
    if "md5" not in df.columns or "dhash" not in df.columns:
        raise ValueError("duplicate_audit.csv must contain md5 and dhash")
    # Conservative group: exact hash if duplicated, else perceptual hash.
    md5_counts = df["md5"].value_counts()
    dhash_counts = df["dhash"].value_counts()
    df["dup_group"] = [
        f"md5::{m}" if md5_counts[m] > 1 else f"dhash::{h}" if dhash_counts[h] > 1 else f"unique::{Path(p).stem}"
        for m, h, p in zip(df["md5"], df["dhash"], df["path"])
    ]
    out = MANIFEST_ROOT / "duplicate_groups.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)
    return df

def make_group_safe_standard_split(group_df, test_size=0.25, seed=SEED):
    df = group_df[group_df["label"].isin([0, 1])].copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train, idx_test = next(splitter.split(df, df["label"], groups=df["dup_group"]))
    train = df.iloc[idx_train].copy()
    test = df.iloc[idx_test].copy()
    train["split_group_safe"] = "train"
    test["split_group_safe"] = "test"
    out = pd.concat([train, test], ignore_index=True)
    path = MANIFEST_ROOT / f"group_safe_split_seed{seed}.csv"
    out.to_csv(path, index=False)
    display(out.groupby(["split_group_safe", "class_name"]).size().reset_index(name="n"))
    leak = out.groupby("dup_group")["split_group_safe"].nunique().gt(1).sum()
    print("Groups crossing train/test:", leak)
    return out


# 03_leave_one_defect_type_out_protocol

# %% 03_leave_one_defect_type_out_protocol
def make_lodo_splits(type_manifest, seed=SEED, calib_size=0.25, test_size=0.25):
    df = type_manifest[type_manifest["label"].isin([0, 1])].copy()
    defect_types = sorted([x for x in df["defect_type"].dropna().unique() if x != "good"])
    all_splits = []
    for heldout in defect_types:
        good = df[df["label"].eq(0)].copy()
        known_abn = df[(df["label"].eq(1)) & (~df["defect_type"].eq(heldout))].copy()
        unknown_abn = df[(df["label"].eq(1)) & (df["defect_type"].eq(heldout))].copy()

        good_train, good_temp = train_test_split(good, test_size=calib_size + test_size, random_state=seed)
        good_calib, good_test = train_test_split(good_temp, test_size=test_size / (calib_size + test_size), random_state=seed)
        known_train, known_temp = train_test_split(
            known_abn,
            test_size=calib_size + test_size,
            random_state=seed,
            stratify=known_abn["defect_type"] if known_abn["defect_type"].nunique() > 1 else None,
        )
        known_calib, known_test = train_test_split(
            known_temp,
            test_size=test_size / (calib_size + test_size),
            random_state=seed,
            stratify=known_temp["defect_type"] if known_temp["defect_type"].nunique() > 1 else None,
        )
        unknown_test = unknown_abn.copy()

        parts = [
            (good_train, "train", "good_train"),
            (known_train, "train", "known_abnormal_train"),
            (good_calib, "calibration", "good_calibration"),
            (known_calib, "calibration", "known_abnormal_calibration"),
            (good_test, "known_test", "good_known_test"),
            (known_test, "known_test", "known_abnormal_test"),
            (good_test.copy(), "unknown_test", "good_unknown_test"),
            (unknown_test, "unknown_test", "heldout_unknown_abnormal"),
        ]
        rows = []
        for part, split, role in parts:
            p = part.copy()
            p["heldout_defect_type"] = heldout
            p["lodo_split"] = split
            p["lodo_role"] = role
            rows.append(p)
        split_df = pd.concat(rows, ignore_index=True)
        all_splits.append(split_df)
    out = pd.concat(all_splits, ignore_index=True)
    path = MANIFEST_ROOT / f"lodo_splits_seed{seed}.csv"
    out.to_csv(path, index=False)
    print("Saved:", path)
    display(out.groupby(["heldout_defect_type", "lodo_split", "label", "defect_type"]).size().reset_index(name="n"))
    return out


# 04_robust_label_json_and_mask_audit

# %% 04_robust_label_json_and_mask_audit
def find_label_json_by_stem(image_path, search_roots=None):
    image_path = Path(image_path)
    stem = image_path.stem
    if search_roots is None:
        search_roots = [
            DATA_ROOT / "training",
            DATA_ROOT / "validation",
            Path("/content/cable_dataset_by_defect/training"),
            Path("/content/cable_dataset_by_defect/validation"),
        ]
    candidates = []
    for root in search_roots:
        if not Path(root).exists():
            continue
        candidates.extend(Path(root).rglob(f"{stem}.json"))
        candidates.extend(Path(root).rglob(f"{stem.upper()}.json"))
        candidates.extend(Path(root).rglob(f"{stem.lower()}.json"))
    for c in candidates:
        if c.suffix.lower() == ".json":
            return str(c)
    return None

def read_aihub_annotation(json_path, image_path=None):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        return {"valid_json": False, "error": "UnicodeDecodeError_probably_image_path"}
    except Exception as e:
        return {"valid_json": False, "error": f"{type(e).__name__}: {e}"}

    anns = data.get("annotations", [])
    n_poly = 0
    n_bbox = 0
    for ann in anns:
        seg = ann.get("segmentation")
        bbox = ann.get("bbox")
        if seg:
            n_poly += 1
        if bbox:
            n_bbox += 1
    return {
        "valid_json": True,
        "n_annotations": len(anns),
        "n_polygon_annotations": n_poly,
        "n_bbox_annotations": n_bbox,
        "has_polygon": n_poly > 0,
        "has_bbox": n_bbox > 0,
    }

def robust_mask_from_json(json_path, image_shape):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for ann in data.get("annotations", []):
        seg = ann.get("segmentation")
        bbox = ann.get("bbox")
        if seg:
            polys = seg if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list) else [seg]
            for poly in polys:
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 1)
        elif bbox:
            x, y, w, h = [int(round(v)) for v in bbox[:4]]
            mask[max(0, y):max(0, y+h), max(0, x):max(0, x+w)] = 1
    return mask

def resize_mask_nearest(mask, size_hw):
    h, w = size_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

def localization_json_audit(manifest_df):
    rows = []
    abnormal = manifest_df[manifest_df["label"].eq(1)].copy()
    for _, r in abnormal.iterrows():
        img_path = r["path"]
        label_json = find_label_json_by_stem(img_path)
        row = {"path": img_path, "label_json": label_json, "json_found": label_json is not None}
        if label_json:
            row.update(read_aihub_annotation(label_json, img_path))
            if row.get("valid_json"):
                img = cv2.imread(img_path)
                if img is not None:
                    mask = robust_mask_from_json(label_json, img.shape)
                    row["image_h"] = int(img.shape[0])
                    row["image_w"] = int(img.shape[1])
                    row["mask_area"] = int(mask.sum())
                    row["empty_mask"] = bool(mask.sum() == 0)
        rows.append(row)
    audit = pd.DataFrame(rows)
    out = MANIFEST_ROOT / "localization_json_audit.csv"
    audit.to_csv(out, index=False)
    summary = {
        "n_abnormal_images": int(len(audit)),
        "json_found_rate": float(audit["json_found"].mean()) if len(audit) else 0,
        "valid_json_rate": float(audit.get("valid_json", pd.Series(dtype=float)).fillna(False).mean()) if len(audit) else 0,
        "polygon_rate": float(audit.get("has_polygon", pd.Series(dtype=float)).fillna(False).mean()) if len(audit) else 0,
        "bbox_rate": float(audit.get("has_bbox", pd.Series(dtype=float)).fillna(False).mean()) if len(audit) else 0,
        "empty_mask_rate": float(audit.get("empty_mask", pd.Series(dtype=float)).fillna(False).mean()) if len(audit) else 0,
    }
    save_json(summary, MANIFEST_ROOT / "localization_json_audit_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return audit


# 05_repeated_runs_and_uncertainty

# %% 05_repeated_runs_and_uncertainty
def bootstrap_metric_ci(y_true, y_score, metric_fn=roc_auc_score, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    if not vals:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}
    vals = np.asarray(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
    }

def add_bootstrap_ci_to_run(run_dir, n_boot=1000):
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        print("No predictions.csv:", run_dir)
        return None
    df = pd.read_csv(pred_path)
    if not {"y_true", "y_score"}.issubset(df.columns):
        print("Missing y_true/y_score:", pred_path)
        return None
    ci = {
        "auroc": bootstrap_metric_ci(df.y_true, df.y_score, roc_auc_score, n_boot=n_boot),
        "auprc": bootstrap_metric_ci(df.y_true, df.y_score, average_precision_score, n_boot=n_boot),
    }
    save_json(ci, run_dir / "bootstrap_ci.json")
    return ci

def summarize_repeated_runs(df):
    group_cols = ["model_name", "paradigm"]
    if "label_ratio" in df.columns:
        group_cols.append("label_ratio")
    agg_cols = [c for c in ["metric_auroc", "metric_auprc", "metric_f1"] if c in df.columns]
    out = df.groupby(group_cols, dropna=False)[agg_cols].agg(["mean", "std", "count"]).reset_index()
    out.to_csv(PAPER_TABLE_ROOT / "repeated_run_summary.csv", index=False)
    display(out)
    return out


# 06_label_efficient_naming_fix

# %% 06_label_efficient_naming_fix
def normalize_paradigm_names_for_paper(df):
    out = df.copy()
    if "paradigm" in out.columns:
        out["paradigm_original"] = out["paradigm"]
        out["paradigm"] = out["paradigm"].replace({
            "label_ratio_supervised": "label_efficient_supervised",
        })
    return out


# 01_dataset_audit

# %% 01_dataset_audit
def list_images(root):
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            label = None
            parts = [x.lower() for x in p.parts]
            if "good" in parts:
                label = 0
            elif "abnormal" in parts:
                label = 1
            split = "unknown"
            for s in ["training", "validation", "test"]:
                if s in parts:
                    split = s
            rows.append({
                "path": str(p),
                "file": p.name,
                "stem": p.stem,
                "split": split,
                "label": label,
                "class_name": "good" if label == 0 else "abnormal" if label == 1 else None,
                "size_bytes": p.stat().st_size,
            })
    return pd.DataFrame(rows)

def file_md5(path, block_size=2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def dhash(path, hash_size=8):
    from PIL import Image
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size))
    arr = np.asarray(img)
    diff = arr[:, 1:] > arr[:, :-1]
    return "".join("1" if v else "0" for v in diff.flatten())

manifest = list_images(DATA_ROOT)
manifest.to_csv(MANIFEST_ROOT / "image_manifest.csv", index=False)
print("Image counts")

display(manifest.groupby(["split", "class_name"], dropna=False).size().reset_index(name="n"))

# Optional execution example. Uncomment to run.
# Exact duplicate audit. Run once; it can take a little time.
# if not (MANIFEST_ROOT / "duplicate_audit.csv").exists():
#     dup = manifest.copy()
#     dup["md5"] = [file_md5(p) for p in dup["path"]]
#     dup["dhash"] = [dhash(p) for p in dup["path"]]
#     dup.to_csv(MANIFEST_ROOT / "duplicate_audit.csv", index=False)
# else:
#     dup = pd.read_csv(MANIFEST_ROOT / "duplicate_audit.csv")
#
# exact_dups = dup[dup.duplicated("md5", keep=False)].sort_values("md5")
# near_dups = dup[dup.duplicated("dhash", keep=False)].sort_values("dhash")
# exact_dups.to_csv(MANIFEST_ROOT / "exact_duplicates.csv", index=False)
# near_dups.to_csv(MANIFEST_ROOT / "near_duplicates_dhash.csv", index=False)
# print("Exact duplicate rows:", len(exact_dups))
# print("Near duplicate rows by dhash:", len(near_dups))


# 02_splits

# %% 02_splits
def make_standard_split(manifest_df):
    df = manifest_df[manifest_df["label"].isin([0, 1])].copy()
    df["split_standard"] = df["split"].map({"training": "train", "validation": "test"}).fillna(df["split"])
    return df

def make_label_ratio_splits(df, ratios=(0.01, 0.05, 0.10, 0.20), seed=SEED):
    rng = np.random.default_rng(seed)
    train = df[df["split_standard"] == "train"].copy()
    out = []
    for ratio in ratios:
        temp = train.copy()
        temp["label_ratio"] = ratio
        temp["is_labeled"] = False
        for label in [0, 1]:
            idx = temp[temp["label"] == label].index.to_numpy()
            k = max(1, int(len(idx) * ratio))
            chosen = rng.choice(idx, size=k, replace=False)
            temp.loc[chosen, "is_labeled"] = True
        out.append(temp)
    return pd.concat(out, ignore_index=True)

standard_split = make_standard_split(manifest)
standard_split.to_csv(MANIFEST_ROOT / "split_standard.csv", index=False)
semi_split = make_label_ratio_splits(standard_split)
semi_split.to_csv(MANIFEST_ROOT / "split_label_ratio_ratios.csv", index=False)

print("Standard split")
display(standard_split.groupby(["split_standard", "class_name"]).size().reset_index(name="n"))
print("Label-ratio supervised labeled counts")
display(semi_split.groupby(["label_ratio", "class_name", "is_labeled"]).size().reset_index(name="n"))


# 03_preprocessing_ablation

# %% 03_preprocessing_ablation
from torchvision import transforms

PREPROCESSING = {
    "resize224_imagenet": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "resize256_tensor": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "resize384_imagenet": transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


# 04-06 shared datasets/loaders

# %% shared datasets/loaders for 04-06
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets

class BinaryImagePathDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"]), row["path"]

def make_binary_loaders(preprocess_name="resize224_imagenet", batch_size=32, num_workers=2):
    transform = PREPROCESSING[preprocess_name]
    train_df = standard_split[standard_split["split_standard"] == "train"].copy()
    test_df = standard_split[standard_split["split_standard"] == "test"].copy()
    train_ds = BinaryImagePathDataset(train_df, transform=transform)
    test_ds = BinaryImagePathDataset(test_df, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader

def save_training_history(tracker, history):
    pd.DataFrame(history).to_csv(tracker.artifact("training_history.csv"), index=False)

def save_anomalib_test_results(tracker, test_results):
    if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
        metrics = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
            for k, v in test_results[0].items()
        }
    elif isinstance(test_results, dict):
        metrics = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
            for k, v in test_results.items()
        }
    else:
        metrics = {"raw_test_results": str(test_results)}
    tracker.save_metrics(metrics)
    save_json({"test_results": str(test_results)}, tracker.artifact("anomalib_test_results_raw.json"))
    return metrics


# 04_supervised_models

# %% 04_supervised_models
from torchvision import models

def build_supervised_model(model_name, num_classes=2, pretrained=True):
    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        input_shape = (1, 3, 224, 224)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        input_shape = (1, 3, 224, 224)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        input_shape = (1, 3, 224, 224)
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        input_shape = (1, 3, 224, 224)
    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        input_shape = (1, 3, 224, 224)
    else:
        raise ValueError(f"Unsupported supervised model: {model_name}")
    return model, input_shape

def train_supervised_model(
    model_name="resnet18",
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    overwrite=False,
):
    experiment_name = f"standard_{preprocess_name}_ep{epochs}_bs{batch_size}"
    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "04_supervised_models",
            "paradigm": "supervised",
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "label_definition": "good=0, abnormal=1",
        },
        overwrite=overwrite,
    )
    if (tracker.artifact("model_state_dict.pt").exists()
            and tracker.artifact("training_history.csv").exists()
            and not overwrite):
        print("skipped:", tracker.run_dir)
        return tracker

    clear_gpu()
    train_loader, test_loader = make_binary_loaders(preprocess_name, batch_size=batch_size)
    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_seen = 0
        for x, y, _paths in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_seen += bs
        epoch_loss = total_loss / max(n_seen, 1)
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": epoch_loss, "elapsed_sec": elapsed})
        print(f"{model_name} epoch {epoch}/{epochs} loss={epoch_loss:.5f} sec={elapsed:.1f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            tracker.save_checkpoint(model, "best_model_state_dict.pt")

    tracker.save_checkpoint(model, "model_state_dict.pt")
    save_training_history(tracker, history)
    save_json({"input_shape": input_shape}, tracker.artifact("model_meta.json"))
    if "evaluate_and_save_supervised" in globals():
        metrics, _ = evaluate_and_save_supervised(
            model, test_loader, model_name, experiment_name,
            config=tracker.config, checkpoint=False
        )
        save_efficiency_report(model, tracker, input_shape=input_shape)
        print(metrics)
    return tracker

def evaluate_saved_supervised_run(model_name, experiment_name, preprocess_name="resize224_imagenet", batch_size=32):
    tracker = ExperimentTracker(model_name, experiment_name, config={"section": "04_supervised_models"})
    model, input_shape = build_supervised_model(model_name, pretrained=False)
    state_path = tracker.artifact("best_model_state_dict.pt")
    if not state_path.exists():
        state_path = tracker.artifact("model_state_dict.pt")
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model = model.to(DEVICE)
    _train_loader, test_loader = make_binary_loaders(preprocess_name, batch_size=batch_size)
    metrics, pred_df = evaluate_and_save_supervised(
        model, test_loader, model_name, experiment_name,
        config=load_json(tracker.config_path, default={}), checkpoint=False
    )
    save_efficiency_report(model, tracker, input_shape=input_shape)
    return metrics, pred_df

resnet_tracker = train_supervised_model("resnet18", epochs=5, batch_size=32)

eff_tracker = train_supervised_model("efficientnet_b0", epochs=5, batch_size=32)


# 05_unsupervised_anomaly_models

# %% 05_unsupervised_anomaly_models
class NormalOnlyImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted([str(p) for p in Path(root).glob("*.jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, path

def make_autoencoder_loaders(preprocess_name="resize256_tensor", train_batch_size=16, eval_batch_size=32):
    transform = PREPROCESSING[preprocess_name]
    train_ds = NormalOnlyImageDataset(DATA_ROOT / "training" / "good", transform=transform)
    test_df = standard_split[standard_split["split_standard"] == "test"].copy()
    test_ds = BinaryImagePathDataset(test_df, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

class SimpleConvAutoencoder(nn.Module):
    def __init__(self, latent_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_reconstruction_anomaly_model(
    model_name="simple_conv_autoencoder",
    preprocess_name="resize256_tensor",
    epochs=5,
    train_batch_size=16,
    eval_batch_size=32,
    lr=1e-3,
    overwrite=False,
):
    experiment_name = f"normal_only_{preprocess_name}_ep{epochs}_bs{train_batch_size}"
    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "05_unsupervised_anomaly_models",
            "paradigm": "reconstruction_anomaly",
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "lr": lr,
            "score": "mean_pixel_mse",
        },
        overwrite=overwrite,
    )
    if (tracker.artifact("model_state_dict.pt").exists()
            and tracker.artifact("training_history.csv").exists()
            and not overwrite):
        print("Existing reconstruction checkpoint found:", tracker.run_dir)
        return tracker

    clear_gpu()
    train_loader, test_loader = make_autoencoder_loaders(preprocess_name, train_batch_size, eval_batch_size)
    model = SimpleConvAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_seen = 0
        for x, _paths in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_seen += bs
        epoch_loss = total_loss / max(n_seen, 1)
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": epoch_loss, "elapsed_sec": elapsed})
        print(f"{model_name} epoch {epoch}/{epochs} loss={epoch_loss:.6f} sec={elapsed:.1f}")

    tracker.save_checkpoint(model, "model_state_dict.pt")
    save_training_history(tracker, history)
    save_json({"input_shape": (1, 3, 256, 256)}, tracker.artifact("model_meta.json"))
    if "evaluate_and_save_reconstruction" in globals():
        metrics, _ = evaluate_and_save_reconstruction(
            model, test_loader, model_name, experiment_name,
            config=tracker.config, checkpoint=False
        )
        save_efficiency_report(model, tracker, input_shape=(1, 3, 256, 256))
        print(metrics)
    return tracker

def train_anomalib_embedding_model(
    method="patchcore",
    backbone="resnet18",
    layers=None,
    coreset_sampling_ratio=0.005,
    num_neighbors=1,
    train_batch_size=1,
    eval_batch_size=1,
    accelerator=None,
    show_progress=False,
    overwrite=False,
):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore, Padim, Stfpm
    from lightning.pytorch.callbacks import TQDMProgressBar

    method = method.lower()
    if layers is None:
        layers = ["layer2"] if method == "patchcore" else ["layer1", "layer2", "layer3"]
    experiment_name = f"normal_only_{backbone}_{'-'.join(layers)}_bs{train_batch_size}"
    tracker = ExperimentTracker(
        f"{method}_{backbone}",
        experiment_name,
        config={
            "section": "05_unsupervised_anomaly_models",
            "paradigm": "embedding_anomaly",
            "method": method,
            "backbone": backbone,
            "layers": layers,
            "coreset_sampling_ratio": coreset_sampling_ratio if method == "patchcore" else None,
            "num_neighbors": num_neighbors if method == "patchcore" else None,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "show_progress": show_progress,
            "label_definition": "train only normal images; test good=0, abnormal=1",
        },
        overwrite=overwrite,
    )
    if tracker.metrics_path.exists() and not overwrite:
        print("Existing anomalib metrics found:", tracker.run_dir)
        return tracker

    clear_gpu()
    dm = Folder(
        name=f"{method}_lng_cable",
        root=str(DATA_ROOT),
        normal_dir="training/good",
        normal_test_dir="validation/good",
        abnormal_dir="validation/abnormal",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=0,
    )
    dm.setup()
    if method == "patchcore":
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
    elif method == "padim":
        model = Padim(backbone=backbone, layers=layers)
    elif method == "stfpm":
        model = Stfpm(backbone=backbone, layers=layers)
    else:
        raise ValueError("method must be one of: patchcore, padim, stfpm")
    callbacks = [TQDMProgressBar(refresh_rate=20)] if show_progress else []
    engine = Engine(
        max_epochs=1,
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=show_progress,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm, ckpt_path=None)
    train_sec = time.perf_counter() - t0

    manual_ckpt_path = tracker.artifact(f"{method}_manual_after_fit.ckpt")
    try:
        engine.trainer.save_checkpoint(str(manual_ckpt_path))
        checkpoint_info = {"manual_checkpoint_path": str(manual_ckpt_path)}
    except Exception as e:
        checkpoint_info = {
            "manual_checkpoint_path": None,
            "checkpoint_warning": f"{type(e).__name__}: {e}",
        }

    test_results = engine.test(model=model, datamodule=dm)
    metrics = save_anomalib_test_results(tracker, test_results)
    metrics["train_elapsed_sec"] = train_sec
    tracker.save_metrics(metrics)
    save_json(checkpoint_info, tracker.artifact("checkpoint_info.json"))
    print(metrics)
    return tracker

# Optional execution example. Uncomment to run.
# Examples:
# ae_tracker = train_reconstruction_anomaly_model(epochs=5)

patch_tracker = train_anomalib_embedding_model(
    "patchcore",
    layers=["layer2"],
    train_batch_size=1,
    eval_batch_size=1,
    show_progress=False
)

padim_tracker = train_anomalib_embedding_model("padim", layers=["layer2"], train_batch_size=4, eval_batch_size=4,show_progress=False)

stfpm_tracker = train_anomalib_embedding_model("stfpm", layers=["layer1", "layer2"], train_batch_size=4, eval_batch_size=4,show_progress=False)


# 06_label_ratio_models

# %% 06_label_ratio_models
def make_labeled_subset_loader(label_ratio, preprocess_name="resize224_imagenet", batch_size=32, num_workers=2):
    split_df = semi_split[
        (semi_split["label_ratio"] == label_ratio)
        & (semi_split["is_labeled"])
    ].copy()
    transform = PREPROCESSING[preprocess_name]
    ds = BinaryImagePathDataset(split_df, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return loader, split_df

def train_label_ratio_supervised(
    model_name="resnet18",
    label_ratio=0.05,
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    overwrite=False,
):
    experiment_name = f"semi_label{label_ratio:g}_{preprocess_name}_ep{epochs}_bs{batch_size}"
    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "06_label_ratio_models",
            "paradigm": "label_ratio_supervised",
            "label_ratio": label_ratio,
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        overwrite=overwrite,
    )
    if tracker.metrics_path.exists() and not overwrite:
        print("Existing label-ratio supervised result found:", tracker.run_dir)
        return tracker

    clear_gpu()
    train_loader, labeled_df = make_labeled_subset_loader(label_ratio, preprocess_name, batch_size)
    _full_train_loader, test_loader = make_binary_loaders(preprocess_name, batch_size=batch_size)
    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_seen = 0
        for x, y, _paths in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_seen += bs
        epoch_loss = total_loss / max(n_seen, 1)
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": epoch_loss, "elapsed_sec": elapsed, "n_labeled": len(labeled_df)})
        print(f"{model_name} label_ratio={label_ratio:g} epoch {epoch}/{epochs} loss={epoch_loss:.5f}")

    tracker.save_checkpoint(model, "model_state_dict.pt")
    save_training_history(tracker, history)
    labeled_df.to_csv(tracker.artifact("labeled_subset.csv"), index=False)
    if "evaluate_and_save_supervised" in globals():
        metrics, _ = evaluate_and_save_supervised(
            model, test_loader, model_name, experiment_name,
            config=tracker.config, checkpoint=False
        )
        save_efficiency_report(model, tracker, input_shape=input_shape)
        print(metrics)
    return tracker

def run_label_ratio_grid(
    model_name="resnet18",
    ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.0),
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
):
    trackers = []
    global semi_split
    # Add 50% and 100% splits if they are not already in semi_split.
    existing = set(np.round(semi_split["label_ratio"].unique(), 6)) if "semi_split" in globals() else set()
    needed = [r for r in ratios if round(r, 6) not in existing]
    if needed:
        extra = make_label_ratio_splits(standard_split, ratios=tuple(needed))
        semi_split = pd.concat([semi_split, extra], ignore_index=True)
        semi_split.to_csv(MANIFEST_ROOT / "split_label_ratio_ratios.csv", index=False)
    for r in ratios:
        trackers.append(train_label_ratio_supervised(
            model_name=model_name,
            label_ratio=r,
            preprocess_name=preprocess_name,
            epochs=epochs,
            batch_size=batch_size,
        ))
    return trackers

semi_trackers = run_label_ratio_grid("resnet18", ratios=(0.01, 0.05, 0.10, 0.20), epochs=5)


# 07_metrics_image_level

# %% 07_metrics_image_level
def choose_threshold_youden(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float(np.nan)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])

def compute_binary_metrics(y_true, y_score, threshold=None, positive_label=1):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
        threshold = choose_threshold_youden(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(tn / (tn + fp + 1e-12)),
        "f1": float(f1),
        "fnr": float(fn / (tp + fn + 1e-12)),
        "fpr": float(fp / (tn + fp + 1e-12)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
    }
    return metrics, y_pred

def save_binary_evaluation(tracker, y_true, y_score, paths=None, threshold=None, extra_cols=None):
    metrics, y_pred = compute_binary_metrics(y_true, y_score, threshold)
    df = pd.DataFrame({
        "path": paths if paths is not None else [None] * len(y_true),
        "y_true": np.asarray(y_true).astype(int),
        "y_score": np.asarray(y_score).astype(float),
        "y_pred": y_pred.astype(int),
    })
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    tracker.save_predictions(df)
    tracker.save_metrics(metrics)
    pd.DataFrame(confusion_matrix(df.y_true, df.y_pred, labels=[0, 1]),
                 index=["true_normal", "true_abnormal"],
                 columns=["pred_normal", "pred_abnormal"]).to_csv(tracker.artifact("confusion_matrix.csv"))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, roc_thr = roc_curve(y_true, y_score)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(tracker.artifact("roc_curve.csv"), index=False)
        prec, rec, pr_thr = precision_recall_curve(y_true, y_score)
        pr_thr = np.append(pr_thr, np.nan)
        pd.DataFrame({"precision": prec, "recall": rec, "threshold": pr_thr}).to_csv(tracker.artifact("pr_curve.csv"), index=False)
    return metrics, df


# %% supervised adapter
def collect_supervised_scores(model, dataloader, device=DEVICE):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch["image"].to(device)
                y = batch["label"]
                paths = batch.get("path") or batch.get("image_path") or [None] * len(y)
            else:
                x, y = batch[0].to(device), batch[1]
                paths = batch[2] if len(batch) > 2 else [None] * len(y)
            logits = model(x)
            if logits.ndim == 1 or logits.shape[1] == 1:
                score = torch.sigmoid(logits.reshape(-1))
            else:
                score = torch.softmax(logits, dim=1)[:, 1]
            for p, yy, ss in zip(paths, y, score.detach().cpu()):
                rows.append({"path": p, "y_true": int(yy), "y_score": float(ss)})
    return pd.DataFrame(rows)

def evaluate_and_save_supervised(model, dataloader, model_name, experiment_name, config=None, checkpoint=True):
    tracker = ExperimentTracker(model_name, experiment_name, config=config)
    if tracker.done():
        print("skipped:", tracker.run_dir)
        return load_json(tracker.metrics_path), pd.read_csv(tracker.pred_path)
    df = collect_supervised_scores(model, dataloader)
    metrics, pred_df = save_binary_evaluation(tracker, df.y_true, df.y_score, paths=df.path)
    if checkpoint:
        tracker.save_checkpoint(model, "model_state_dict.pt")
    print(metrics)
    return metrics, pred_df


# %% reconstruction anomaly adapter
def collect_reconstruction_scores(model, dataloader, device=DEVICE):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch["image"].to(device)
                y = batch["label"]
                paths = batch.get("path") or batch.get("image_path") or [None] * len(y)
            else:
                x, y = batch[0].to(device), batch[1]
                paths = batch[2] if len(batch) > 2 else [None] * len(y)
            recon = model(x)
            score = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
            for p, yy, ss in zip(paths, y, score):
                rows.append({"path": p, "y_true": int(yy), "y_score": float(ss)})
    return pd.DataFrame(rows)

def evaluate_and_save_reconstruction(model, dataloader, model_name, experiment_name, config=None, checkpoint=True):
    tracker = ExperimentTracker(model_name, experiment_name, config=config)
    if tracker.done():
        print("skipped:", tracker.run_dir)
        return load_json(tracker.metrics_path), pd.read_csv(tracker.pred_path)
    df = collect_reconstruction_scores(model, dataloader)
    metrics, pred_df = save_binary_evaluation(tracker, df.y_true, df.y_score, paths=df.path)
    if checkpoint:
        tracker.save_checkpoint(model, "model_state_dict.pt")
    print(metrics)
    return metrics, pred_df


# %% anomalib adapter
def normalize_anomalib_prediction_item(item):
    """Convert one anomalib prediction item to a flat dict when possible."""
    out = {}
    if isinstance(item, dict):
        for k, v in item.items():
            if torch.is_tensor(v):
                v = v.detach().cpu()
                if v.numel() == 1:
                    v = v.item()
            out[k] = v
    else:
        out["raw"] = item
    return out

def save_anomalib_test_results(tracker, test_results):
    if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
        metrics = {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                   for k, v in test_results[0].items()}
    elif isinstance(test_results, dict):
        metrics = {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                   for k, v in test_results.items()}
    else:
        metrics = {"raw_test_results": str(test_results)}
    tracker.save_metrics(metrics)
    save_json({"test_results": str(test_results)}, tracker.artifact("anomalib_test_results_raw.json"))
    return metrics


# 08_metrics_localization

# %% 08_metrics_localization
def resize_map_to_mask(pred_map, mask):
    import cv2
    pred = np.asarray(pred_map).squeeze().astype(np.float32)
    gt = np.asarray(mask).squeeze()
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
    gt = (gt > 0).astype(np.uint8)
    return pred, gt

def compute_localization_metrics(pred_maps, masks, threshold=0.5):
    rows = []
    for i, (pm, m) in enumerate(zip(pred_maps, masks)):
        pred, gt = resize_map_to_mask(pm, m)
        y_true = gt.reshape(-1)
        y_score = pred.reshape(-1)
        y_bin = (y_score >= threshold).astype(np.uint8)
        inter = np.logical_and(y_bin == 1, y_true == 1).sum()
        union = np.logical_or(y_bin == 1, y_true == 1).sum()
        pred_sum = (y_bin == 1).sum()
        gt_sum = (y_true == 1).sum()
        rows.append({
            "idx": i,
            "pixel_auroc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
            "pixel_auprc": average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
            "iou": inter / (union + 1e-12),
            "dice": (2 * inter) / (pred_sum + gt_sum + 1e-12),
            "mask_area": int(gt_sum),
            "pred_area": int(pred_sum),
        })
    return pd.DataFrame(rows)


# 09_efficiency_latency_model_size

# %% 09_efficiency_latency_model_size
def count_parameters(model):
    return {
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }

def estimate_state_dict_size_mb(model):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / (1024**2)

def measure_latency_ms(model, input_shape=(1, 3, 224, 224), device=DEVICE, warmup=10, runs=50):
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / runs * 1000

def save_efficiency_report(model, tracker, input_shape=(1, 3, 224, 224)):
    report = {}
    report.update(count_parameters(model))
    report["state_dict_size_mb"] = float(estimate_state_dict_size_mb(model))
    report["latency_ms_per_image"] = float(measure_latency_ms(model, input_shape=input_shape))
    if torch.cuda.is_available():
        report["gpu_max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2)
    save_json(report, tracker.artifact("efficiency.json"))
    return report


# 10_Result

# %% 10_figures_tables
def collect_all_metrics(run_root=RUN_ROOT):
    rows = []
    for metrics_path in Path(run_root).glob("*/metrics.json"):
        metrics = load_json(metrics_path, default={})
        config = load_json(metrics_path.parent / "config.json", default={})
        row = {}
        row.update(config)
        row.update(metrics)
        row["run_dir"] = str(metrics_path.parent)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(TABLE_ROOT / "all_model_metrics.csv", index=False)
    return df

def make_paper_table(df):
    cols = [
        "model_name", "experiment_name", "accuracy", "balanced_accuracy",
        "precision", "recall_sensitivity", "specificity", "f1",
        "auroc", "auprc", "fnr", "fpr", "mcc",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if c not in ["model_name", "experiment_name"] and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(4)
    out.to_csv(TABLE_ROOT / "metrics.csv", index=False)
    return out

all_metrics = collect_all_metrics()
if not all_metrics.empty:
    display(make_paper_table(all_metrics))

# %% output_setup
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="report")
except Exception:
    sns = None

from PIL import Image, ImageOps, ImageDraw

TABLE_ROOT = TABLE_ROOT
FIG_ROOT = FIG_ROOT / "figure"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)

METRIC_ALIASES = {
    "auroc": ["auroc", "AUROC", "image_AUROC", "image_AUROC_epoch"],
    "auprc": ["auprc", "AUPRC", "image_AUPRC", "image_AUPRC_epoch"],
    "f1": ["f1", "F1", "image_F1Score", "image_F1Score_epoch"],
    "accuracy": ["accuracy", "Accuracy", "image_Accuracy", "image_Accuracy_epoch"],
    "precision": ["precision", "Precision", "image_Precision", "image_Precision_epoch"],
    "recall_sensitivity": ["recall_sensitivity", "recall", "Recall", "image_Recall", "image_Recall_epoch"],
    "specificity": ["specificity", "Specificity"],
    "fnr": ["fnr", "FNR"],
    "fpr": ["fpr", "FPR"],
    "mcc": ["mcc", "MCC"],
}

def coerce_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def canonical_metric(row, key):
    for k in METRIC_ALIASES.get(key, [key]):
        if k in row and pd.notna(row[k]):
            return coerce_float(row[k])
    return np.nan

def read_run_artifacts(run_dir):
    run_dir = Path(run_dir)
    config = load_json(run_dir / "config.json", default={})
    metrics = load_json(run_dir / "metrics.json", default={})
    efficiency = load_json(run_dir / "efficiency.json", default={})
    row = {}
    row.update(config)
    row.update(metrics)
    for k, v in efficiency.items():
        row[f"eff_{k}"] = v
    row["run_dir"] = str(run_dir)
    row["has_predictions"] = (run_dir / "predictions.csv").exists()
    row["has_confusion_matrix"] = (run_dir / "confusion_matrix.csv").exists()
    row["has_roc_curve"] = (run_dir / "roc_curve.csv").exists()
    row["has_pr_curve"] = (run_dir / "pr_curve.csv").exists()
    row["has_efficiency"] = (run_dir / "efficiency.json").exists()
    for metric in METRIC_ALIASES:
        row[f"metric_{metric}"] = canonical_metric(row, metric)
    return row

def collect_results(run_root=RUN_ROOT):
    rows = []
    for run_dir in sorted(Path(run_root).glob("*")):
        if run_dir.is_dir() and (run_dir / "metrics.json").exists():
            rows.append(read_run_artifacts(run_dir))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(TABLE_ROOT / "results_long.csv", index=False)
    return df

results = collect_results()
display(results.head())


# %% result_quality_audit
def audit_result_quality(df):
    rows = []
    required_metrics = ["metric_auroc", "metric_auprc", "metric_f1"]
    for _, r in df.iterrows():
        issues = []
        if not r.get("has_predictions", False) and r.get("paradigm") not in ["embedding_anomaly"]:
            issues.append("missing predictions.csv")
        if not r.get("has_confusion_matrix", False) and r.get("paradigm") not in ["embedding_anomaly"]:
            issues.append("missing confusion_matrix.csv")
        if all(pd.isna(r.get(m)) for m in required_metrics):
            issues.append("no canonical AUROC/AUPRC/F1 metric found")
        if pd.notna(r.get("metric_accuracy")) and pd.notna(r.get("metric_auroc")):
            if r["metric_accuracy"] > 0.85 and r["metric_auroc"] < 0.60:
                issues.append("accuracy high but AUROC low: check label direction or threshold")
        if pd.notna(r.get("metric_f1")) and pd.notna(r.get("metric_recall_sensitivity")):
            if r["metric_f1"] > 0.95 and r["metric_recall_sensitivity"] < 0.70:
                issues.append("F1/recall inconsistency: check metric calculation")
        rows.append({
            "model_name": r.get("model_name"),
            "experiment_name": r.get("experiment_name"),
            "paradigm": r.get("paradigm"),
            "n_issues": len(issues),
            "issues": "; ".join(issues),
            "run_dir": r.get("run_dir"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_ROOT / "result_audit.csv", index=False)
    return out

quality_audit = audit_result_quality(results)
display(quality_audit.sort_values(["n_issues", "model_name"], ascending=[False, True]))

# %% image_level_metrics
def make_image_level_table(df):
    table = pd.DataFrame({
        "Model": df.get("model_name"),
        "Experiment": df.get("experiment_name"),
        "Paradigm": df.get("paradigm"),
        "Preprocess": df.get("preprocessing"),
        "AUROC": df.get("metric_auroc"),
        "AUPRC": df.get("metric_auprc"),
        "F1": df.get("metric_f1"),
        "Accuracy": df.get("metric_accuracy"),
        "Sensitivity": df.get("metric_recall_sensitivity"),
        "Specificity": df.get("metric_specificity"),
        "FNR": df.get("metric_fnr"),
        "FPR": df.get("metric_fpr"),
        "MCC": df.get("metric_mcc"),
        "Threshold": df.get("threshold"),
    })
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    table[numeric_cols] = table[numeric_cols].round(4)
    table = table.sort_values(["Paradigm", "AUROC", "F1"], ascending=[True, False, False])
    table.to_csv(TABLE_ROOT / "1_image_level_metrics.csv", index=False)
    return table

table1 = make_image_level_table(results)
display(table1)

# %% 2_efficiency
def make_efficiency_table(df):
    cols = {
        "Model": df.get("model_name"),
        "Experiment": df.get("experiment_name"),
        "Paradigm": df.get("paradigm"),
        "Latency ms/img": df.get("eff_latency_ms_per_image"),
        "State dict MB": df.get("eff_state_dict_size_mb"),
        "Total params": df.get("eff_params_total"),
        "Trainable params": df.get("eff_params_trainable"),
        "GPU max memory MB": df.get("eff_gpu_max_memory_allocated_mb"),
    }
    table = pd.DataFrame(cols)
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    table[numeric_cols] = table[numeric_cols].round(3)
    table.to_csv(TABLE_ROOT / "table2_efficiency.csv", index=False)
    return table

table2 = make_efficiency_table(results)
display(table2)

# %% 1_model_comparison_bars
def plot_metric_bars(df, metrics=("metric_auroc", "metric_f1", "metric_auprc")):
    plot_df = df.copy()
    label = plot_df["model_name"].astype(str) + "\n" + plot_df["experiment_name"].astype(str).str.slice(0, 24)
    plot_df["label"] = label
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), max(4, 0.4 * len(plot_df))))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        temp = plot_df[["label", metric, "paradigm"]].dropna().sort_values(metric, ascending=True)
        colors = None
        if sns is not None:
            sns.barplot(data=temp, y="label", x=metric, hue="paradigm", dodge=False, ax=ax)
            ax.legend(loc="lower right", fontsize=8)
        else:
            ax.barh(temp["label"], temp[metric], color="steelblue")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(metric.replace("metric_", "").upper())
        ax.set_ylabel("")
        ax.set_title(metric.replace("metric_", "").upper())
    plt.tight_layout()
    path = FIG_ROOT / "fig1_model_metric_bars.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)

plot_metric_bars(results)

# %% figure_2_roc_pr_curves
def plot_roc_pr_curves(run_root=RUN_ROOT, max_runs=20):
    run_dirs = [p for p in sorted(Path(run_root).glob("*")) if p.is_dir()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_roc = 0
    n_pr = 0
    for run_dir in run_dirs[:max_runs]:
        label = run_dir.name.replace("__", " | ")
        roc_path = run_dir / "roc_curve.csv"
        pr_path = run_dir / "pr_curve.csv"
        if roc_path.exists():
            roc = pd.read_csv(roc_path)
            axes[0].plot(roc["fpr"], roc["tpr"], linewidth=1.5, label=label)
            n_roc += 1
        if pr_path.exists():
            pr = pd.read_csv(pr_path)
            axes[1].plot(pr["recall"], pr["precision"], linewidth=1.5, label=label)
            n_pr += 1
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curves")
    if n_roc:
        axes[0].legend(fontsize=7, loc="lower right")
    if n_pr:
        axes[1].legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    path = FIG_ROOT / "fig2_roc_pr_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)

plot_roc_pr_curves()

# %% 3_efficiency_tradeoff
def plot_efficiency_tradeoff(df):
    temp = df.copy()
    temp = temp.dropna(subset=["metric_auroc", "eff_latency_ms_per_image"])
    if temp.empty:
        print("No efficiency data found.")
        return
    plt.figure(figsize=(7, 5))
    if sns is not None:
        sns.scatterplot(
            data=temp,
            x="eff_latency_ms_per_image",
            y="metric_auroc",
            hue="paradigm",
            size="eff_state_dict_size_mb",
            sizes=(40, 300),
        )
    else:
        plt.scatter(temp["eff_latency_ms_per_image"], temp["metric_auroc"], s=80)
    for _, r in temp.iterrows():
        plt.text(r["eff_latency_ms_per_image"], r["metric_auroc"], str(r["model_name"]), fontsize=8)
    plt.xlabel("Inference latency (ms/image)")
    plt.ylabel("Image-level AUROC")
    plt.title("Accuracy-efficiency trade-off")
    plt.ylim(0, 1.03)
    plt.tight_layout()
    path = FIG_ROOT / "fig3_efficiency_tradeoff.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)

plot_efficiency_tradeoff(results)

# %% 4_label_ratio_supervised
def plot_label_ratio_curve(df):
    temp = df[df.get("paradigm").eq("label_ratio_supervised")].copy() if "paradigm" in df else pd.DataFrame()
    if temp.empty:
        print("No label-ratio supervised label-ratio results found.")
        return
    temp["label_ratio"] = pd.to_numeric(temp["label_ratio"], errors="coerce")
    plt.figure(figsize=(7, 5))
    y_metric = "metric_auroc" if temp["metric_auroc"].notna().any() else "metric_f1"
    if sns is not None:
        sns.lineplot(data=temp, x="label_ratio", y=y_metric, hue="model_name", marker="o")
    else:
        for name, sub in temp.groupby("model_name"):
            plt.plot(sub["label_ratio"], sub[y_metric], marker="o", label=name)
        plt.legend()
    plt.xscale("log")
    plt.xlabel("Labeled training ratio")
    plt.ylabel(y_metric.replace("metric_", "").upper())
    plt.title("Effect of labeled defect data")
    plt.ylim(0, 1.03)
    plt.tight_layout()
    path = FIG_ROOT / "fig4_label_ratio_curve.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)

plot_label_ratio_curve(results)

# %% figure_5_failure_cases_from_predictions
def make_thumb(path, size=(180, 120), caption=None):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    if caption:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, size[1] - 22, size[0], size[1]], fill=(255, 255, 255))
        draw.text((4, size[1] - 18), caption[:32], fill=(0, 0, 0))
    return canvas

def plot_failure_grid(run_dir, top_k=12):
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        print("No predictions.csv:", run_dir)
        return
    df = pd.read_csv(pred_path)
    if not {"path", "y_true", "y_score", "y_pred"}.issubset(df.columns):
        print("predictions.csv missing required columns:", pred_path)
        return
    df["error_type"] = np.select(
        [
            (df.y_true == 1) & (df.y_pred == 0),
            (df.y_true == 0) & (df.y_pred == 1),
        ],
        ["FN", "FP"],
        default="correct",
    )
    df["error_severity"] = np.where(df.y_true == 1, 1 - df.y_score, df.y_score)
    failures = df[df.error_type != "correct"].sort_values("error_severity", ascending=False).head(top_k)
    if failures.empty:
        failures = df.iloc[(df.y_score - 0.5).abs().argsort()].head(top_k)
    thumbs = []
    for _, r in failures.iterrows():
        if isinstance(r["path"], str) and Path(r["path"]).exists():
            caption = f"{r.error_type} y={int(r.y_true)} s={r.y_score:.2f}"
            thumbs.append(make_thumb(r["path"], caption=caption))
    if not thumbs:
        print("No readable image paths for:", run_dir)
        return
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    w, h = thumbs[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(thumbs):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    out = FIG_ROOT / f"failure_grid_{run_dir.name}.png"
    grid.save(out)
    display(grid)
    print("Saved:", out)

# Optional execution example. Uncomment to run.
# Example: plot worst cases for the first run with predictions
# pred_runs = [p for p in Path(RUN_ROOT).glob("*") if (p / "predictions.csv").exists()]
# if pred_runs:
#     plot_failure_grid(pred_runs[0])
# else:
#     print("No prediction-based runs found yet.")


# 11_Additional Analysis


# 01_supervised_coverage

# %% 01_supervised_coverage
def run_full_supervised_baselines(epochs=5, batch_size=32):
    trackers = {}
    for model_name in ["resnet18", "efficientnet_b0", "convnext_tiny"]:
        trackers[model_name] = train_supervised_model(
            model_name=model_name,
            preprocess_name="resize224_imagenet",
            epochs=epochs,
            batch_size=batch_size,
        )
    return trackers

supervised_trackers = run_full_supervised_baselines(epochs=5, batch_size=32)


# 02_dynamic_anomalib_resolver

import inspect

def resolve_anomalib_model_class(method):
    method = method.lower().replace("-", "_")
    candidates = {
        "efficientad": ["EfficientAd", "EfficientAD", "Efficientad"],
        "efficient_ad": ["EfficientAd", "EfficientAD", "Efficientad"],
        "reverse_distillation": ["ReverseDistillation", "ReverseDistillationModel"],
        "reverse_distillation_model": ["ReverseDistillation", "ReverseDistillationModel"],
        "fastflow": ["Fastflow", "FastFlow"],
        "fast_flow": ["Fastflow", "FastFlow"],
        "draem": ["Draem", "DRAEM"],
        "winclip": ["WinClip", "WinCLIP", "Winclip"],
    }.get(method, [method])

    modules = []
    try:
        import anomalib.models as m
        modules.append(m)
    except Exception:
        pass
    try:
        import anomalib.models.image as mi
        modules.append(mi)
    except Exception:
        pass

    for module in modules:
        for name in candidates:
            if hasattr(module, name):
                return getattr(module, name)
    for module in modules:
        for attr in dir(module):
            if attr.lower() in [c.lower() for c in candidates]:
                return getattr(module, attr)

    available = sorted(set(
        name for module in modules for name in dir(module)
        if not name.startswith("_")
    ))
    raise ImportError(
        f"Could not resolve Anomalib class for method={method}. "
        f"Candidates={candidates}. Available names include: {available[:80]}"
    )

def instantiate_anomalib_model(method, backbone="resnet18", layers=None, image_size=256):
    cls = resolve_anomalib_model_class(method)
    name = cls.__name__.lower()
    sig = inspect.signature(cls)
    kwargs = {}

    # Add only parameters accepted by the installed Anomalib class.
    if "backbone" in sig.parameters:
        kwargs["backbone"] = backbone
    if "layers" in sig.parameters and layers is not None:
        kwargs["layers"] = layers
    if "input_size" in sig.parameters:
        kwargs["input_size"] = (image_size, image_size)
    if "image_size" in sig.parameters:
        kwargs["image_size"] = (image_size, image_size)

    # Method-specific safe defaults.
    if "efficient" in name:
        # EfficientAD generally has custom defaults; avoid over-specifying.
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if "fast" in name:
        if "flow_steps" in sig.parameters:
            kwargs["flow_steps"] = 8
    if "draem" in name:
        # DRAEM may use anomaly-source arguments in some versions; defaults are safest.
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if "win" in name and "clip" in name:
        # WinCLIP is usually zero/few-shot and may not need backbone/layers.
        kwargs = {k: v for k, v in kwargs.items() if k not in ["backbone", "layers"]}

    print("Instantiating", cls, "with", kwargs)
    return cls(**kwargs)


# 03_train_advanced_anomalib_model

# %% 03_train_advanced_anomalib_model
def train_advanced_anomalib_model(
    method,
    backbone="resnet18",
    layers=None,
    train_batch_size=2,
    eval_batch_size=2,
    image_size=256,
    accelerator=None,
    show_progress=False,
    overwrite=False,
):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from lightning.pytorch.callbacks import TQDMProgressBar

    method_clean = method.lower().replace("-", "_")
    layers = layers or (["layer2"] if backbone.startswith("resnet") else None)
    experiment_name = f"advanced_{method_clean}_{backbone}_bs{train_batch_size}"

    tracker = ExperimentTracker(
        f"{method_clean}_{backbone}",
        experiment_name,
        config={
            "section": "advanced_models",
            "paradigm": "embedding_anomaly" if method_clean != "winclip" else "zero_shot_anomaly",
            "method": method_clean,
            "backbone": backbone,
            "layers": layers,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "image_size": image_size,
            "show_progress": show_progress,
        },
        overwrite=overwrite,
    )
    if tracker.metrics_path.exists() and not overwrite:
        print("skipped:", tracker.run_dir)
        return tracker

    clear_gpu()
    dm = Folder(
        name=f"{method_clean}_lng_cable",
        root=str(DATA_ROOT),
        normal_dir="training/good",
        normal_test_dir="validation/good",
        abnormal_dir="validation/abnormal",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=0,
    )
    dm.setup()

    model = instantiate_anomalib_model(method_clean, backbone=backbone, layers=layers, image_size=image_size)
    callbacks = [TQDMProgressBar(refresh_rate=20)] if show_progress else []
    engine = Engine(
        max_epochs=1,
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=show_progress,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    t0 = time.perf_counter()
    fit_status = "fit_completed"
    try:
        engine.fit(model=model, datamodule=dm, ckpt_path=None)
    except Exception as e:
        fit_status = f"fit_skipped_or_failed: {type(e).__name__}: {e}"
        print(fit_status)
    train_sec = time.perf_counter() - t0

    test_results = engine.test(model=model, datamodule=dm)
    metrics = save_anomalib_test_results(tracker, test_results)
    metrics["train_elapsed_sec"] = train_sec
    metrics["fit_status"] = fit_status
    tracker.save_metrics(metrics)

    try:
        ckpt_path = tracker.artifact(f"{method_clean}_manual_after_test.ckpt")
        engine.trainer.save_checkpoint(str(ckpt_path))
        save_json({"manual_checkpoint_path": str(ckpt_path)}, tracker.artifact("checkpoint_info.json"))
    except Exception as e:
        save_json(
            {"manual_checkpoint_path": None, "checkpoint_warning": f"{type(e).__name__}: {e}"},
            tracker.artifact("checkpoint_info.json"),
        )

    print(metrics)
    return tracker


# 04_run_advanced_model_suite

# %% 04_run_advanced_model_suite
def run_advanced_model_suite():
    trackers = {}
    candidate_settings = [
        ("efficientad", {"train_batch_size": 2, "eval_batch_size": 2}),
        ("reverse_distillation", {"train_batch_size": 2, "eval_batch_size": 2}),
        ("fastflow", {"train_batch_size": 2, "eval_batch_size": 2}),
        ("draem", {"train_batch_size": 2, "eval_batch_size": 2}),
        ("winclip", {"train_batch_size": 1, "eval_batch_size": 1}),
    ]
    for method, kwargs in candidate_settings:
        try:
            trackers[method] = train_advanced_anomalib_model(method, show_progress=False, **kwargs)
        except Exception as e:
            print(f"FAILED {method}: {type(e).__name__}: {e}")
            trackers[method] = None
    return trackers

# Optional execution example. Uncomment to run.
# Example:
# advanced_trackers = run_advanced_model_suite()

results = collect_results()

display(make_image_level_table(results))


# 11_Additional Analysis_2

# %% make_defect_type_preserving_dataset

import os, zipfile, shutil, re
from pathlib import Path
from tqdm.auto import tqdm

TYPE_DATASET_PATH = Path("/content/cable_dataset_by_defect")
TYPE_MARKER_DIR = TYPE_DATASET_PATH / "_unzip_markers"
TYPE_MARKER_DIR.mkdir(parents=True, exist_ok=True)

def archive_to_defect_type(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    if "Good" in parts:
        return "good"
    if len(parts) >= 4:
        return "_".join(parts[2:-1])
    return stem

def unzip_type_preserving_archives(set_type, force=False):
    folder_prefix = "TS_" if set_type == "Training" else "VS_"
    source_folder = Path(GDRIVE_DATA_ROOT) / f"{set_type}_image"
    files_to_unzip = [f for f in TARGET_CABLE_FILES if f.startswith(folder_prefix)]
    split = set_type.lower()

    for filename in tqdm(files_to_unzip, desc=f"type-preserving unzip {set_type}"):
        source_zip_path = source_folder / filename
        if not source_zip_path.exists():
            print("Missing:", source_zip_path)
            continue
        defect_type = archive_to_defect_type(filename)
        if defect_type == "good":
            dest = TYPE_DATASET_PATH / split / "good"
        else:
            dest = TYPE_DATASET_PATH / split / "abnormal" / defect_type
        dest.mkdir(parents=True, exist_ok=True)
        marker = TYPE_MARKER_DIR / f"{set_type}_{filename}.done"
        if marker.exists() and not force:
            continue
        with zipfile.ZipFile(source_zip_path, "r") as z:
            z.extractall(dest)
        marker.write_text(f"source={source_zip_path}\ndest={dest}\n", encoding="utf-8")

def summarize_type_dataset(root=TYPE_DATASET_PATH):
    rows = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            parts = p.parts
            split = "training" if "training" in parts else "validation" if "validation" in parts else "unknown"
            label = 0 if "good" in parts else 1
            defect_type = "good"
            if "abnormal" in parts:
                idx = parts.index("abnormal")
                defect_type = parts[idx + 1] if idx + 1 < len(parts) else "abnormal_unknown"
            rows.append({"path": str(p), "split": split, "label": label, "defect_type": defect_type})
    df = pd.DataFrame(rows)
    out = MANIFEST_ROOT / "type_preserving_manifest.csv"
    df.to_csv(out, index=False)
    display(df.groupby(["split", "label", "defect_type"]).size().reset_index(name="n"))
    print("Saved:", out)
    return df

# Optional execution example. Uncomment to run.
# Run once if needed:
# unzip_type_preserving_archives("Training")
# unzip_type_preserving_archives("Validation")
# type_manifest = summarize_type_dataset()


# 01_unknown_defect_split

# %% unknown_defect_split
def make_unknown_defect_split(type_manifest, heldout_defect_type):
    df = type_manifest.copy()
    train = df[df["split"].eq("training")].copy()
    test = df[df["split"].eq("validation")].copy()
    train_known = train[(train["label"].eq(0)) | (~train["defect_type"].eq(heldout_defect_type))].copy()
    test_unknown = test[(test["label"].eq(0)) | (test["defect_type"].eq(heldout_defect_type))].copy()
    train_known["split_unknown"] = "train_known"
    test_unknown["split_unknown"] = "test_heldout"
    out = pd.concat([train_known, test_unknown], ignore_index=True)
    path = MANIFEST_ROOT / f"unknown_defect_split__heldout_{heldout_defect_type}.csv"
    out.to_csv(path, index=False)
    print("Saved:", path)
    display(out.groupby(["split_unknown", "label", "defect_type"]).size().reset_index(name="n"))
    return out

def make_unknown_defect_loaders(split_df, preprocess_name="resize224_imagenet", batch_size=32):
    transform = PREPROCESSING[preprocess_name]
    train_df = split_df[split_df["split_unknown"].eq("train_known")].copy()
    test_df = split_df[split_df["split_unknown"].eq("test_heldout")].copy()
    train_ds = BinaryImagePathDataset(train_df, transform)
    test_ds = BinaryImagePathDataset(test_df, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def train_unknown_defect_supervised(model_name, heldout_defect_type, type_manifest, epochs=5, batch_size=32):
    split_df = make_unknown_defect_split(type_manifest, heldout_defect_type)
    train_loader, test_loader = make_unknown_defect_loaders(split_df, batch_size=batch_size)
    experiment_name = f"unknown_heldout_{heldout_defect_type}_ep{epochs}"
    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "unknown_defect_split",
            "paradigm": "supervised_unknown_defect",
            "heldout_defect_type": heldout_defect_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "preprocessing": "resize224_imagenet",
        },
    )
    if tracker.metrics_path.exists():
        print("Existing unknown-defect result found:", tracker.run_dir)
        return tracker
    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        n = 0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1)})
        print(model_name, heldout_defect_type, epoch, history[-1])
    tracker.save_checkpoint(model, "model_state_dict.pt")
    save_training_history(tracker, history)
    metrics, _ = evaluate_and_save_supervised(model, test_loader, model_name, experiment_name, config=tracker.config, checkpoint=False)
    save_efficiency_report(model, tracker, input_shape=input_shape)
    print(metrics)
    return tracker

# Optional execution example. Uncomment to run.
# Example:
# type_manifest = pd.read_csv(MANIFEST_ROOT / "type_preserving_manifest.csv")

unknown_tracker = train_unknown_defect_supervised("resnet18", "CableInstall_Defect", type_manifest)


# 02_localization_from_json_masks

# %% 04_localization_from_json_masks
import cv2
import json
import numpy as np
from pathlib import Path

def guess_label_path(image_path):
    p = Path(image_path)
    candidates = []
    s = str(p)
    candidates.append(s.replace("/training/good/", "/training/good_label/").replace(".jpg", ".json"))
    candidates.append(s.replace("/training/abnormal/", "/training/abnormal_label/").replace(".jpg", ".json"))
    candidates.append(s.replace("/validation/good/", "/validation/good_label/").replace(".jpg", ".json"))
    candidates.append(s.replace("/validation/abnormal/", "/validation/abnormal_label/").replace(".jpg", ".json"))
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def mask_from_aihub_json(json_path, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"Warning: Could not decode or parse JSON from {json_path}: {e}")
        return mask # Return an empty mask if the JSON is invalid

    anns = data.get("annotations", [])
    for ann in anns:
        seg = ann.get("segmentation")
        bbox = ann.get("bbox")
        if seg:
            if isinstance(seg, list) and len(seg) > 0:
                # COCO polygon list or flat polygon.
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    # Check if polygon points are valid (at least 3 points, i.e., 6 coordinates)
                    if len(poly) % 2 == 0 and len(poly) >= 6:
                        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 1)
                    else:
                        print(f"Warning: Invalid segmentation polygon data in {json_path} for annotation {ann.get('id', 'N/A')}")
            else:
                 print(f"Warning: Unexpected segmentation format in {json_path} for annotation {ann.get('id', 'N/A')}")
        elif bbox:
            # Ensure bbox has 4 elements and they are convertible to int
            if isinstance(bbox, list) and len(bbox) >= 4:
                try:
                    x, y, w, h = [int(v) for v in bbox[:4]]
                    # Ensure bbox coordinates are valid (non-negative)
                    if x >= 0 and y >= 0 and w >= 0 and h >= 0:
                        mask[y:y+h, x:x+w] = 1
                    else:
                        print(f"Warning: Invalid bbox coordinates in {json_path} for annotation {ann.get('id', 'N/A')}")
                except ValueError as ve:
                    print(f"Warning: Could not parse bbox coordinates in {json_path} for annotation {ann.get('id', 'N/A')}: {ve}")
            else:
                print(f"Warning: Invalid bbox format in {json_path} for annotation {ann.get('id', 'N/A')}")
    return mask

def create_mask_audit_for_predictions(run_dir, max_images=200):
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        print("No predictions.csv:", run_dir)
        return pd.DataFrame()

    pred_df = pd.read_csv(pred_path)
    rows = []
    processed_images = 0
    for idx, r in pred_df.iterrows():
        if processed_images >= max_images:
            break
        path = r.get("path")
        if not isinstance(path, str) or not Path(path).exists():
            rows.append({"path": path, "has_label_json": False, "mask_area": np.nan, "error": "Image file not found or path invalid"})
            processed_images += 1
            continue

        label_path = guess_label_path(path)
        if label_path is None:
            rows.append({"path": path, "has_label_json": False, "mask_area": np.nan, "error": "No corresponding label JSON found"})
            processed_images += 1
            continue

        try:
            img = cv2.imread(path)
            if img is None:
                rows.append({"path": path, "label_json": label_path, "has_label_json": True, "mask_area": np.nan, "error": "Could not read image file with OpenCV"})
                processed_images += 1
                continue

            mask = mask_from_aihub_json(label_path, img.shape)
            rows.append({"path": path, "label_json": label_path, "has_label_json": True, "mask_area": int(mask.sum()), "error": None})
            processed_images += 1
        except Exception as e:
            # Catch any other unexpected errors during image/mask processing
            rows.append({"path": path, "label_json": label_path, "has_label_json": True, "mask_area": np.nan, "error": f"Error during image or mask processing: {e}"})
            processed_images += 1

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "mask_audit.csv", index=False)
    display(out.head())
    print("Mask JSON coverage:", out["has_label_json"].mean() if len(out) else 0)
    return out

# Optional execution example. Uncomment to run.
# Example:
# pred_runs = [p for p in Path(RUN_ROOT).glob("*") if (p / "predictions.csv").exists()]
# if pred_runs:
#      mask_audit = create_mask_audit_for_predictions(pred_runs[0])


# 03_Result_Tracker

# %% 00_diagnose_missing_results
def diagnose_run_coverage(df):
    print("Completed runs by paradigm")
    if "paradigm" in df:
        display(df.groupby("paradigm", dropna=False).size().reset_index(name="n_runs"))
    else:
        print("No paradigm column found.")

    expected = ["supervised", "reconstruction_anomaly", "embedding_anomaly", "label_ratio_supervised"]
    present = set(df["paradigm"].dropna()) if "paradigm" in df else set()
    missing = [x for x in expected if x not in present]
    print("Missing paradigms:", missing)

    artifact_cols = ["has_predictions", "has_confusion_matrix", "has_roc_curve", "has_pr_curve", "has_efficiency"]
    cols = [c for c in artifact_cols if c in df.columns]
    if cols:
        display(df[["model_name", "experiment_name", "paradigm"] + cols])

    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    if metric_cols:
        null_report = df[["model_name", "experiment_name", "paradigm"] + metric_cols].copy()
        display(null_report)

diagnose_run_coverage(results if "results" in globals() else collect_results())

# Optional execution example. Uncomment to run.
# %% run_minimum_experiments
# def run_minimum__experiments():
#     trackers = {}
#
    # Supervised baselines
#     trackers["resnet18"] = train_supervised_model(
#         "resnet18",
#         preprocess_name="resize224_imagenet",
#         epochs=5,
#         batch_size=32,
#     )
#     trackers["efficientnet_b0"] = train_supervised_model(
#         "efficientnet_b0",
#         preprocess_name="resize224_imagenet",
#         epochs=5,
#         batch_size=32,
#     )
#
    # Reconstruction anomaly baseline
#     trackers["simple_conv_autoencoder"] = train_reconstruction_anomaly_model(
#         epochs=5,
#         train_batch_size=16,
#         eval_batch_size=32,
#     )
#
    # Embedding anomaly baselines. PatchCore is intentionally low-memory.
#     trackers["patchcore"] = train_anomalib_embedding_model(
#         "patchcore",
#         layers=["layer2"],
#         train_batch_size=1,
#         eval_batch_size=1,
#         show_progress=False,
#     )
#     trackers["padim"] = train_anomalib_embedding_model(
#         "padim",
#         layers=["layer2"],
#         train_batch_size=4,
#         eval_batch_size=4,
#         show_progress=False,
#     )
#
    # Label-efficiency experiment
#     trackers["semi_resnet18"] = run_label_ratio_grid(
#         "resnet18",
#         ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.0),
#         preprocess_name="resize224_imagenet",
#         epochs=5,
#         batch_size=32,
#     )
#     return trackers

trackers = run_minimum__experiments()

trackers

results = collect_results()
results

quality_audit = audit_result_quality(results)
display(make_image_level_table(results))

plot_metric_bars(results)

plot_efficiency_tradeoff(results)

resnet_tracker = train_supervised_model("resnet18", epochs=5, batch_size=32)

eff_tracker = train_supervised_model("efficientnet_b0", epochs=5, batch_size=32)

ae_tracker = train_reconstruction_anomaly_model(epochs=5)

semi_trackers = run_label_ratio_grid(
    "resnet18",
    ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.0),
    epochs=5
)


# 12_Additional Analysis_E2_group_calibrated

# %% E2_constants
PROTOCOL_VERSION = "E2_group_calibrated"
SEEDS_V2 = [11, 22, 33, 44, 55]

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def protocol_experiment_name(base, seed, extra=None):
    pieces = [PROTOCOL_VERSION, f"seed{seed}", base]
    if extra:
        pieces.append(str(extra))
    return "_".join(pieces)


# 01_split_validation_and_reporting

# %% split_validation_and_reporting
def split_report(df, split_col, defect_col="defect_type", group_col="dup_group", archive_col="source_archive"):
    rows = []
    for split_name, part in df.groupby(split_col):
        n = len(part)
        n_normal = int((part["label"] == 0).sum())
        n_abnormal = int((part["label"] == 1).sum())
        rows.append({
            "split": split_name,
            "n": int(n),
            "n_normal": n_normal,
            "n_abnormal": n_abnormal,
            "abnormal_prevalence": float(n_abnormal / max(n, 1)),
            "n_defect_types": int(part.loc[part["label"].eq(1), defect_col].nunique()) if defect_col in part else np.nan,
            "n_groups": int(part[group_col].nunique()) if group_col in part else np.nan,
            "n_source_archives": int(part[archive_col].nunique()) if archive_col in part else np.nan,
        })
    summary = pd.DataFrame(rows)
    display(summary)

    if defect_col in df.columns:
        display(df.groupby([split_col, "label", defect_col]).size().reset_index(name="n"))
    if archive_col in df.columns:
        display(df.groupby([split_col, archive_col]).size().reset_index(name="n").head(50))
    if group_col in df.columns:
        crossing = df.groupby(group_col)[split_col].nunique().gt(1).sum()
        print("Duplicate/near-duplicate group crossing count:", int(crossing))
    return summary

def validate_protocol_split(df, split_col, required_splits=("train", "calibration", "test"), defect_col="defect_type", group_col="dup_group"):
    issues = []
    present = set(df[split_col].dropna().unique())
    for s in required_splits:
        if s not in present:
            issues.append(f"missing_split:{s}")
            continue
        part = df[df[split_col].eq(s)]
        if (part["label"] == 0).sum() == 0:
            issues.append(f"{s}:missing_normal")
        if (part["label"] == 1).sum() == 0:
            issues.append(f"{s}:missing_abnormal")
    if group_col in df.columns:
        crossing = df.groupby(group_col)[split_col].nunique().gt(1).sum()
        if crossing > 0:
            issues.append(f"duplicate_group_crossing:{int(crossing)}")
    if defect_col in df.columns and "test" in present:
        if df[df[split_col].eq("test") & df["label"].eq(1)][defect_col].nunique() == 0:
            issues.append("test:missing_defect_type")
    return issues

# %% stratified_group_candidate_split
def choose_stratified_group_split(
    df,
    seed=SEED,
    test_size=0.20,
    calib_size=0.20,
    group_col="dup_group",
    defect_col="defect_type",
    n_candidates=200,
):
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    data = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
    data["target_key"] = np.where(data["label"].eq(0), "good", data[defect_col].fillna("abnormal"))
    global_prev = float(data["label"].mean())
    global_types = set(data.loc[data["label"].eq(1), defect_col].dropna().unique())
    rng = np.random.default_rng(seed)

    best = None
    best_penalty = float("inf")
    for i in range(n_candidates):
        candidate_seed = int(rng.integers(0, 1_000_000_000))
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size + calib_size, random_state=candidate_seed)
        train_idx, temp_idx = next(outer.split(data, data["label"], groups=data[group_col]))
        train = data.iloc[train_idx].copy()
        temp = data.iloc[temp_idx].copy()
        inner_test_size = test_size / (test_size + calib_size)
        inner = GroupShuffleSplit(n_splits=1, test_size=inner_test_size, random_state=candidate_seed + 1)
        calib_rel, test_rel = next(inner.split(temp, temp["label"], groups=temp[group_col]))
        calib = temp.iloc[calib_rel].copy()
        test = temp.iloc[test_rel].copy()

        parts = {"train": train, "calibration": calib, "test": test}
        penalty = 0.0
        for split_name, part in parts.items():
            n = len(part)
            n_pos = int(part["label"].sum())
            n_neg = int((part["label"] == 0).sum())
            if n_pos == 0 or n_neg == 0:
                penalty += 1000
            penalty += abs(float(part["label"].mean()) - global_prev) * 10
            if split_name in ["calibration", "test"]:
                types = set(part.loc[part["label"].eq(1), defect_col].dropna().unique())
                missing_types = len(global_types - types)
                penalty += missing_types * 2
        if penalty < best_penalty:
            best_penalty = penalty
            best = (candidate_seed, parts)

    candidate_seed, parts = best
    out_parts = []
    for split_name, part in parts.items():
        p = part.copy()
        p["protocol_split"] = split_name
        p["protocol_version"] = PROTOCOL_VERSION
        p["protocol_seed"] = seed
        p["split_candidate_seed"] = candidate_seed
        out_parts.append(p)
    out = pd.concat(out_parts, ignore_index=True)
    issues = validate_protocol_split(out, "protocol_split", group_col=group_col, defect_col=defect_col)
    out_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print("Split issues:", issues if issues else "none")
    split_report(out, "protocol_split", defect_col=defect_col, group_col=group_col)
    return out

def make_E2_split(seed=SEED, n_candidates=200):
    groups = build_duplicate_groups()
    if "source_archive" not in groups.columns:
        groups["source_archive"] = groups["path"].map(lambda p: Path(p).parts[-3] if len(Path(p).parts) >= 3 else "unknown")
    if "defect_type" not in groups.columns:
        groups["defect_type"] = np.where(groups["label"].eq(0), "good", groups.get("class_name", "abnormal"))
    return choose_stratified_group_split(groups, seed=seed, n_candidates=n_candidates)

# %% loaders_from_protocol_split
def make_loader_from_df(split_df, split_name, preprocess_name="resize224_imagenet", batch_size=32, shuffle=False, num_workers=2):
    part = split_df[split_df["protocol_split"].eq(split_name)].copy()
    transform = PREPROCESSING[preprocess_name]
    ds = BinaryImagePathDataset(part, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ), part

def make_label_efficient_train_df(split_df, label_ratio, seed):
    train = split_df[split_df["protocol_split"].eq("train")].copy()
    sampled = []
    for label, part in train.groupby("label"):
        n = max(1, int(math.ceil(len(part) * label_ratio)))
        sampled.append(part.sample(n=min(n, len(part)), random_state=seed))
    out = pd.concat(sampled, ignore_index=True)
    out["label_ratio"] = label_ratio
    return out

def make_loader_from_arbitrary_df(df, preprocess_name="resize224_imagenet", batch_size=32, shuffle=False, num_workers=2):
    transform = PREPROCESSING[preprocess_name]
    ds = BinaryImagePathDataset(df.copy(), transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# %% common_calibrated_score_saving
def save_calibrated_score_evaluation(
    tracker,
    calib_scores,
    test_scores,
    threshold_method="youden",
    extra_config=None,
):
    calib_scores = calib_scores.copy()
    test_scores = test_scores.copy()
    calib_scores.to_csv(tracker.artifact("predictions_calibration.csv"), index=False)
    test_scores[["path", "y_true", "y_score"]].to_csv(tracker.artifact("scores_test_unthresholded.csv"), index=False)
    metrics, pred_df = evaluate_with_frozen_threshold(
        tracker,
        calib_scores,
        test_scores,
        threshold_method=threshold_method,
        score_col="y_score",
    )
    cfg = load_json(tracker.config_path, default={})
    cfg.update({
        "protocol_version": PROTOCOL_VERSION,
        "threshold_source": "calibration",
        "threshold_method": threshold_method,
        "group_safe_split": True,
        "has_calibration_predictions": True,
    })
    if extra_config:
        cfg.update(extra_config)
    save_json(cfg, tracker.config_path)
    return metrics, pred_df

def truthy_config_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


# 02_supervised_E2

# %% supervised_E2
def train_supervised_model_E2(
    model_name="resnet18",
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    seed=SEED,
    threshold_method="youden",
    overwrite=False,
):
    set_all_seeds(seed)
    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    issues = validate_protocol_split(split_df, "protocol_split")
    if issues:
        raise RuntimeError(f"Invalid protocol split for seed {seed}: {issues}")

    experiment_name = protocol_experiment_name(
        f"supervised_{preprocess_name}_ep{epochs}_bs{batch_size}",
        seed,
        model_name,
    )
    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "04_supervised_models",
            "paradigm": "supervised",
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    if tracker.metrics_path.exists() and not overwrite:
        print("Existing E2 supervised result found:", tracker.run_dir)
        return tracker

    clear_gpu()
    train_loader, _ = make_loader_from_df(split_df, "train", preprocess_name, batch_size, shuffle=True)
    calib_loader, _ = make_loader_from_df(split_df, "calibration", preprocess_name, batch_size, shuffle=False)
    test_loader, _ = make_loader_from_df(split_df, "test", preprocess_name, batch_size, shuffle=False)

    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_seen = 0
        for x, y, _paths in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_seen += bs
        epoch_loss = total_loss / max(n_seen, 1)
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": epoch_loss, "elapsed_sec": elapsed})
        print(f"{model_name} seed={seed} epoch {epoch}/{epochs} loss={epoch_loss:.5f} sec={elapsed:.1f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            tracker.save_checkpoint(model, "best_model_state_dict.pt")

    tracker.save_checkpoint(model, "model_state_dict.pt")
    save_training_history(tracker, history)
    save_json({"input_shape": input_shape}, tracker.artifact("model_meta.json"))

    calib_scores = collect_supervised_scores(model, calib_loader)
    test_scores = collect_supervised_scores(model, test_loader)
    metrics, _ = save_calibrated_score_evaluation(
        tracker,
        calib_scores,
        test_scores,
        threshold_method=threshold_method,
        extra_config={
            "seed": seed,
            "n_test_normal": int((test_scores.y_true == 0).sum()),
            "n_test_abnormal": int((test_scores.y_true == 1).sum()),
        },
    )
    save_efficiency_report(model, tracker, input_shape=input_shape)
    print(metrics)
    return tracker


# 03_label_efficient_E2

# %% label_efficient_E2
def train_label_efficient_supervised_E2(
    model_name="resnet18",
    label_ratio=0.05,
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    seed=SEED,
    threshold_method="youden",
    overwrite=False,
):
    set_all_seeds(seed)
    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    labeled_train = make_label_efficient_train_df(split_df, label_ratio, seed)

    experiment_name = protocol_experiment_name(
        f"label_efficient_ratio{label_ratio:g}_{preprocess_name}_ep{epochs}_bs{batch_size}",
        seed,
        model_name,
    )
    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "06_label_efficient_supervised_models",
            "paradigm": "label_efficient_supervised",
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "label_ratio": label_ratio,
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "uses_unlabeled_data": False,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    if tracker.metrics_path.exists() and not overwrite:
        print("Existing E2 label-efficient result found:", tracker.run_dir)
        return tracker

    clear_gpu()
    train_loader = make_loader_from_arbitrary_df(labeled_train, preprocess_name, batch_size, shuffle=True)
    calib_loader, _ = make_loader_from_df(split_df, "calibration", preprocess_name, batch_size, shuffle=False)
    test_loader, _ = make_loader_from_df(split_df, "test", preprocess_name, batch_size, shuffle=False)

    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_seen = 0
        for x, y, _paths in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_seen += bs
        epoch_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "elapsed_sec": time.perf_counter() - t0})
        print(f"{model_name} label_ratio={label_ratio:g} seed={seed} epoch {epoch}/{epochs} loss={epoch_loss:.5f}")

    tracker.save_checkpoint(model, "model_state_dict.pt")
    save_training_history(tracker, history)
    save_json({"input_shape": input_shape, "n_labeled_train": int(len(labeled_train))}, tracker.artifact("model_meta.json"))

    calib_scores = collect_supervised_scores(model, calib_loader)
    test_scores = collect_supervised_scores(model, test_loader)
    metrics, _ = save_calibrated_score_evaluation(
        tracker,
        calib_scores,
        test_scores,
        threshold_method=threshold_method,
        extra_config={
            "seed": seed,
            "n_labeled_train": int(len(labeled_train)),
            "n_test_normal": int((test_scores.y_true == 0).sum()),
            "n_test_abnormal": int((test_scores.y_true == 1).sum()),
        },
    )
    save_efficiency_report(model, tracker, input_shape=input_shape)
    print(metrics)
    return tracker

# %% repeated_run_entrypoints
def run_E2_supervised_suite(
    model_names=("resnet18", "efficientnet_b0", "convnext_tiny"),
    seeds=SEEDS_V2[:3],
    epochs=5,
    batch_size=32,
    overwrite=False,
):
    trackers = []
    for seed in seeds:
        make_E2_split(seed=seed)
        for model_name in model_names:
            trackers.append(train_supervised_model_E2(
                model_name=model_name,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                overwrite=overwrite,
            ))
    return trackers

def run_E2_label_efficiency_suite(
    model_name="resnet18",
    ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.00),
    seeds=SEEDS_V2[:3],
    epochs=5,
    batch_size=32,
    overwrite=False,
):
    trackers = []
    for seed in seeds:
        make_E2_split(seed=seed)
        for ratio in ratios:
            trackers.append(train_label_efficient_supervised_E2(
                model_name=model_name,
                label_ratio=ratio,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                overwrite=overwrite,
            ))
    return trackers


# 04_calibrate_existing_prediction_artifacts

# %% calibrate_existing_prediction_artifacts
def calibrate_existing_score_files(
    model_name,
    experiment_name,
    calibration_csv,
    test_csv,
    threshold_method="youden",
    paradigm="embedding_anomaly",
    seed=None,
    overwrite=True,
):
    calib = pd.read_csv(calibration_csv)
    test = pd.read_csv(test_csv)
    required = {"path", "y_true", "y_score"}
    if not required.issubset(calib.columns) or not required.issubset(test.columns):
        raise ValueError("Both calibration_csv and test_csv must contain path, y_true, y_score")
    tracker = ExperimentTracker(
        model_name,
        protocol_experiment_name(experiment_name, seed or "noseed", "calibrated"),
        config={
            "section": "07_metrics_image_level",
            "paradigm": paradigm,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "threshold_source": "calibration",
            "threshold_method": threshold_method,
            "group_safe_split": True,
            "score_source": "external_prediction_csv",
        },
        overwrite=overwrite,
    )
    return save_calibrated_score_evaluation(tracker, calib, test, threshold_method=threshold_method)


# 05_anomalib_E2

# %% anomalib_E2_score_export
def link_or_copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        import shutil
        shutil.copy2(src, dst)

def materialize_E2_anomalib_folder(split_df, seed, root=None):
    if root is None:
        root = PROJECT_ROOT / "E2_anomalib_folders" / f"seed{seed}"
    root = Path(root)
    mapping = {
        ("train", 0): root / "train" / "good",
        ("calibration", 0): root / "calibration" / "good",
        ("calibration", 1): root / "calibration" / "abnormal",
        ("test", 0): root / "test" / "good",
        ("test", 1): root / "test" / "abnormal",
    }
    for dst in mapping.values():
        dst.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for _, row in split_df.iterrows():
        split = row["protocol_split"]
        label = int(row["label"])
        if (split, label) not in mapping:
            continue
        src = Path(row["path"])
        rel_name = f"{src.stem}_{hashlib.md5(str(src).encode('utf-8')).hexdigest()[:10]}{src.suffix}"
        dst = mapping[(split, label)] / rel_name
        link_or_copy_file(src, dst)
        r = row.to_dict()
        r["materialized_path"] = str(dst)
        manifest_rows.append(r)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = root / "E2_anomalib_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("Saved Anomalib folder view:", root)
    split_report(manifest.rename(columns={"materialized_path": "path"}), "protocol_split")
    return root, manifest

def extract_scalar_from_prediction_item(item, keys):
    for key in keys:
        value = None
        if isinstance(item, dict) and key in item:
            value = item[key]
        elif hasattr(item, key):
            value = getattr(item, key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
            return float(value.reshape(-1)[0].item())
        if isinstance(value, (np.ndarray, list, tuple)):
            arr = np.asarray(value).reshape(-1)
            if len(arr):
                return float(arr[0])
        try:
            return float(value)
        except Exception:
            pass
    return np.nan

def extract_path_from_prediction_item(item):
    path_keys = ["image_path", "path", "image_paths", "paths"]
    for key in path_keys:
        value = None
        if isinstance(item, dict) and key in item:
            value = item[key]
        elif hasattr(item, key):
            value = getattr(item, key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value):
            return str(value[0])
        if isinstance(value, np.ndarray) and value.size:
            return str(value.reshape(-1)[0])
        return str(value)
    return None

def anomalib_predictions_to_score_df(predictions, materialized_manifest=None):
    rows = []
    for item in predictions:
        if isinstance(item, list):
            items = item
        else:
            items = [item]
        for x in items:
            path = extract_path_from_prediction_item(x)
            y_true = extract_scalar_from_prediction_item(x, ["gt_label", "label", "labels", "target"])
            y_score = extract_scalar_from_prediction_item(x, ["pred_score", "anomaly_score", "score", "scores"])
            rows.append({"materialized_path": path, "y_true": y_true, "y_score": y_score})
    df = pd.DataFrame(rows)
    if materialized_manifest is not None and not df.empty:
        meta = materialized_manifest[["path", "materialized_path", "label", "protocol_split"]].copy()
        meta["materialized_path"] = meta["materialized_path"].astype(str)
        df["materialized_path"] = df["materialized_path"].astype(str)
        df = df.merge(meta, on="materialized_path", how="left", suffixes=("", "_meta"))
        df["path"] = df["path"].fillna(df["materialized_path"])
        if "label" in df:
            df["y_true"] = df["y_true"].fillna(df["label"])
    if "path" not in df.columns:
        df["path"] = df["materialized_path"]
    df = df[["path", "y_true", "y_score"]].copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_score"] = df["y_score"].astype(float)
    return df

def build_anomalib_folder_datamodule(folder_root, eval_split="calibration", train_batch_size=1, eval_batch_size=1, num_workers=0):
    from anomalib.data import Folder
    return Folder(
        name=f"lng_cable_{PROTOCOL_VERSION}_{eval_split}",
        root=str(folder_root),
        normal_dir="train/good",
        normal_test_dir=f"{eval_split}/good",
        abnormal_dir=f"{eval_split}/abnormal",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

def train_anomalib_embedding_model_E2(
    method="patchcore",
    backbone="resnet18",
    layers=None,
    coreset_sampling_ratio=0.005,
    num_neighbors=1,
    train_batch_size=1,
    eval_batch_size=1,
    seed=SEED,
    threshold_method="youden",
    accelerator=None,
    show_progress=False,
    overwrite=False,
):
    from anomalib.engine import Engine
    from anomalib.models import Patchcore, Padim, Stfpm
    from lightning.pytorch.callbacks import TQDMProgressBar

    set_all_seeds(seed)
    method = method.lower()
    if layers is None:
        layers = ["layer2"] if method == "patchcore" else ["layer1", "layer2", "layer3"]

    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    folder_root, materialized_manifest = materialize_E2_anomalib_folder(split_df, seed)

    experiment_name = protocol_experiment_name(
        f"{method}_{backbone}_{'-'.join(layers)}_bs{train_batch_size}",
        seed,
        "embedding_anomaly",
    )
    tracker = ExperimentTracker(
        f"{method}_{backbone}",
        experiment_name,
        config={
            "section": "05_unsupervised_anomaly_models",
            "paradigm": "embedding_anomaly",
            "method": method,
            "backbone": backbone,
            "layers": layers,
            "coreset_sampling_ratio": coreset_sampling_ratio if method == "patchcore" else None,
            "num_neighbors": num_neighbors if method == "patchcore" else None,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    if tracker.metrics_path.exists() and tracker.pred_path.exists() and not overwrite:
        print("Existing E2 anomalib result found:", tracker.run_dir)
        return tracker

    clear_gpu()
    dm_train_calib = build_anomalib_folder_datamodule(
        folder_root,
        eval_split="calibration",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    dm_train_calib.setup()

    if method == "patchcore":
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
    elif method == "padim":
        model = Padim(backbone=backbone, layers=layers)
    elif method == "stfpm":
        model = Stfpm(backbone=backbone, layers=layers)
    else:
        raise ValueError("method must be one of: patchcore, padim, stfpm")

    callbacks = [TQDMProgressBar(refresh_rate=20)] if show_progress else []
    engine = Engine(
        max_epochs=1,
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=show_progress,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm_train_calib, ckpt_path=None)
    train_sec = time.perf_counter() - t0
        # Predict calibration scores.
    calib_predictions = engine.predict(model=model, datamodule=dm_train_calib)
    calib_scores = anomalib_predictions_to_score_df(calib_predictions, materialized_manifest)
    calib_scores = calib_scores[calib_scores["path"].isin(
        materialized_manifest.loc[materialized_manifest["protocol_split"].eq("calibration"), "path"]
    )].copy()

    # Predict held-out test scores using the same fitted model/memory bank.
    dm_test = build_anomalib_folder_datamodule(
        folder_root,
        eval_split="test",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    dm_test.setup()
    test_predictions = engine.predict(model=model, datamodule=dm_test)
    test_scores = anomalib_predictions_to_score_df(test_predictions, materialized_manifest)
    test_scores = test_scores[test_scores["path"].isin(
        materialized_manifest.loc[materialized_manifest["protocol_split"].eq("test"), "path"]
    )].copy()

    if calib_scores.empty or test_scores.empty:
        calib_scores.to_csv(tracker.artifact("debug_calibration_scores_empty.csv"), index=False)
        test_scores.to_csv(tracker.artifact("debug_test_scores_empty.csv"), index=False)
        raise RuntimeError(
            "Anomalib prediction parsing produced empty calibration/test scores. "
            "Inspect Engine.predict output keys and update anomalib_predictions_to_score_df()."
        )

    metrics, _ = save_calibrated_score_evaluation(
        tracker,
        calib_scores,
        test_scores,
        threshold_method=threshold_method,
        extra_config={
            "seed": seed,
            "train_elapsed_sec": train_sec,
            "n_test_normal": int((test_scores.y_true == 0).sum()),
            "n_test_abnormal": int((test_scores.y_true == 1).sum()),
        },
    )
    metrics["train_elapsed_sec"] = train_sec
    tracker.save_metrics(metrics)
    print(metrics)
    return tracker

def train_patchcore_E2_and_export_scores(seed=11, layers=("layer2",), train_batch_size=1, eval_batch_size=1, overwrite=False, show_progress=False):
    return train_anomalib_embedding_model_E2(
        method="patchcore",
        layers=list(layers),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        overwrite=overwrite,
        show_progress=show_progress,
    )

# %% lodo_result_validation_helpers
def split_known_unknown_lodo_results(pred_df, lodo_df, heldout_defect_type):
    cols = ["path", "heldout_defect_type", "lodo_split", "lodo_role", "defect_type"]
    meta = lodo_df[lodo_df["heldout_defect_type"].eq(heldout_defect_type)][cols].drop_duplicates("path")
    merged = pred_df.merge(meta, on="path", how="left")
    known = merged[merged["lodo_split"].eq("known_test")].copy()
    unknown = merged[merged["lodo_split"].eq("unknown_test")].copy()
    return known, unknown

def validate_lodo_artifact(pred_df, heldout_defect_type):
    issues = []
    if pred_df["y_true"].nunique() < 2:
        issues.append("single_class_prediction_set")
    if (pred_df["y_true"] == 1).sum() == 0:
        issues.append("no_abnormal_samples")
    if (pred_df["y_true"] == 0).sum() == 0:
        issues.append("no_normal_samples")
    if "defect_type" in pred_df and heldout_defect_type not in set(pred_df["defect_type"].dropna()):
        issues.append("heldout_defect_type_missing")
    return issues

# %% localization_metric_from_saved_maps
def compute_localization_metrics_from_arrays(pred_maps, gt_masks, threshold=None):
    flat_scores = []
    flat_true = []
    dice_vals = []
    iou_vals = []
    region_recalls = []
    for score_map, mask in zip(pred_maps, gt_masks):
        score = np.asarray(score_map, dtype=float)
        gt = np.asarray(mask, dtype=np.uint8)
        if gt.shape != score.shape:
            gt = resize_mask_nearest(gt, score.shape)
        flat_scores.append(score.reshape(-1))
        flat_true.append(gt.reshape(-1))
        if threshold is not None:
            pred = (score >= threshold).astype(np.uint8)
            inter = int((pred & gt).sum())
            pred_sum = int(pred.sum())
            gt_sum = int(gt.sum())
            dice_vals.append(float(2 * inter / max(pred_sum + gt_sum, 1)))
            iou_vals.append(float(inter / max(int((pred | gt).sum()), 1)))
            region_recalls.append(float(inter / max(gt_sum, 1)) if gt_sum > 0 else np.nan)
    y_true = np.concatenate(flat_true)
    y_score = np.concatenate(flat_scores)
    out = {
        "pixel_auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan,
        "pixel_auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan,
    }
    if threshold is not None:
        out.update({
            "dice": float(np.nanmean(dice_vals)),
            "iou": float(np.nanmean(iou_vals)),
            "region_recall": float(np.nanmean(region_recalls)),
        })
    return out


# 13_Additional Analysis_E3_group_calibrated

# %% E3_t_constants
E3 = "E3_t_E2"
REQUIRED_COMPLETE_FILES = [
    "config.json",
    "metrics.json",
    "predictions.csv",
    "predictions_calibration.csv",
    "threshold_protocol.json",
]

def run_dir_for(model_name, experiment_name):
    return RUN_ROOT / f"{model_name}__{experiment_name}"

def artifact_complete(run_dir, required_files=REQUIRED_COMPLETE_FILES):
    run_dir = Path(run_dir)
    missing = [name for name in required_files if not (run_dir / name).exists()]
    if missing:
        return False, missing
    try:
        metrics = load_json(run_dir / "metrics.json", default={})
        config = load_json(run_dir / "config.json", default={})
        pred = pd.read_csv(run_dir / "predictions.csv")
        calib = pd.read_csv(run_dir / "predictions_calibration.csv")
    except Exception as e:
        return False, [f"unreadable_artifact:{type(e).__name__}"]
    issues = []
    if config.get("protocol_version") != PROTOCOL_VERSION:
        issues.append("wrong_protocol_version")
    if config.get("threshold_source") != "calibration":
        issues.append("threshold_not_calibrated")
    if pred.empty:
        issues.append("empty_predictions")
    if calib.empty:
        issues.append("empty_calibration_predictions")
    if pred["y_true"].nunique() < 2:
        issues.append("single_class_test")
    if pd.isna(metrics.get("auroc")) and pd.isna(metrics.get("image_AUROC")):
        issues.append("missing_auroc")
    if pd.isna(metrics.get("auprc")) and pd.isna(metrics.get("image_AUPRC")):
        issues.append("missing_auprc")
    return len(issues) == 0, issues

def mark_run_status(run_dir, status, issues=None, extra=None):
    payload = {
        "status": status,
        "updated_at": now_id(),
        "issues": issues or [],
    }
    if extra:
        payload.update(extra)
    save_json(payload, Path(run_dir) / "run_status.json")

def should_skip_run(model_name, experiment_name, overwrite=False):
    run_dir = run_dir_for(model_name, experiment_name)
    if overwrite:
        return False
    ok, issues = artifact_complete(run_dir)
    if ok:
        print("skipped:", run_dir)
        mark_run_status(run_dir, "complete")
        return True
    if run_dir.exists():
        print("rerun:", run_dir)
        print("Issues:", issues)
        mark_run_status(run_dir, "incomplete", issues)
    return False


# 01_recover_defect_subtypes

# %% recover_defect_subtypes
def _safe_json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def infer_defect_type_from_path(path):
    p = Path(path)
    parts = [x for x in p.parts]
    lower = [x.lower() for x in parts]
    if "good" in lower:
        return "good"
    for token in reversed(parts):
        t = str(token)
        if any(k.lower() in t.lower() for k in ["defect", "damage", "abnormal", "cable"]):
            return t.replace(".zip", "")
    parent = p.parent.name
    if parent.lower() not in {"good", "abnormal", "training", "validation"}:
        return parent
    return "abnormal_unknown"

def extract_defect_type_from_json(json_path):
    data = _safe_json_load(json_path)
    if not data:
        return None
    candidates = []
    for key in ["defect_type", "damage_type", "class", "class_name", "category", "category_name"]:
        if isinstance(data, dict) and data.get(key):
            candidates.append(str(data[key]))
    for ann in data.get("annotations", []) if isinstance(data, dict) else []:
        for key in ["defect_type", "damage_type", "class", "class_name", "category", "category_name", "label"]:
            if ann.get(key):
                candidates.append(str(ann[key]))
    for item in candidates:
        if item and item.lower() not in {"good", "normal", "none", "0"}:
            return item
    return None

def recover_defect_subtypes(manifest_df):
    rows = []
    for _, row in manifest_df.iterrows():
        r = row.to_dict()
        path = r.get("path")
        if int(r.get("label", 0)) == 0:
            r["defect_type_recovered"] = "good"
        else:
            label_json = find_label_json_by_stem(path) if "find_label_json_by_stem" in globals() else None
            from_json = extract_defect_type_from_json(label_json) if label_json else None
            from_archive = r.get("source_archive") or r.get("archive") or None
            inferred = from_json or from_archive or infer_defect_type_from_path(path)
            r["defect_type_recovered"] = str(inferred).replace(".zip", "")
            r["label_json"] = label_json
            r["defect_type_source"] = "json" if from_json else "archive_or_path"
        rows.append(r)
    out = pd.DataFrame(rows)
    out["defect_type"] = out["defect_type_recovered"]
    out_path = MANIFEST_ROOT / f"{E3}_defect_type_manifest.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    display(out.groupby(["label", "defect_type"]).size().reset_index(name="n"))
    return out


# 02_label_budget_keep_all_normals

# %% label_budget_keep_all_normals
def make_label_budget_train_df_keep_all_normals(split_df, abnormal_label_ratio, seed):
    train = split_df[split_df["protocol_split"].eq("train")].copy()
    normal = train[train["label"].eq(0)].copy()
    abnormal = train[train["label"].eq(1)].copy()
    if abnormal.empty:
        raise RuntimeError("No abnormal training samples in protocol split.")
    n_abn = max(1, int(math.ceil(len(abnormal) * abnormal_label_ratio)))
    abnormal_sample = abnormal.sample(n=min(n_abn, len(abnormal)), random_state=seed)
    out = pd.concat([normal, abnormal_sample], ignore_index=True)
    out["abnormal_label_ratio"] = abnormal_label_ratio
    out["normal_sampling"] = "all_normals_kept"
    return out

def train_label_budget_keep_all_normals_E2(
    model_name="resnet18",
    abnormal_label_ratio=0.05,
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    seed=SEED,
    threshold_method="youden",
    overwrite=False,
):
    set_all_seeds(seed)
    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    train_df = make_label_budget_train_df_keep_all_normals(split_df, abnormal_label_ratio, seed)
    experiment_name = protocol_experiment_name(
        f"label_budget_abnratio{abnormal_label_ratio:g}_allnormal_{preprocess_name}_ep{epochs}_bs{batch_size}",
        seed,
        model_name,
    )
    if should_skip_run(model_name, experiment_name, overwrite=overwrite):
        return ExperimentTracker(model_name, experiment_name, config={"section": "06_label_budget_models"}, overwrite=False)

    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "06_label_budget_models",
            "paradigm": "label_budget_supervised",
            "protocol_version": PROTOCOL_VERSION,
            "version": E3,
            "seed": seed,
            "abnormal_label_ratio": abnormal_label_ratio,
            "normal_sampling": "all_normals_kept",
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "uses_unlabeled_data": False,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    mark_run_status(tracker.run_dir, "running")
    clear_gpu()
    train_loader = make_loader_from_arbitrary_df(train_df, preprocess_name, batch_size, shuffle=True)
    calib_loader, _ = make_loader_from_df(split_df, "calibration", preprocess_name, batch_size, shuffle=False)
    test_loader, _ = make_loader_from_df(split_df, "test", preprocess_name, batch_size, shuffle=False)

    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.perf_counter()
            total_loss = 0.0
            n_seen = 0
            for x, y, _paths in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                bs = x.size(0)
                total_loss += float(loss.item()) * bs
                n_seen += bs
            history.append({
                "epoch": epoch,
                "train_loss": total_loss / max(n_seen, 1),
                "elapsed_sec": time.perf_counter() - t0,
            })
            print(f"{model_name} abnormal_label_ratio={abnormal_label_ratio:g} seed={seed} epoch {epoch}/{epochs}")

        tracker.save_checkpoint(model, "model_state_dict.pt")
        save_training_history(tracker, history)
        train_df.to_csv(tracker.artifact("labeled_train_subset.csv"), index=False)
        save_json({"input_shape": input_shape, "n_labeled_train": int(len(train_df))}, tracker.artifact("model_meta.json"))
        calib_scores = collect_supervised_scores(model, calib_loader)
        test_scores = collect_supervised_scores(model, test_loader)
        metrics, _ = save_calibrated_score_evaluation(
            tracker,
            calib_scores,
            test_scores,
            threshold_method=threshold_method,
            extra_config={
                "seed": seed,
                "n_labeled_train": int(len(train_df)),
                "n_labeled_normal": int((train_df.label == 0).sum()),
                "n_labeled_abnormal": int((train_df.label == 1).sum()),
                "n_test_normal": int((test_scores.y_true == 0).sum()),
                "n_test_abnormal": int((test_scores.y_true == 1).sum()),
            },
        )
        save_efficiency_report(model, tracker, input_shape=input_shape)
        ok, issues = artifact_complete(tracker.run_dir)
        mark_run_status(tracker.run_dir, "complete" if ok else "incomplete", issues)
        print(metrics)
        return tracker
    except Exception as e:
        mark_run_status(tracker.run_dir, "failed", [f"{type(e).__name__}: {e}"])
        raise


# 03_anomaly_same_split_suite

# %% anomaly_same_split_suite
def run_E2_embedding_anomaly_suite(
    methods=("patchcore", "padim", "stfpm"),
    seeds=SEEDS_V2[:3],
    train_batch_size=1,
    eval_batch_size=1,
    overwrite=False,
):
    trackers = []
    method_layers = {
        "patchcore": ["layer2"],
        "padim": ["layer1", "layer2", "layer3"],
        "stfpm": ["layer1", "layer2"],
    }
    for seed in seeds:
        make_E2_split(seed=seed)
        for method in methods:
            layers = method_layers.get(method, ["layer2"])
            experiment_name = protocol_experiment_name(
                f"{method}_resnet18_{'-'.join(layers)}_bs{train_batch_size}",
                seed,
                "embedding_anomaly",
            )
            model_name = f"{method}_resnet18"
            if should_skip_run(model_name, experiment_name, overwrite=overwrite):
                trackers.append(ExperimentTracker(model_name, experiment_name, config={"section": "05_unsupervised_anomaly_models"}))
                continue
            trackers.append(train_anomalib_embedding_model_E2(
                method=method,
                backbone="resnet18",
                layers=layers,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                seed=seed,
                overwrite=overwrite,
            ))
    return trackers

def run_label_budget_allnormal_suite(
    model_name="resnet18",
    ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.00),
    seeds=SEEDS_V2[:3],
    epochs=5,
    batch_size=32,
    overwrite=False,
):
    trackers = []
    for seed in seeds:
        make_E2_split(seed=seed)
        for ratio in ratios:
            trackers.append(train_label_budget_keep_all_normals_E2(
                model_name=model_name,
                abnormal_label_ratio=ratio,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                overwrite=overwrite,
            ))
    return trackers


# 03_lodo_after_defect_recovery_guard

# %% lodo_after_defect_recovery_guard
def build_lodo_after_defect_recovery(seed=SEED):
    base = pd.read_csv(MANIFEST_ROOT / "manifest.csv") if (MANIFEST_ROOT / "manifest.csv").exists() else standard_split.copy()
    recovered = recover_defect_subtypes(base)
    counts = recovered[recovered["label"].eq(1)].groupby("defect_type").size()
    usable_types = counts[counts >= 5].index.tolist()
    if len(usable_types) < 2:
        raise RuntimeError(
            "Not enough recovered defect subtypes for LODO. "
            "Inspect E3_t_E2_defect_type_manifest.csv."
        )
    usable = recovered[(recovered["label"].eq(0)) | (recovered["defect_type"].isin(usable_types))].copy()
    return make_lodo_splits(usable, seed=seed)


# 04_robustness_analysis

# %% robustness_analysis_after_core_runs
def make_robustness_plan_table():
    rows = [
        {"analysis": "seed robustness", "status": "required", "implementation": "SEEDS=[11,22,33,44,55], report mean+/-sd"},
        {"analysis": "threshold robustness", "status": "recommended", "implementation": "compare youden, fixed_fpr, normal_percentile on calibration only"},
        {"analysis": "preprocessing robustness", "status": "recommended", "implementation": "resize224_imagenet vs resize256_tensor where model supports it"},
        {"analysis": "label-budget robustness", "status": "required", "implementation": "keep all normals; vary abnormal labels only"},
        {"analysis": "unknown-defect robustness", "status": "required after defect recovery", "implementation": "LODO known-test and unknown-test separately"},
        {"analysis": "localization", "status": "postpone", "implementation": "include only after image-level paper is complete and anomaly maps are saved"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_ROOT / "robustness_analysis_plan.csv", index=False)
    display(out)
    return out

# %% robust_artifact_completion
def _read_threshold_protocol(run_dir):
    path = Path(run_dir) / "threshold_protocol.json"
    return load_json(path, default={}) if path.exists() else {}

def _metric_value(metrics, *keys):
    for key in keys:
        value = metrics.get(key)
        if value is not None and not pd.isna(value):
            return value
    return np.nan

def artifact_complete(run_dir, required_files=REQUIRED_COMPLETE_FILES):
    run_dir = Path(run_dir)
    missing = [name for name in required_files if not (run_dir / name).exists()]
    if missing:
        return False, missing

    try:
        metrics = load_json(run_dir / "metrics.json", default={})
        config = load_json(run_dir / "config.json", default={})
        threshold_protocol = _read_threshold_protocol(run_dir)
        pred = pd.read_csv(run_dir / "predictions.csv")
        calib = pd.read_csv(run_dir / "predictions_calibration.csv")
    except Exception as e:
        return False, [f"unreadable_artifact:{type(e).__name__}"]

    issues = []
    protocol_value = (
        config.get("protocol_version")
        or metrics.get("protocol_version")
        or threshold_protocol.get("protocol_version")
    )
    exp_name = str(config.get("experiment_name", run_dir.name))
    if protocol_value not in {PROTOCOL_VERSION, globals().get("PROTOCOL_VERSION_V2"), globals().get("PROTOCOL_VERSION_V3")}:
        if "E2_group_calibrated" not in exp_name and "E4_defect_type_group_calibrated" not in exp_name:
            issues.append("wrong_protocol_version")

    threshold_source = (
        config.get("threshold_source")
        or metrics.get("threshold_source")
        or threshold_protocol.get("threshold_source")
    )
    valid_threshold_sources = {
        "calibration",
        "normal_calibration_only",
    }
    if threshold_source not in valid_threshold_sources:
        issues.append("threshold_not_calibrated")

    if pred.empty:
        issues.append("empty_predictions")
    if calib.empty:
        issues.append("empty_calibration_predictions")
    if "y_true" in pred.columns and pred["y_true"].nunique() < 2:
        issues.append("single_class_test")

    auroc = _metric_value(metrics, "auroc", "AUROC", "image_AUROC", "metric_auroc")
    auprc = _metric_value(metrics, "auprc", "AUPRC", "image_AUPRC", "metric_auprc")
    if pd.isna(auroc):
        issues.append("missing_auroc")
    if pd.isna(auprc):
        issues.append("missing_auprc")

    return len(issues) == 0, issues

def should_skip_run(model_name, experiment_name, overwrite=False):
    run_dir = RUN_ROOT / f"{model_name}__{experiment_name}"
    if overwrite:
        return False
    ok, issues = artifact_complete(run_dir)
    if ok:
        print("skipped:", run_dir)
        mark_run_status(run_dir, "complete") if "mark_run_status" in globals() else None
        return True
    if run_dir.exists():
        print("rerun:", run_dir)
        print("Issues:", issues)
        mark_run_status(run_dir, "incomplete", issues) if "mark_run_status" in globals() else None
    return False


# 05_pytorch_resume_fix

# %% pytorch_resume_fix
def trusted_torch_load(path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_resume_checkpoint(
    tracker,
    model,
    optimizer,
    scheduler=None,
    expected=None,
):
    checkpoint_path = tracker.artifact("resume_checkpoint.pt")
    if not checkpoint_path.exists():
        return {"loaded": False, "start_epoch": 1, "history": []}

    try:
        checkpoint = trusted_torch_load(checkpoint_path, map_location=DEVICE)
    except Exception as e:
        print("Resume checkpoint could not be loaded; restarting this run from epoch 1.")
        print("Checkpoint path:", checkpoint_path)
        print("Reason:", type(e).__name__, e)
        mark_run_status(tracker.run_dir, "resume_checkpoint_unreadable", [f"{type(e).__name__}: {e}"]) if "mark_run_status" in globals() else None
        return {"loaded": False, "start_epoch": 1, "history": []}

    if expected is not None:
        validate_checkpoint_metadata(checkpoint, expected)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    completed_epoch = int(checkpoint.get("epoch", 0))
    print(f"Resuming from epoch {completed_epoch + 1}")
    return {
        "loaded": True,
        "start_epoch": completed_epoch + 1,
        "history": checkpoint.get("history", []),
    }

# %% cached_split_and_json_index
JSON_STEM_INDEX_CACHE = {}

def get_E2_split_cached(seed=SEED):
    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    if split_path.exists():
        print("Using cached protocol split:", split_path)
        return pd.read_csv(split_path)
    return make_E2_split(seed=seed)

def build_label_json_stem_index(search_roots=None, cache_key="default", force=False):
    if not force and cache_key in JSON_STEM_INDEX_CACHE:
        return JSON_STEM_INDEX_CACHE[cache_key]

    if search_roots is None:
        search_roots = [
            DATA_ROOT,
            Path("/content/cable_dataset"),
            Path("/content/cable_dataset_by_defect"),
        ]

    index_path = MANIFEST_ROOT / "label_json_stem_index.csv"
    if index_path.exists() and not force:
        idx_df = pd.read_csv(index_path)
        index = dict(zip(idx_df["stem"].astype(str), idx_df["json_path"].astype(str)))
        JSON_STEM_INDEX_CACHE[cache_key] = index
        print("Loaded cached JSON index:", index_path, "n=", len(index))
        return index

    rows = []
    print("Building JSON stem index. This runs once.")
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for json_path in root.rglob("*.json"):
            rows.append({"stem": json_path.stem, "json_path": str(json_path)})
    idx_df = pd.DataFrame(rows).drop_duplicates("stem")
    idx_df.to_csv(index_path, index=False)
    index = dict(zip(idx_df["stem"].astype(str), idx_df["json_path"].astype(str)))
    JSON_STEM_INDEX_CACHE[cache_key] = index
    print("Saved JSON index:", index_path, "n=", len(index))
    return index

def find_label_json_by_stem_fast(image_path, index=None):
    if index is None:
        index = build_label_json_stem_index()
    stem = Path(image_path).stem
    return index.get(stem) or index.get(stem.upper()) or index.get(stem.lower())

def recover_defect_subtypes_fast(manifest_df, force_rebuild_index=False, save_every=500):
    output_path = MANIFEST_ROOT / f"{E3}_defect_type_manifest_fast.csv"
    partial_path = MANIFEST_ROOT / f"{E3}_defect_type_manifest_fast.partial.csv"
    if output_path.exists() and not force_rebuild_index:
        print("Using cached recovered defect manifest:", output_path)
        return pd.read_csv(output_path)

    index = build_label_json_stem_index(force=force_rebuild_index)
    rows = []
    n = len(manifest_df)
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(manifest_df.iterrows(), start=1):
        r = row.to_dict()
        path = r.get("path")
        if int(r.get("label", 0)) == 0:
            r["defect_type_recovered"] = "good"
            r["defect_type_source"] = "normal_label"
            r["label_json"] = None
        else:
            label_json = find_label_json_by_stem_fast(path, index=index)
            from_json = extract_defect_type_from_json(label_json) if label_json and "extract_defect_type_from_json" in globals() else None
            from_archive = r.get("source_archive") or r.get("archive") or None
            inferred = from_json or from_archive or infer_defect_type_from_path(path)
            r["defect_type_recovered"] = str(inferred).replace(".zip", "")
            r["defect_type_source"] = "json" if from_json else "archive_or_path"
            r["label_json"] = label_json
        rows.append(r)

        if i % save_every == 0 or i == n:
            elapsed = time.perf_counter() - t0
            print(f"Recovered defect types: {i}/{n} ({elapsed/60:.1f} min)")
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    out = pd.DataFrame(rows)
    out["defect_type"] = out["defect_type_recovered"]
    out.to_csv(output_path, index=False)
    print("Saved:", output_path)
    display(out.groupby(["label", "defect_type"]).size().reset_index(name="n"))
    return out


# 06_execution

# Optional execution example. Uncomment to run.
# 1. Build and validate one split:
# split_v2 = make_E2_split(seed=11)

# Optional execution example. Uncomment to run.
# 2. Run minimum supervised E2 baselines:
# supervised_v2 = run_E2_supervised_suite(
#      model_names=("resnet18", "efficientnet_b0", "convnext_tiny"),
#      seeds=[11, 22, 33],
#      epochs=5,
#      batch_size=32,
# )

# Optional execution example. Uncomment to run.
# 3. Run label-efficiency curves:
# label_eff_v2 = run_E2_label_efficiency_suite(
#      model_name="resnet18",
#      ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.00),
#      seeds=[11, 22, 33],
#      epochs=5,
#      batch_size=32,
# )

# Optional execution example. Uncomment to run.
# patch_tracker = train_patchcore_E2_and_export_scores(
#     seed=11,
#     layers=("layer2",),
#     train_batch_size=1,
#     eval_batch_size=1,
#     overwrite=False,
#     show_progress=False,
# )


# 07_adjustment

# %% resumable_training_checkpoint
def validate_checkpoint_metadata(checkpoint, expected):
    saved = checkpoint.get("extra", {})
    mismatches = []
    for key, expected_value in expected.items():
        saved_value = saved.get(key)
        if saved_value != expected_value:
            mismatches.append(f"{key}: saved={saved_value}, expected={expected_value}")
    if mismatches:
        raise RuntimeError(
            "Resume checkpoint configuration mismatch:\n" + "\n".join(mismatches)
        )

def save_resume_checkpoint(
    tracker,
    model,
    optimizer,
    epoch,
    history,
    scheduler=None,
    extra=None,
):
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "extra": extra or {},
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint_path = tracker.artifact("resume_checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_resume_checkpoint(
    tracker,
    model,
    optimizer,
    scheduler=None,
    expected=None,
):
    checkpoint_path = tracker.artifact("resume_checkpoint.pt")
    if not checkpoint_path.exists():
        return {"loaded": False, "start_epoch": 1, "history": []}

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if expected is not None:
        validate_checkpoint_metadata(checkpoint, expected)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    completed_epoch = int(checkpoint.get("epoch", 0))
    print(f"Resuming from epoch {completed_epoch + 1}")
    return {
        "loaded": True,
        "start_epoch": completed_epoch + 1,
        "history": checkpoint.get("history", []),
    }

def _dedupe_history_append(history, epoch_record):
    epoch = int(epoch_record["epoch"])
    history = [row for row in history if int(row["epoch"]) != epoch]
    history.append(epoch_record)
    return sorted(history, key=lambda row: int(row["epoch"]))

# %% patched_supervised_E2_with_resume
def train_supervised_model_E2(
    model_name="resnet18",
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    seed=SEED,
    threshold_method="youden",
    overwrite=False,
):
    set_all_seeds(seed)
    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    issues = validate_protocol_split(split_df, "protocol_split")
    if issues:
        raise RuntimeError(f"Invalid protocol split for seed {seed}: {issues}")

    experiment_name = protocol_experiment_name(
        f"supervised_{preprocess_name}_ep{epochs}_bs{batch_size}",
        seed,
        model_name,
    )
    if "should_skip_run" in globals() and should_skip_run(model_name, experiment_name, overwrite=overwrite):
        return ExperimentTracker(model_name, experiment_name, config={"section": "04_supervised_models"}, overwrite=False)

    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "04_supervised_models",
            "paradigm": "supervised",
            "protocol_version": PROTOCOL_VERSION,
            "version": globals().get("E3", "E3_t_E2"),
            "seed": seed,
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    if "mark_run_status" in globals():
        mark_run_status(tracker.run_dir, "running")

    clear_gpu()
    train_loader, _ = make_loader_from_df(split_df, "train", preprocess_name, batch_size, shuffle=True)
    calib_loader, _ = make_loader_from_df(split_df, "calibration", preprocess_name, batch_size, shuffle=False)
    test_loader, _ = make_loader_from_df(split_df, "test", preprocess_name, batch_size, shuffle=False)

    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "version": globals().get("E3", "E3_t_E2"),
        "model_name": model_name,
        "seed": seed,
        "preprocessing": preprocess_name,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    resume_state = load_resume_checkpoint(tracker, model, optimizer, expected=expected)
    start_epoch = resume_state["start_epoch"]
    history = resume_state["history"]

    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            t0 = time.perf_counter()
            total_loss = 0.0
            n_seen = 0
            for x, y, _paths in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                bs = x.size(0)
                total_loss += float(loss.item()) * bs
                n_seen += bs
            epoch_record = {
                "epoch": epoch,
                "train_loss": total_loss / max(n_seen, 1),
                "elapsed_sec": time.perf_counter() - t0,
            }
            history = _dedupe_history_append(history, epoch_record)
            print(f"{model_name} seed={seed} epoch {epoch}/{epochs} loss={epoch_record['train_loss']:.6f}")
            save_resume_checkpoint(tracker, model, optimizer, epoch, history, extra=expected)
            save_training_history(tracker, history)

        tracker.save_checkpoint(model, "model_state_dict.pt")
        save_json({"input_shape": input_shape}, tracker.artifact("model_meta.json"))
        calib_scores = collect_supervised_scores(model, calib_loader)
        test_scores = collect_supervised_scores(model, test_loader)
        metrics, _ = save_calibrated_score_evaluation(
            tracker,
            calib_scores,
            test_scores,
            threshold_method=threshold_method,
            extra_config={
                "seed": seed,
                "n_test_normal": int((test_scores.y_true == 0).sum()),
                "n_test_abnormal": int((test_scores.y_true == 1).sum()),
            },
        )
        save_efficiency_report(model, tracker, input_shape=input_shape)
        if "artifact_complete" in globals() and "mark_run_status" in globals():
            ok, complete_issues = artifact_complete(tracker.run_dir)
            mark_run_status(tracker.run_dir, "complete" if ok else "incomplete", complete_issues)
        print(metrics)
        return tracker
    except Exception as e:
        if "mark_run_status" in globals():
            mark_run_status(tracker.run_dir, "failed", [f"{type(e).__name__}: {e}"])
        raise

# %% patched_label_budget_with_resume
def train_label_budget_keep_all_normals_E2(
    model_name="resnet18",
    abnormal_label_ratio=0.05,
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    seed=SEED,
    threshold_method="youden",
    overwrite=False,
):
    set_all_seeds(seed)
    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    train_df = make_label_budget_train_df_keep_all_normals(split_df, abnormal_label_ratio, seed)
    experiment_name = protocol_experiment_name(
        f"label_budget_abnratio{abnormal_label_ratio:g}_allnormal_{preprocess_name}_ep{epochs}_bs{batch_size}",
        seed,
        model_name,
    )
    if "should_skip_run" in globals() and should_skip_run(model_name, experiment_name, overwrite=overwrite):
        return ExperimentTracker(model_name, experiment_name, config={"section": "06_label_budget_models"}, overwrite=False)

    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "06_label_budget_models",
            "paradigm": "label_budget_supervised",
            "protocol_version": PROTOCOL_VERSION,
            "version": globals().get("E3", "E3_t_E2"),
            "seed": seed,
            "abnormal_label_ratio": abnormal_label_ratio,
            "normal_sampling": "all_normals_kept",
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "uses_unlabeled_data": False,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    if "mark_run_status" in globals():
        mark_run_status(tracker.run_dir, "running")

    clear_gpu()
    train_loader = make_loader_from_arbitrary_df(train_df, preprocess_name, batch_size, shuffle=True)
    calib_loader, _ = make_loader_from_df(split_df, "calibration", preprocess_name, batch_size, shuffle=False)
    test_loader, _ = make_loader_from_df(split_df, "test", preprocess_name, batch_size, shuffle=False)

    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "version": globals().get("E3", "E3_t_E2"),
        "model_name": model_name,
        "seed": seed,
        "abnormal_label_ratio": abnormal_label_ratio,
        "preprocessing": preprocess_name,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    resume_state = load_resume_checkpoint(tracker, model, optimizer, expected=expected)
    start_epoch = resume_state["start_epoch"]
    history = resume_state["history"]

    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            t0 = time.perf_counter()
            total_loss = 0.0
            n_seen = 0
            for x, y, _paths in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                bs = x.size(0)
                total_loss += float(loss.item()) * bs
                n_seen += bs
            epoch_record = {
                "epoch": epoch,
                "train_loss": total_loss / max(n_seen, 1),
                "elapsed_sec": time.perf_counter() - t0,
            }
            history = _dedupe_history_append(history, epoch_record)
            print(
                f"{model_name} abnormal_label_ratio={abnormal_label_ratio:g} "
                f"seed={seed} epoch {epoch}/{epochs} loss={epoch_record['train_loss']:.6f}"
            )
            save_resume_checkpoint(tracker, model, optimizer, epoch, history, extra=expected)
            save_training_history(tracker, history)

        tracker.save_checkpoint(model, "model_state_dict.pt")
        train_df.to_csv(tracker.artifact("labeled_train_subset.csv"), index=False)
        save_json({"input_shape": input_shape, "n_labeled_train": int(len(train_df))}, tracker.artifact("model_meta.json"))
        calib_scores = collect_supervised_scores(model, calib_loader)
        test_scores = collect_supervised_scores(model, test_loader)
        metrics, _ = save_calibrated_score_evaluation(
            tracker,
            calib_scores,
            test_scores,
            threshold_method=threshold_method,
            extra_config={
                "seed": seed,
                "n_labeled_train": int(len(train_df)),
                "n_labeled_normal": int((train_df.label == 0).sum()),
                "n_labeled_abnormal": int((train_df.label == 1).sum()),
                "n_test_normal": int((test_scores.y_true == 0).sum()),
                "n_test_abnormal": int((test_scores.y_true == 1).sum()),
            },
        )
        save_efficiency_report(model, tracker, input_shape=input_shape)
        if "artifact_complete" in globals() and "mark_run_status" in globals():
            ok, complete_issues = artifact_complete(tracker.run_dir)
            mark_run_status(tracker.run_dir, "complete" if ok else "incomplete", complete_issues)
        print(metrics)
        return tracker
    except Exception as e:
        if "mark_run_status" in globals():
            mark_run_status(tracker.run_dir, "failed", [f"{type(e).__name__}: {e}"])
        raise

# %% patched_anomalib_E2_skip_and_checkpoint
def train_anomalib_embedding_model_E2(
    method="patchcore",
    backbone="resnet18",
    layers=None,
    coreset_sampling_ratio=0.005,
    num_neighbors=1,
    train_batch_size=1,
    eval_batch_size=1,
    seed=SEED,
    threshold_method="youden",
    accelerator=None,
    show_progress=False,
    overwrite=False,
):
    from anomalib.engine import Engine
    from anomalib.models import Patchcore, Padim, Stfpm
    from lightning.pytorch.callbacks import TQDMProgressBar

    set_all_seeds(seed)
    method = method.lower()
    if layers is None:
        layers = ["layer2"] if method == "patchcore" else ["layer1", "layer2", "layer3"]

    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    folder_root, materialized_manifest = materialize_E2_anomalib_folder(split_df, seed)

    experiment_name = protocol_experiment_name(
        f"{method}_{backbone}_{'-'.join(layers)}_bs{train_batch_size}",
        seed,
        "embedding_anomaly",
    )
    model_name_for_tracker = f"{method}_{backbone}"
    if "should_skip_run" in globals() and should_skip_run(model_name_for_tracker, experiment_name, overwrite=overwrite):
        return ExperimentTracker(model_name_for_tracker, experiment_name, config={"section": "05_unsupervised_anomaly_models"}, overwrite=False)

    tracker = ExperimentTracker(
        model_name_for_tracker,
        experiment_name,
        config={
            "section": "05_unsupervised_anomaly_models",
            "paradigm": "embedding_anomaly",
            "method": method,
            "backbone": backbone,
            "layers": layers,
            "coreset_sampling_ratio": coreset_sampling_ratio if method == "patchcore" else None,
            "num_neighbors": num_neighbors if method == "patchcore" else None,
            "protocol_version": PROTOCOL_VERSION,
            "version": globals().get("E3", "E3_t_E2"),
            "seed": seed,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    if "mark_run_status" in globals():
        mark_run_status(tracker.run_dir, "running")

    clear_gpu()
    dm_train_calib = build_anomalib_folder_datamodule(
        folder_root,
        eval_split="calibration",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    dm_train_calib.setup()

    if method == "patchcore":
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
    elif method == "padim":
        model = Padim(backbone=backbone, layers=layers)
    elif method == "stfpm":
        model = Stfpm(backbone=backbone, layers=layers)
    else:
        raise ValueError("method must be one of: patchcore, padim, stfpm")

    callbacks = [TQDMProgressBar(refresh_rate=20)] if show_progress else []
    engine = Engine(
        max_epochs=1,
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=show_progress,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    try:
        t0 = time.perf_counter()
        engine.fit(model=model, datamodule=dm_train_calib, ckpt_path=None)
        train_sec = time.perf_counter() - t0

        manual_checkpoint_path = tracker.artifact(f"{method}_{backbone}_after_fit.ckpt")
        try:
            engine.trainer.save_checkpoint(str(manual_checkpoint_path))
            save_json(
                {
                    "manual_checkpoint_path": str(manual_checkpoint_path),
                    "protocol_version": PROTOCOL_VERSION,
                    "version": globals().get("E3", "E3_t_E2"),
                    "seed": seed,
                    "method": method,
                    "backbone": backbone,
                },
                tracker.artifact("checkpoint_info.json"),
            )
        except Exception as error:
            save_json(
                {
                    "manual_checkpoint_path": None,
                    "checkpoint_warning": f"{type(error).__name__}: {error}",
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": seed,
                },
                tracker.artifact("checkpoint_info.json"),
            )

        calib_predictions = engine.predict(model=model, datamodule=dm_train_calib)
        calib_scores = anomalib_predictions_to_score_df(calib_predictions, materialized_manifest)
        calib_scores = calib_scores[calib_scores["path"].isin(
            materialized_manifest.loc[materialized_manifest["protocol_split"].eq("calibration"), "path"]
        )].copy()

        dm_test = build_anomalib_folder_datamodule(
            folder_root,
            eval_split="test",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )
        dm_test.setup()
        test_predictions = engine.predict(model=model, datamodule=dm_test)
        test_scores = anomalib_predictions_to_score_df(test_predictions, materialized_manifest)
        test_scores = test_scores[test_scores["path"].isin(
            materialized_manifest.loc[materialized_manifest["protocol_split"].eq("test"), "path"]
        )].copy()

        if calib_scores.empty or test_scores.empty:
            calib_scores.to_csv(tracker.artifact("debug_calibration_scores_empty.csv"), index=False)
            test_scores.to_csv(tracker.artifact("debug_test_scores_empty.csv"), index=False)
            raise RuntimeError("Anomalib prediction parsing produced empty calibration/test scores.")

        metrics, _ = save_calibrated_score_evaluation(
            tracker,
            calib_scores,
            test_scores,
            threshold_method=threshold_method,
            extra_config={
                "seed": seed,
                "train_elapsed_sec": train_sec,
                "n_test_normal": int((test_scores.y_true == 0).sum()),
                "n_test_abnormal": int((test_scores.y_true == 1).sum()),
            },
        )
        metrics["train_elapsed_sec"] = train_sec
        tracker.save_metrics(metrics)
        if "artifact_complete" in globals() and "mark_run_status" in globals():
            ok, complete_issues = artifact_complete(tracker.run_dir)
            mark_run_status(tracker.run_dir, "complete" if ok else "incomplete", complete_issues)
        print(metrics)
        return tracker
    except Exception as e:
        if "mark_run_status" in globals():
            mark_run_status(tracker.run_dir, "failed", [f"{type(e).__name__}: {e}"])
        raise

# %% faster_anomalib_skip_before_materialize
def materialize_E2_anomalib_folder(split_df, seed, root=None):
    if root is None:
        root = PROJECT_ROOT / "E2_anomalib_folders" / f"seed{seed}"
    root = Path(root)
    manifest_path = root / "E2_anomalib_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        expected = {"train", "calibration", "test"}
        if "protocol_split" in manifest.columns and expected.issubset(set(manifest["protocol_split"].unique())):
            print("cached:", root)
            return root, manifest

    mapping = {
        ("train", 0): root / "train" / "good",
        ("calibration", 0): root / "calibration" / "good",
        ("calibration", 1): root / "calibration" / "abnormal",
        ("test", 0): root / "test" / "good",
        ("test", 1): root / "test" / "abnormal",
    }
    for dst in mapping.values():
        dst.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(split_df.iterrows(), start=1):
        split = row["protocol_split"]
        label = int(row["label"])
        if (split, label) not in mapping:
            continue
        src = Path(row["path"])
        rel_name = f"{src.stem}_{hashlib.md5(str(src).encode('utf-8')).hexdigest()[:10]}{src.suffix}"
        dst = mapping[(split, label)] / rel_name
        link_or_copy_file(src, dst)
        r = row.to_dict()
        r["materialized_path"] = str(dst)
        manifest_rows.append(r)
        if i % 1000 == 0:
            print(f"Materialized Anomalib view rows: {i}/{len(split_df)} ({(time.perf_counter()-t0)/60:.1f} min)")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    print("Saved Anomalib folder view:", root)
    return root, manifest

def train_anomalib_embedding_model_E2(
    method="patchcore",
    backbone="resnet18",
    layers=None,
    coreset_sampling_ratio=0.005,
    num_neighbors=1,
    train_batch_size=1,
    eval_batch_size=1,
    seed=SEED,
    threshold_method="youden",
    accelerator=None,
    show_progress=False,
    overwrite=False,
):
    from anomalib.engine import Engine
    from anomalib.models import Patchcore, Padim, Stfpm
    from lightning.pytorch.callbacks import TQDMProgressBar

    set_all_seeds(seed)
    method = method.lower()
    if layers is None:
        layers = ["layer2"] if method == "patchcore" else ["layer1", "layer2", "layer3"]

    experiment_name = protocol_experiment_name(
        f"{method}_{backbone}_{'-'.join(layers)}_bs{train_batch_size}",
        seed,
        "embedding_anomaly",
    )
    model_name_for_tracker = f"{method}_{backbone}"

    # Important: check completion before split loading/materialization.
    if should_skip_run(model_name_for_tracker, experiment_name, overwrite=overwrite):
        return ExperimentTracker(model_name_for_tracker, experiment_name, config={"section": "05_unsupervised_anomaly_models"}, overwrite=False)

    split_df = get_E2_split_cached(seed=seed)
    folder_root, materialized_manifest = materialize_E2_anomalib_folder(split_df, seed)

    tracker = ExperimentTracker(
        model_name_for_tracker,
        experiment_name,
        config={
            "section": "05_unsupervised_anomaly_models",
            "paradigm": "embedding_anomaly",
            "method": method,
            "backbone": backbone,
            "layers": layers,
            "coreset_sampling_ratio": coreset_sampling_ratio if method == "patchcore" else None,
            "num_neighbors": num_neighbors if method == "patchcore" else None,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    mark_run_status(tracker.run_dir, "running") if "mark_run_status" in globals() else None

    clear_gpu()
    print(f"[{method} seed={seed}] setup datamodule")
    dm_train_calib = build_anomalib_folder_datamodule(
        folder_root,
        eval_split="calibration",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    dm_train_calib.setup()

    print(f"[{method} seed={seed}] instantiate model")
    if method == "patchcore":
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
    elif method == "padim":
        model = Padim(backbone=backbone, layers=layers)
    elif method == "stfpm":
        model = Stfpm(backbone=backbone, layers=layers)
    else:
        raise ValueError("method must be one of: patchcore, padim, stfpm")

    callbacks = [TQDMProgressBar(refresh_rate=20)] if show_progress else []
    engine = Engine(
        max_epochs=1,
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=show_progress,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    try:
        print(f"[{method} seed={seed}] fit start")
        t0 = time.perf_counter()
        engine.fit(model=model, datamodule=dm_train_calib, ckpt_path=None)
        train_sec = time.perf_counter() - t0
        print(f"[{method} seed={seed}] fit done in {train_sec/60:.1f} min")

        manual_checkpoint_path = tracker.artifact(f"{method}_{backbone}_after_fit.ckpt")
        try:
            engine.trainer.save_checkpoint(str(manual_checkpoint_path))
            save_json(
                {
                    "manual_checkpoint_path": str(manual_checkpoint_path),
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": seed,
                    "method": method,
                    "backbone": backbone,
                },
                tracker.artifact("checkpoint_info.json"),
            )
        except Exception as error:
            save_json(
                {
                    "manual_checkpoint_path": None,
                    "checkpoint_warning": f"{type(error).__name__}: {error}",
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": seed,
                },
                tracker.artifact("checkpoint_info.json"),
            )

        print(f"[{method} seed={seed}] predict calibration start")
        t1 = time.perf_counter()
        calib_predictions = engine.predict(model=model, datamodule=dm_train_calib)
        calib_scores = anomalib_predictions_to_score_df(calib_predictions, materialized_manifest)
        calib_scores = calib_scores[calib_scores["path"].isin(
            materialized_manifest.loc[materialized_manifest["protocol_split"].eq("calibration"), "path"]
        )].copy()
        print(f"[{method} seed={seed}] predict calibration done in {(time.perf_counter()-t1)/60:.1f} min")

        print(f"[{method} seed={seed}] predict test start")
        dm_test = build_anomalib_folder_datamodule(
            folder_root,
            eval_split="test",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )
        dm_test.setup()
        t2 = time.perf_counter()
        test_predictions = engine.predict(model=model, datamodule=dm_test)
        test_scores = anomalib_predictions_to_score_df(test_predictions, materialized_manifest)
        test_scores = test_scores[test_scores["path"].isin(
            materialized_manifest.loc[materialized_manifest["protocol_split"].eq("test"), "path"]
        )].copy()
        print(f"[{method} seed={seed}] predict test done in {(time.perf_counter()-t2)/60:.1f} min")

        if calib_scores.empty or test_scores.empty:
            calib_scores.to_csv(tracker.artifact("debug_calibration_scores_empty.csv"), index=False)
            test_scores.to_csv(tracker.artifact("debug_test_scores_empty.csv"), index=False)
            raise RuntimeError("Anomalib prediction parsing produced empty calibration/test scores.")

        metrics, _ = save_calibrated_score_evaluation(
            tracker,
            calib_scores,
            test_scores,
            threshold_method=threshold_method,
            extra_config={
                "seed": seed,
                "train_elapsed_sec": train_sec,
                "n_test_normal": int((test_scores.y_true == 0).sum()),
                "n_test_abnormal": int((test_scores.y_true == 1).sum()),
            },
        )
        metrics["train_elapsed_sec"] = train_sec
        tracker.save_metrics(metrics)
        ok, complete_issues = artifact_complete(tracker.run_dir)
        mark_run_status(tracker.run_dir, "complete" if ok else "incomplete", complete_issues) if "mark_run_status" in globals() else None
        print(metrics)
        return tracker
    except Exception as e:
        mark_run_status(tracker.run_dir, "failed", [f"{type(e).__name__}: {e}"]) if "mark_run_status" in globals() else None
        raise

def train_patchcore_E2_and_export_scores(
    seed=11,
    layers=("layer2",),
    train_batch_size=1,
    eval_batch_size=1,
    overwrite=False,
    show_progress=False,
):
    return train_anomalib_embedding_model_E2(
        method="patchcore",
        layers=list(layers),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        overwrite=overwrite,
        show_progress=show_progress,
    )

def run_E2_embedding_anomaly_suite(
    methods=("patchcore", "padim", "stfpm"),
    seeds=SEEDS_V2[:3],
    train_batch_size=1,
    eval_batch_size=1,
    overwrite=False,
    show_progress=False,
):
    trackers = []
    method_layers = {
        "patchcore": ["layer2"],
        "padim": ["layer1", "layer2", "layer3"],
        "stfpm": ["layer1", "layer2"],
    }
    for seed in seeds:
        for method in methods:
            layers = method_layers.get(method, ["layer2"])
            trackers.append(train_anomalib_embedding_model_E2(
                method=method,
                backbone="resnet18",
                layers=layers,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                seed=seed,
                overwrite=overwrite,
                show_progress=show_progress,
            ))
    return trackers

# Optional execution example. Uncomment to run.
# duplicate_groups = build_duplicate_groups()
# recovered_manifest = recover_defect_subtypes_fast(duplicate_groups)

# Optional execution example. Uncomment to run.
# patch_tracker = train_patchcore_E2_and_export_scores(
#      seed=11,
#      layers=("layer2",),
#      train_batch_size=1,
#      eval_batch_size=1,
#      overwrite=False,
#      show_progress=False,
# )


# 08_re-execution

def materialize_E2_anomalib_folder(split_df, seed, root=None):
    if root is None:
        root = PROJECT_ROOT / "E2_anomalib_folders" / f"seed{seed}"
    root = Path(root)
    manifest_path = root / "E2_anomalib_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        expected = {"train", "calibration", "test"}
        if "protocol_split" in manifest.columns and expected.issubset(set(manifest["protocol_split"].unique())):
            print("cached:", root)
            return root, manifest

    mapping = {
        ("train", 0): root / "train" / "good",
        ("calibration", 0): root / "calibration" / "good",
        ("calibration", 1): root / "calibration" / "abnormal",
        ("test", 0): root / "test" / "good",
        ("test", 1): root / "test" / "abnormal",
    }
    for dst in mapping.values():
        dst.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(split_df.iterrows(), start=1):
        split = row["protocol_split"]
        label = int(row["label"])
        if (split, label) not in mapping:
            continue
        src = Path(row["path"])
        rel_name = f"{src.stem}_{hashlib.md5(str(src).encode('utf-8')).hexdigest()[:10]}{src.suffix}"
        dst = mapping[(split, label)] / rel_name
        link_or_copy_file(src, dst)
        r = row.to_dict()
        r["materialized_path"] = str(dst)
        manifest_rows.append(r)
        if i % 1000 == 0:
            print(f"Materialized Anomalib view rows: {i}/{len(split_df)} ({(time.perf_counter()-t0)/60:.1f} min)")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    print("Saved Anomalib folder view:", root)
    return root, manifest

def train_anomalib_embedding_model_E2(
    method="patchcore",
    backbone="resnet18",
    layers=None,
    coreset_sampling_ratio=0.005,
    num_neighbors=1,
    train_batch_size=1,
    eval_batch_size=1,
    seed=SEED,
    threshold_method="youden",
    accelerator=None,
    show_progress=False,
    overwrite=False,
):
    from anomalib.engine import Engine
    from anomalib.models import Patchcore, Padim, Stfpm
    from lightning.pytorch.callbacks import TQDMProgressBar

    set_all_seeds(seed)
    method = method.lower()
    if layers is None:
        layers = ["layer2"] if method == "patchcore" else ["layer1", "layer2", "layer3"]

    experiment_name = protocol_experiment_name(
        f"{method}_{backbone}_{'-'.join(layers)}_bs{train_batch_size}",
        seed,
        "embedding_anomaly",
    )
    model_name_for_tracker = f"{method}_{backbone}"
    if should_skip_run(model_name_for_tracker, experiment_name, overwrite=overwrite):
        return ExperimentTracker(model_name_for_tracker, experiment_name, config={"section": "05_unsupervised_anomaly_models"}, overwrite=False)

    split_df = get_E2_split_cached(seed=seed)
    folder_root, materialized_manifest = materialize_E2_anomalib_folder(split_df, seed)

    tracker = ExperimentTracker(
        model_name_for_tracker,
        experiment_name,
        config={
            "section": "05_unsupervised_anomaly_models",
            "paradigm": "embedding_anomaly",
            "method": method,
            "backbone": backbone,
            "layers": layers,
            "coreset_sampling_ratio": coreset_sampling_ratio if method == "patchcore" else None,
            "num_neighbors": num_neighbors if method == "patchcore" else None,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "checkpoint_reuse": False,
        },
        overwrite=overwrite,
    )
    mark_run_status(tracker.run_dir, "running") if "mark_run_status" in globals() else None

    clear_gpu()
    print(f"[{method} seed={seed}] setup datamodule")
    dm_train_calib = build_anomalib_folder_datamodule(
        folder_root,
        eval_split="calibration",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    dm_train_calib.setup()

    print(f"[{method} seed={seed}] instantiate model")
    if method == "patchcore":
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
    elif method == "padim":
        model = Padim(backbone=backbone, layers=layers)
    elif method == "stfpm":
        model = Stfpm(backbone=backbone, layers=layers)
    else:
        raise ValueError("method must be one of: patchcore, padim, stfpm")

    callbacks = [TQDMProgressBar(refresh_rate=20)] if show_progress else []
    engine = Engine(
        max_epochs=1,
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=show_progress,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    try:
        print(f"[{method} seed={seed}] fit start")
        t0 = time.perf_counter()
        engine.fit(model=model, datamodule=dm_train_calib, ckpt_path=None)
        train_sec = time.perf_counter() - t0
        print(f"[{method} seed={seed}] fit done in {train_sec/60:.1f} min")

        manual_checkpoint_path = tracker.artifact(f"{method}_{backbone}_after_fit.ckpt")
        try:
            engine.trainer.save_checkpoint(str(manual_checkpoint_path))
            save_json(
                {
                    "manual_checkpoint_path": str(manual_checkpoint_path),
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": seed,
                    "method": method,
                    "backbone": backbone,
                },
                tracker.artifact("checkpoint_info.json"),
            )
        except Exception as error:
            save_json(
                {
                    "manual_checkpoint_path": None,
                    "checkpoint_warning": f"{type(error).__name__}: {error}",
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": seed,
                },
                tracker.artifact("checkpoint_info.json"),
            )

        print(f"[{method} seed={seed}] predict calibration start")
        t1 = time.perf_counter()
        calib_predictions = engine.predict(model=model, datamodule=dm_train_calib)
        calib_scores = anomalib_predictions_to_score_df(calib_predictions, materialized_manifest)
        calib_scores = calib_scores[calib_scores["path"].isin(
            materialized_manifest.loc[materialized_manifest["protocol_split"].eq("calibration"), "path"]
        )].copy()
        print(f"[{method} seed={seed}] predict calibration done in {(time.perf_counter()-t1)/60:.1f} min")

        print(f"[{method} seed={seed}] predict test start")
        dm_test = build_anomalib_folder_datamodule(
            folder_root,
            eval_split="test",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )
        dm_test.setup()
        t2 = time.perf_counter()
        test_predictions = engine.predict(model=model, datamodule=dm_test)
        test_scores = anomalib_predictions_to_score_df(test_predictions, materialized_manifest)
        test_scores = test_scores[test_scores["path"].isin(
            materialized_manifest.loc[materialized_manifest["protocol_split"].eq("test"), "path"]
        )].copy()
        print(f"[{method} seed={seed}] predict test done in {(time.perf_counter()-t2)/60:.1f} min")

        if calib_scores.empty or test_scores.empty:
            calib_scores.to_csv(tracker.artifact("debug_calibration_scores_empty.csv"), index=False)
            test_scores.to_csv(tracker.artifact("debug_test_scores_empty.csv"), index=False)
            raise RuntimeError("Anomalib prediction parsing produced empty calibration/test scores.")

        metrics, _ = save_calibrated_score_evaluation(
            tracker,
            calib_scores,
            test_scores,
            threshold_method=threshold_method,
            extra_config={
                "seed": seed,
                "train_elapsed_sec": train_sec,
                "n_test_normal": int((test_scores.y_true == 0).sum()),
                "n_test_abnormal": int((test_scores.y_true == 1).sum()),
            },
        )
        metrics["train_elapsed_sec"] = train_sec
        tracker.save_metrics(metrics)
        ok, complete_issues = artifact_complete(tracker.run_dir)
        mark_run_status(tracker.run_dir, "complete" if ok else "incomplete", complete_issues) if "mark_run_status" in globals() else None
        print(metrics)
        return tracker
    except Exception as e:
        mark_run_status(tracker.run_dir, "failed", [f"{type(e).__name__}: {e}"]) if "mark_run_status" in globals() else None
        raise

def train_patchcore_E2_and_export_scores(
    seed=11,
    layers=("layer2",),
    train_batch_size=1,
    eval_batch_size=1,
    overwrite=False,
    show_progress=False,
):
    return train_anomalib_embedding_model_E2(
        method="patchcore",
        layers=list(layers),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        overwrite=overwrite,
        show_progress=show_progress,
    )

def run_E2_embedding_anomaly_suite(
    methods=("patchcore", "padim", "stfpm"),
    seeds=SEEDS_V2[:3],
    train_batch_size=1,
    eval_batch_size=1,
    overwrite=False,
    show_progress=False,
):
    trackers = []
    method_layers = {
        "patchcore": ["layer2"],
        "padim": ["layer1", "layer2", "layer3"],
        "stfpm": ["layer1", "layer2"],
    }
    for seed in seeds:
        for method in methods:
            layers = method_layers.get(method, ["layer2"])
            trackers.append(train_anomalib_embedding_model_E2(
                method=method,
                backbone="resnet18",
                layers=layers,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                seed=seed,
                overwrite=overwrite,
                show_progress=show_progress,
            ))
    return trackers

# Optional execution example. Uncomment to run.
# duplicate_groups = build_duplicate_groups()
# recovered_manifest = recover_defect_subtypes_fast(duplicate_groups)

# Optional execution example. Uncomment to run.
# patch_tracker = train_patchcore_E2_and_export_scores(
#      seed=11,
#      layers=("layer2",),
#      train_batch_size=1,
#      eval_batch_size=1,
#      overwrite=False,
#      show_progress=False,
# )

# Optional execution example. Uncomment to run.
# A. Run only once to recover defect metadata:
# recover_defect_subtypes = recover_defect_subtypes_fast
# recovered_manifest = recover_defect_subtypes(standard_split)

# Optional execution example. Uncomment to run.
# B. Run core E2 models with resume-safe skipping:
# sup = run_E2_supervised_suite(
#      model_names=("resnet18", "efficientnet_b0", "convnext_tiny"),
#      seeds=[11, 22, 33],
#      epochs=5,
#      batch_size=32,
#      overwrite=False,
# )

# Optional execution example. Uncomment to run.
# C. Run anomaly models on exactly the same E2 splits:
# anom = run_E2_embedding_anomaly_suite(
#      methods=("patchcore", "padim", "stfpm"),
#      seeds=[11, 22, 33],
#      train_batch_size=1,
#      eval_batch_size=1,
#      overwrite=False,
#      show_progress=False,
# )

# Optional execution example. Uncomment to run.
# D. Run the corrected label-budget experiment:
# lb = run_label_budget_allnormal_suite(
#     model_name="resnet18",
#     ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.00),
#     seeds=[11, 22, 33],
#     epochs=5,
#     batch_size=32,
#     overwrite=False,
# )


# 14_Additional Analysis_E4


# 01_Setting

# %% E4_defect_type_recovery

import json
import re
from pathlib import Path
from collections import Counter

PROTOCOL_VERSION = "E4_defect_type_group_calibrated"
SEEDS_V3 = [11, 22, 33, 44, 55]


GENERIC_DEFECT_NAMES = {
    "",
    "abnormal",
    "defect",
    "defective",
    "bad",
    "ng",
    "cable_damage",
    "cable damage",
    "unknown",
    "none",
    "nan",
}


def normalize_defect_name(value):
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    value = Path(value).stem
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    # Remove common archive/split words.
    removable = {
        "training", "validation", "train", "valid", "val",
        "image", "images", "label", "labels", "json",
        "source", "원천데이터", "라벨링데이터",
    }

    tokens = [
        token for token in value.split("_")
        if token.lower() not in removable
    ]
    value = "_".join(tokens)

    if value.lower() in GENERIC_DEFECT_NAMES:
        return None

    return value or None


def category_names_from_aihub_json(json_path):
    if json_path is None or not Path(json_path).exists():
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    category_map = {}

    # Standard COCO categories.
    for category in data.get("categories", []) or []:
        if not isinstance(category, dict):
            continue

        category_id = category.get("id")
        name = (
            category.get("name")
            or category.get("category_name")
            or category.get("label")
            or category.get("class_name")
        )

        if category_id is not None and name is not None:
            category_map[category_id] = str(name)

    extracted = []

    # Standard annotations.
    for annotation in data.get("annotations", []) or []:
        if not isinstance(annotation, dict):
            continue

        candidate_fields = [
            annotation.get("category_name"),
            annotation.get("class_name"),
            annotation.get("label"),
            annotation.get("defect_type"),
            annotation.get("damage_type"),
            annotation.get("type"),
        ]

        category_id = annotation.get("category_id")
        if category_id in category_map:
            candidate_fields.append(category_map[category_id])

        for value in candidate_fields:
            normalized = normalize_defect_name(value)
            if normalized:
                extracted.append(normalized)

    nested_objects = []

    for key in [
        "object", "objects", "labels", "label",
        "metadata", "meta", "data", "result",
    ]:
        value = data.get(key)
        if isinstance(value, list):
            nested_objects.extend(value)
        elif isinstance(value, dict):
            nested_objects.append(value)

    for obj in nested_objects:
        if not isinstance(obj, dict):
            continue

        for key in [
            "category_name", "class_name", "label",
            "defect_type", "damage_type", "type",
            "category", "name",
        ]:
            normalized = normalize_defect_name(obj.get(key))
            if normalized:
                extracted.append(normalized)
    extracted = [
        x for x in extracted
        if x and x.lower() not in GENERIC_DEFECT_NAMES
    ]

    return sorted(set(extracted))


def archive_candidates_from_path(image_path):
    path = Path(image_path)
    candidates = []

    for parent in list(path.parents)[:8]:
        normalized = normalize_defect_name(parent.name)
        if normalized:
            candidates.append(normalized)

    preferred = [
        x for x in candidates
        if any(
            token in x.lower()
            for token in [
                "damage", "defect", "install", "connector",
                "cable", "break", "bend", "fix", "twist",
                "scratch", "crack", "detach", "missing",
            ]
        )
    ]

    return preferred or candidates


def infer_defect_type_for_row(row):
    if int(row["label"]) == 0:
        return "good", "normal_label"

    image_path = row["path"]

    json_path = find_label_json_by_stem(image_path)
    json_categories = category_names_from_aihub_json(json_path)

    if json_categories:
        # If multiple annotations exist, retain a stable combined category.
        return "+".join(sorted(json_categories)), "json_category"

    # Try existing metadata columns.
    for col in [
        "defect_type",
        "class_name",
        "source_archive",
        "archive_name",
        "category_name",
    ]:
        if col in row.index:
            value = normalize_defect_name(row.get(col))
            if value and value.lower() not in GENERIC_DEFECT_NAMES:
                return value, f"column:{col}"

    folder_candidates = archive_candidates_from_path(image_path)
    if folder_candidates:
        return folder_candidates[0], "folder_or_archive"

    return "unresolved_abnormal", "unresolved"


def recover_defect_types(manifest_df):
    enriched = manifest_df.copy()

    inferred = enriched.apply(
        infer_defect_type_for_row,
        axis=1,
        result_type="expand",
    )

    enriched["defect_type_recovered"] = inferred[0]
    enriched["defect_type_source"] = inferred[1]

    # Use recovered values as the official defect type.
    enriched["defect_type"] = enriched["defect_type_recovered"]

    output_path = MANIFEST_ROOT / "defect_type_enriched_manifest.csv"
    enriched.to_csv(output_path, index=False)

    print("Saved:", output_path)

    summary = (
        enriched.groupby(
            ["label", "defect_type", "defect_type_source"],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
        .sort_values(["label", "n"], ascending=[True, False])
    )

    display(summary)

    abnormal = enriched[enriched["label"].eq(1)]

    print("\nNumber of recovered abnormal defect types:")
    print(abnormal["defect_type"].nunique())

    print("\nUnresolved abnormal images:")
    print(int(abnormal["defect_type"].eq("unresolved_abnormal").sum()))

    print("\nRecovery source distribution:")
    display(
        abnormal["defect_type_source"]
        .value_counts(dropna=False)
        .rename_axis("source")
        .reset_index(name="n")
    )

    return enriched

# Optional execution example. Uncomment to run.
# Use the duplicate audit/group dataframe as the base.
# duplicate_groups = build_duplicate_groups()

# Optional execution example. Uncomment to run.
# type_manifest_v3 = recover_defect_types(duplicate_groups)

# Optional execution example. Uncomment to run.
# display(
#     type_manifest_v3[
#         type_manifest_v3["label"].eq(1)
#     ]["defect_type"].value_counts()
# )

# %% E4_constants
PROTOCOL_VERSION_V3 = "E4_defect_type_group_calibrated"
SEEDS_V3 = [11, 22, 33, 44, 55]

GENERIC_DEFECT_NAMES_V3 = {
    "",
    "abnormal",
    "defect",
    "defective",
    "bad",
    "ng",
    "unknown",
    "none",
    "nan",
    "케이블양품",
    "양품",
    "정상",
    "normal",
    "good",
}


BASE_ABNORMAL_LABELS_V3 = {
    "케이블손상",
    "cable_damage",
    "cable damage",
}

def normalize_defect_name_v3(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    value = Path(value).stem
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    removable = {
        "training", "validation", "train", "valid", "val",
        "image", "images", "label", "labels", "json",
        "source", "원천데이터", "라벨링데이터",
    }
    tokens = [token for token in value.split("_") if token.lower() not in removable]
    value = "_".join(tokens)
    if value.lower() in GENERIC_DEFECT_NAMES_V3 or value in GENERIC_DEFECT_NAMES_V3:
        return None
    return value or None

def split_combined_defect_label_v3(value):
    if value is None:
        return []
    pieces = re.split(r"[+/,;|]", str(value))
    out = []
    for piece in pieces:
        normalized = normalize_defect_name_v3(piece)
        if normalized:
            out.append(normalized)
    return sorted(set(out))

def category_names_from_aihub_json_v3(json_path):
    if json_path is None or not Path(json_path).exists():
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    category_map = {}
    for category in data.get("categories", []) or []:
        if not isinstance(category, dict):
            continue
        category_id = category.get("id")
        name = (
            category.get("name")
            or category.get("category_name")
            or category.get("label")
            or category.get("class_name")
        )
        if category_id is not None and name is not None:
            category_map[category_id] = str(name)

    extracted = []
    for annotation in data.get("annotations", []) or []:
        if not isinstance(annotation, dict):
            continue
        candidate_fields = [
            annotation.get("category_name"),
            annotation.get("class_name"),
            annotation.get("label"),
            annotation.get("defect_type"),
            annotation.get("damage_type"),
            annotation.get("type"),
        ]
        category_id = annotation.get("category_id")
        if category_id in category_map:
            candidate_fields.append(category_map[category_id])
        for value in candidate_fields:
            for label in split_combined_defect_label_v3(value):
                extracted.append(label)

    nested_objects = []
    for key in ["object", "objects", "labels", "label", "metadata", "meta", "data", "result"]:
        value = data.get(key)
        if isinstance(value, list):
            nested_objects.extend(value)
        elif isinstance(value, dict):
            nested_objects.append(value)

    for obj in nested_objects:
        if not isinstance(obj, dict):
            continue
        for key in [
            "category_name", "class_name", "label",
            "defect_type", "damage_type", "type",
            "category", "name",
        ]:
            for label in split_combined_defect_label_v3(obj.get(key)):
                extracted.append(label)

    return sorted(set(extracted))

def infer_defect_labels_for_row_v3(row):
    if int(row["label"]) == 0:
        return ["good"], "normal_label"

    image_path = row["path"]
    json_path = find_label_json_by_stem(image_path) if "find_label_json_by_stem" in globals() else None
    json_labels = category_names_from_aihub_json_v3(json_path)
    if json_labels:
        return json_labels, "json_category"

    for col in ["defect_type", "class_name", "source_archive", "archive_name", "category_name"]:
        if col in row.index:
            labels = split_combined_defect_label_v3(row.get(col))
            labels = [x for x in labels if x.lower() not in GENERIC_DEFECT_NAMES_V3]
            if labels:
                return labels, f"column:{col}"

    if "archive_candidates_from_path" in globals():
        labels = []
        for candidate in archive_candidates_from_path(image_path):
            labels.extend(split_combined_defect_label_v3(candidate))
        labels = sorted(set(labels))
        if labels:
            return labels, "folder_or_archive"

    return ["unresolved_abnormal"], "unresolved"

def recover_defect_types_v3(manifest_df):
    enriched = manifest_df.copy()
    labels_and_sources = enriched.apply(infer_defect_labels_for_row_v3, axis=1, result_type="expand")
    enriched["defect_labels"] = labels_and_sources[0].map(lambda xs: json.dumps(xs, ensure_ascii=False))
    enriched["defect_type_source"] = labels_and_sources[1]

    def primary_label(labels_json):
        labels = json.loads(labels_json)
        if labels == ["good"]:
            return "good"
        non_base = [x for x in labels if x not in BASE_ABNORMAL_LABELS_V3]
        if non_base:
            return "+".join(sorted(non_base))
        return "+".join(sorted(labels))

    def combined_label(labels_json):
        labels = json.loads(labels_json)
        if labels == ["good"]:
            return "good"
        return "+".join(sorted(labels))

    enriched["defect_type_combined"] = enriched["defect_labels"].map(combined_label)
    enriched["defect_type_primary"] = enriched["defect_labels"].map(primary_label)
    enriched["defect_type"] = enriched["defect_type_primary"]

    output_path = MANIFEST_ROOT / "defect_type_enriched_manifest_v3.csv"
    enriched.to_csv(output_path, index=False)
    print("Saved:", output_path)

    display(
        enriched.groupby(["label", "defect_type_combined", "defect_type_primary", "defect_type_source"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["label", "n"], ascending=[True, False])
    )

    abnormal = enriched[enriched["label"].eq(1)]
    atomic_counts = Counter()
    for labels_json in abnormal["defect_labels"]:
        for label in json.loads(labels_json):
            atomic_counts[label] += 1
    atomic_df = pd.DataFrame(
        [{"atomic_defect_label": k, "n": v, "is_base_abnormal_label": k in BASE_ABNORMAL_LABELS_V3}
         for k, v in atomic_counts.items()]
    ).sort_values("n", ascending=False)
    atomic_df.to_csv(MANIFEST_ROOT / "defect_atomic_label_counts_v3.csv", index=False)
    display(atomic_df)
    return enriched

# %% E4_lodo_eligibility
def evaluate_lodo_eligibility_v3(type_manifest, min_samples_per_type=5):
    df = type_manifest.copy()
    abnormal = df[df["label"].eq(1)].copy()

    primary_counts = abnormal["defect_type_primary"].value_counts().rename_axis("defect_type").reset_index(name="n")
    primary_counts["eligible_primary_lodo"] = primary_counts["n"] >= min_samples_per_type

    atomic_rows = []
    for _, row in abnormal.iterrows():
        for label in json.loads(row["defect_labels"]):
            if label in BASE_ABNORMAL_LABELS_V3:
                continue
            atomic_rows.append({"path": row["path"], "atomic_defect_label": label})
    atomic_df = pd.DataFrame(atomic_rows)
    if atomic_df.empty:
        atomic_counts = pd.DataFrame(columns=["atomic_defect_label", "n", "eligible_rare_holdout"])
    else:
        atomic_counts = atomic_df["atomic_defect_label"].value_counts().rename_axis("atomic_defect_label").reset_index(name="n")
        atomic_counts["eligible_rare_holdout"] = atomic_counts["n"] >= min_samples_per_type

    display(primary_counts)
    display(atomic_counts)

    summary = {
        "min_samples_per_type": int(min_samples_per_type),
        "n_primary_defect_types": int(primary_counts["defect_type"].nunique()),
        "n_primary_eligible_types": int(primary_counts["eligible_primary_lodo"].sum()) if not primary_counts.empty else 0,
        "n_non_base_atomic_types": int(atomic_counts["atomic_defect_label"].nunique()) if not atomic_counts.empty else 0,
        "n_non_base_atomic_eligible_types": int(atomic_counts["eligible_rare_holdout"].sum()) if not atomic_counts.empty else 0,
        "full_lodo_recommended": bool(
            (primary_counts["eligible_primary_lodo"].sum() >= 2) if not primary_counts.empty else False
        ),
        "rare_defect_holdout_possible": bool(
            (atomic_counts["eligible_rare_holdout"].sum() >= 1) if not atomic_counts.empty else False
        ),
    }
    save_json(summary, MANIFEST_ROOT / "lodo_eligibility_v3_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return primary_counts, atomic_counts, summary


# %% E4_safe_lodo_or_rare_holdout
def make_primary_lodo_splits_v3(type_manifest, seed=SEED, min_samples_per_type=5):
    primary_counts, _atomic_counts, summary = evaluate_lodo_eligibility_v3(type_manifest, min_samples_per_type)
    eligible = primary_counts.loc[primary_counts["eligible_primary_lodo"], "defect_type"].tolist()
    if len(eligible) < 2:
        report = {
            "status": "skipped",
            "reason": "less_than_two_eligible_primary_defect_types",
            "eligible_primary_types": eligible,
            "note": "Do not claim full leave-one-defect-type-out generalization.",
        }
        save_json(report, MANIFEST_ROOT / "primary_lodo_v3_skip_report.json")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return None

    all_splits = []
    df = type_manifest[type_manifest["label"].isin([0, 1])].copy()
    for heldout in eligible:
        good = df[df["label"].eq(0)].copy()
        known_abn = df[(df["label"].eq(1)) & (~df["defect_type_primary"].eq(heldout))].copy()
        unknown_abn = df[(df["label"].eq(1)) & (df["defect_type_primary"].eq(heldout))].copy()
        if known_abn.empty or unknown_abn.empty:
            continue

        good_train, good_temp = train_test_split(good, test_size=0.4, random_state=seed)
        good_calib, good_test = train_test_split(good_temp, test_size=0.5, random_state=seed)
        known_train, known_temp = train_test_split(known_abn, test_size=0.4, random_state=seed)
        known_calib, known_test = train_test_split(known_temp, test_size=0.5, random_state=seed)

        parts = [
            (good_train, "train", "good_train"),
            (known_train, "train", "known_abnormal_train"),
            (good_calib, "calibration", "good_calibration"),
            (known_calib, "calibration", "known_abnormal_calibration"),
            (good_test, "known_test", "good_known_test"),
            (known_test, "known_test", "known_abnormal_test"),
            (good_test.copy(), "unknown_test", "good_unknown_test"),
            (unknown_abn, "unknown_test", "heldout_unknown_abnormal"),
        ]
        for part, split, role in parts:
            p = part.copy()
            p["heldout_defect_type"] = heldout
            p["lodo_split"] = split
            p["lodo_role"] = role
            all_splits.append(p)

    if not all_splits:
        return None
    out = pd.concat(all_splits, ignore_index=True)
    out_path = MANIFEST_ROOT / f"primary_lodo_splits_v3_seed{seed}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    display(out.groupby(["heldout_defect_type", "lodo_split", "label"]).size().reset_index(name="n"))
    return out

def make_rare_codefect_holdout_v3(type_manifest, seed=SEED, min_samples_per_type=5):
    _primary_counts, atomic_counts, summary = evaluate_lodo_eligibility_v3(type_manifest, min_samples_per_type)
    if atomic_counts.empty:
        return None
    eligible = atomic_counts.loc[atomic_counts["eligible_rare_holdout"], "atomic_defect_label"].tolist()
    if not eligible:
        report = {
            "status": "skipped",
            "reason": "no_eligible_non_base_atomic_defect",
            "note": "Rare co-defect holdout is unavailable.",
        }
        save_json(report, MANIFEST_ROOT / "rare_codefect_holdout_v3_skip_report.json")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return None

    df = type_manifest[type_manifest["label"].isin([0, 1])].copy()
    all_splits = []
    for heldout in eligible:
        has_heldout = df["defect_labels"].map(lambda s: heldout in json.loads(s))
        good = df[df["label"].eq(0)].copy()
        known_abn = df[(df["label"].eq(1)) & (~has_heldout)].copy()
        rare_abn = df[(df["label"].eq(1)) & (has_heldout)].copy()
        if known_abn.empty or rare_abn.empty:
            continue

        good_train, good_temp = train_test_split(good, test_size=0.4, random_state=seed)
        good_calib, good_test = train_test_split(good_temp, test_size=0.5, random_state=seed)
        known_train, known_temp = train_test_split(known_abn, test_size=0.4, random_state=seed)
        known_calib, known_test = train_test_split(known_temp, test_size=0.5, random_state=seed)

        parts = [
            (good_train, "train", "good_train"),
            (known_train, "train", "known_abnormal_train"),
            (good_calib, "calibration", "good_calibration"),
            (known_calib, "calibration", "known_abnormal_calibration"),
            (good_test, "known_test", "good_known_test"),
            (known_test, "known_test", "known_abnormal_test"),
            (good_test.copy(), "rare_codefect_test", "good_rare_codefect_test"),
            (rare_abn, "rare_codefect_test", "heldout_rare_codefect_abnormal"),
        ]
        for part, split, role in parts:
            p = part.copy()
            p["heldout_atomic_defect"] = heldout
            p["rare_holdout_split"] = split
            p["rare_holdout_role"] = role
            all_splits.append(p)

    if not all_splits:
        return None
    out = pd.concat(all_splits, ignore_index=True)
    out_path = MANIFEST_ROOT / f"rare_codefect_holdout_v3_seed{seed}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    display(out.groupby(["heldout_atomic_defect", "rare_holdout_split", "label"]).size().reset_index(name="n"))
    return out


# 02_execution

# Optional execution example. Uncomment to run.
# v3_execution
# duplicate_groups = build_duplicate_groups()

# Optional execution example. Uncomment to run.
# type_manifest_v3 = recover_defect_types_v3(duplicate_groups)
# primary_counts, atomic_counts, lodo_summary = evaluate_lodo_eligibility_v3(type_manifest_v3, min_samples_per_type=5)


# 03_Full LODO

# Optional execution example. Uncomment to run.
### 03_Full LODO
# primary_lodo_v3 = make_primary_lodo_splits_v3(type_manifest_v3, seed=11, min_samples_per_type=5)

# Optional execution example. Uncomment to run.
# Weaker fallback for rare co-defects:
# rare_holdout_v3 = make_rare_codefect_holdout_v3(type_manifest_v3, seed=11, min_samples_per_type=5)


# 04_robustness_plan

# Optional execution example. Uncomment to run.
# G.robustness_plan
# robustness_plan = make_robustness_plan_table()


# 05_Adjustment

# %% restore_protocol_versions
PROTOCOL_VERSION_V2 = "E2_group_calibrated"
PROTOCOL_VERSION_V3 = "E4_defect_type_group_calibrated"

# Keep old v2 helper functions working unless a v3-specific function is being run.
PROTOCOL_VERSION = PROTOCOL_VERSION_V2

def protocol_version_is(value, expected):
    return str(value) == str(expected)

def collect_results_by_protocol_versions(versions=(PROTOCOL_VERSION_V2, PROTOCOL_VERSION_V3)):
    df = collect_results()
    if df.empty:
        return df
    if "protocol_version" not in df.columns:
        df["protocol_version"] = np.nan
    out = df[df["protocol_version"].isin(list(versions))].copy()
    out.to_csv(TABLE_ROOT / "results_E2_v3_only.csv", index=False)
    return out


# %% robust_seed_and_completeness_diagnostics
def artifact_complete_relaxed_for_review(run_dir):

    run_dir = Path(run_dir)
    required = [
        "config.json",
        "metrics.json",
        "predictions.csv",
        "predictions_calibration.csv",
        "threshold_protocol.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return False, missing
    try:
        cfg = load_json(run_dir / "config.json", default={})
        met = load_json(run_dir / "metrics.json", default={})
        pred = pd.read_csv(run_dir / "predictions.csv")
        calib = pd.read_csv(run_dir / "predictions_calibration.csv")
    except Exception as e:
        return False, [f"unreadable:{type(e).__name__}"]

    issues = []
    if cfg.get("threshold_source") != "calibration" and met.get("threshold_source") != "calibration":
        issues.append("threshold_not_calibrated")
    if pred.empty:
        issues.append("empty_test_predictions")
    if calib.empty:
        issues.append("empty_calibration_predictions")
    if "y_true" in pred.columns and pred["y_true"].nunique() < 2:
        issues.append("single_class_test")
    auroc = met.get("auroc", met.get("image_AUROC", np.nan))
    auprc = met.get("auprc", met.get("image_AUPRC", np.nan))
    if pd.isna(auroc):
        issues.append("missing_auroc")
    if pd.isna(auprc):
        issues.append("missing_auprc")
    return len(issues) == 0, issues

def extract_seed_for_review(row):
    seed = row.get("seed")
    try:
        if pd.notna(seed) and int(seed) != 42:
            return int(seed)
    except Exception:
        pass
    text = f"{row.get('experiment_name', '')} {row.get('run_id', '')}"
    match = re.search(r"seed(\d+)", str(text))
    if match:
        return int(match.group(1))
    try:
        return int(seed)
    except Exception:
        return np.nan

def condition_key_for_review(row):
    paradigm = row.get("paradigm")
    model = row.get("model_name")
    method = row.get("method")
    label_ratio = row.get("abnormal_label_ratio")
    if pd.isna(label_ratio):
        label_ratio = row.get("label_ratio")
    protocol = row.get("protocol_version")
    return f"{protocol} | {model} | {paradigm} | {method} | ratio={label_ratio}"

def complete_seed_count_diagnostic_review(df=None):
    if df is None:
        df = collect_results_by_protocol_versions()
    if df.empty:
        print("No E2/v3 results found.")
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        ok, issues = artifact_complete_relaxed_for_review(row["run_dir"])
        rows.append({
            "condition": condition_key_for_review(row),
            "seed_recovered": extract_seed_for_review(row),
            "artifact_complete": ok,
            "issues": "; ".join(issues),
            "run_dir": row["run_dir"],
        })
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLE_ROOT / "complete_seed_detail_review.csv", index=False)

    summary = (
        detail[detail["artifact_complete"]]
        .groupby("condition")["seed_recovered"]
        .nunique()
        .reset_index(name="n_complete_seeds")
        .sort_values(["n_complete_seeds", "condition"])
    )
    summary["has_three_complete_seeds"] = summary["n_complete_seeds"] >= 3
    summary.to_csv(TABLE_ROOT / "complete_seed_summary_review.csv", index=False)
    display(summary)
    return summary


# %% safer_v3_lodo_judgment
def lodo_feasibility_review_v3(type_manifest_v3, min_unknown=5, min_known_train_abnormal=100):
    df = type_manifest_v3[type_manifest_v3["label"].isin([0, 1])].copy()
    if "defect_type_primary" not in df.columns:
        raise ValueError("Run recover_defect_types_v3(...) first.")

    rows = []
    for heldout, unknown_abn in df[df["label"].eq(1)].groupby("defect_type_primary"):
        known_abn = df[(df["label"].eq(1)) & (~df["defect_type_primary"].eq(heldout))]
        rows.append({
            "heldout_defect_type": heldout,
            "n_unknown_abnormal": int(len(unknown_abn)),
            "n_known_train_candidate_abnormal": int(len(known_abn)),
            "is_base_abnormal_label": heldout in BASE_ABNORMAL_LABELS_V3,
            "statistically_usable": bool(
                len(unknown_abn) >= min_unknown
                and len(known_abn) >= min_known_train_abnormal
                and heldout not in BASE_ABNORMAL_LABELS_V3
            ),
        })
    out = pd.DataFrame(rows).sort_values("n_unknown_abnormal", ascending=False)
    out.to_csv(MANIFEST_ROOT / "lodo_feasibility_review_v3.csv", index=False)
    display(out)

    usable = out[out["statistically_usable"]]
    summary = {
        "full_lodo_for_report": bool(len(usable) >= 2),
        "rare_codefect_holdout_for_report": bool(len(usable) >= 1),
        "usable_heldout_types": usable["heldout_defect_type"].tolist(),
    }
    save_json(summary, MANIFEST_ROOT / "lodo_feasibility_review_v3_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return out, summary


# 06_execution

# Optional execution example. Uncomment to run.
# duplicate_groups = build_duplicate_groups()

# Optional execution example. Uncomment to run.
# type_manifest_v3 = recover_defect_types_v3(duplicate_groups)

# %% analysis_finalization_constants
ANALYSIS_PROTOCOLS = {
    "E2_group_calibrated",
    "E4_defect_type_group_calibrated",
}

ANALYSIS_SUPERVISED_MODELS = ["resnet18", "efficientnet_b0", "convnext_tiny"]
ANALYSIS_ANOMALY_MODELS = ["patchcore_resnet18", "padim_resnet18", "stfpm_resnet18"]
ANALYSIS_LABEL_BUDGET_MODEL = "resnet18"
ANALYSIS_LABEL_BUDGET_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
ANALYSIS_SEEDS_MINIMUM = [11, 22, 33]

ANALYSIS_TABLE_ROOT = TABLE_ROOT / "analysis_final"
ANALYSIS_FIG_ROOT = FIG_ROOT / "analysis_final"
ANALYSIS_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
ANALYSIS_FIG_ROOT.mkdir(parents=True, exist_ok=True)

# %% canonicalize_final_runs
def infer_seed_from_text(text, fallback=np.nan):
    match = re.search(r"seed(\d+)", str(text))
    if match:
        return int(match.group(1))
    try:
        if pd.notna(fallback):
            return int(fallback)
    except Exception:
        pass
    return np.nan

def canonicalize_final_paradigm(row):
    paradigm = row.get("paradigm")
    if isinstance(paradigm, str) and paradigm and paradigm != "nan":
        if paradigm == "label_ratio_supervised":
            return "legacy_exclude"
        if paradigm == "label_efficient_supervised":
            return "legacy_label_efficient_exclude"
        return paradigm

    exp = str(row.get("experiment_name", "")).lower()
    model = str(row.get("model_name", "")).lower()
    method = str(row.get("method", "")).lower()

    if "label_budget" in exp or "defect_label_budget" in exp:
        return "defect_label_budget_supervised"
    if "supervised" in exp and any(m in model for m in ANALYSIS_SUPERVISED_MODELS):
        return "supervised"
    if method in {"patchcore", "padim", "stfpm"}:
        return "embedding_anomaly"
    if any(m in model for m in ["patchcore", "padim", "stfpm"]):
        return "embedding_anomaly"
    return "unknown_or_legacy"

def canonicalize_final_model(row):
    model = str(row.get("model_name", ""))
    exp = str(row.get("experiment_name", "")).lower()
    method = str(row.get("method", "")).lower()
    if method in {"patchcore", "padim", "stfpm"}:
        return f"{method}_resnet18"
    if "patchcore" in model.lower() or "patchcore" in exp:
        return "patchcore_resnet18"
    if "padim" in model.lower() or "padim" in exp:
        return "padim_resnet18"
    if "stfpm" in model.lower() or "stfpm" in exp:
        return "stfpm_resnet18"
    return model

def canonicalize_label_budget_ratio(row):
    for col in ["abnormal_label_ratio", "label_ratio"]:
        value = row.get(col)
        if pd.notna(value):
            try:
                return round(float(value), 4)
            except Exception:
                pass
    exp = str(row.get("experiment_name", ""))
    match = re.search(r"abnratio([0-9.]+)", exp)
    if match:
        return round(float(match.group(1)), 4)
    match = re.search(r"ratio([0-9.]+)", exp)
    if match:
        return round(float(match.group(1)), 4)
    return np.nan

def canonicalize_protocol_version(row):
    version = row.get("protocol_version")
    if isinstance(version, str) and version in ANALYSIS_PROTOCOLS:
        return version
    exp = str(row.get("experiment_name", ""))
    for version_name in ANALYSIS_PROTOCOLS:
        if version_name in exp:
            return version_name
    return version

def build_final_run_index(run_root=RUN_ROOT):
    df = collect_results(run_root)
    if df.empty:
        return df
    df = df.copy()
    df["protocol_version_canonical"] = df.apply(canonicalize_protocol_version, axis=1)
    df["paradigm_canonical"] = df.apply(canonicalize_final_paradigm, axis=1)
    df["model_canonical"] = df.apply(canonicalize_final_model, axis=1)
    df["seed_canonical"] = df.apply(
        lambda r: infer_seed_from_text(f"{r.get('experiment_name', '')} {r.get('run_id', '')}", r.get("seed")),
        axis=1,
    )
    df["label_budget_ratio_canonical"] = df.apply(canonicalize_label_budget_ratio, axis=1)

    # Recompute completion using the relaxed review function if available.
    complete_flags = []
    complete_issues = []
    for _, row in df.iterrows():
        if "artifact_complete_relaxed_for_review" in globals():
            ok, issues = artifact_complete_relaxed_for_review(row["run_dir"])
        elif "artifact_complete" in globals():
            ok, issues = artifact_complete(row["run_dir"])
        else:
            required = ["metrics.json", "predictions.csv", "predictions_calibration.csv", "threshold_protocol.json"]
            run_dir = Path(row["run_dir"])
            issues = [x for x in required if not (run_dir / x).exists()]
            ok = len(issues) == 0
        complete_flags.append(bool(ok))
        complete_issues.append("; ".join(issues))
    df["artifact_complete_final"] = complete_flags
    df["artifact_issues_final"] = complete_issues

    df.to_csv(ANALYSIS_TABLE_ROOT / "all_runs_canonicalized.csv", index=False)
    return df

def final_analysis_runs(df=None, allow_v3_label_budget=True):
    if df is None:
        df = build_final_run_index()
    if df.empty:
        return df

    keep = df["protocol_version_canonical"].isin(ANALYSIS_PROTOCOLS)
    keep &= df["artifact_complete_final"]
    keep &= df["paradigm_canonical"].isin([
        "supervised",
        "embedding_anomaly",
        "defect_label_budget_supervised",
        "label_budget_supervised",
    ])

    final = df[keep].copy()
    # Use one name in report tables.
    final["paradigm_final"] = final["paradigm_canonical"].replace({
        "label_budget_supervised": "defect_label_budget_supervised",
    })

    # Keep only required model families.
    is_supervised = final["paradigm_final"].eq("supervised") & final["model_canonical"].isin(ANALYSIS_SUPERVISED_MODELS)
    is_anomaly = final["paradigm_final"].eq("embedding_anomaly") & final["model_canonical"].isin(ANALYSIS_ANOMALY_MODELS)
    is_budget = (
        final["paradigm_final"].eq("defect_label_budget_supervised")
        & final["model_canonical"].eq(ANALYSIS_LABEL_BUDGET_MODEL)
        & final["label_budget_ratio_canonical"].isin(ANALYSIS_LABEL_BUDGET_RATIOS)
    )
    final = final[is_supervised | is_anomaly | is_budget].copy()

    final = deduplicate_final_runs(final)
    final.to_csv(ANALYSIS_TABLE_ROOT / "final_analysis_runs.csv", index=False)
    return final

def protocol_priority_for_final(version):
    priority = {
        "E4_defect_type_group_calibrated": 3,
        "E2_group_calibrated": 2,
    }
    return priority.get(str(version), 0)

def run_modified_time(path):
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0

def dedup_ratio_key(value):
    if pd.isna(value):
        return "none"
    return f"{float(value):.4f}"

def deduplicate_final_runs(final):
    if final.empty:
        return final

    work = final.copy()
    work["protocol_priority"] = work["protocol_version_canonical"].map(protocol_priority_for_final)
    work["run_modified_time"] = work["run_dir"].map(run_modified_time)
    work["dedup_ratio_key"] = work["label_budget_ratio_canonical"].map(dedup_ratio_key)
    work["dedup_key"] = (
        work["paradigm_final"].astype(str)
        + " | " + work["model_canonical"].astype(str)
        + " | seed=" + work["seed_canonical"].astype(str)
        + " | ratio=" + work["dedup_ratio_key"].astype(str)
    )

    before = len(work)
    duplicates = work[work.duplicated("dedup_key", keep=False)].copy()
    if not duplicates.empty:
        duplicates = duplicates.sort_values(
            ["dedup_key", "protocol_priority", "run_modified_time"],
            ascending=[True, False, False],
        )
        duplicates.to_csv(ANALYSIS_TABLE_ROOT / "duplicate_final_run_candidates.csv", index=False)

    selected = (
        work.sort_values(
            ["protocol_priority", "run_modified_time"],
            ascending=[False, False],
        )
        .drop_duplicates(
            subset=[
                "paradigm_final",
                "model_canonical",
                "seed_canonical",
                "dedup_ratio_key",
            ],
            keep="first",
        )
        .copy()
    )

    selected_keys = set(selected["dedup_key"])
    excluded = work[
        work["dedup_key"].isin(set(duplicates["dedup_key"]) if not duplicates.empty else set())
        & ~work.index.isin(selected.index)
    ].copy()
    excluded.to_csv(ANALYSIS_TABLE_ROOT / "duplicate_final_runs_excluded.csv", index=False)

    audit = {
        "n_before_dedup": int(before),
        "n_after_dedup": int(len(selected)),
        "n_excluded_duplicates": int(before - len(selected)),
        "expected_main_final_runs": int(
            len(ANALYSIS_SUPERVISED_MODELS) * len(ANALYSIS_SEEDS_MINIMUM)
            + len(ANALYSIS_ANOMALY_MODELS) * len(ANALYSIS_SEEDS_MINIMUM)
            + len(ANALYSIS_LABEL_BUDGET_RATIOS) * len(ANALYSIS_SEEDS_MINIMUM)
        ),
        "duplicate_candidates_csv": str(ANALYSIS_TABLE_ROOT / "duplicate_final_run_candidates.csv"),
        "excluded_duplicates_csv": str(ANALYSIS_TABLE_ROOT / "duplicate_final_runs_excluded.csv"),
    }
    save_json(audit, ANALYSIS_TABLE_ROOT / "final_run_dedup_audit.json")
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    return selected.drop(columns=["dedup_ratio_key", "dedup_key"], errors="ignore")

### 07_missing_condition_plan
def required_final_conditions():
    rows = []
    for model in ANALYSIS_SUPERVISED_MODELS:
        for seed in ANALYSIS_SEEDS_MINIMUM:
            rows.append({
                "family": "supervised",
                "model_canonical": model,
                "seed": seed,
                "label_budget_ratio": np.nan,
                "rerun_call": f'train_supervised_model_E2(model_name="{model}", seed={seed}, epochs=5, batch_size=32, overwrite=False)',
            })
    for model in ANALYSIS_ANOMALY_MODELS:
        method = model.split("_")[0]
        for seed in ANALYSIS_SEEDS_MINIMUM:
            rows.append({
                "family": "embedding_anomaly",
                "model_canonical": model,
                "seed": seed,
                "label_budget_ratio": np.nan,
                "rerun_call": f'train_anomalib_embedding_model_E2(method="{method}", backbone="resnet18", seed={seed}, train_batch_size=1, eval_batch_size=1, overwrite=False)',
            })
    for ratio in ANALYSIS_LABEL_BUDGET_RATIOS:
        for seed in ANALYSIS_SEEDS_MINIMUM:
            rows.append({
                "family": "defect_label_budget_supervised",
                "model_canonical": ANALYSIS_LABEL_BUDGET_MODEL,
                "seed": seed,
                "label_budget_ratio": ratio,
                "rerun_call": f'train_label_budget_keep_all_normals_E2(model_name="resnet18", abnormal_label_ratio={ratio}, seed={seed}, epochs=5, batch_size=32, overwrite=False)',
            })
    return pd.DataFrame(rows)

def diagnose_missing_final_conditions(final=None):
    if final is None:
        final = final_analysis_runs()
    required = required_final_conditions()
    have_rows = []
    for _, r in final.iterrows():
        family = r["paradigm_final"]
        ratio = r["label_budget_ratio_canonical"] if family == "defect_label_budget_supervised" else np.nan
        have_rows.append({
            "family": family,
            "model_canonical": r["model_canonical"],
            "seed": int(r["seed_canonical"]) if pd.notna(r["seed_canonical"]) else np.nan,
            "label_budget_ratio": ratio,
        })
    have = pd.DataFrame(have_rows).drop_duplicates() if have_rows else pd.DataFrame(columns=required.columns)

    def key_tuple(row):
        ratio = row["label_budget_ratio"]
        ratio_key = "nan" if pd.isna(ratio) else f"{float(ratio):.4f}"
        return (row["family"], row["model_canonical"], int(row["seed"]), ratio_key)

    have_keys = set()
    if not have.empty:
        for _, row in have.iterrows():
            if pd.notna(row["seed"]):
                have_keys.add(key_tuple(row))

    missing_rows = []
    for _, row in required.iterrows():
        if key_tuple(row) not in have_keys:
            missing_rows.append(row.to_dict())
    missing = pd.DataFrame(missing_rows)
    missing.to_csv(ANALYSIS_TABLE_ROOT / "missing_final_conditions.csv", index=False)

    coverage = (
        have.groupby(["family", "model_canonical", "label_budget_ratio"], dropna=False)["seed"]
        .nunique()
        .reset_index(name="n_complete_seeds")
        if not have.empty else pd.DataFrame()
    )
    coverage.to_csv(ANALYSIS_TABLE_ROOT / "final_condition_seed_coverage.csv", index=False)
    display(coverage)
    if missing.empty:
        print("All required final conditions have the minimum seeds.")
    else:
        print("Missing final conditions:", len(missing))
        display(missing[["family", "model_canonical", "seed", "label_budget_ratio"]])
    return missing, coverage

def print_rerun_block(missing):
    if missing is None or missing.empty:
        print("# No reruns needed.")
        return
    print("# Copy/paste this rerun block. Existing complete runs will be skipped.")
    for call in missing["rerun_call"].drop_duplicates():
        print(call)

# %% reports
def mean_sd(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return ""
    if len(vals) == 1:
        return f"{vals.iloc[0]:.4f}"
    return f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"

def metric_col(df, name):
    candidates = [f"metric_{name}", name, name.upper()]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def make_main_performance_table(final=None):
    if final is None:
        final = final_analysis_runs()
    main = final[final["paradigm_final"].isin(["supervised", "embedding_anomaly"])].copy()
    rows = []
    group_cols = ["paradigm_final", "model_canonical"]
    metrics = ["auroc", "auprc", "f1", "fnr", "recall_sensitivity", "specificity", "accuracy", "mcc"]
    for (paradigm, model), g in main.groupby(group_cols):
        row = {
            "Paradigm": "Supervised" if paradigm == "supervised" else "Normal-only anomaly",
            "Model": model,
            "Seeds": int(g["seed_canonical"].nunique()),
        }
        for metric in metrics:
            col = metric_col(g, metric)
            row[metric.upper() if metric != "recall_sensitivity" else "Sensitivity"] = mean_sd(g[col]) if col else ""
        lat_col = "eff_latency_ms_per_image" if "eff_latency_ms_per_image" in g.columns else None
        row["Latency ms/img"] = mean_sd(g[lat_col]) if lat_col else ""
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["Paradigm", "Model"])
    table.to_csv(ANALYSIS_TABLE_ROOT / "table_main_performance_mean_sd.csv", index=False)
    display(table)
    return table

def make_label_budget_table(final=None):
    if final is None:
        final = final_analysis_runs()
    budget = final[final["paradigm_final"].eq("defect_label_budget_supervised")].copy()
    rows = []
    for ratio, g in budget.groupby("label_budget_ratio_canonical"):
        row = {
            "Abnormal label ratio": ratio,
            "Seeds": int(g["seed_canonical"].nunique()),
            "Normal train n": mean_sd(g["n_labeled_normal"]) if "n_labeled_normal" in g else "",
            "Abnormal train n": mean_sd(g["n_labeled_abnormal"]) if "n_labeled_abnormal" in g else "",
        }
        for metric in ["auroc", "auprc", "f1", "fnr", "recall_sensitivity", "specificity"]:
            col = metric_col(g, metric)
            row[metric.upper() if metric != "recall_sensitivity" else "Sensitivity"] = mean_sd(g[col]) if col else ""
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("Abnormal label ratio")
    table.to_csv(ANALYSIS_TABLE_ROOT / "table_label_budget_mean_sd.csv", index=False)
    display(table)
    return table

# %% reports
def plot_label_budget_crossover(final=None):
    if final is None:
        final = final_analysis_runs()
    budget = final[final["paradigm_final"].eq("defect_label_budget_supervised")].copy()
    if budget.empty:
        print("empty")
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = (
        budget.groupby("label_budget_ratio_canonical")
        .agg(
            auroc_mean=("metric_auroc", "mean"),
            auroc_sd=("metric_auroc", "std"),
            f1_mean=("metric_f1", "mean"),
            f1_sd=("metric_f1", "std"),
        )
        .reset_index()
        .sort_values("label_budget_ratio_canonical")
    )
    x = summary["label_budget_ratio_canonical"].astype(float) * 100
    ax.errorbar(x, summary["auroc_mean"], yerr=summary["auroc_sd"], marker="o", label="Label-budget AUROC")
    ax.errorbar(x, summary["f1_mean"], yerr=summary["f1_sd"], marker="s", label="Label-budget F1")

    anomaly = final[final["paradigm_final"].eq("embedding_anomaly")]
    if not anomaly.empty:
        for model, g in anomaly.groupby("model_canonical"):
            if "metric_auroc" in g:
                ax.axhline(g["metric_auroc"].mean(), linestyle="--", linewidth=1.2, label=f"{model} AUROC")

    ax.set_xlabel("Labeled abnormal training samples (%)")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Annotation-budget crossover")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = ANALYSIS_FIG_ROOT / "fig_label_budget_crossover.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)
    return path

def plot_fnr_vs_label_budget(final=None):
    if final is None:
        final = final_analysis_runs()
    budget = final[final["paradigm_final"].eq("defect_label_budget_supervised")].copy()
    if budget.empty or "metric_fnr" not in budget.columns:
        print("No label-budget FNR runs to plot.")
        return None
    summary = (
        budget.groupby("label_budget_ratio_canonical")
        .agg(fnr_mean=("metric_fnr", "mean"), fnr_sd=("metric_fnr", "std"))
        .reset_index()
        .sort_values("label_budget_ratio_canonical")
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = summary["label_budget_ratio_canonical"].astype(float) * 100
    ax.errorbar(x, summary["fnr_mean"], yerr=summary["fnr_sd"], marker="o", color="crimson")
    ax.set_xlabel("Labeled abnormal training samples (%)")
    ax.set_ylabel("False negative rate")
    ax.set_title("False negatives versus annotation budget")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = ANALYSIS_FIG_ROOT / "fig_fnr_vs_annotation_budget.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)
    return path


# %% final_analysis_and_next_step
def build_final_analysis_report():
    all_runs = build_final_run_index()
    final = final_analysis_runs(all_runs)
    missing, coverage = diagnose_missing_final_conditions(final)

    paradigms = set(final["paradigm_final"].dropna()) if not final.empty else set()
    has_supervised = "supervised" in paradigms and any(final["model_canonical"].isin(ANALYSIS_SUPERVISED_MODELS))
    has_embedding_anomaly = "embedding_anomaly" in paradigms and any(final["model_canonical"].isin(ANALYSIS_ANOMALY_MODELS))
    has_label_budget = "defect_label_budget_supervised" in paradigms

    report = {
        "n_final_runs": int(len(final)),
        "has_supervised": bool(has_supervised),
        "has_embedding_anomaly": bool(has_embedding_anomaly),
        "has_label_budget": bool(has_label_budget),
        "conditions_without_required_seed": int(len(missing)),
        "ready_for_main_results_tables": bool(
            has_supervised and has_embedding_anomaly and has_label_budget and len(missing) == 0
        ),
        "final_runs_csv": str(ANALYSIS_TABLE_ROOT / "final_analysis_runs.csv"),
        "missing_conditions_csv": str(ANALYSIS_TABLE_ROOT / "missing_final_conditions.csv"),
    }
    save_json(report, ANALYSIS_TABLE_ROOT / "build_final_analysis_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if missing.empty:
        make_main_performance_table(final)
        make_label_budget_table(final)
        plot_label_budget_crossover(final)
        plot_fnr_vs_label_budget(final)
    else:
        print_rerun_block(missing)
    return report, final, missing

# Optional execution example. Uncomment to run.
# report, final_runs, missing_conditions = build_final_analysis_report()


# Additional Analysis_E5

# %% E5_constants
E5_ROOT = PROJECT_ROOT / "E5"
TABLE_ROOT = E5_ROOT / "tables"
FIG_ROOT = E5_ROOT / "figures"
TABLE_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

FINE_LABEL_COUNTS = [4, 8, 16, 24, 32, 64]
LOW_LABEL_REPEATS = [0, 1, 2, 3, 4]


# 01_normal_only_threshold_variants

# %% normal_only_threshold_variants
def threshold_from_normal_scores_only(normal_scores, method="p95", k=3.0):
    normal_scores = np.asarray(normal_scores, dtype=float)
    normal_scores = normal_scores[~np.isnan(normal_scores)]
    if len(normal_scores) == 0:
        return float("nan")
    method = str(method).lower()
    if method in {"p95", "percentile95", "95"}:
        return float(np.percentile(normal_scores, 95))
    if method in {"p99", "percentile99", "99"}:
        return float(np.percentile(normal_scores, 99))
    if method in {"mean+3sd", "mean_plus_3sd"}:
        return float(normal_scores.mean() + k * normal_scores.std(ddof=1))
    raise ValueError(f"Unknown normal-only threshold method: {method}")

def evaluate_existing_run_with_normal_only_threshold(run_dir, method="p95", overwrite=False):
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json", default={})
    calib_path = run_dir / "predictions_calibration.csv"
    test_path = run_dir / "scores_test_unthresholded.csv"
    if not test_path.exists():
        test_path = run_dir / "predictions.csv"
    if not calib_path.exists() or not test_path.exists():
        print("Missing prediction files:", run_dir)
        return None

    calib = pd.read_csv(calib_path)
    test = pd.read_csv(test_path)
    if not {"y_true", "y_score"}.issubset(calib.columns) or not {"y_true", "y_score"}.issubset(test.columns):
        print("Missing y_true/y_score:", run_dir)
        return None

    normal_scores = calib.loc[calib["y_true"].eq(0), "y_score"]
    threshold = threshold_from_normal_scores_only(normal_scores, method=method)

    model_name = cfg.get("model_name", run_dir.name.split("__")[0])
    base_exp = cfg.get("experiment_name", run_dir.name.split("__", 1)[-1])
    seed = cfg.get("seed")
    if seed == 42:
        seed = extract_seed_for_review(cfg) if "extract_seed_for_review" in globals() else seed

    tracker = ExperimentTracker(
        model_name,
        f"{base_exp}_normal_only_threshold_{method}",
        config={
            **cfg,
            "paradigm": "embedding_anomaly_normal_threshold",
            "threshold_source": "normal_calibration_only",
            "threshold_method": method,
            "abnormal_labels_for_training": 0,
            "abnormal_labels_for_threshold_calibration": 0,
            "source_run_dir": str(run_dir),
            "seed": seed,
            "E5": True,
        },
        overwrite=overwrite,
    )
    if tracker.metrics_path.exists() and tracker.pred_path.exists() and not overwrite:
        print("skipped:", tracker.run_dir)
        return tracker

    save_json(
        {
            "threshold": threshold,
            "threshold_source": "normal_calibration_only",
            "threshold_method": method,
            "calibration_n_normal": int((calib["y_true"] == 0).sum()),
            "calibration_n_abnormal_used_for_threshold": 0,
            "note": "Abnormal calibration labels were not used for threshold selection.",
        },
        tracker.artifact("threshold_protocol.json"),
    )
    metrics, pred_df = save_binary_evaluation(
        tracker,
        test["y_true"],
        test["y_score"],
        paths=test["path"] if "path" in test.columns else None,
        threshold=threshold,
    )
    metrics["threshold_source"] = "normal_calibration_only"
    metrics["threshold_method"] = method
    metrics["abnormal_labels_for_threshold_calibration"] = 0
    tracker.save_metrics(metrics)
    print(metrics)
    return tracker

def create_normal_only_threshold_variants(methods=("p95", "p99"), final_runs=None):
    if final_runs is None:
        final_runs = final_analysis_runs()
    anomaly = final_runs[final_runs["paradigm_final"].eq("embedding_anomaly")].copy()
    trackers = []
    for _, row in anomaly.iterrows():
        for method in methods:
            trackers.append(evaluate_existing_run_with_normal_only_threshold(row["run_dir"], method=method))
    return trackers

# %% fixed_count_label_budget
def make_label_count_train_df_keep_all_normals(split_df, abnormal_label_count, seed, subset_repeat=0):
    train = split_df[split_df["protocol_split"].eq("train")].copy()
    normal = train[train["label"].eq(0)].copy()
    abnormal = train[train["label"].eq(1)].copy()
    if abnormal.empty:
        raise RuntimeError("No abnormal training samples in protocol split.")
    n_abn = min(int(abnormal_label_count), len(abnormal))
    draw_seed = int(seed) * 1000 + int(subset_repeat)
    abnormal_sample = abnormal.sample(n=n_abn, random_state=draw_seed)
    out = pd.concat([normal, abnormal_sample], ignore_index=True)
    out["abnormal_label_count"] = n_abn
    out["subset_repeat"] = subset_repeat
    out["normal_sampling"] = "all_normals_kept"
    return out

def train_label_count_keep_all_normals_E2(
    model_name="resnet18",
    abnormal_label_count=32,
    subset_repeat=0,
    preprocess_name="resize224_imagenet",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    seed=SEED,
    threshold_method="youden",
    overwrite=False,
):
    set_all_seeds(seed + int(subset_repeat))
    split_path = MANIFEST_ROOT / f"{PROTOCOL_VERSION}_split_seed{seed}.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else make_E2_split(seed=seed)
    train_df = make_label_count_train_df_keep_all_normals(split_df, abnormal_label_count, seed, subset_repeat)

    experiment_name = protocol_experiment_name(
        f"label_count_abn{int(abnormal_label_count)}_draw{subset_repeat}_allnormal_{preprocess_name}_ep{epochs}_bs{batch_size}",
        seed,
        model_name,
    )
    if "should_skip_run" in globals() and should_skip_run(model_name, experiment_name, overwrite=overwrite):
        return ExperimentTracker(model_name, experiment_name, config={"section": "06_label_count_budget_models"}, overwrite=False)

    tracker = ExperimentTracker(
        model_name,
        experiment_name,
        config={
            "section": "06_label_count_budget_models",
            "paradigm": "defect_label_count_budget_supervised",
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "subset_repeat": subset_repeat,
            "abnormal_label_count": int(abnormal_label_count),
            "normal_sampling": "all_normals_kept",
            "preprocessing": preprocess_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "threshold_source": "calibration",
            "group_safe_split": True,
            "E5": True,
        },
        overwrite=overwrite,
    )
    if "mark_run_status" in globals():
        mark_run_status(tracker.run_dir, "running")

    clear_gpu()
    train_loader = make_loader_from_arbitrary_df(train_df, preprocess_name, batch_size, shuffle=True)
    calib_loader, _ = make_loader_from_df(split_df, "calibration", preprocess_name, batch_size, shuffle=False)
    test_loader, _ = make_loader_from_df(split_df, "test", preprocess_name, batch_size, shuffle=False)

    model, input_shape = build_supervised_model(model_name)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "model_name": model_name,
        "seed": seed,
        "subset_repeat": subset_repeat,
        "abnormal_label_count": int(abnormal_label_count),
        "preprocessing": preprocess_name,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    resume_state = load_resume_checkpoint(tracker, model, optimizer, expected=expected) if "load_resume_checkpoint" in globals() else {"start_epoch": 1, "history": []}
    history = resume_state["history"]

    try:
        for epoch in range(resume_state["start_epoch"], epochs + 1):
            model.train()
            t0 = time.perf_counter()
            total_loss = 0.0
            n_seen = 0
            for x, y, _paths in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                bs = x.size(0)
                total_loss += float(loss.item()) * bs
                n_seen += bs
            epoch_record = {
                "epoch": epoch,
                "train_loss": total_loss / max(n_seen, 1),
                "elapsed_sec": time.perf_counter() - t0,
            }
            history = _dedupe_history_append(history, epoch_record) if "_dedupe_history_append" in globals() else history + [epoch_record]
            print(f"{model_name} n_abnormal={abnormal_label_count} draw={subset_repeat} seed={seed} epoch {epoch}/{epochs} loss={epoch_record['train_loss']:.6f}")
            if "save_resume_checkpoint" in globals():
                save_resume_checkpoint(tracker, model, optimizer, epoch, history, extra=expected)
            save_training_history(tracker, history)

        tracker.save_checkpoint(model, "model_state_dict.pt")
        train_df.to_csv(tracker.artifact("labeled_train_subset.csv"), index=False)
        save_json({"input_shape": input_shape, "n_labeled_train": int(len(train_df))}, tracker.artifact("model_meta.json"))
        calib_scores = collect_supervised_scores(model, calib_loader)
        test_scores = collect_supervised_scores(model, test_loader)
        metrics, _ = save_calibrated_score_evaluation(
            tracker,
            calib_scores,
            test_scores,
            threshold_method=threshold_method,
            extra_config={
                "seed": seed,
                "subset_repeat": subset_repeat,
                "abnormal_label_count": int(abnormal_label_count),
                "n_labeled_train": int(len(train_df)),
                "n_labeled_normal": int((train_df.label == 0).sum()),
                "n_labeled_abnormal": int((train_df.label == 1).sum()),
                "n_test_normal": int((test_scores.y_true == 0).sum()),
                "n_test_abnormal": int((test_scores.y_true == 1).sum()),
            },
        )
        save_efficiency_report(model, tracker, input_shape=input_shape)
        if "artifact_complete" in globals() and "mark_run_status" in globals():
            ok, issues = artifact_complete(tracker.run_dir)
            mark_run_status(tracker.run_dir, "complete" if ok else "incomplete", issues)
        print(metrics)
        return tracker
    except Exception as e:
        if "mark_run_status" in globals():
            mark_run_status(tracker.run_dir, "failed", [f"{type(e).__name__}: {e}"])
        raise

def run_fine_crossover_label_count_suite(
    counts=FINE_LABEL_COUNTS,
    seeds=(11, 22, 33),
    repeats=LOW_LABEL_REPEATS,
    epochs=5,
    batch_size=32,
    overwrite=False,
):
    trackers = []
    for seed in seeds:
        make_E2_split(seed=seed)
        for count in counts:
            for repeat in repeats:
                trackers.append(train_label_count_keep_all_normals_E2(
                    model_name="resnet18",
                    abnormal_label_count=count,
                    subset_repeat=repeat,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    overwrite=overwrite,
                ))
    return trackers


# 02_make_100pct_label_budget_alias_from_supervised

# %% make_100pct_label_budget_alias_from_supervised
def copy_artifact_if_exists(src_dir, dst_dir, name):
    src = Path(src_dir) / name
    dst = Path(dst_dir) / name
    if src.exists():
        import shutil
        shutil.copy2(src, dst)

def create_100pct_budget_alias_from_supervised(overwrite=False):
    all_runs = build_final_run_index()
    sup = all_runs[
        (all_runs["model_canonical"].eq("resnet18"))
        & (all_runs["paradigm_canonical"].eq("supervised"))
        & (all_runs["artifact_complete_final"])
    ].copy()
    trackers = []
    for _, row in sup.iterrows():
        seed = int(row["seed_canonical"])
        source_dir = Path(row["run_dir"])
        tracker = ExperimentTracker(
            "resnet18",
            protocol_experiment_name(
                "label_budget_abnratio1_allnormal_ALIAS_FROM_FULL_SUPERVISED_resize224_imagenet_ep5_bs32",
                seed,
                "resnet18",
            ),
            config={
                "section": "06_label_budget_models",
                "paradigm": "defect_label_budget_supervised",
                "protocol_version": row["protocol_version_canonical"],
                "seed": seed,
                "abnormal_label_ratio": 1.0,
                "normal_sampling": "all_normals_kept",
                "source_run_dir": str(source_dir),
                "alias_from_full_supervised": True,
                "threshold_source": "calibration",
                "group_safe_split": True,
                "E5": True,
            },
            overwrite=overwrite,
        )
        if tracker.metrics_path.exists() and tracker.pred_path.exists() and not overwrite:
            print("Existing 100% alias:", tracker.run_dir)
            trackers.append(tracker)
            continue
        for name in [
            "metrics.json",
            "predictions.csv",
            "predictions_calibration.csv",
            "threshold_protocol.json",
            "confusion_matrix.csv",
            "roc_curve.csv",
            "pr_curve.csv",
            "efficiency.json",
            "model_state_dict.pt",
        ]:
            copy_artifact_if_exists(source_dir, tracker.run_dir, name)
        metrics = load_json(tracker.metrics_path, default={})
        metrics["alias_from_full_supervised"] = True
        metrics["abnormal_label_ratio"] = 1.0
        tracker.save_metrics(metrics)
        save_json({"status": "complete", "issues": [], "source_run_dir": str(source_dir)}, tracker.artifact("run_status.json"))
        trackers.append(tracker)
    return trackers


# 03_fine_crossover_tables_and_figures

# %% fine_crossover_tables_and_figures
def collect_label_count_budget_results(run_root=RUN_ROOT):
    df = collect_results(run_root)
    if df.empty:
        return df
    df = df[df.get("paradigm").eq("defect_label_count_budget_supervised")].copy()
    if df.empty:
        return df
    df["seed_recovered"] = df.apply(lambda r: infer_seed_from_text(r.get("experiment_name", ""), r.get("seed")), axis=1)
    df["abnormal_label_count"] = pd.to_numeric(df["abnormal_label_count"], errors="coerce")
    df["subset_repeat"] = pd.to_numeric(df["subset_repeat"], errors="coerce")
    df.to_csv(TABLE_ROOT / "label_count_budget_results_long.csv", index=False)
    return df

def make_fine_crossover_table():
    df = collect_label_count_budget_results()
    if df.empty:
        print("empty")
        return df
    rows = []
    for count, g in df.groupby("abnormal_label_count"):
        rows.append({
            "Abnormal labels": int(count),
            "Runs": int(len(g)),
            "Seeds": int(g["seed_recovered"].nunique()),
            "Subset draws": int(g["subset_repeat"].nunique()),
            "AUROC": mean_sd(g["metric_auroc"]),
            "AUPRC": mean_sd(g["metric_auprc"]),
            "F1": mean_sd(g["metric_f1"]),
            "FNR": mean_sd(g["metric_fnr"]),
        })
    table = pd.DataFrame(rows).sort_values("Abnormal labels")
    table.to_csv(TABLE_ROOT / "table_fine_crossover_label_counts.csv", index=False)
    display(table)
    return table

def plot_fine_crossover_against_patchcore():
    df = collect_label_count_budget_results()
    if df.empty:
        print("empty")
        return None
    final = final_analysis_runs()
    patch = final[
        (final["paradigm_final"].eq("embedding_anomaly"))
        & (final["model_canonical"].eq("patchcore_resnet18"))
    ]
    patch_fnr = patch["metric_fnr"].mean() if not patch.empty and "metric_fnr" in patch else np.nan
    patch_auroc = patch["metric_auroc"].mean() if not patch.empty and "metric_auroc" in patch else np.nan

    summary = (
        df.groupby("abnormal_label_count")
        .agg(
            auroc_mean=("metric_auroc", "mean"),
            auroc_sd=("metric_auroc", "std"),
            fnr_mean=("metric_fnr", "mean"),
            fnr_sd=("metric_fnr", "std"),
        )
        .reset_index()
        .sort_values("abnormal_label_count")
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = summary["abnormal_label_count"]
    axes[0].errorbar(x, summary["auroc_mean"], yerr=summary["auroc_sd"], marker="o")
    if pd.notna(patch_auroc):
        axes[0].axhline(patch_auroc, linestyle="--", color="gray", label="PatchCore")
    axes[0].set_xlabel("Labeled abnormal training images")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Fine-grained crossover: AUROC")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(x, summary["fnr_mean"], yerr=summary["fnr_sd"], marker="o", color="crimson")
    if pd.notna(patch_fnr):
        axes[1].axhline(patch_fnr, linestyle="--", color="gray", label="PatchCore")
    axes[1].set_xlabel("Labeled abnormal training images")
    axes[1].set_ylabel("FNR")
    axes[1].set_title("Fine-grained crossover: FNR")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path = FIG_ROOT / "fig_fine_label_count_crossover.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)
    return path


# 04_gradcam_shortcut_check

# %% gradcam_shortcut_check
def choose_gradcam_target_layer(model, model_name):
    name = model_name.lower()
    if "resnet" in name:
        return model.layer4[-1]
    if "efficientnet" in name:
        return model.features[-1]
    if "convnext" in name:
        return model.features[-1]
    raise ValueError(f"No target layer rule for {model_name}")

def load_supervised_model_from_run(run_dir):
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json", default={})
    model_name = cfg.get("model_name", run_dir.name.split("__")[0])
    model, _input_shape = build_supervised_model(model_name, pretrained=False)
    state_path = run_dir / "model_state_dict.pt"
    if not state_path.exists():
        state_path = run_dir / "best_model_state_dict.pt"
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model, model_name

def select_prediction_examples(pred_df, n_each=3):
    df = pred_df.copy()
    df["case_type"] = np.select(
        [
            (df.y_true.eq(1) & df.y_pred.eq(1)),
            (df.y_true.eq(0) & df.y_pred.eq(0)),
            (df.y_true.eq(0) & df.y_pred.eq(1)),
            (df.y_true.eq(1) & df.y_pred.eq(0)),
        ],
        ["TP", "TN", "FP", "FN"],
        default="other",
    )
    out = []
    for case_type in ["TP", "TN", "FP", "FN"]:
        part = df[df["case_type"].eq(case_type)].copy()
        if part.empty:
            continue
        if case_type in ["TP", "FP"]:
            part = part.sort_values("y_score", ascending=False)
        else:
            part = part.sort_values("y_score", ascending=True)
        out.append(part.head(n_each))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def make_gradcam_shortcut_check(run_dir, n_each=3, preprocess_name="resize224_imagenet"):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as e:
        raise ImportError("Install grad-cam first: pip install grad-cam") from e

    run_dir = Path(run_dir)
    pred = pd.read_csv(run_dir / "predictions.csv")
    examples = select_prediction_examples(pred, n_each=n_each)
    if examples.empty:
        print("empty")
        return None

    model, model_name = load_supervised_model_from_run(run_dir)
    target_layer = choose_gradcam_target_layer(model, model_name)
    transform = PREPROCESSING[preprocess_name]
    out_dir = FIG_ROOT / "gradcam" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for i, row in examples.iterrows():
            img_path = row["path"]
            pil = Image.open(img_path).convert("RGB")
            x = transform(pil).unsqueeze(0).to(DEVICE)
            rgb = np.asarray(pil.resize((x.shape[-1], x.shape[-2]))) / 255.0
            grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
            overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)

            save_path = out_dir / f"{i:02d}_{row['case_type']}_true{int(row.y_true)}_pred{int(row.y_pred)}.png"
            Image.fromarray(overlay).save(save_path)

            attention_ratio = np.nan
            if "find_label_json_by_stem" in globals() and "robust_mask_from_json" in globals():
                json_path = find_label_json_by_stem(img_path)
                if json_path:
                    mask = robust_mask_from_json(json_path, np.asarray(pil).shape)
                    mask = resize_mask_nearest(mask, grayscale_cam.shape)
                    total = float(grayscale_cam.sum() + 1e-12)
                    attention_ratio = float(grayscale_cam[mask.astype(bool)].sum() / total)

            records.append({
                "path": img_path,
                "case_type": row["case_type"],
                "y_true": int(row.y_true),
                "y_pred": int(row.y_pred),
                "y_score": float(row.y_score),
                "gradcam_path": str(save_path),
                "attention_localization_ratio": attention_ratio,
            })
    record_df = pd.DataFrame(records)
    record_df.to_csv(out_dir / "gradcam_examples.csv", index=False)
    display(record_df)
    print("Saved Grad-CAM examples:", out_dir)
    return record_d_read_threshold_protocol

def run_gradcam_for_best_supervised_model(final_runs=None, n_each=3):
    if final_runs is None:
        final_runs = final_analysis_runs()
    sup = final_runs[final_runs["paradigm_final"].eq("supervised")].copy()
    if sup.empty:
        print("No supervised final runs.")
        return None
    best = sup.sort_values("metric_auroc", ascending=False).iloc[0]
    print("Grad-CAM target run:", best["run_dir"])
    return make_gradcam_shortcut_check(best["run_dir"], n_each=n_each)

# %% analysis_completion_report
def analysis_completion_report():
    final = final_analysis_runs()
    normal_threshold_runs = collect_results()
    n_no = int((normal_threshold_runs.get("paradigm", pd.Series(dtype=str)) == "embedding_anomaly_normal_threshold").sum()) if not normal_threshold_runs.empty else 0
    fine = collect_label_count_budget_results()
    alias_runs = final[
        (final["paradigm_final"].eq("defect_label_budget_supervised"))
        & (final["label_budget_ratio_canonical"].eq(1.0))
        & (final.get("alias_from_full_supervised", False).astype(str).str.lower().isin(["true", "1"]))
    ] if not final.empty and "alias_from_full_supervised" in final.columns else pd.DataFrame()
    summary = {
        "normal_only_threshold_variants": n_no,
        "fine_label_count_runs": int(len(fine)) if fine is not None and not fine.empty else 0,
        "has_100pct_alias_from_full_supervised": bool(len(alias_runs) > 0),
    }
    save_json(summary, TABLE_ROOT / "analysis_completion_report.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

# %% analysis_finalization_constants
ANALYSIS_PROTOCOLS = {
    "E2_group_calibrated",
    "E4_defect_type_group_calibrated",
}

ANALYSIS_SUPERVISED_MODELS = ["resnet18", "efficientnet_b0", "convnext_tiny"]
ANALYSIS_ANOMALY_MODELS = ["patchcore_resnet18", "padim_resnet18", "stfpm_resnet18"]
ANALYSIS_LABEL_BUDGET_MODEL = "resnet18"
ANALYSIS_LABEL_BUDGET_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
ANALYSIS_SEEDS_MINIMUM = [11, 22, 33]

ANALYSIS_TABLE_ROOT = TABLE_ROOT / "analysis_final"
ANALYSIS_FIG_ROOT = FIG_ROOT / "analysis_final"
ANALYSIS_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
ANALYSIS_FIG_ROOT.mkdir(parents=True, exist_ok=True)


# %% canonicalize_final_runs
def infer_seed_from_text(text, fallback=np.nan):
    match = re.search(r"seed(\d+)", str(text))
    if match:
        return int(match.group(1))
    try:
        if pd.notna(fallback):
            return int(fallback)
    except Exception:
        pass
    return np.nan

def canonicalize_final_paradigm(row):
    paradigm = row.get("paradigm")
    if isinstance(paradigm, str) and paradigm and paradigm != "nan":
        if paradigm == "label_ratio_supervised":
            return "legacy_exclude"
        if paradigm == "label_efficient_supervised":
            return "legacy_label_efficient_exclude"
        return paradigm

    exp = str(row.get("experiment_name", "")).lower()
    model = str(row.get("model_name", "")).lower()
    method = str(row.get("method", "")).lower()

    if "label_budget" in exp or "defect_label_budget" in exp:
        return "defect_label_budget_supervised"
    if "supervised" in exp and any(m in model for m in ANALYSIS_SUPERVISED_MODELS):
        return "supervised"
    if method in {"patchcore", "padim", "stfpm"}:
        return "embedding_anomaly"
    if any(m in model for m in ["patchcore", "padim", "stfpm"]):
        return "embedding_anomaly"
    return "unknown_or_legacy"

def canonicalize_final_model(row):
    model = str(row.get("model_name", ""))
    exp = str(row.get("experiment_name", "")).lower()
    method = str(row.get("method", "")).lower()
    if method in {"patchcore", "padim", "stfpm"}:
        return f"{method}_resnet18"
    if "patchcore" in model.lower() or "patchcore" in exp:
        return "patchcore_resnet18"
    if "padim" in model.lower() or "padim" in exp:
        return "padim_resnet18"
    if "stfpm" in model.lower() or "stfpm" in exp:
        return "stfpm_resnet18"
    return model

def canonicalize_label_budget_ratio(row):
    for col in ["abnormal_label_ratio", "label_ratio"]:
        value = row.get(col)
        if pd.notna(value):
            try:
                return round(float(value), 4)
            except Exception:
                pass
    exp = str(row.get("experiment_name", ""))
    match = re.search(r"abnratio([0-9.]+)", exp)
    if match:
        return round(float(match.group(1)), 4)
    match = re.search(r"ratio([0-9.]+)", exp)
    if match:
        return round(float(match.group(1)), 4)
    return np.nan

def canonicalize_protocol_version(row):
    version = row.get("protocol_version")
    if isinstance(version, str) and version in ANALYSIS_PROTOCOLS:
        return version
    exp = str(row.get("experiment_name", ""))
    for version_name in ANALYSIS_PROTOCOLS:
        if version_name in exp:
            return version_name
    return version

def build_final_run_index(run_root=RUN_ROOT):
    df = collect_results(run_root)
    if df.empty:
        return df
    df = df.copy()
    df["protocol_version_canonical"] = df.apply(canonicalize_protocol_version, axis=1)
    df["paradigm_canonical"] = df.apply(canonicalize_final_paradigm, axis=1)
    df["model_canonical"] = df.apply(canonicalize_final_model, axis=1)
    df["seed_canonical"] = df.apply(
        lambda r: infer_seed_from_text(f"{r.get('experiment_name', '')} {r.get('run_id', '')}", r.get("seed")),
        axis=1,
    )
    df["label_budget_ratio_canonical"] = df.apply(canonicalize_label_budget_ratio, axis=1)

    # Recompute completion using the relaxed review function if available.
    complete_flags = []
    complete_issues = []
    for _, row in df.iterrows():
        if "artifact_complete_relaxed_for_review" in globals():
            ok, issues = artifact_complete_relaxed_for_review(row["run_dir"])
        elif "artifact_complete" in globals():
            ok, issues = artifact_complete(row["run_dir"])
        else:
            required = ["metrics.json", "predictions.csv", "predictions_calibration.csv", "threshold_protocol.json"]
            run_dir = Path(row["run_dir"])
            issues = [x for x in required if not (run_dir / x).exists()]
            ok = len(issues) == 0
        complete_flags.append(bool(ok))
        complete_issues.append("; ".join(issues))
    df["artifact_complete_final"] = complete_flags
    df["artifact_issues_final"] = complete_issues

    df.to_csv(ANALYSIS_TABLE_ROOT / "all_runs_canonicalized.csv", index=False)
    return df

def final_analysis_runs(df=None, allow_v3_label_budget=True):
    if df is None:
        df = build_final_run_index()
    if df.empty:
        return df

    keep = df["protocol_version_canonical"].isin(ANALYSIS_PROTOCOLS)
    keep &= df["artifact_complete_final"]
    keep &= df["paradigm_canonical"].isin([
        "supervised",
        "embedding_anomaly",
        "defect_label_budget_supervised",
        "label_budget_supervised",
    ])

    final = df[keep].copy()
    # Use one name in report tables.
    final["paradigm_final"] = final["paradigm_canonical"].replace({
        "label_budget_supervised": "defect_label_budget_supervised",
    })

    # Keep only required model families.
    is_supervised = final["paradigm_final"].eq("supervised") & final["model_canonical"].isin(ANALYSIS_SUPERVISED_MODELS)
    is_anomaly = final["paradigm_final"].eq("embedding_anomaly") & final["model_canonical"].isin(ANALYSIS_ANOMALY_MODELS)
    is_budget = (
        final["paradigm_final"].eq("defect_label_budget_supervised")
        & final["model_canonical"].eq(ANALYSIS_LABEL_BUDGET_MODEL)
        & final["label_budget_ratio_canonical"].isin(ANALYSIS_LABEL_BUDGET_RATIOS)
    )
    final = final[is_supervised | is_anomaly | is_budget].copy()

    final = deduplicate_final_runs(final)
    final.to_csv(ANALYSIS_TABLE_ROOT / "final_analysis_runs.csv", index=False)
    return final

def protocol_priority_for_final(version):
    priority = {
        "E4_defect_type_group_calibrated": 3,
        "E2_group_calibrated": 2,
    }
    return priority.get(str(version), 0)

def run_modified_time(path):
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0

def dedup_ratio_key(value):
    if pd.isna(value):
        return "none"
    return f"{float(value):.4f}"

def deduplicate_final_runs(final):
    if final.empty:
        return final

    work = final.copy()
    work["protocol_priority"] = work["protocol_version_canonical"].map(protocol_priority_for_final)
    work["run_modified_time"] = work["run_dir"].map(run_modified_time)
    work["dedup_ratio_key"] = work["label_budget_ratio_canonical"].map(dedup_ratio_key)
    work["dedup_key"] = (
        work["paradigm_final"].astype(str)
        + " | " + work["model_canonical"].astype(str)
        + " | seed=" + work["seed_canonical"].astype(str)
        + " | ratio=" + work["dedup_ratio_key"].astype(str)
    )

    before = len(work)
    duplicates = work[work.duplicated("dedup_key", keep=False)].copy()
    if not duplicates.empty:
        duplicates = duplicates.sort_values(
            ["dedup_key", "protocol_priority", "run_modified_time"],
            ascending=[True, False, False],
        )
        duplicates.to_csv(ANALYSIS_TABLE_ROOT / "duplicate_final_run_candidates.csv", index=False)

    selected = (
        work.sort_values(
            ["protocol_priority", "run_modified_time"],
            ascending=[False, False],
        )
        .drop_duplicates(
            subset=[
                "paradigm_final",
                "model_canonical",
                "seed_canonical",
                "dedup_ratio_key",
            ],
            keep="first",
        )
        .copy()
    )

    selected_keys = set(selected["dedup_key"])
    excluded = work[
        work["dedup_key"].isin(set(duplicates["dedup_key"]) if not duplicates.empty else set())
        & ~work.index.isin(selected.index)
    ].copy()
    excluded.to_csv(ANALYSIS_TABLE_ROOT / "duplicate_final_runs_excluded.csv", index=False)

    audit = {
        "n_before_dedup": int(before),
        "n_after_dedup": int(len(selected)),
        "n_excluded_duplicates": int(before - len(selected)),
        "expected_main_final_runs": int(
            len(ANALYSIS_SUPERVISED_MODELS) * len(ANALYSIS_SEEDS_MINIMUM)
            + len(ANALYSIS_ANOMALY_MODELS) * len(ANALYSIS_SEEDS_MINIMUM)
            + len(ANALYSIS_LABEL_BUDGET_RATIOS) * len(ANALYSIS_SEEDS_MINIMUM)
        ),
        "duplicate_candidates_csv": str(ANALYSIS_TABLE_ROOT / "duplicate_final_run_candidates.csv"),
        "excluded_duplicates_csv": str(ANALYSIS_TABLE_ROOT / "duplicate_final_runs_excluded.csv"),
    }
    save_json(audit, ANALYSIS_TABLE_ROOT / "final_run_dedup_audit.json")
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    return selected.drop(columns=["dedup_ratio_key", "dedup_key"], errors="ignore")


# %% missing_condition_plan
def required_final_conditions():
    rows = []
    for model in ANALYSIS_SUPERVISED_MODELS:
        for seed in ANALYSIS_SEEDS_MINIMUM:
            rows.append({
                "family": "supervised",
                "model_canonical": model,
                "seed": seed,
                "label_budget_ratio": np.nan,
                "rerun_call": f'train_supervised_model_E2(model_name="{model}", seed={seed}, epochs=5, batch_size=32, overwrite=False)',
            })
    for model in ANALYSIS_ANOMALY_MODELS:
        method = model.split("_")[0]
        for seed in ANALYSIS_SEEDS_MINIMUM:
            rows.append({
                "family": "embedding_anomaly",
                "model_canonical": model,
                "seed": seed,
                "label_budget_ratio": np.nan,
                "rerun_call": f'train_anomalib_embedding_model_E2(method="{method}", backbone="resnet18", seed={seed}, train_batch_size=1, eval_batch_size=1, overwrite=False)',
            })
    for ratio in ANALYSIS_LABEL_BUDGET_RATIOS:
        for seed in ANALYSIS_SEEDS_MINIMUM:
            rows.append({
                "family": "defect_label_budget_supervised",
                "model_canonical": ANALYSIS_LABEL_BUDGET_MODEL,
                "seed": seed,
                "label_budget_ratio": ratio,
                "rerun_call": f'train_label_budget_keep_all_normals_E2(model_name="resnet18", abnormal_label_ratio={ratio}, seed={seed}, epochs=5, batch_size=32, overwrite=False)',
            })
    return pd.DataFrame(rows)

def diagnose_missing_final_conditions(final=None):
    if final is None:
        final = final_analysis_runs()
    required = required_final_conditions()
    have_rows = []
    for _, r in final.iterrows():
        family = r["paradigm_final"]
        ratio = r["label_budget_ratio_canonical"] if family == "defect_label_budget_supervised" else np.nan
        have_rows.append({
            "family": family,
            "model_canonical": r["model_canonical"],
            "seed": int(r["seed_canonical"]) if pd.notna(r["seed_canonical"]) else np.nan,
            "label_budget_ratio": ratio,
        })
    have = pd.DataFrame(have_rows).drop_duplicates() if have_rows else pd.DataFrame(columns=required.columns)

    def key_tuple(row):
        ratio = row["label_budget_ratio"]
        ratio_key = "nan" if pd.isna(ratio) else f"{float(ratio):.4f}"
        return (row["family"], row["model_canonical"], int(row["seed"]), ratio_key)

    have_keys = set()
    if not have.empty:
        for _, row in have.iterrows():
            if pd.notna(row["seed"]):
                have_keys.add(key_tuple(row))

    missing_rows = []
    for _, row in required.iterrows():
        if key_tuple(row) not in have_keys:
            missing_rows.append(row.to_dict())
    missing = pd.DataFrame(missing_rows)
    missing.to_csv(ANALYSIS_TABLE_ROOT / "missing_final_conditions.csv", index=False)

    coverage = (
        have.groupby(["family", "model_canonical", "label_budget_ratio"], dropna=False)["seed"]
        .nunique()
        .reset_index(name="n_complete_seeds")
        if not have.empty else pd.DataFrame()
    )
    coverage.to_csv(ANALYSIS_TABLE_ROOT / "final_condition_seed_coverage.csv", index=False)
    display(coverage)
    if missing.empty:
        print("All required final conditions have the minimum seeds.")
    else:
        print("Missing final conditions:", len(missing))
        display(missing[["family", "model_canonical", "seed", "label_budget_ratio"]])
    return missing, coverage

def print_rerun_block(missing):
    if missing is None or missing.empty:
        print("# No reruns needed.")
        return
    print("# Copy/paste this rerun block. Existing complete runs will be skipped.")
    for call in missing["rerun_call"].drop_duplicates():
        print(call)


# %% report_tables
def mean_sd(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return ""
    if len(vals) == 1:
        return f"{vals.iloc[0]:.4f}"
    return f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"

def metric_col(df, name):
    candidates = [f"metric_{name}", name, name.upper()]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def make_main_performance_table(final=None):
    if final is None:
        final = final_analysis_runs()
    main = final[final["paradigm_final"].isin(["supervised", "embedding_anomaly"])].copy()
    rows = []
    group_cols = ["paradigm_final", "model_canonical"]
    metrics = ["auroc", "auprc", "f1", "fnr", "recall_sensitivity", "specificity", "accuracy", "mcc"]
    for (paradigm, model), g in main.groupby(group_cols):
        row = {
            "Paradigm": "Supervised" if paradigm == "supervised" else "Normal-only anomaly",
            "Model": model,
            "Seeds": int(g["seed_canonical"].nunique()),
        }
        for metric in metrics:
            col = metric_col(g, metric)
            row[metric.upper() if metric != "recall_sensitivity" else "Sensitivity"] = mean_sd(g[col]) if col else ""
        lat_col = "eff_latency_ms_per_image" if "eff_latency_ms_per_image" in g.columns else None
        row["Latency ms/img"] = mean_sd(g[lat_col]) if lat_col else ""
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["Paradigm", "Model"])
    table.to_csv(ANALYSIS_TABLE_ROOT / "table_main_performance_mean_sd.csv", index=False)
    display(table)
    return table

def make_label_budget_table(final=None):
    if final is None:
        final = final_analysis_runs()
    budget = final[final["paradigm_final"].eq("defect_label_budget_supervised")].copy()
    rows = []
    for ratio, g in budget.groupby("label_budget_ratio_canonical"):
        row = {
            "Abnormal label ratio": ratio,
            "Seeds": int(g["seed_canonical"].nunique()),
            "Normal train n": mean_sd(g["n_labeled_normal"]) if "n_labeled_normal" in g else "",
            "Abnormal train n": mean_sd(g["n_labeled_abnormal"]) if "n_labeled_abnormal" in g else "",
        }
        for metric in ["auroc", "auprc", "f1", "fnr", "recall_sensitivity", "specificity"]:
            col = metric_col(g, metric)
            row[metric.upper() if metric != "recall_sensitivity" else "Sensitivity"] = mean_sd(g[col]) if col else ""
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("Abnormal label ratio")
    table.to_csv(ANALYSIS_TABLE_ROOT / "table_label_budget_mean_sd.csv", index=False)
    display(table)
    return table


# %% report_figures
def plot_label_budget_crossover(final=None):
    if final is None:
        final = final_analysis_runs()
    budget = final[final["paradigm_final"].eq("defect_label_budget_supervised")].copy()
    if budget.empty:
        print("empty")
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = (
        budget.groupby("label_budget_ratio_canonical")
        .agg(
            auroc_mean=("metric_auroc", "mean"),
            auroc_sd=("metric_auroc", "std"),
            f1_mean=("metric_f1", "mean"),
            f1_sd=("metric_f1", "std"),
        )
        .reset_index()
        .sort_values("label_budget_ratio_canonical")
    )
    x = summary["label_budget_ratio_canonical"].astype(float) * 100
    ax.errorbar(x, summary["auroc_mean"], yerr=summary["auroc_sd"], marker="o", label="Label-budget AUROC")
    ax.errorbar(x, summary["f1_mean"], yerr=summary["f1_sd"], marker="s", label="Label-budget F1")

    anomaly = final[final["paradigm_final"].eq("embedding_anomaly")]
    if not anomaly.empty:
        for model, g in anomaly.groupby("model_canonical"):
            if "metric_auroc" in g:
                ax.axhline(g["metric_auroc"].mean(), linestyle="--", linewidth=1.2, label=f"{model} AUROC")

    ax.set_xlabel("Labeled abnormal training samples (%)")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Annotation-budget crossover")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = ANALYSIS_FIG_ROOT / "fig_label_budget_crossover.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)
    return path

def plot_fnr_vs_label_budget(final=None):
    if final is None:
        final = final_analysis_runs()
    budget = final[final["paradigm_final"].eq("defect_label_budget_supervised")].copy()
    if budget.empty or "metric_fnr" not in budget.columns:
        print("No label-budget FNR runs to plot.")
        return None
    summary = (
        budget.groupby("label_budget_ratio_canonical")
        .agg(fnr_mean=("metric_fnr", "mean"), fnr_sd=("metric_fnr", "std"))
        .reset_index()
        .sort_values("label_budget_ratio_canonical")
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = summary["label_budget_ratio_canonical"].astype(float) * 100
    ax.errorbar(x, summary["fnr_mean"], yerr=summary["fnr_sd"], marker="o", color="crimson")
    ax.set_xlabel("Labeled abnormal training samples (%)")
    ax.set_ylabel("False negative rate")
    ax.set_title("False negatives versus annotation budget")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = ANALYSIS_FIG_ROOT / "fig_fnr_vs_annotation_budget.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)
    return path


# %% final_analysis_and_next_step
def build_final_analysis_report():
    all_runs = build_final_run_index()
    final = final_analysis_runs(all_runs)
    missing, coverage = diagnose_missing_final_conditions(final)

    paradigms = set(final["paradigm_final"].dropna()) if not final.empty else set()
    has_supervised = "supervised" in paradigms and any(final["model_canonical"].isin(ANALYSIS_SUPERVISED_MODELS))
    has_embedding_anomaly = "embedding_anomaly" in paradigms and any(final["model_canonical"].isin(ANALYSIS_ANOMALY_MODELS))
    has_label_budget = "defect_label_budget_supervised" in paradigms

    report = {
        "n_final_runs": int(len(final)),
        "has_supervised": bool(has_supervised),
        "has_embedding_anomaly": bool(has_embedding_anomaly),
        "has_label_budget": bool(has_label_budget),
        "conditions_without_required_seed": int(len(missing)),
        "ready_for_main_results_tables": bool(
            has_supervised and has_embedding_anomaly and has_label_budget and len(missing) == 0
        ),
        "final_runs_csv": str(ANALYSIS_TABLE_ROOT / "final_analysis_runs.csv"),
        "missing_conditions_csv": str(ANALYSIS_TABLE_ROOT / "missing_final_conditions.csv"),
    }
    save_json(report, ANALYSIS_TABLE_ROOT / "build_final_analysis_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if missing.empty:
        make_main_performance_table(final)
        make_label_budget_table(final)
        plot_label_budget_crossover(final)
        plot_fnr_vs_label_budget(final)
    else:
        print_rerun_block(missing)
    return report, final, missing


# 05_execution

# Optional execution example. Uncomment to run.
# report, final_runs, missing_conditions = build_final_analysis_report()

# Optional execution example. Uncomment to run.
### 02_Make anomaly-threshold comparison without abnormal calibration labels:
# final_runs = final_analysis_runs()
# normal_threshold_trackers = create_normal_only_threshold_variants(methods=("p95", "p99"), final_runs=final_runs)

# Optional execution example. Uncomment to run.
### 03_Fix 100% label-budget inconsistency by aliasing full supervised ResNet:
# alias_100 = create_100pct_budget_alias_from_supervised(overwrite=False)

### 04_fine_crossover_checkpoint_fix
def trusted_torch_load(path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def quarantine_bad_resume_checkpoint(checkpoint_path, reason=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    quarantine_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + f".bad_{now_id()}"
    )
    checkpoint_path.rename(quarantine_path)
    print("Moved unreadable resume checkpoint to:", quarantine_path)
    if reason:
        save_json(
            {"bad_checkpoint": str(quarantine_path), "reason": str(reason)},
            checkpoint_path.parent / "bad_resume_checkpoint_info.json",
        )
    return quarantine_path

def load_resume_checkpoint(
    tracker,
    model,
    optimizer,
    scheduler=None,
    expected=None,
):
    checkpoint_path = tracker.artifact("resume_checkpoint.pt")
    if not checkpoint_path.exists():
        return {"loaded": False, "start_epoch": 1, "history": []}

    try:
        checkpoint = trusted_torch_load(checkpoint_path, map_location=DEVICE)
    except Exception as error:
        print("Could not read resume checkpoint; restarting this condition from epoch 1.")
        print("Checkpoint:", checkpoint_path)
        print("Reason:", type(error).__name__, error)
        quarantine_bad_resume_checkpoint(checkpoint_path, reason=f"{type(error).__name__}: {error}")
        if "mark_run_status" in globals():
            mark_run_status(
                tracker.run_dir,
                "resume_checkpoint_quarantined",
                [f"{type(error).__name__}: {error}"],
            )
        return {"loaded": False, "start_epoch": 1, "history": []}

    if expected is not None and "validate_checkpoint_metadata" in globals():
        try:
            validate_checkpoint_metadata(checkpoint, expected)
        except Exception as error:
            print("Resume checkpoint metadata mismatch; restarting this condition from epoch 1.")
            print("Reason:", error)
            quarantine_bad_resume_checkpoint(checkpoint_path, reason=f"metadata_mismatch: {error}")
            return {"loaded": False, "start_epoch": 1, "history": []}

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    except Exception as error:
        print("Resume checkpoint contents are incompatible; restarting from epoch 1.")
        print("Reason:", type(error).__name__, error)
        quarantine_bad_resume_checkpoint(checkpoint_path, reason=f"{type(error).__name__}: {error}")
        return {"loaded": False, "start_epoch": 1, "history": []}

    completed_epoch = int(checkpoint.get("epoch", 0))
    print(f"Resuming from epoch {completed_epoch + 1}")
    return {
        "loaded": True,
        "start_epoch": completed_epoch + 1,
        "history": checkpoint.get("history", []),
    }

print("Fine crossover checkpoint loader patched for PyTorch 2.6.")

# Optional execution example. Uncomment to run.
# fine = run_fine_crossover_label_count_suite(
#     counts=(8, 16, 24, 32, 64),
#     seeds=(11, 22, 33),
#     repeats=(0, 1, 2),
#     epochs=5,
#     batch_size=32,
#     overwrite=False,
# )

# Optional execution example. Uncomment to run.
# 4. Generate revised tables/figures:
# fine_table = make_fine_crossover_table()
# fine_fig = plot_fine_crossover_against_patchcore()

# %% gradcam_return_fix
def make_gradcam_shortcut_check(run_dir, n_each=3, preprocess_name="resize224_imagenet"):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as e:
        raise ImportError("Install grad-cam first: pip install grad-cam") from e

    run_dir = Path(run_dir)
    pred = pd.read_csv(run_dir / "predictions.csv")
    examples = select_prediction_examples(pred, n_each=n_each)
    if examples.empty:
        print("empty")
        return None

    model, model_name = load_supervised_model_from_run(run_dir)
    target_layer = choose_gradcam_target_layer(model, model_name)
    transform = PREPROCESSING[preprocess_name]
    out_dir = FIG_ROOT / "gradcam" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for i, row in examples.iterrows():
            img_path = row["path"]
            pil = Image.open(img_path).convert("RGB")
            x = transform(pil).unsqueeze(0).to(DEVICE)
            rgb = np.asarray(pil.resize((x.shape[-1], x.shape[-2]))) / 255.0
            grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
            overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)

            save_path = out_dir / f"{i:02d}_{row['case_type']}_true{int(row.y_true)}_pred{int(row.y_pred)}.png"
            Image.fromarray(overlay).save(save_path)

            attention_ratio = np.nan
            if "find_label_json_by_stem" in globals() and "robust_mask_from_json" in globals():
                json_path = find_label_json_by_stem(img_path)
                if json_path:
                    mask = robust_mask_from_json(json_path, np.asarray(pil).shape)
                    mask = resize_mask_nearest(mask, grayscale_cam.shape)
                    total = float(grayscale_cam.sum() + 1e-12)
                    attention_ratio = float(grayscale_cam[mask.astype(bool)].sum() / total)

            records.append({
                "path": img_path,
                "case_type": row["case_type"],
                "y_true": int(row.y_true),
                "y_pred": int(row.y_pred),
                "y_score": float(row.y_score),
                "gradcam_path": str(save_path),
                "attention_localization_ratio": attention_ratio,
            })

    record_df = pd.DataFrame(records)
    record_df.to_csv(out_dir / "gradcam_examples.csv", index=False)
    display(record_df)
    print("Saved Grad-CAM examples:", out_dir)
    return record_df

# Optional execution example. Uncomment to run.
# 5. Grad-CAM shortcut sanity check:
# gradcam_records = run_gradcam_for_best_supervised_model(final_runs=final_analysis_runs(), n_each=3)

# Optional execution example. Uncomment to run.
# 6. report:
# summary = analysis_completion_report()

# Optional execution example. Uncomment to run.
# %% optional_fast_subset_for_compute (# Take almost 40 hours)
# Full setting is 90 runs:
#    This is compute-heavy: 6 counts x 3 seeds x 5 draws = 90 ResNet runs.
#    If too expensive, use counts=(8,16,24,32,64), seeds=(11,22,33), repeats=(0,1,2).
# fine = run_fine_crossover_label_count_suite(
#      counts=(4, 8, 16, 24, 32, 64),
#      seeds=(11, 22, 33),
#      repeats=(0, 1, 2, 3, 4),
#      epochs=5,
#      batch_size=32,
#      overwrite=False,
# )

# Optional execution example. Uncomment to run.
# 4. Generate revised tables/figures:
# fine_table = make_fine_crossover_table()
# fine_fig = plot_fine_crossover_against_patchcore()

# Optional execution example. Uncomment to run.
# 5. Grad-CAM shortcut sanity check:
# gradcam_records = run_gradcam_for_best_supervised_model(final_runs=final_analysis_runs(), n_each=3)

# Optional execution example. Uncomment to run.
# 6. report:
# summary = analysis_completion_report()


# Execution Examples The notebook defines the reusable analysis functions above. Run the following examples selectively ac

# Build and validate one duplicate-aware calibrated split.
# split_df = make_E2_split(seed=11)

# Run supervised baselines.
# supervised_runs = run_E2_supervised_suite(
#     model_names=("resnet18", "efficientnet_b0", "convnext_tiny"),
#     seeds=[11, 22, 33],
#     epochs=5,
#     batch_size=32,
#     overwrite=OVERWRITE_EXISTING,
# )

# Run normal-only anomaly baselines on the same splits.
# anomaly_runs = run_E2_embedding_anomaly_suite(
#     methods=("patchcore", "padim", "stfpm"),
#     seeds=[11, 22, 33],
#     train_batch_size=1,
#     eval_batch_size=1,
#     overwrite=OVERWRITE_EXISTING,
#     show_progress=False,
# )

# Run percentage-based abnormal training-label budget experiments.
# label_budget_runs = run_label_budget_allnormal_suite(
#     model_name="resnet18",
#     ratios=(0.01, 0.05, 0.10, 0.20, 0.50, 1.00),
#     seeds=[11, 22, 33],
#     epochs=5,
#     batch_size=32,
#     overwrite=OVERWRITE_EXISTING,
# )

# Run the fine label-count crossover experiment.
# This is compute-heavy. Start with the reduced setting before running the full grid.
# fine_runs = run_fine_crossover_label_count_suite(
#     counts=(8, 16, 24, 32, 64),
#     seeds=(11, 22, 33),
#     repeats=(0, 1, 2),
#     epochs=5,
#     batch_size=32,
#     overwrite=OVERWRITE_EXISTING,
# )

# Full fine crossover grid used for the final analysis.
# fine_runs_full = run_fine_crossover_label_count_suite(
#     counts=(4, 8, 16, 24, 32, 64),
#     seeds=(11, 22, 33),
#     repeats=(0, 1, 2, 3, 4),
#     epochs=5,
#     batch_size=32,
#     overwrite=OVERWRITE_EXISTING,
# )

# Summarize saved results.
# final_runs = final_analysis_runs()
# main_table = make_main_performance_table(final_runs)
# label_budget_table = make_label_budget_table(final_runs)
# fine_table = make_fine_crossover_table()
