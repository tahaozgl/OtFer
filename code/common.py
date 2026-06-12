"""
common.py
---------
Paylaşımlı yardımcı fonksiyonlar:
- YOLO formatındaki etiketleri okuma
- Bbox'lardan görüntü yamalarını (patches) kesme
- HOG + renk histogramı özellik çıkarımı
- Eğitim / değerlendirme için ortak metrik raporları
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import hog


# --- Sabitler --------------------------------------------------------------
PROJECT_ROOT = Path("/sessions/sleepy-brave-volta/mnt/outputs/weed_crop_project")
DATASET_ROOT = Path("/sessions/sleepy-brave-volta/mnt/WeedCrop.v1i.yolov5pytorch")

CLASS_NAMES = ["crop", "weed"]  # 0 -> bitki, 1 -> ot
PATCH_SIZE = (96, 96)            # SVM/KNN için tüm yamaları bu boyuta resize ediyoruz


# --- Veri yükleme ----------------------------------------------------------
@dataclass
class BBox:
    cls: int
    cx: float  # normalize merkez x
    cy: float  # normalize merkez y
    w: float   # normalize genişlik
    h: float   # normalize yükseklik

    def to_pixel(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = int(max(0, (self.cx - self.w / 2) * img_w))
        y1 = int(max(0, (self.cy - self.h / 2) * img_h))
        x2 = int(min(img_w, (self.cx + self.w / 2) * img_w))
        y2 = int(min(img_h, (self.cy + self.h / 2) * img_h))
        return x1, y1, x2, y2


def read_yolo_label(label_path: Path) -> List[BBox]:
    boxes: List[BBox] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append(BBox(cls, cx, cy, w, h))
    return boxes


def list_split(split: str) -> List[Path]:
    img_dir = DATASET_ROOT / split / "images"
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_patches(
    split: str,
    max_per_class: int | None = None,
    min_box_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bir split (train/valid/test) için tüm bbox'ları kesip resize edilmiş yamalar listesi döner.
    Çıkış: patches (N, H, W, 3) BGR uint8, labels (N,) int.
    """
    images = list_split(split)
    patches: List[np.ndarray] = []
    labels: List[int] = []
    counts = {0: 0, 1: 0}

    for img_path in images:
        label_path = DATASET_ROOT / split / "labels" / (img_path.stem + ".txt")
        boxes = read_yolo_label(label_path)
        if not boxes:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for b in boxes:
            x1, y1, x2, y2 = b.to_pixel(w, h)
            if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
                continue
            if max_per_class is not None and counts[b.cls] >= max_per_class:
                continue
            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patch = cv2.resize(patch, PATCH_SIZE, interpolation=cv2.INTER_AREA)
            patches.append(patch)
            labels.append(b.cls)
            counts[b.cls] += 1

    X = np.stack(patches, axis=0) if patches else np.empty((0, *PATCH_SIZE, 3), dtype=np.uint8)
    y = np.array(labels, dtype=np.int64)
    return X, y


# --- Özellik çıkarımı ------------------------------------------------------
def color_histogram(img_bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    """HSV uzayında 3 kanallı histogram (her kanal `bins` bin), L1-normalize."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    ranges = [(0, 180), (0, 256), (0, 256)]
    for c, (lo, hi) in enumerate(ranges):
        h = cv2.calcHist([hsv], [c], None, [bins], [lo, hi]).flatten()
        feats.append(h)
    feats = np.concatenate(feats).astype(np.float32)
    s = feats.sum()
    if s > 0:
        feats /= s
    return feats


def hog_features(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    f = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return f.astype(np.float32)


def extract_features(patches: np.ndarray) -> np.ndarray:
    """
    Her yamanın HOG + renk histogramı özelliklerini birleştirip (N, D) matrisi döner.
    """
    feats = []
    for p in patches:
        f = np.concatenate([hog_features(p), color_histogram(p)])
        feats.append(f)
    return np.stack(feats, axis=0).astype(np.float32) if feats else np.empty((0, 0), dtype=np.float32)


# --- Metrik yardımcıları ---------------------------------------------------
def evaluate_classifier(name: str, model, X_test, y_test, train_time: float) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix
    )
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    report = {
        "model": name,
        "n_test": int(len(y_test)),
        "accuracy": float(acc),
        "precision_per_class": [float(p) for p in prec],
        "recall_per_class": [float(r) for r in rec],
        "f1_per_class": [float(v) for v in f1],
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "train_time_sec": float(train_time),
        "inference_time_sec_total": float(inf_time),
        "inference_ms_per_sample": float(inf_time / max(1, len(y_test)) * 1000.0),
    }
    return report


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def ensure_dirs():
    for sub in ("features", "models", "results", "figures"):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)
