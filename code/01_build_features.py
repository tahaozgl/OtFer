"""
01_build_features.py
--------------------
Train/Valid/Test split'leri için bbox yamalarını çıkarır,
HOG + renk histogramı özelliklerini hesaplayıp NumPy dosyalarına kaydeder.
SVM ve KNN bu özellik setini birebir aynı şekilde kullanır → adil karşılaştırma.
"""

import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    PROJECT_ROOT, ensure_dirs, load_patches, extract_features, save_json,
)

def main():
    ensure_dirs()
    feat_dir = PROJECT_ROOT / "features"
    summary = {}
    for split in ("train", "valid", "test"):
        t0 = time.perf_counter()
        X_img, y = load_patches(split)
        t_patch = time.perf_counter() - t0

        t0 = time.perf_counter()
        X = extract_features(X_img)
        t_feat = time.perf_counter() - t0

        np.save(feat_dir / f"{split}_X.npy", X)
        np.save(feat_dir / f"{split}_y.npy", y)
        # raw resized patches (sadece görselleştirme için)
        if split == "test":
            np.save(feat_dir / "test_patches.npy", X_img)

        unique, counts = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        summary[split] = {
            "n_patches": int(len(y)),
            "feature_dim": int(X.shape[1]) if X.size else 0,
            "patch_extract_sec": round(t_patch, 2),
            "feature_extract_sec": round(t_feat, 2),
            "class_counts": dist,
        }
        print(f"[{split}] patches={len(y)} feat_dim={X.shape[1] if X.size else 0} "
              f"dist={dist} (patch {t_patch:.1f}s, feat {t_feat:.1f}s)")

    save_json(summary, PROJECT_ROOT / "results" / "feature_summary.json")
    print("Saved features to:", feat_dir)


if __name__ == "__main__":
    main()
