"""
02_train_svm.py
---------------
İki SVM varyantı eğitir:
  * svm_imbalanced: doğal sınıf dağılımı (class_weight=None)
  * svm_balanced  : class_weight='balanced'

Hız nedeniyle RBF SVM yerine LinearSVC kullanılabilir; veri boyutuna göre
otomatik seçim yaparız (büyük N → LinearSVC, küçük N → RBF SVC).
"""

import sys, time, json
from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT_ROOT, evaluate_classifier, save_json


def make_svm(class_weight, n_train: int):
    if n_train > 4000:
        # Büyük veri → LinearSVC (çok daha hızlı). 1 vs rest.
        base = LinearSVC(C=1.0, class_weight=class_weight, max_iter=5000, dual="auto")
        return Pipeline([("scaler", StandardScaler(with_mean=True)), ("svm", base)])
    else:
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight=class_weight)),
        ])


def main():
    feat_dir = PROJECT_ROOT / "features"
    X_train = np.load(feat_dir / "train_X.npy")
    y_train = np.load(feat_dir / "train_y.npy")
    X_test = np.load(feat_dir / "test_X.npy")
    y_test = np.load(feat_dir / "test_y.npy")

    print(f"Train: {X_train.shape}, dist: {np.bincount(y_train).tolist()}")
    print(f"Test : {X_test.shape},  dist: {np.bincount(y_test).tolist()}")

    results = []
    for variant, cw in (("svm_imbalanced", None), ("svm_balanced", "balanced")):
        model = make_svm(cw, len(X_train))
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_train = time.perf_counter() - t0

        rep = evaluate_classifier(variant, model, X_test, y_test, t_train)
        rep["class_weight"] = "balanced" if cw == "balanced" else "none"
        rep["model_type"] = type(model.named_steps["svm"]).__name__
        results.append(rep)

        joblib.dump(model, PROJECT_ROOT / "models" / f"{variant}.joblib")
        print(f"[{variant}] acc={rep['accuracy']:.3f} "
              f"macroF1={rep['macro_f1']:.3f} train={t_train:.1f}s "
              f"inf={rep['inference_ms_per_sample']:.2f}ms/sample")
        print("  per-class P/R/F1:")
        for c, name in enumerate(["crop", "weed"]):
            print(f"   {name}: P={rep['precision_per_class'][c]:.3f} "
                  f"R={rep['recall_per_class'][c]:.3f} F1={rep['f1_per_class'][c]:.3f}")

    save_json({"results": results}, PROJECT_ROOT / "results" / "svm_metrics.json")
    print("Saved SVM results to results/svm_metrics.json")


if __name__ == "__main__":
    main()
