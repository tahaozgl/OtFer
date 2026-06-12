import sys, time
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT_ROOT, evaluate_classifier, save_json


def best_k_search(X_train, y_train, X_val, y_val, ks=(3, 5, 7), weights="uniform"):
    best_k, best_f1, best_model = None, -1, None
    for k in ks:
        m = Pipeline([("scaler", StandardScaler()),
                      ("knn", KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights=weights))])
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        f1m = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1m > best_f1:
            best_f1, best_k, best_model = f1m, k, m
    return best_model, best_k, best_f1


def oversample(X, y, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_max = counts.max()
    idxs = []
    for c in classes:
        ci = np.where(y == c)[0]
        if len(ci) < n_max:
            extra = rng.choice(ci, size=n_max - len(ci), replace=True)
            ci = np.concatenate([ci, extra])
        idxs.append(ci)
    sel = np.concatenate(idxs); rng.shuffle(sel)
    return X[sel], y[sel]


def main():
    feat_dir = PROJECT_ROOT / "features"
    X_train = np.load(feat_dir / "train_X.npy"); y_train = np.load(feat_dir / "train_y.npy")
    X_val   = np.load(feat_dir / "valid_X.npy"); y_val   = np.load(feat_dir / "valid_y.npy")
    X_test  = np.load(feat_dir / "test_X.npy");  y_test  = np.load(feat_dir / "test_y.npy")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    results = []

    # 1) Imbalanced
    t0 = time.perf_counter()
    model_im, k_im, f1_val = best_k_search(X_train, y_train, X_val, y_val)
    t_train = time.perf_counter() - t0
    rep = evaluate_classifier("knn_imbalanced", model_im, X_test, y_test, t_train)
    rep["best_k"] = int(k_im); rep["valid_macro_f1"] = float(f1_val); rep["balancing"] = "none"
    results.append(rep)
    joblib.dump(model_im, PROJECT_ROOT / "models" / "knn_imbalanced.joblib")
    print(f"[knn_imbalanced] best_k={k_im}  acc={rep['accuracy']:.3f}  macroF1={rep['macro_f1']:.3f}  train={t_train:.1f}s  inf={rep['inference_ms_per_sample']:.2f}ms/sample")
    print(f"   crop F1={rep['f1_per_class'][0]:.3f}  weed F1={rep['f1_per_class'][1]:.3f}")

    # 2) Balanced (oversample + distance weights)
    Xb, yb = oversample(X_train, y_train)
    print(f"Balanced shape: {Xb.shape}, dist: {np.bincount(yb).tolist()}")
    t0 = time.perf_counter()
    model_b, k_b, f1_val = best_k_search(Xb, yb, X_val, y_val, weights="distance")
    t_train = time.perf_counter() - t0
    rep = evaluate_classifier("knn_balanced", model_b, X_test, y_test, t_train)
    rep["best_k"] = int(k_b); rep["valid_macro_f1"] = float(f1_val)
    rep["balancing"] = "minority_oversample+distance"
    results.append(rep)
    joblib.dump(model_b, PROJECT_ROOT / "models" / "knn_balanced.joblib")
    print(f"[knn_balanced]   best_k={k_b}  acc={rep['accuracy']:.3f}  macroF1={rep['macro_f1']:.3f}  train={t_train:.1f}s  inf={rep['inference_ms_per_sample']:.2f}ms/sample")
    print(f"   crop F1={rep['f1_per_class'][0]:.3f}  weed F1={rep['f1_per_class'][1]:.3f}")

    save_json({"results": results}, PROJECT_ROOT / "results" / "knn_metrics.json")
    print("Saved knn_metrics.json")


if __name__ == "__main__":
    main()
