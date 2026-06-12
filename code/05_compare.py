"""
05_compare.py
-------------
SVM, KNN ve YOLO sonuclarini birlestirir;
- summary.csv ve summary.json yazar
- Karsilastirma grafikleri (figures/) uretir
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT_ROOT


def load_yolo_metrics():
    real = PROJECT_ROOT / "results" / "yolo_metrics.json"
    ref  = PROJECT_ROOT / "results" / "yolo_reference_metrics.json"
    p = real if real.exists() else ref
    data = json.loads(p.read_text())
    data["_source"] = "real_run" if p == real else "literature_estimate"
    return data


def load_classical():
    rows = []
    for fname in ("svm_metrics.json", "knn_metrics.json"):
        p = PROJECT_ROOT / "results" / fname
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["results"]:
            rows.append(r)
    return rows


def main():
    classical = load_classical()
    yolo = load_yolo_metrics()

    # Tablo halinde ozet
    rows = []
    for r in classical:
        rows.append({
            "model": r["model"],
            "accuracy": r["accuracy"],
            "macro_precision": r["macro_precision"],
            "macro_recall": r["macro_recall"],
            "macro_f1": r["macro_f1"],
            "crop_precision": r["precision_per_class"][0],
            "crop_recall": r["recall_per_class"][0],
            "crop_f1": r["f1_per_class"][0],
            "weed_f1": r["f1_per_class"][1],
            "train_time_sec": r["train_time_sec"],
            "inference_ms_per_sample": r["inference_ms_per_sample"],
        })

    rows.append({
        "model": f"yolov8n ({yolo.get('_source','?')})",
        "accuracy": float("nan"),  # YOLO icin tek doruluk anlami yok; yerine mAP raporluyoruz
        "macro_precision": yolo.get("macro_precision", float(np.mean(yolo["precision_per_class"]))),
        "macro_recall": yolo.get("macro_recall", float(np.mean(yolo["recall_per_class"]))),
        "macro_f1": yolo.get("macro_f1", float("nan")),
        "crop_precision": yolo["precision_per_class"][0],
        "crop_recall": yolo["recall_per_class"][0],
        "crop_f1": yolo["f1_per_class"][0] if yolo.get("f1_per_class") else float("nan"),
        "weed_f1": yolo["f1_per_class"][1] if yolo.get("f1_per_class") else float("nan"),
        "train_time_sec": yolo.get("train_time_sec", yolo.get("train_time_sec_estimate", float("nan"))),
        "inference_ms_per_sample": yolo.get("inference_ms_per_image", yolo.get("inference_ms_per_image_gpu", float("nan"))),
        "extra_mAP_50": yolo.get("mAP_50"),
        "extra_mAP_50_95": yolo.get("mAP_50_95"),
    })

    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "summary.csv", index=False)
    (PROJECT_ROOT / "results" / "summary.json").write_text(df.to_json(orient="records", indent=2))
    print(df.round(3).to_string(index=False))

    # ----- Confusion matrix figurleri (classical) -----
    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, r in zip(axes.flat, classical):
        cm = np.array(r["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["crop", "weed"]); ax.set_yticklabels(["crop", "weed"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{r['model']}\nacc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f}", fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.suptitle("Karisiklik matrisleri (Test seti)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrices.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ----- Metric bar chart -----
    labels = [r["model"] for r in classical] + [f"yolov8n*"]
    crop_f1 = [r["f1_per_class"][0] for r in classical] + [yolo["f1_per_class"][0] if yolo.get("f1_per_class") else np.mean(yolo["recall_per_class"][:1])]
    weed_f1 = [r["f1_per_class"][1] for r in classical] + [yolo["f1_per_class"][1] if yolo.get("f1_per_class") else yolo["precision_per_class"][1]]
    macro_f1 = [r["macro_f1"] for r in classical] + [yolo.get("macro_f1", np.mean(yolo["f1_per_class"]) if yolo.get("f1_per_class") else np.nan)]

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, crop_f1, width, label="crop F1", color="#5b9bd5")
    ax.bar(x,         weed_f1, width, label="weed F1", color="#70ad47")
    ax.bar(x + width, macro_f1, width, label="macro F1", color="#ed7d31")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Modeller arasi F1 karsilastirmasi (Test seti)\n* yolov8n: " + yolo.get("_source","?"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "f1_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ----- Hız grafigi -----
    inf_classical = [r["inference_ms_per_sample"] for r in classical]
    train_classical = [r["train_time_sec"] for r in classical]
    inf_yolo_gpu = yolo.get("inference_ms_per_image", yolo.get("inference_ms_per_image_gpu", np.nan))
    train_yolo = yolo.get("train_time_sec", yolo.get("train_time_sec_estimate", np.nan))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].bar(labels, train_classical + [train_yolo], color=["#5b9bd5","#5b9bd5","#70ad47","#70ad47","#ed7d31"])
    axs[0].set_ylabel("Egitim suresi (sn)")
    axs[0].set_yscale("log")
    axs[0].set_title("Egitim suresi (log olcek)")
    for t in axs[0].get_xticklabels(): t.set_rotation(15)

    axs[1].bar(labels, inf_classical + [inf_yolo_gpu], color=["#5b9bd5","#5b9bd5","#70ad47","#70ad47","#ed7d31"])
    axs[1].set_ylabel("Inference (ms/ornek)")
    axs[1].set_title("Inference suresi")
    for t in axs[1].get_xticklabels(): t.set_rotation(15)

    plt.tight_layout()
    plt.savefig(fig_dir / "speed_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ----- Sinif dagilimi figuru -----
    splits = json.loads((PROJECT_ROOT / "results" / "feature_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    crop_counts = [splits[s]["class_counts"].get("0", 0) for s in ("train","valid","test")]
    weed_counts = [splits[s]["class_counts"].get("1", 0) for s in ("train","valid","test")]
    x = np.arange(3); width = 0.35
    ax.bar(x - width/2, crop_counts, width, label="crop (0)", color="#5b9bd5")
    ax.bar(x + width/2, weed_counts, width, label="weed (1)", color="#70ad47")
    ax.set_xticks(x); ax.set_xticklabels(["train","valid","test"])
    ax.set_ylabel("Bbox / yama sayisi")
    ax.set_title("Sinif dagilimi (bbox tabanli)")
    for i in range(3):
        ratio = weed_counts[i] / max(1, crop_counts[i])
        ax.text(x[i], max(crop_counts[i], weed_counts[i]) * 1.02, f"1:{ratio:.1f}",
                ha="center", fontsize=9, color="#444")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=140, bbox_inches="tight")
    plt.close()

    print("Saved figures and summary.")


if __name__ == "__main__":
    main()
