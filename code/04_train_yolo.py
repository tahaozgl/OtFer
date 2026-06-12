"""
04_train_yolo.py
----------------
YOLOv8 (nano) ile WeedCrop veri setini egitir ve test eder.
Bu script GPU'lu yerel makinede calistirilmak uzere yazilmistir
(sandbox'ta torch+ultralytics kurulumu zaman asimina takiliyor).

Calistirma adimlari (yerelde):
    pip install ultralytics
    python 04_train_yolo.py --epochs 50 --imgsz 640 --batch 16

Sonuclar weed_crop_project/results/yolo_metrics.json icine kaydedilir,
boylece 05_compare.py uc modeli birlikte raporlayabilir.
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path

# Bu dosyayi sandbox'ta da import edebilelim diye geç import
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data.yaml"),
                        help="data.yaml yolu (WeedCrop.v1i.yolov5pytorch icindeki)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt", help="Baslangic agirligi")
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent.parent / "models" / "yolo_runs"))
    parser.add_argument("--name", default="weedcrop")
    parser.add_argument("--device", default=0, help="GPU id veya 'cpu'")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    t0 = time.perf_counter()
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        verbose=True,
        plots=True,
    )
    train_time = time.perf_counter() - t0

    # Test set'te degerlendirme (split='test' YOLOv8 1.1+ destekler; aksi halde val kullanilir)
    eval_t0 = time.perf_counter()
    try:
        metrics = model.val(data=args.data, split="test", device=args.device)
    except TypeError:
        metrics = model.val(data=args.data, device=args.device)
    eval_time = time.perf_counter() - eval_t0

    # Inference suresi olcumu (orneklem)
    sample_dir = Path(args.data).parent / "test" / "images"
    sample_imgs = sorted(sample_dir.glob("*.jpg"))[:50]
    inf_t0 = time.perf_counter()
    if sample_imgs:
        _ = model.predict([str(p) for p in sample_imgs], verbose=False)
    inf_time = time.perf_counter() - inf_t0
    inf_ms_per_image = (inf_time / max(1, len(sample_imgs))) * 1000.0

    # Metrikleri JSON'a kaydet
    box = metrics.box  # ultralytics BoxResults
    out = {
        "model": "yolov8n",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(args.device),
        "train_time_sec": float(train_time),
        "eval_time_sec": float(eval_time),
        "inference_ms_per_image": float(inf_ms_per_image),
        "n_inference_images": len(sample_imgs),
        # mAP & per-class metrikler
        "mAP_50": float(box.map50),
        "mAP_50_95": float(box.map),
        "precision_per_class": [float(x) for x in box.p],
        "recall_per_class": [float(x) for x in box.r],
        "f1_per_class": [float(x) for x in (box.f1 if hasattr(box, "f1") else [])],
        "class_names": ["crop", "weed"],
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "yolo_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Saved:", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
