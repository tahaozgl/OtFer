# OtFer
Deep Learning vs. Classical ML for Autonomous Fertilization

WeedCrop — YOLO / SVM / KNN Karsilastirma Calismasi
Otomatik gubreleme sistemleri icin bitki/ot ayrimi konusunda uc yontemin karsilastirildigi calisma.

Hazir ciktiar
`weedcrop\_yolo\_svm\_knn\_rapor.docx` — Word rapor (ana ciktiariniz)
`weedcrop\_yolo\_svm\_knn\_rapor.pdf` — ayni rapor PDF olarak

Kod
`code/01\_build\_features.py` — train/valid/test bbox'lardan HOG+renk ozelliklerini cikarir
`code/02\_train\_svm.py` — iki SVM varyantini egitir
`code/03\_train\_knn.py` — iki KNN varyantini egitir
`code/04\_train\_yolo.py` — YOLOv8n egitim+eval scripti (GPU'lu makinenizde calistirin)
`code/05\_compare.py` — uc modelin sonuclarini birlestirip grafikleri uretir
`code/06\_make\_report.js` — Word raporunu olusturur (node 06_make_report.js)

Yerelde YOLO calistirmak icin
```bash
pip install ultralytics
# data\_yolo.yaml icindeki "path:" satirini guncelleyin
python code/04\_train\_yolo.py --data code/data\_yolo.yaml --epochs 50 --imgsz 640 --batch 16 --device 0
# sonra raporu yenileyin:
python code/05\_compare.py
node code/06\_make\_report.js
```
YOLO calisinca `results/yolo\_metrics.json` olusur ve `05\_compare.py` literatur tahmini
yerine gercek degerleri otomatik kullanir.

Kisaca sonuclar
Model         Accuracy MacroF1 CropF1 WeedF1
svm_imbalanced	  0.886	0.666	0.395	0.937
svm_balanced	    0.879	0.737	0.544	0.930
knn_imbalanced	  0.882	0.644	0.352	0.935
knn_balanced	    0.785	0.603	0.335	0.872
yolov8n*	      	  -   0.790	0.700	0.880
(*) Literatur tahmini — gercegi yerelde calistirinca otomatik guncellenir.
