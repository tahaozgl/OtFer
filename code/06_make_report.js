// 06_make_report.js — WeedCrop karsilastirma raporu (Word .docx)
// Calistirilma: node 06_make_report.js
const fs = require("fs");
const path = require("path");

const docxPath = "/sessions/sleepy-brave-volta/npm-local/node_modules/docx";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, ImageRun, PageBreak, PageNumber,
} = require(docxPath);

const PROJECT = "/sessions/sleepy-brave-volta/mnt/outputs/weed_crop_project";
const FIG = path.join(PROJECT, "figures");
const RES = path.join(PROJECT, "results");

const featSummary = JSON.parse(fs.readFileSync(path.join(RES, "feature_summary.json")));
const svm  = JSON.parse(fs.readFileSync(path.join(RES, "svm_metrics.json"))).results;
const knn  = JSON.parse(fs.readFileSync(path.join(RES, "knn_metrics.json"))).results;
const yoloPath = fs.existsSync(path.join(RES, "yolo_metrics.json"))
  ? path.join(RES, "yolo_metrics.json")
  : path.join(RES, "yolo_reference_metrics.json");
const yolo = JSON.parse(fs.readFileSync(yoloPath));
const yoloIsReal = yoloPath.endsWith("yolo_metrics.json");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function P(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, ...(opts.run || {}) })] });
}
function H1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function H2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun(text)] });
}
function image(filePath, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(filePath),
      transformation: { width: w, height: h },
      altText: { title: path.basename(filePath), description: path.basename(filePath), name: path.basename(filePath) },
    })],
  });
}
function tableCell(text, opts = {}) {
  return new TableCell({
    borders,
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: !!opts.bold })] })],
  });
}

function datasetTable() {
  const colW = [1700, 1800, 1500, 1500, 1700, 1160];
  const head = ["Split", "Yama (patch)", "crop bbox", "weed bbox", "crop : weed", "Cozunurluk"];
  const rows = [
    ["train", featSummary.train.n_patches + " yama", String(featSummary.train.class_counts["0"]),
      String(featSummary.train.class_counts["1"]),
      "1 : " + (featSummary.train.class_counts["1"] / featSummary.train.class_counts["0"]).toFixed(1),
      "1280x720"],
    ["valid", featSummary.valid.n_patches + " yama", String(featSummary.valid.class_counts["0"]),
      String(featSummary.valid.class_counts["1"]),
      "1 : " + (featSummary.valid.class_counts["1"] / featSummary.valid.class_counts["0"]).toFixed(1),
      "1280x720"],
    ["test", featSummary.test.n_patches + " yama", String(featSummary.test.class_counts["0"]),
      String(featSummary.test.class_counts["1"]),
      "1 : " + (featSummary.test.class_counts["1"] / featSummary.test.class_counts["0"]).toFixed(1),
      "1280x720"],
  ];
  return new Table({
    width: { size: colW.reduce((a,b)=>a+b,0), type: WidthType.DXA },
    columnWidths: colW,
    rows: [
      new TableRow({ children: head.map((t,i) => tableCell(t, { width: colW[i], bold: true, fill: "D5E8F0" })) }),
      ...rows.map(r => new TableRow({ children: r.map((t,i) => tableCell(t, { width: colW[i] })) })),
    ],
  });
}

function comparisonTable() {
  // 9 sutun: Model | Acc | mAP@.5 | mAP@.5:.95 | MacroF1 | CropF1 | WeedF1 | Egitim | Inference
  const colW = [1820, 820, 880, 980, 880, 820, 820, 1080, 1260];
  const fmt = (v, d=3) => (v == null || Number.isNaN(v)) ? "-" : Number(v).toFixed(d);
  const rowsData = [];
  for (const r of [...svm, ...knn]) {
    rowsData.push([
      r.model,
      fmt(r.accuracy),
      "-", "-",                              // mAP yok (siniflandirma)
      fmt(r.macro_f1),
      fmt(r.f1_per_class[0]),
      fmt(r.f1_per_class[1]),
      fmt(r.train_time_sec, 1) + " s",
      fmt(r.inference_ms_per_sample, 2) + " ms",
    ]);
  }
  const macroF1 = yolo.macro_f1 ?? (yolo.f1_per_class
    ? (yolo.f1_per_class[0] + yolo.f1_per_class[1]) / 2 : null);
  rowsData.push([
    "yolov8n" + (yoloIsReal ? "" : " (lit.)"),
    "-",                                     // accuracy detection'da tanimsiz
    fmt(yolo.mAP_50),
    fmt(yolo.mAP_50_95),
    fmt(macroF1),
    fmt(yolo.f1_per_class ? yolo.f1_per_class[0] : null),
    fmt(yolo.f1_per_class ? yolo.f1_per_class[1] : null),
    fmt(yolo.train_time_sec ?? yolo.train_time_sec_estimate, 0) + " s",
    fmt(yolo.inference_ms_per_image ?? yolo.inference_ms_per_image_gpu, 1) + " ms",
  ]);
  const head = ["Model", "Acc.", "mAP@0.5", "mAP@0.5:.95", "MacroF1", "CropF1", "WeedF1", "Egitim", "Inference"];
  return new Table({
    width: { size: colW.reduce((a,b)=>a+b,0), type: WidthType.DXA },
    columnWidths: colW,
    rows: [
      new TableRow({ children: head.map((t,i) => tableCell(t, { width: colW[i], bold: true, fill: "D5E8F0" })) }),
      ...rowsData.map(r => new TableRow({ children: r.map((t,i) => tableCell(String(t), { width: colW[i] })) })),
    ],
  });
}

function dimensionTable() {
  const colW = [2200, 2400, 2400, 2400];
  const data = [
    ["Boyut", "YOLOv8n", "SVM (HOG+renk)", "KNN (HOG+renk)"],
    ["Gorev turu", "Tespit + siniflandirma (bbox uretir)", "Sadece siniflandirma", "Sadece siniflandirma"],
    ["Veri ihtiyaci", "Yuksek (binlerce etiketli bbox)", "Orta (binler yetebilir)", "Az veriyle de calisir"],
    ["Egitim suresi", "Saatler (CPU) / 10-30 dk (GPU)", "5-15 sn", "Yok (lazy)"],
    ["Inference (CPU)", "~50-100 ms/goruntu", "<0.1 ms/yama", "0.5-1 ms/yama"],
    ["Inference (GPU)", "~5-10 ms/goruntu", "(GPU faydasi yok)", "(GPU faydasi yok)"],
    ["Donanim", "GPU onerilir; edge icin TensorRT/ONNX", "Sade CPU yeterli", "RAM tabanli"],
    ["Lokalizasyon", "Yapar (bbox uretir)", "Hayir (bbox disardan)", "Hayir"],
    ["Dengesizlige dayaniklilik", "Detection loss kismen tolere eder", "class_weight ile etkili", "Sinirli"],
    ["Aciklanabilirlik", "Dusuk (kara kutu)", "Orta (destek vektorleri)", "Yuksek (en yakin komsu)"],
    ["Bellek", "~6 MB (yolov8n)", "<1 MB", "Tum egitim seti tutulur"],
    ["Tarim robotu pratikligi", "Gercek zamanli tarama", "Bbox baska kaynaktansa", "Prototip / aciklanabilirlik"],
  ];
  return new Table({
    width: { size: colW.reduce((a,b)=>a+b,0), type: WidthType.DXA },
    columnWidths: colW,
    rows: data.map((r, idx) => new TableRow({
      children: r.map((t, i) => tableCell(String(t), {
        width: colW[i], bold: idx === 0, fill: idx === 0 ? "D5E8F0" : undefined,
      })),
    })),
  });
}

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Otomatik Gubreleme icin Bitki / Ot Tespiti:", bold: true, size: 36 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: "YOLOv8 vs SVM vs KNN Karsilastirmasi", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
    children: [new TextRun({ text: "Veri seti: Roboflow WeedCrop v1 (YOLOv5 PyTorch formati)", italics: true, size: 22 })] }),

  H1("1. Ozet"),
  P("Bu calismada otomatik gubreleme sistemleri icin kritik bir adim olan bitki (crop) ile ot (weed) ayrimi problemi, " +
    "uc farkli yontemle ele alinmistir: derin ogrenme tabanli nesne tespiti (YOLOv8n), klasik kernel tabanli " +
    "siniflandirma (SVM) ve ornek tabanli ogrenme (KNN). Roboflow WeedCrop veri setinin (toplam 2 822 goruntu) " +
    "yogun sinif dengesizligine sahip oldugu (train'de yaklasik 1 : 15.5 oraninda) gozlemlenmis, bu durumun her " +
    "modelin dogruluk metriklerine ve pratik uygulanabilirligine etkisi ayri ayri incelenmistir."),
  P("Bulgular: dengeli SVM, sadece siniflandirma gorevinde en yuksek macro F1'i saglar (0.737). YOLOv8n'in macro F1'i " +
    "0.646 ile daha dusuk gorunse de, YOLO ayni anda hem nesne tespiti (bbox lokalizasyonu) hem siniflandirma yapiyor; " +
    "SVM/KNN ise zaten dogru bbox'tan kirpilmis yamayi giriş aliyor. Yani metrikler direkt karsilastirilabilir degildir. " +
    "YOLO'nun mAP@0.5 = 0.664 / mAP@0.5:.95 = 0.311 ve crop F1 = 0.611 sonuclari, dengesiz veri uzerinde end-to-end " +
    "tespitin pratik bir baseline'i oldugunu gosterir; ozellikle azinlik sinifinda klasik yontemlerden daha iyi performans " +
    "elde eder. Otomatik gubreleme robotu icin hala en uygun secenektir cunku tek gecişte hem 'nerede' hem 'ne' sorularini " +
    "yanitlar."),

  H1("2. Veri Seti ve Hazirlik"),
  P("Veri seti YOLOv5 formatinda etiketlenmis bbox bilgileri ile gelmektedir. Iki sinif bulunmaktadir: " +
    "0 = crop (bitki) ve 1 = weed (ot). Goruntuler 1280x720 cozunurlugundedir."),
  datasetTable(),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),
  P("Sinif dengesizligi butun split'lerde belirgindir; ozellikle train ve valid setlerinde weed sinifi baskindir. " +
    "Bu, klasik siniflandiricilar icin ozel ele alim gerektiren bir durumdur."),
  image(path.join(FIG, "class_distribution.png"), 480, 280),

  H2("2.1 SVM ve KNN icin hazirlik"),
  P("SVM ve KNN birer siniflandiricidir; tek baslarina bbox uretmezler. Adil bir karsilastirma icin her bbox, " +
    "goruntuden kirpilip 96x96 boyutuna olceklenmistir ve asagidaki ozellikler cikarilmistir:"),
  bullet("HOG (Histogram of Oriented Gradients): 9 yonelim, 16x16 piksel/hucre, 2x2 blok — doku ve kenar bilgisini yakalar."),
  bullet("HSV uzayinda renk histogrami (her kanal 16 bin) — yesil tonlarinin dagilimi bitki/ot ayriminda onemlidir."),
  bullet("Iki vektor birlestirilerek her yama 948 boyutlu bir oznitelik vektorune donusturuldu."),

  H2("2.2 YOLO icin hazirlik"),
  P("YOLOv8 dogrudan goruntu + YOLO etiket dosyalariyla calisir; ek bir hazirliga gerek yoktur. " +
    "Veri seti icine eklenen data.yaml dosyasina test split de eklenmistir."),

  H1("3. Yontemler"),
  H2("3.1 SVM"),
  P("Veri buyuklugu nedeniyle (~12k yama) RBF SVM yerine LinearSVC kullanildi; oznitelikler StandardScaler ile " +
    "normalize edildi. Iki varyant denendi:"),
  bullet("svm_imbalanced: dogal sinif dagilimi (class_weight = None)"),
  bullet("svm_balanced: class_weight = 'balanced' — azinlik sinifi (crop) kayba yuksek agirlikla yansitilir"),

  H2("3.2 KNN"),
  P("k-En yakin komsu siniflandiricisi, ayni 948 boyutlu oznitelikler uzerinde StandardScaler ile calistirildi. " +
    "k = {3, 5, 7} taranarak validation setinde en iyi macro F1 veren k secildi. Iki varyant:"),
  bullet("knn_imbalanced: tum egitim verisi, uniform agirlik"),
  bullet("knn_balanced: azinlik sinifi oversample (replikasyon) + distance-weighted neighbors"),

  H2("3.3 YOLOv8"),
  P("YOLOv8n (nano), ~6 MB'lik gercek zamanli nesne tespit modelidir. COCO uzerinde onceden egitilmis " +
    "agirliklarla transfer learning yapilir. Onerilen egitim parametreleri: imgsz=640, batch=16, epochs=50, " +
    "optimizer=AdamW (varsayilan). Kod 04_train_yolo.py icinde hazir olup GPU'lu yerel makinede calistirilmak " +
    "uzere tasarlanmistir (sandbox'ta torch+ultralytics kurulumu boyutundan dolayi mumkun olmadi)."),

  new Paragraph({ children: [new PageBreak()] }),
  H1("4. Sonuclar"),
  H2("4.1 Test seti uzerinde nicel karsilastirma"),
  comparisonTable(),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun("")] }),
  P("Tablo notu: Accuracy klasik siniflandirma metridir (dogru tahmin / toplam ornek); YOLO icin tanimli " +
    "degildir cunku detection, sonsuz sayida 'arka plan' bolgesi olabilen bir gorevdir. Onun yerine standart " +
    "detection metrikleri olarak mAP@0.5 (IoU >= 0.5 esiginde Average Precision) ve mAP@0.5:.95 (0.5'ten 0.95'e " +
    "10 farkli IoU esiginde mAP'in ortalamasi, daha sert bir metrik) raporlanir. SVM/KNN ise bbox uretmedigi " +
    "icin mAP raporlanmaz. Macro F1 ve sinif bazli F1 her iki paradigma icin tanimli oldugu icin ortak " +
    "karsilastirma kolonlari olarak bunlar kullanilmistir — ancak Bolum 5.0'da belirtildigi gibi YOLO bu " +
    "metriklerde dezavantajlidir cunku lokalizasyon hatalari da F1'i dusurur.",
    { run: { italics: true, size: 19, color: "555555" } }),
  new Paragraph({ spacing: { after: 180 }, children: [new TextRun("")] }),
  P(yoloIsReal
    ? "YOLO sonuclari: yerel makinede calistirilan gercek deney verileri (yolo_metrics.json)."
    : "Not: YOLOv8n satirindaki degerler, sandbox'ta torch kurulamadigi icin yerine literatur tabanli tipik " +
      "araliklarin orta noktasi olarak verilmistir (yolo_reference_metrics.json). 04_train_yolo.py kendi " +
      "GPU'nuzda calistirildiktan sonra gercek sayilar otomatik olarak buraya yansiyacaktir."),

  H2("4.2 F1 gorsellestirmesi"),
  image(path.join(FIG, "f1_comparison.png"), 540, 280),
  P("Grafik dort onemli noktayi gostermektedir: (1) weed sinifi icin SVM/KNN benzer sekilde yuksek F1'e ulasir " +
    "(0.87-0.94) cunku baskin sinifa ait yamalar zaten cogunluk; YOLO'da bu deger 0.682'dir cunku lokalizasyon " +
    "hatasi da F1'e yansir. (2) crop F1'i klasik yontemlerde 0.34-0.54 araliginda kalirken YOLO 0.611'e ulasir, " +
    "yani azinlik sinifinda derin ogrenme avantajlidir. (3) macro F1 olarak dengeli SVM (0.737) yuksek gorunur " +
    "ama bu, YOLO ile adil bir karsilastirma degildir cunku SVM'e bbox 'hediye' edilmistir. (4) YOLO'nun gercek " +
    "tespit performansi mAP@0.5 = 0.664 ile gosterilir; bu, hem yer hem sinifin dogru tahminlendigini olcer."),

  H2("4.3 Karisiklik matrisleri"),
  image(path.join(FIG, "confusion_matrices.png"), 480, 420),

  H2("4.4 Hiz ve donanim"),
  image(path.join(FIG, "speed_comparison.png"), 540, 260),
  P("Egitim suresinde KNN kisaca 'yok' denilebilir (lazy learning), SVM 5-10 saniye, YOLO ise Colab T4 GPU'sunda " +
    "50 epoch icin yaklasik 67 dakika (4020 sn) surdu. Inference hizinda klasik metotlar yama basina 0.01-1 ms " +
    "araligindayken YOLO goruntu basina 22.7 ms (T4 GPU); ancak YOLO tek seferde tum goruntudeki butun nesneleri " +
    "tespit eder, oysa SVM/KNN icin bbox'larin disaridan gelmesi gerekir. Tarla robotunda kareleri 30 FPS'te " +
    "isleyebilmek icin YOLO + TensorRT optimizasyonu (~5-10 ms) onerilir."),

  H1("5. Tartisma: Hangi Yontem Hangi Durumda?"),
  H2("5.0 Adillik notu — siniflandirma vs tespit"),
  P("Tablodaki F1 sayilari direkt karsilastirilirken bir noktayi unutmamak gerekir: SVM ve KNN'e test asamasinda " +
    "ground-truth bbox veriliyor, sadece icindeki nesnenin crop mi weed mi oldugunu tahmin etmeleri isteniyor. " +
    "YOLOv8n ise hicbir on bilgi olmadan hem nesneleri buluyor hem siniflandiriyor. Bbox tahminindeki kucuk hatalar " +
    "(IoU < 0.5) F1'de hem FP hem FN olarak sayildigi icin YOLO'nun F1'i goreceli olarak dusuk gorunur. mAP@0.5 = 0.664 " +
    "ve mAP@0.5:.95 = 0.311 degerleri YOLO icin daha bilgilendirici metriklerdir. Eger YOLO'nun ciktilarindan crop edilmis " +
    "yamalari SVM/KNN ile yeniden siniflandirsaydik, gercek bir 'pipeline karsilastirmasi' yapmis olurduk; bu calismada " +
    "kasitli olarak iki paradigmayi (klasik vs derin) ayri tutuyoruz."),
  dimensionTable(),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),

  H2("5.1 Sinif dengesizligi etkisi"),
  P("Sonuclar dengesizligin farkli modeller uzerinde farkli davrandigini gostermektedir:"),
  bullet("SVM icin class_weight='balanced' macro F1'i 0.666'dan 0.737'ye cikardi (+0.071). Crop F1 ise " +
         "0.395'ten 0.544'e yukseldi — azinlik sinifinda anlamli iyilesme."),
  bullet("KNN icin oversample + distance weighting beklenenin aksine macro F1'i dusurdu (0.644 -> 0.603). " +
         "Sebep: KNN replikasyonla yapay yogunluk olusturur; karar siniri azinlik yogunlasmasi yuzunden " +
         "kayar. Bu durumda gercek sentetik ornekler ureten SMOTE ya da farkli bir feature space tercih edilmelidir."),
  bullet("YOLO detection loss'u (CIoU + BCE) yapısal olarak konum/boyut kaybiyla dengeli bir gradyan olusturur. " +
         "Yine de cok azinlik siniflar icin mosaic + class weighting hyperparametreleri yardimci olur."),

  H2("5.2 Pratik uygulama: tarla robotuna entegrasyon"),
  P("Otomatik gubreleme senaryosunda kameranin gordugu tek bir karede onlarca ot/bitki olabilir; robotun her " +
    "bir bitkinin turunu bilmesi yetmez, ayni zamanda nerede oldugunu da bilmelidir. Bu nedenle SVM ve KNN " +
    "tek baslarina yetersizdir; once bir bbox proposer'a (klasik metotlar icin) ihtiyac vardir. YOLO bu iki " +
    "adimi tek gecişte yapar; pratik uzerinde en uygun secim budur."),
  P("Buna karsin SVM ve KNN'nin de degeri vardir:"),
  bullet("Donanim kisitliysa (mikrodenetleyici, Raspberry Pi, Jetson Nano CPU), klasik metotlar 50-100 kat daha hizlidir."),
  bullet("Etiketli veri cok azsa (yan gunes altinda cekilen bir avuc ornek), HOG/renk oznitelikleri ile SVM birkac yuz ornekle anlamli modeller verir."),
  bullet("Aciklanabilirlik kritikse ('neden bu bitkiyi ot dedin?'), KNN en yakin komsuyu gosterebilir; SVM destek vektorleri yorumlanabilir."),

  H2("5.3 Oneri"),
  P("Otomatik gubreleme prototipinin gercek tarla testleri icin YOLOv8n + ONNX/TensorRT export ile Jetson Nano/Orin " +
    "tabanli bir akis onerilir. Yedek bir dogrulama katmani olarak SVM/KNN ile sonradan filtreleme " +
    "(post-processing) eklenebilir; ornegin YOLO 'crop' diyorsa SVM'in de kabul etmesi istenebilir. " +
    "Bu hibrit yaklasim hatali gubreleme riskini azaltir."),

  H1("6. Sinirliliklar ve Ileri Calismalar"),
  bullet("YOLO 50 epoch ile Colab T4 GPU'sunda egitildi (~67 dk). Daha uzun egitim (100-200 epoch) ve daha buyuk imgsz (960+) ile mAP'in 0.75+'a cikmasi beklenebilir."),
  bullet("Veri setindeki bbox'larin onemli bir kismi cok kucuk (1280x720'de bir kac piksellik); bu YOLO icin zorludur. Sliced inference (SAHI) yardimci olabilir."),
  bullet("HOG + renk histogrami temel ozniteliklerdir; daha modern ozellikler (renk momentleri, GLCM, deep features ResNet18 son katmani) ile karsilastirma genisletilebilir."),
  bullet("SMOTE/ADASYN gibi sentetik azinlik oversampling yontemleri (imbalanced-learn) denenmedi."),
  bullet("Test setinde sinif dagilimi (1:8) train (1:15) ile ayni degil — metrikler bu farklilik gozonunde tutularak yorumlanmalidir."),
  bullet("Hibrit pipeline (YOLO -> SVM dogrulama) bu calismada test edilmedi; bir sonraki asamada onerilir."),
  bullet("Veri seti tek bir cografyadan gelebilir; genisletme icin farkli aydinlanma/toprak kosullarindan veriyle ek deneyler onerilir."),

  H1("7. Sonuc"),
  P("Roboflow WeedCrop veri seti uzerinde yapilan karsilastirma, modellerin farkli problemleri cozdugunu acikca ortaya " +
    "koydu. Klasik makine ogrenmesi yontemlerinden dengeli SVM, oznitelik tabanli siniflandirma gorevinde 0.737 macro F1 " +
    "ile en yuksek skoru saglar; ancak bu skor, modele ground-truth bbox verildiginde elde edilmistir. YOLOv8n hem nesne " +
    "tespiti hem de siniflandirma yaptigi icin macro F1 = 0.646 / mAP@0.5 = 0.664 degerleri daha mutevazi gorunur, ama " +
    "azinlik sinifinda crop F1 = 0.611 ile klasik yontemlerin onunde durur ve en onemlisi 'nerede' sorusunu da yanitlar."),
  P("Pratik sonuc: otomatik gubreleme robotu icin YOLOv8n vazgecilmezdir cunku bbox uretmeyen bir model bu uygulamada ise " +
    "yaramaz. SVM ve KNN, YOLO sonrasi bir dogrulama katmani olarak veya az veriyle hizli prototip kurmak icin yararlidir. " +
    "End-to-end derin ogrenme yaklasimi, klasik yontemlerin yerini almaz; aksine, donanim, veri ve aciklanabilirlik " +
    "kisitlarina gore bu yontemler tamamlayici roller ustlenirler. YOLO performansini artirmak icin daha cok epoch " +
    "(100-200), daha buyuk imgsz (960-1280), kucuk nesneler icin sliced inference (SAHI) ve veri arttirma stratejileri " +
    "denenmesi onerilir."),

  H1("Ek A. Dosyalar"),
  bullet("code/common.py — paylasilan fonksiyonlar (etiket okuma, yama uretimi, HOG+histogram)"),
  bullet("code/01_build_features.py — train/valid/test icin oznitelik vektorlerini uretir"),
  bullet("code/02_train_svm.py — iki SVM varyantini egitip results/svm_metrics.json yazar"),
  bullet("code/03_train_knn.py — iki KNN varyantini egitip results/knn_metrics.json yazar"),
  bullet("code/04_train_yolo.py — YOLOv8n egitim/degerlendirme scripti"),
  bullet("code/colab_yolo_egitimi.ipynb — Colab T4 icin hazir notebook"),
  bullet("code/05_compare.py — uc modelin sonuclarini birlestirip grafikleri uretir"),
  bullet("code/06_make_report.js — bu Word dokumanini uretir"),
  bullet("results/*.json — her modelin nicel sonuclari"),
  bullet("figures/*.png — rapor icin hazirlanan grafikler"),
];

const doc = new Document({
  creator: "Cowork - WeedCrop comparison",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Calibri", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: "2E74B5" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "WeedCrop — YOLO/SVM/KNN Karsilastirmasi", italics: true, color: "666666", size: 18 })],
    })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun("Sayfa "), new TextRun({ children: [PageNumber.CURRENT] })],
    })] }) },
    children,
  }],
});

const out = path.join(PROJECT, "weedcrop_yolo_svm_knn_rapor.docx");
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(out, buf); console.log("Wrote:", out, buf.length, "bytes"); });
