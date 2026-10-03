# Edge AI Object Detection

Donanıma duyarlı, edge için optimize edilmiş YOLOX-Nano nesne tespiti — yüksek lisans tezi uygulaması.
Hedef: COCO mAP'i bütçe içinde tutarak modeli **Raspberry Pi 5** ve **NVIDIA Jetson Orin Nano**
üzerinde daha hızlı ve daha az enerjiyle çalıştırmak.

Zincir: donanıma duyarlı makro-mimari araması → yapısal kanal budama → ONNX (FP32, canonical)
→ statik INT8 PTQ (QDQ) / QAT → Stage-1 (ONNX Runtime CPU) ve Stage-2 (ncnn, TensorRT) benchmark.

Ayrıntılı iş akışı ve kararlar: [`project_information/`](project_information/)

## Klasör yapısı

| Klasör | İçerik | Git |
| --- | --- | --- |
| `YOLOX/` | YOLOX fork'u (`Bey3fendi/YOLOX`, dal `edge-ai-thesis`), submodule | submodule |
| `notebooks/` | Colab notebook'ları | izlenir |
| `scripts/` | Yerel/cihaz betikleri (benchmark, dönüşüm, ölçüm) | izlenir |
| `results/` | Ölçüm CSV'leri, değerlendirme logları, özet tablolar | izlenir |
| `project_information/` | İş akışı belgeleri ve şema (asıl kopya Obsidian vault'ta) | izlenir |
| `artifacts/{pytorch,onnx,ncnn,tensorrt}/` | Ağırlıklar, ONNX grafikleri, ncnn/TensorRT dosyaları | izlenmez |
| `datasets/` | COCO alt kümeleri, kalibrasyon görselleri | izlenmez |
| `logs/` | Eğitim ve telemetri logları | izlenmez |

## Notebook adlandırma

`NN_asama_konu.ipynb` — `NN` iş akışındaki sırayı verir, ad ne yaptığını söyler.

| Ön ek | Aşama |
| --- | --- |
| `00_` | Pilot ve bütçe kalibrasyonu (ör. `00_pilot_1epoch_sure_olcumu.ipynb`) |
| `01_` | Baseline reprodüksiyonu |
| `02_` | Makro-mimari araması |
| `03_` | Seçilen mimarinin fine-tune'u |
| `04_` | Yapısal budama ve oran taraması |
| `05_` | ONNX dışa aktarımı ve eşdeğerlik kontrolü |
| `06_` | INT8 PTQ / QAT |

Aynı aşamada birden fazla notebook varsa ek harf kullanılır (`04a_`, `04b_`).
Her notebook başında amaç, girdi/çıktı artefaktları ve dayandığı karar(lar) yazılır.

## Sonuçlar

- `results/baseline/` — YOLOX-Nano baseline (COCO val2017, 416, PyTorch FP32, RTX GPU):
  **mAP@[.5:.95] 25,8 · AP50 41,4**. `benchmark_results_v0.csv` eski 24 sütunlu şemadadır;
  yeni ölçümler Karar 3'teki şemayı kullanır.

## Lisans

YOLOX Apache-2.0 lisanslıdır. Bu deponun kendi kodu için henüz lisans seçilmedi.
