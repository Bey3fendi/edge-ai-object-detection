# AGENTS.md — Edge AI Object Detection

Bu depoda çalışan ajanlar için bağlayıcı kurallar. Gerekçeler ve ayrıntılar
`project_information/Thesis_Edge_Object_Detection_Workflow.md` içindeki **Decision log**'dadır.
Belgelerin asıl kopyası tez yazarının Obsidian vault'undadır; buradaki kopyayı elle değiştirme,
değişiklik gerekiyorsa söyle.

## Kapsam
- Temel model **YOLOX-Nano** (Apache-2.0); referans COCO val2017 mAP@[.5:.95] = 25,8 (416).
- Hedef cihazlar yalnızca **Raspberry Pi 5** ve **Jetson Orin Nano**. Eğitim ve backpropagation
  cihazlarda yapılmaz; Colab'da (veya iş istasyonunda) yapılır.
- YOLOX değişiklikleri yalnızca `YOLOX/` submodule'ünde (fork `Bey3fendi/YOLOX`, dal `edge-ai-thesis`)
  ayrı commit olarak yapılır.

## Kararlar (2026-10-03)
1. **Canonical model:** `yolox_fp32.onnx` asıl referanstır. `yolox_int8_qdq.onnx` ondan türetilir ve
   Stage-1'de ek veri olarak ayrıca ölçülür. Her ölçüm satırı hangi artefaktı kullandığını yazar.
2. **Doğruluk kapıları:** mAP@[.5:.95] (val2017, tam 5000 görsel; mAP@.5 yardımcı), tüm modellerde aynı
   ön/son işlem ve NMS. INT8 kaybı FP32'ye göre ≤ 2,0 puan; aşılırsa QAT en fazla 2 × 10 epoch; sonra
   mixed precision; sonra olduğu gibi raporla. Toplam kayıp (orijinal Nano'ya göre) ≤ 4,0 puan.
   Aşama bütçeleri: NAS ≤ 1,0 · budama ≤ 1,0 · INT8 ≤ 2,0.
3. **Ölçüm protokolü:** USB-C güç ölçer (giriş, toplam kart gücü) cihazlar arası tek ölçüttür; idle güç
   her oturumda ölçülür. J/frame toplam ve artımsal raporlanır. tegrastats yalnız Jetson, ayrı sütun.
   100 çıkarım ısınma → 60 sn × 3 tekrar, ortalama ± std. FPS: yalnız model (ana) + uçtan uca (ayrı).
4. **Stage-2:** TensorRT INT8 QDQ modelden (explicit) + TensorRT FP16. ncnn QDQ ONNX'i **desteklemez**:
   ncnn INT8, FP32 pnnx modelinden `ncnn2table` + `ncnn2int8` ile, aynı 500 kalibrasyon görseliyle.
   ncnn FP32 satırı dönüşüm doğrulamasıdır. Her INT8 modelin mAP'i kendi cihazında ölçülür.
5. **Makro-mimari araması (NAS) ve budama:** NAS kanal genişliğine dokunmaz — genişlik budamanın işidir.
   Arama uzayı: çözünürlük 320/416/512 · aktivasyon SiLU/ReLU/HardSwish · CSP derinliği mevcut/−1 ·
   DW kernel 3/5. Adaylar pretrained Nano'dan, 10k COCO alt kümesinde 10 epoch. Seçim ölçütü
   **RPi 5 + ONNX Runtime CPU** gecikmesi; tek mimari; en fazla 2 tur. Budama oranı 10/20/30/40 %
   taranır. Ablasyon: baseline · yalnız NAS · yalnız budama · NAS + budama.
   **Karar 6 ile güncellendi:** bu ızgara araması (nas_r1b, nas_r2) önçalışmadır.
6. **Gerçek NAS (2026-10-16, danışman görüşmesi):** one-shot ağırlık paylaşımlı süper ağ
   (`exps/nas/supernet.py`): her DW katmanında kernel {3, 5, 7} (30 katman) · dark3/dark4 derinlik
   {1, 2, 3} · çözünürlük 320–512 (adım 32). Genişlik aramada yok. Süper ağ tam train2017 ile sandwich
   (en büyük + en küçük + 2 rastgele; inplace distillation yok — BigNAS'tan açık sapma), 60 epoch + önceden
   yazılı uzatma kuralı. **val2017 A/B** (`scripts/make_splits.py`): A arama (A_fit1k, BN kalibreli) ve tüm
   yöntem kararları; **B mühürlü** — NAS, budama, nicemleme kararları bitene kadar hiçbir kodda açılmaz.
   Nihai değerlendirme: gönderim kabulü doğrulanırsa COCO test-dev, değilse B. Gecikme: cihaz LUT'u
   (`scripts/latency_lut.py`). Arama: NSGA-II 50 × 20. Sıralama referansı: 10 aday sıfırdan 60 epoch.
   Final + orijinal Nano sıfırdan 300 epoch × 3 tohum. Hesaplama Colab L4, tek GPU.
7. **Kesintiye dayanıklılık (zorunlu, tüm uzun koşular):** tam durum (ham + EMA, optimizer, AMP ölçekleyici,
   RNG, epoch/iter) doğrulanmış depoya (`scripts/ckpt_store.py`: benzersiz dosya, sha256 + yükleme denetimi,
   önceki sağlam kayıt silinmez); tohumlu veri sırası ve örnek başına artırma (`scripts/nas_data.py`).
   Uzun koşudan önce kesinti testi (eğitim ortasında ve kayıt yazılırken öldürme) geçmeli.

## Teknik kurallar
- Yalnız **yapısal** kanal budama (Torch-Pruning DepGraph). Başlık koruması
  `model.head.cls_preds / reg_preds / obj_preds` modülleriyle yapılır (YOLOX'ta 85 kanallı katman yoktur).
- Budanmış model `torch.save(model)` ile tam nesne olarak kaydedilir; ONNX'e `tools/export_onnx.py`
  ile değil, kaydedilen nesne doğrudan dışa aktarılır ve PyTorch/ORT çıktı eşdeğerliği kontrol edilir.
- Statik giriş şekli `1×3×H×W`; dağıtım artefaktlarında dinamik eksen yok.
- ONNX **opset 13** (NAS adayları dahil tüm dışa aktarımlar; per-channel QDQ için gereken en düşük opset).
- ONNX Runtime CPU ölçümlerinde `intra_op_num_threads = 4`, iki cihazda ve NAS gecikme ölçümünde aynı.
- Notebook'lar `notebooks/NN_asama_konu.ipynb` adlandırmasıyla tutulur; ölçümler `results/` altına CSV olarak.
- Ölçülmemiş değeri ölçülmüş gibi yazma; tablo şablonlarındaki semboller (F1, W1 …) veri değildir.
