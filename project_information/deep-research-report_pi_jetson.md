# Uç Yapay Zekâ Ortamları İçin Donanıma Duyarlı Nesne Tespit Modeli: Uçtan Uca Tez Yol Haritası

> **Kapsam güncellemesi:** Bu revizyonda deneysel donanım kapsamı yalnızca **Raspberry Pi** ve **NVIDIA Jetson Orin Nano** olarak sınırlandırılmıştır. Tüm yöntem, dağıtım ve benchmark tasarımı bu iki platforma göre düzenlenmiştir.

## Tezin teknik çerçevesi ve önce düzeltilmesi gereken noktalar

`yontem.txt` dosyasındaki yaklaşım teknik açıdan doğru eksene oturmuş durumda: sıfırdan yeni bir dedektör icat etmek yerine, mevcut ve güçlü bir nesne tespit modelini **NAS + budama + düşük hassasiyetli çıkarım** ile uç donanıma göre yeniden biçimlendirmek ve ortaya bir *hardware-aware variant* çıkarmak hedefleniyor. Bu revizyonda deneysel karşılaştırma **Raspberry Pi ve Jetson Orin Nano** ile sınırlandırılmıştır. Sürekli öğrenme ise ana deney yüküne sokulmayıp teorik olarak ele alınmaya devam etmektedir. fileciteturn0file0

Tez öneri formundaki asıl risk ise şu cümlelerde: “**yeni bir nesne tespit mimarisi oluşturulacak**” ve “**özgün bir model ortaya çıkarılması**”. Formun geri kalanında NAS, kuantizasyon, budama, CPU/GPU karşılaştırması ve sürekli öğrenmenin teorik olarak incelenmesi zaten tanımlanmış olsa da, bu iki ifade jüri tarafından “mimarinin temel bloklarını sıfırdan icat edeceksiniz” şeklinde yorumlanabilir. fileciteturn0file1

Bu nedenle tezde savunulması gereken iddia şu olmalı:

> **Bu tezde sıfırdan yeni bir nesne tespit ailesi icat edilmemekte; açık kaynaklı ve önceden eğitilmiş bir gerçek-zamanlı dedektörün mimari arama, yapısal budama ve kuantizasyon yoluyla heterojen uç donanımlar için optimize edilmiş yeni bir donanıma-duyarlı varyantı geliştirilmektedir.**

Bu daha güçlü bir iddiadır; çünkü gerçekten ölçülebilir bir araştırma sorusuna dönüşür:

| Araştırma sorusu | Ölçülebilir çıktı |
|---|---|
| NAS gerçekten fayda sağlıyor mu? | ΔmAP, ΔMAC/FLOPs, Δlatency, ΔFPS/W |
| Structured pruning gerçekten donanımda hız kazandırıyor mu? | sparsity/MAC azalması → gerçek latency azalması |
| INT8/PTQ ne kadar doğruluk kaybettiriyor? | mAP FP32 − mAP INT8 |
| QAT bu kaybı geri kazanıyor mu? | mAP QAT − mAP PTQ |
| Donanıma özel compiler ne kazandırıyor? | Stage-2 / Stage-1 speedup |
| Raspberry Pi CPU ve Jetson GPU tabanlı dağıtım yolları enerji açısından nasıl ayrışıyor? | FPS/W ve J/frame |
| Aynı model Raspberry Pi ve Jetson üzerinde aynı şekilde mi ölçekleniyor? | latency ve enerji Pareto eğrileri |

Bu yaklaşım, yüklediğiniz FPGA/GPU karşılaştırma çalışmalarındaki önemli bir metodolojik sorunu da doğrudan çözüyor. Uppsala çalışması, farklı platformlarda farklı precision, model ve hatta bazı deneylerde farklı veri setlerinin kullanılmasının doğrudan karşılaştırmanın geçerliliğini zayıflattığını açıkça kabul ediyor; gelecekte aynı network/modelin iki platforma da aktarılmasını öneriyor. fileciteturn0file4 Aynı şekilde yüklediğiniz IEEE çalışması aynı CNN'i aynı görüntülerle farklı donanımlara taşıyarak güç, inference süresi, doğruluk ve geliştirme kolaylığını karşılaştırıyor. fileciteturn0file2

Bu yüzden tezin **özgün katkısını “hangi teknik daha iyi?” değil, aşağıdaki deneysel zincir oluşturmalıdır**:

```text
Permissive açık kaynak temel model
          ↓
COCO baseline reprodüksiyonu
          ↓
Hardware-aware NAS
          ↓
Tam eğitim / fine-tuning
          ↓
Structured channel/filter pruning
          ↓
Fine-tuning
          ↓
PTQ INT8
          ↓
Gerekirse QAT INT8
          ↓
Canonical ONNX
          ↓
┌───────────────────────────┬───────────────────────────────┐
│ Stage 1                   │ Stage 2                       │
│ Ortak CPU runtime         │ Platforma özel maksimum      │
│ ONNX Runtime CPU EP       │ optimize runtime              │
├───────────────────────────┼───────────────────────────────┤
│ Raspberry Pi              │ ncnn / ARM NEON / INT8        │
│ Jetson Orin Nano          │ TensorRT GPU / FP16 / INT8    │
└───────────────────────────┴───────────────────────────────┘
          ↓
mAP / latency / FPS / Watt / FPS-W / J-frame / RAM
          ↓
Pareto analizi + ablation study
```

Bu tez yapısı hem öneri formunuzdaki “bütüncül optimizasyon” iddiasını gerçekleştirir hem de “sıfırdan model icadı” gibi gereksiz ve savunması zor bir hedefi ortadan kaldırır. fileciteturn0file0 fileciteturn0file1

## Veri seti ve temel model seçimi

### Ana benchmark veri seti COCO olmalı

Bu tez için **COCO 2017 ana veri seti olmalıdır**. COCO'nun temel avantajı yalnızca büyük olması değil; karmaşık günlük sahneler, farklı nesne ölçekleri ve çoklu nesne bağlamı üzerine kurulmuş olması ve standart nesne tespit değerlendirmesi sunmasıdır. COCO 2017 detection benchmark'ında 80 nesne kategorisi bulunur; `train2017` yaklaşık 118 bin, `val2017` ise 5 bin görüntüdür. citeturn10search6turn10search9turn10academia18

Pascal VOC daha küçük ve deney döngülerini hızlandırabilir; fakat **tezin nihai doğruluk iddiasını VOC üzerinde kurmayın**. VOC'u yalnızca “pipeline smoke test” veya erken geliştirme aşamasında kullanmak mantıklıdır. Nihai model seçimi ve raporlanan mAP, COCO val2017 üzerinden yapılmalıdır.

Önerdiğim veri kullanım planı:

```text
COCO train2017
│
├─ NAS proxy subset:
│    sabit ve sınıf dağılımı korunmuş 10k–20k görüntü
│
├─ Full training:
│    train2017 tamamı
│
├─ INT8 calibration:
│    train2017 içinden sabit temsilî calibration subset
│
└─ Final evaluation:
     COCO val2017'nin 5.000 görüntüsünün TAMAMI
```

En kritik kural: **NAS sırasında hızlı sonuç almak için kullanılan proxy subset üzerinde nihai akademik sonuç raporlanmamalıdır.** Proxy yalnızca aday mimari sıralamak içindir.

COCO metriklerinde ana sonuç:

\[
mAP = AP_{50:95}
\]

olmalı. Bunun yanında `AP50`, `AP75`, `AP_S`, `AP_M`, `AP_L` raporlanmalıdır. Özellikle edge optimizasyonunda düşük çözünürlüğe geçildiğinde küçük nesne başarımı `AP_S` ilk bozulan metriklerden biri olabileceği için bunu ayrıca göstermelisiniz.

### Lisans konusunda önemli düzeltme

“Hiçbir kısıtlama olmaksızın %100 açık kaynak” ifadesini akademik metinde aynen kullanmayın. **MIT, BSD veya Apache-2.0 dahi kelimenin literal anlamıyla ‘kısıtlamasız’ değildir; lisans bildirimi/attribution gibi şartları vardır.** Ayrıca kod lisansı ile pretrained ağırlıkların hukuki statüsü aynı şey değildir. Torchvision da pretrained ağırlıkların kendi lisans/şartlarının bulunabileceğini açıkça hatırlatmaktadır. citeturn1search1

Bu nedenle tez şartını şöyle yazın:

> **Temel model; kaynak kodu ve eğitim betikleri permissive açık kaynak lisans altında bulunan, resmi pretrained ağırlıkları ücretsiz ve kayıt/paywall gerektirmeden indirilebilen ve mimari kaynak kodu araştırmacı tarafından değiştirilebilen bir model olacaktır.**

Bu tanımla **YOLOv8'i ana model olarak seçmem**. Ultralytics'in güncel açık kaynak dağıtımı AGPL-3.0 ile lisanslanıyor ve ayrıca Enterprise lisans seçeneği sunuyor. AGPL açık kaynak bir lisanstır, fakat güçlü copyleft yükümlülükleri nedeniyle sizin “MIT/Apache tarzı serbest müdahale ve yeniden kullanım” kriterinizle uyuşmuyor. citeturn4search0

### Önerilen temel modeller

| Model | Kod lisansı | Resmî pretrained | Boyut/COCO referansı | NAS/pruning uygunluğu | Kararım |
|---|---|---:|---:|---|---|
| **YOLOX-Nano** | Apache-2.0 | Evet | 0.91 M param., ~1.08 GFLOPs, COCO AP ~25.8 @416 | Çok iyi | **Birinci tercih** |
| **RTMDet-Tiny / MMDetection** | Apache-2.0 | Evet | Tiny aile; yüksek accuracy/parameter dengesi | Mükemmel | İkinci aday |
| **SSDLite320-MobileNetV3-Large** | Torchvision BSD-3-Clause kod | Evet | 3.44 M param., 0.58 GFLOPs, COCO mAP 21.3 | Orta/iyi | Kontrol modeli |

YOLOX-Nano'nun resmi kaynak kodu Apache-2.0'dır; resmi repo pretrained ağırlıkları sağlar ve ONNX, TensorRT, ncnn ve OpenVINO deployment yollarını içerir. YOLOX-Nano resmi model tablosunda yaklaşık 0.91 milyon parametre, 1.08 GFLOPs ve 416×416 girdide COCO AP 25.8 olarak veriliyor. YOLOX ayrıca anchor-free tasarım, decoupled head ve SimOTA kullanır. citeturn14search7turn14academia49turn1search3

Bu tez için **YOLOX-Nano en mantıklı ana modeldir**. Sebebi yalnızca küçük olması değil: Pi üzerinde çalışabilecek kadar hafif, Jetson'da GPU avantajını gösterecek kadar gerçek bir CNN iş yükü, ncnn/TensorRT dönüşümlerini uygulanabilir tutacak kadar geleneksel operatörlerden oluşan bir mimari ve resmi olarak çoklu deployment backend'i destekliyor. citeturn14search7

RTMDet-Tiny ikinci güçlü seçenektir. MMDetection kod tabanı Apache-2.0'dır ve modüler biçimde backbone, neck ve head bileşenlerini değiştirmeye imkân verir. RTMDet ailesi tiny'den büyük modellere kadar özellikle parametre-doğruluk dengesi için tasarlanmıştır. citeturn11search0turn14academia48 RTMDet konfigürasyonunda `deepen_factor` ve `widen_factor` gibi parametrelerin açıkça tanımlı olması mimari arama açısından özellikle avantajlıdır. citeturn0search13

Burada bir lisans ayrıntısı önemli: RTMDet'i **MMDetection içinden kullanın**. OpenMMLab geliştiricilerinin kendi açıklamasına göre bazı MMYOLO hızlı-eğitim bileşenleri YOLOv5 kökenli lisans şartları taşırken, RTMDet'in MMDetection implementasyonu Apache-2.0 tarafında tutulmuştur. citeturn11search9

SSDLite320-MobileNetV3-Large ise iyi bir kontrol modelidir. Torchvision'ın resmi pretrained COCO ağırlıkları 3.44 milyon parametre, yaklaşık 0.58 GFLOPs ve COCO-val2017 üzerinde 21.3 box mAP raporlar. citeturn14search1turn14search10

**Benim nihai seçimim:**

> **Ana tez modeli: YOLOX-Nano.**  
> **Literatür/benchmark karşılaştırması: RTMDet-Tiny.**  
> **Klasik mobil kontrol modeli: SSDLite320-MobileNetV3-Large.**

Üç model üzerinde bütün NAS + pruning + QAT zincirini çalıştırmayın. Bu tez kapsamını patlatır. Önce üç modeli aynı workstation'da baseline olarak karşılaştırın; sonra **yalnızca seçilen YOLOX-Nano üzerinde bütün optimizasyon zincirini** uygulayın.

Ayrıca checkpoint konusunda lisans açısından kurşun geçirmez olmak istiyorsanız tez ekinde şunları saklayın:

```text
model_name
repository commit SHA
LICENSE dosyası
pretrained checkpoint dosya adı
checkpoint SHA256
indirildiği tarih
dataset sürümü
eğitim config'i
```

Böylece “hangi kod ve hangi ağırlık kullanıldı?” sorusunun cevabı deneyden yıllar sonra bile verilebilir.

## NAS, budama ve kuantizasyonla model sıkıştırma zinciri

Optimizasyon sırası rastgele olmamalı. Önerdiğim tez zinciri:

```text
Baseline
   ↓
Hardware-aware NAS
   ↓
Selected architecture full retraining
   ↓
Structured pruning
   ↓
Fine-tuning
   ↓
INT8 PTQ
   ↓
Accuracy gate
   ├── kabul edilebilir → deploy
   └── fazla kayıp → QAT → deploy
```

Bu sıra özellikle tez açısından iyidir, çünkü her optimizasyon adımının marjinal katkısını ayrı ayrı ölçebilirsiniz.

### Baseline'ı kilitlemeden NAS'a başlamayın

İlk işiniz model optimize etmek değil, **resmî baseline'ı yeniden üretmek** olmalıdır.

YOLOX-Nano için:

```text
Model        : YOLOX-Nano
Dataset      : COCO 2017
Input        : 416×416
Precision    : FP32
Checkpoint   : official pretrained
Metric       : COCO AP50:95
```

İlk deneyde kendi ortamınızın mAP değerinin resmî sonuca yakın olduğunu gösterin. Sonra aynı checkpoint'ten ONNX çıkarın ve PyTorch ↔ ONNX output farkını bir “golden set” üzerinde test edin.

Örneğin 100 sabit görüntü üzerinde:

```python
max_abs_diff = max(abs(torch_output - onnx_output))
```

ve detection sonuçlarında IoU/class-score farklarını kaydedin.

Bu adım atlanırsa daha sonra kaybedilen mAP'ın NAS'tan mı, quantization'dan mı, ONNX converter'dan mı geldiğini anlayamazsınız.

### Hardware-aware NAS nasıl yapılmalı?

Burada kritik nokta şudur:

**FLOPs-minimizing NAS yapmayın. Latency-aware NAS yapın.**

FLOPs ile gerçek latency birebir ilişkili değildir. Yüklediğiniz Jetson benchmark çalışması da düşük FLOPs'un genellikle avantaj sağladığını göstermekle birlikte, mimari yapı ve backend optimizasyonunun latency üzerinde belirleyici olduğunu gösteriyor; MobileNetV2 ve ShuffleNetV2 gibi düşük-FLOP modeller TensorRT'den büyük kazanç sağlarken bazı mimariler farklı davranabiliyor. fileciteturn0file3

YOLOX-Nano için search space'i küçük ve deployment-dostu tutun:

| Aranan parametre | Önerilen seçenekler |
|---|---|
| Input resolution | 320, 352, 384, 416 |
| Backbone stage width | baseline × {0.75, 1.0, 1.25} |
| Backbone block repeat | baseline −1 / baseline / +1 |
| Neck width | × {0.75, 1.0, 1.25} |
| Head channel width | × {0.75, 1.0} |
| Conv tipi | Standard / Depthwise |
| Kernel | 3×3; yalnızca desteklenen yerlerde 5×5 |

**Search space'e exotic operatör sokmayın.** Önce **TensorRT ve ncnn** operator destek kesişimini çıkarın; NAS yalnızca iki hedef deployment yolunda güvenli biçimde taşınabilen operatörlerde gezsin. Aksi halde NAS matematiksel olarak iyi bir mimari bulabilir fakat hedef platformlardan birinde verimsiz veya uyumsuz bir graph üretebilir.

Amaç fonksiyonunu yalnızca mAP/FLOPs yapmayın. Çok amaçlı Pareto yaklaşımı daha akademiktir:

\[
\text{maximize}\quad mAP
\]

\[
\text{minimize}\quad
Latency,\ Energy/frame,\ ModelSize,\ PeakRAM
\]

Bir zorunlu kısıt da koyabilirsiniz:

\[
mAP_{candidate} \ge mAP_{baseline} - \delta
\]

Burada örneğin `δ = 1.0 AP` **sizin deney tasarımınızın kabul kriteridir**, evrensel bir standart değildir.

Tek bir skora zorlamak gerekirse:

\[
J =
mAP -
\lambda_L \log\frac{L}{L_0}
-
\lambda_E \log\frac{E}{E_0}
-
\lambda_M \log\frac{M}{M_0}
\]

kullanılabilir; fakat tezde asıl sonuç olarak **Pareto front'u** göstermek daha savunulabilir. Aksi halde λ katsayılarının keyfî seçimi eleştiri konusu olur.

### NAS için hangi kütüphane?

Microsoft NNI açık kaynak MIT lisanslı bir NAS/HPO/model-compression framework'üdür. citeturn5search0 Ancak 2026 itibarıyla bunu tüm tezin model dönüşüm omurgası yapmanızı önermiyorum. NNI'yi **aday mimari oluşturma/search orchestration** için kullanın; model modifikasyonunu düz PyTorch içinde tutun.

Pratik yapı:

```python
def objective(config):
    model = build_yolox_variant(config)

    proxy_train(model, coco_proxy)

    map5095 = evaluate_map(model)

    onnx_path = export_onnx(model)

    latency_pi = benchmark_proxy(onnx_path, "pi")
    latency_jetson = benchmark_proxy(onnx_path, "jetson")

    return {
        "map": map5095,
        "latency_pi": latency_pi,
        "latency_jetson": latency_jetson,
    }
```

50–100 aday ile başlamak, yüksek lisans seviyesinde binlerce mimariyi eğitmeye çalışmaktan çok daha gerçekçidir. İlk turda 10–20 epoch proxy training kullanıp Pareto'daki örneğin en iyi 5 mimariyi tam COCO eğitime alın.

Buradaki önemli metodolojik karar şudur: **ana tez sonucunda her cihaz için farklı NAS mimarisi üretmeyin.** Önce Raspberry Pi ve Jetson'a ortak, tek bir robust mimari seçin. Aksi takdirde Jetson ↔ Pi karşılaştırmasında “donanım farkını mı, mimari farkını mı görüyoruz?” sorusuna cevap veremezsiniz.

Device-specific NAS daha sonra “ek deney” olabilir.

### Pruning: structured ana yöntem, unstructured sadece ablation

Burada net olun:

> **Edge cihaz üzerinde gerçek hız hedefliyorsanız ana yöntem structured pruning olmalıdır.**

Unstructured pruning ağırlıkları sıfırlar ama dense convolution kernel'leri hâlâ aynı tensor boyutlarını işleyebilir. Dolayısıyla yüksek sparsity değeri görmek otomatik olarak latency kazancı anlamına gelmez.

Structured pruning ise tüm channel/filter'ları fiziksel olarak kaldırarak tensor şekillerini ve MAC sayısını küçültür. Filter-level pruning literatürü de doğrudan bu tür yapısal küçültmenin genel DL runtime'larıyla kullanılabilmesini hedefler. citeturn13academia43

Bu aşamada en uygun araçlardan biri **Torch-Pruning / DepGraph**'tır; graph bağımlılıklarını takip ederek birbirine bağlı channel'ların tutarlı biçimde budanmasını sağlar. citeturn5search7

Deney tasarımı:

```text
P0 = %0 structured pruning
P1 = %10 MAC/channel azaltma
P2 = %20
P3 = %30
P4 = %40
```

Her budama sonrası:

```text
prune
↓
5–20 epoch low-LR fine-tune
↓
COCO mAP ölç
↓
ONNX export
↓
gerçek latency ölç
```

Özellikle detection head'in son çıktı boyutlarını rastgele budamayın. Backbone ve neck'ten başlayın. Skip connection, concat ve residual dependency'ler sebebiyle manuel channel silmek yerine dependency-aware araç kullanmak çok daha güvenlidir.

Tezde ayrıca küçük bir unstructured pruning ablation yapılabilir:

```text
unstructured %30 sparsity
vs.
structured ~aynı parameter/MAC reduction
```

ve gerçek latency karşılaştırılır. Böylece “sparsity ≠ speedup” olgusunu kendi donanımınızda ampirik olarak gösterebilirsiniz.

Neural Magic/SparseML'i ana araç olarak seçmeyin. Resmî SparseML repository'si Haziran 2025'te arşivlenip EOL durumuna alınmıştır; 2026'da yeni bir tez altyapısını bunun üzerine kurmak gereksiz teknik risk yaratır. citeturn5search1

### PTQ ve QAT

FP16 ile INT8'i aynı şey gibi yazmayın.

**FP16**, floating-point precision azaltımıdır.  
**INT8 PTQ/QAT**, gerçek kuantizasyon akışıdır.

Önerilen sıra:

```text
NAS + pruned FP32 model
        ↓
Static INT8 PTQ
        ↓
COCO evaluation
        ↓
ΔmAP kabul edilebilir mi?
     /             \
   evet            hayır
    ↓                ↓
 deploy          INT8 QAT
                     ↓
                 deploy
```

CNN'ler için ONNX Runtime dokümantasyonu static quantization'ı genel olarak uygun yöntem olarak tanımlar; calibration verisi ile activation scale'leri çıkarılır ve `QDQ` veya `QOperator` biçimi üretilebilir. QAT'tan gelen modeller için Q/DQ temsili özellikle uygundur. citeturn7search0

Örneğin:

```python
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

quantize_static(
    model_input="model_fp32.onnx",
    model_output="model_int8_qdq.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)
```

Calibration set sınıfları ve görüntü ölçeklerini temsil etmelidir. Calibration görüntülerini **val2017 değerlendirme setinden değil, train tarafında ayrılmış sabit bir subset'ten** seçmek daha temiz bir protokoldür.

PTQ sonrası örneğin:

```text
Baseline          26.0 mAP
NAS               25.9
NAS + prune       25.5
PTQ INT8          23.8   ← fazla kayıp
```

görürseniz QAT'a geçin.

PyTorch'un 2026'daki kuantizasyon yönü **torchao** tarafına kaymıştır; PyTorch dokümanları kuantizasyon geliştirmesinin torchao üzerinde merkezileştiğini belirtmektedir. citeturn6search1turn11search4 TorchAO'nun QAT akışı fake quantization'ı training/fine-tuning sırasında simüle ederek nihai düşük-bit modelin doğruluğunu iyileştirmeyi amaçlar. citeturn11search1

QAT önerim:

```text
başlangıç  : pruned FP32 checkpoint
epoch      : 10–30
LR         : normal fine-tuning LR'nin ~1/10'u
BN         : ilk aşamada train, son aşamada freeze denenebilir
metric     : full COCO val2017
export     : explicit Q/DQ ONNX
```

Ablation tablonuz tam olarak şu sırayı izlemeli:

| Varyant | NAS | Structured prune | PTQ | QAT |
|---|---:|---:|---:|---:|
| B0 | – | – | – | – |
| B1 | ✓ | – | – | – |
| B2 | ✓ | ✓ | – | – |
| B3 | ✓ | ✓ | ✓ | – |
| B4 | ✓ | ✓ | – | ✓ |

Böylece tezin özgün tarafını “üç yöntem kullandım” değil, **her yöntemin doğruluk-hız-enerji üzerindeki marjinal katkısını ölçtüm** şeklinde savunabilirsiniz.

Joint NAS + pruning + quantization literatürde de araştırılmıştır; örneğin APQ mimari, pruning ve quantization politikasını birlikte arar. citeturn13academia37 Ancak yüksek lisans tezi için doğrudan joint-search'e girmek yerine ardışık zincir daha doğru: hesaplama maliyeti daha düşük ve causal ablation çok daha nettir.

## PyTorch'tan edge cihazlara dağıtım boru hattı

### Tek bir “canonical artifact” belirleyin

Tezin deployment omurgası **ONNX olmalıdır**.

ONNX Runtime farklı execution provider'larla aynı ONNX modelini farklı platformlarda çalıştırmak üzere tasarlanmış donanımdan bağımsız bir runtime katmanı sağlar. citeturn7search2

Artifact dizinini baştan disiplinli kurun:

```text
artifacts/
├── pytorch/
│   ├── baseline.pth
│   ├── nas.pth
│   ├── nas_pruned.pth
│   └── nas_pruned_qat.pth
│
├── onnx/
│   ├── model_fp32.onnx
│   └── model_int8_qdq.onnx
│
├── raspberry/
│   ├── model.ncnn.param
│   ├── model.ncnn.bin
│   └── model-int8.*
│
└── jetson/
    ├── model_fp16.plan
    └── model_int8.plan
```

Her artifact yanında:

```text
SHA256
git commit
runtime version
input resolution
precision
calibration set hash
```

saklayın.

### PyTorch → ONNX

YOLOX resmî repository'si zaten ONNX deployment destekler. citeturn14search7 Komut mantığı şu biçimdedir:

```bash
python tools/export_onnx.py \
    -n yolox-nano \
    -c weights/nas_pruned.pth \
    --output-name artifacts/onnx/model_fp32.onnx
```

Exact flag isimleri kullandığınız YOLOX commit'ine göre kilitlenmelidir; tezin `requirements.txt`/container'ı ile aynı commit'i saklayın.

**Dynamic shape yerine ana benchmark için fixed shape öneriyorum:**

```text
batch = 1
C     = 3
H=W   = seçilen NAS resolution
```

Örneğin:

```text
1 × 3 × 416 × 416
```

Fixed shape TensorRT/ncnn tarafında optimizer'ın search alanını küçültür ve benchmarking'i daha deterministik hâle getirir.

ONNX'ten hemen sonra:

```python
import onnx

model = onnx.load("model_fp32.onnx")
onnx.checker.check_model(model)
```

ve ONNX Runtime ile golden-set regression yapın.

### Post-processing'i graph'tan ayırmak daha bilimsel

NMS'yi doğrudan TensorRT plugin'e gömüp Raspberry Pi'de Python NMS çalıştırırsanız, yalnızca neural network runtime'larını değil post-processing implementasyonlarını da kıyaslamış olursunuz.

Bu nedenle Stage-1 için:

```text
preprocess      = aynı kod
network output  = raw predictions
decode          = aynı kod
NMS             = aynı kod
```

kullanın.

Zamanlamayı da ayırın:

\[
T_{E2E}
=
T_{pre}
+
T_{inference}
+
T_{post}
\]

Tezde hem `T_inference` hem `T_E2E` raporlayın.

Stage-2'de hardware-specific fused NMS kullanabilirsiniz; fakat bu durumda ayrıca “native E2E” sonucu olarak raporlayın.

### Raspberry Pi → ncnn

Raspberry Pi CPU branch'i için **ncnn çok uygun**. ncnn BSD-3-Clause lisanslı, ARM NEON ve çok çekirdek için optimize bir edge inference framework'üdür; güncel `pnnx` ONNX veya PyTorch'tan ncnn formatına dönüşümü destekler. citeturn15search0turn9search4

Güncel dönüşüm:

```bash
pip install pnnx

pnnx model_fp32.onnx \
    inputshape=[1,3,416,416]
```

çıktı:

```text
model_fp32.ncnn.param
model_fp32.ncnn.bin
```

ncnn kendi PTQ INT8 akışını da sağlar:

```bash
ncnn2table \
    model.param model.bin \
    imagelist.txt \
    model.table \
    method=kl

ncnn2int8 \
    model.param model.bin \
    model-int8.param model-int8.bin \
    model.table
```

Resmî ncnn dokümanında `ncnn2table` + `ncnn2int8` post-training INT8 zinciri tanımlanmıştır. citeturn15search2

Burada iki INT8 sonucu raporlamak faydalı olacaktır:

```text
Common-QDQ INT8 ONNX
vs.
ncnn-native INT8
```

İkincisi Stage-2'ye aittir.

### Jetson Orin Nano → TensorRT

Yüklediğiniz Jetson benchmark çalışması da PyTorch → ONNX → TensorRT engine zincirini kullanıyor. Çalışmada ONNX parser, TensorRT builder ve engine inference adımları açık biçimde verilmiş. fileciteturn0file3

FP16 örneği:

```bash
trtexec \
  --onnx=model_fp32.onnx \
  --saveEngine=model_fp16.plan \
  --fp16 \
  --shapes=images:1x3x416x416 \
  --warmUp=3000 \
  --duration=60
```

`trtexec`, NVIDIA'nın resmi model build ve benchmark aracıdır; throughput ve latency bilgilerini doğrudan raporlayabilir. citeturn8search1turn8search4

INT8 için **eski “implicit INT8 calibration” yaklaşımını tezin ana yöntemi yapmayın**. Güncel TensorRT açık Q/DQ tabanlı explicit quantization akışını öne çıkarmaktadır. Quantized ONNX'te Q/DQ düğümleri precision bilgisini graph'a taşır. citeturn8search5turn8search12

Bu nedenle:

```bash
trtexec \
  --onnx=model_int8_qdq.onnx \
  --saveEngine=model_int8.plan \
  --shapes=images:1x3x416x416 \
  --warmUp=3000 \
  --duration=60
```

şeklindeki mantık daha temizdir.

TensorRT engine dosyalarını başka cihazda oluşturup kopyalamak yerine **hedef Jetson üzerinde build edin**. TensorRT dokümanları engine'lerin varsayılan olarak platform/GPU/TensorRT sürümüne bağlı olduğunu ve taşınabilirlik kısıtları bulunduğunu belirtir. citeturn16search4turn16search7

## İki aşamalı benchmark tasarımı

Burada kullanıcı talebinizdeki bir noktayı doğrudan düzeltmek gerekiyor.

### AŞAMA 1 artık “CPU–GPU ham güç karşılaştırması” olarak adlandırılmamalı

Bu revizyonda yalnızca Raspberry Pi ve Jetson Orin Nano bulunduğu için metodoloji daha sade hâle geliyor. Yine de Stage-1 tanımı doğru yapılmalıdır.

Eğer iki cihazda da aynı ONNX modelini yalnızca `ONNX Runtime CPUExecutionProvider` ile çalıştırırsanız:

```text
Raspberry Pi      → ARM CPU
Jetson Orin Nano  → ARM CPU
```

karşılaştırırsınız. Jetson GPU bu aşamada kullanılmaz.

Dolayısıyla Stage-1 şu soruya cevap verir:

> “Aynı optimize edilmiş model, aynı taşınabilir CPU runtime'ı altında iki edge platformunda nasıl davranıyor?”

Bu aşama **CPU portability/common-runtime baseline** olarak ele alınmalıdır; CPU–GPU ham hesaplama karşılaştırması değildir. Jetson GPU'nun etkisi Stage-2'de TensorRT ile ayrıca ölçülür.

### AŞAMA 1'i şu şekilde tanımlayın

**AŞAMA 1 — Donanımdan bağımsız ortak runtime / portability benchmark**

Canonical artifact:

```text
aynı .onnx
aynı weights
aynı input resolution
aynı preprocessing
aynı postprocessing
batch = 1
ONNX Runtime CPUExecutionProvider
```

iki cihazda da çalıştırılacak.

Kod:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model_fp32.onnx",
    providers=["CPUExecutionProvider"],
)
```

Burada TensorRT yok, CUDA provider yok, ncnn yok.

Bu deney şunu ölçer:

> “Model aynı taşınabilir runtime ile iki edge platformunun genel amaçlı CPU ortamında ne kadar hızlı/verimli çalışıyor?”

Bu, gerçekten izole edilmiş bir benchmark'tır. ONNX Runtime'ın execution-provider mimarisi donanım backend'lerini ayrıştırmak üzere tasarlanmıştır. citeturn7search2

Stage-1 içinde iki precision çalıştırın:

```text
A1-FP32 : model_fp32.onnx
A1-INT8 : model_int8_qdq.onnx
```

yalnızca iki cihazda da aynı ORT build'in bu INT8 graph'ı sorunsuz çalıştırdığı doğrulanırsa.

**FP16'yı zorunlu ortak Stage-1 precision yapmayın.** FP16'nın ARM CPU ile Jetson GPU'daki native yürütme özellikleri aynı değildir. ONNX Runtime da FP16 dönüşümünün model boyutunu düşürebildiğini ve özellikle GPU'larda performans avantajı sağlayabildiğini belirtir. citeturn7search5 FP16 bu nedenle esas olarak Jetson Stage-2'de anlamlıdır.

Çok istiyorsanız Stage-1'in altında ayrıca bir **diagnostic accelerator baseline** ekleyebilirsiniz:

```text
Pi       → ORT CPU
Jetson   → ORT CUDA EP
```

fakat bunu “hardware-independent” diye adlandırmayın. Bu artık “aynı ONNX graph + cihazın en temel accelerator backend'i” deneyidir.

### AŞAMA 2 donanıma özel maksimum optimizasyon

Burada aynı **mantıksal architecture ve training checkpoint** korunacak, fakat compiler/runtime değişebilecek.

| Cihaz | Stage-1 | Stage-2 |
|---|---|---|
| Raspberry Pi | ONNX Runtime CPU | **ncnn ARM/NEON + INT8** |
| Jetson Orin Nano | ONNX Runtime CPU | **TensorRT FP16 + INT8 Q/DQ** |

Bu deney şu soruya cevap verir:

\[
Gain_{HW}
=
\frac{Performance_{Stage2}}
{Performance_{Stage1}}
\]

Örneğin:

\[
Speedup =
\frac{Latency_{Stage1}}
{Latency_{Stage2}}
\]

Böylece bir cihazın hızlı olması ile **vendor optimizer sayesinde hızlı hâle gelmesi** birbirinden ayrılır.

Bu iki-aşamalı yapı, yüklediğiniz önceki çalışmaların en büyük deneysel zayıflığını doğrudan giderir. Uppsala çalışması farklı precision ve implementation yollarının karşılaştırmayı zorlaştırdığını açıkça belirtmiştir. fileciteturn0file4

### Raspberry Pi konfigürasyonu

Raspberry Pi modeli tezde mutlaka tam olarak belirtilmeli.

Eğer donanım henüz alınmadıysa CPU sınıfı benchmark için **Raspberry Pi 5** mantıklı seçimdir. BCM2712, dört Cortex-A76 çekirdeğini 2.4 GHz'e kadar çalıştırır. citeturn9search6

Deney manifesti:

```text
Board        : Raspberry Pi 5
RAM          : 8 GB   (örnek; gerçekte hangisiyse)
OS           : Raspberry Pi OS 64-bit / Ubuntu
Kernel       : ...
CPU governor : performance
Cooling      : Active Cooler
Threads      : 1 ve 4 ayrı deney
Runtime      : ORT ... / ncnn ...
```

CPU thermal throttling mutlaka engellenmelidir. Aktif soğutma ve sıcaklık log'u deneyin parçası olmalıdır.

### Jetson Orin Nano konfigürasyonu

“Jetson Orin Nano” tek başına yeterli tanım değildir. 4 GB / 8 GB / Super ve farklı güç modları farklı sonuç üretir. NVIDIA'nın güncel power-mode tablolarında normal Orin Nano 8 GB ile Super varyantının güç ve clock seçenekleri farklıdır. citeturn12search3

Tez tablosuna mutlaka:

```text
Jetson SKU
RAM
JetPack
Jetson Linux
CUDA
TensorRT
power mode
GPU clock
memory clock
cooling
```

yazın.

Başlangıçta:

```bash
sudo nvpmodel -q
```

ile modu kaydedin.

Maksimum benchmark modunu seçtikten sonra bütün testlerde değiştirmeyin.

TensorRT yalnızca Stage-2'de kullanılmalıdır.

## Ölçüm protokolü, enerji hesabı ve beklenen darboğazlar

### Her cihazda aynı timing protokolünü kullanın

Ana gerçek-zamanlı benchmark:

```text
batch              = 1
stream              = 1
input resolution    = NAS ile seçilen sabit değer
dataset             = COCO val2017
warm-up             = en az 100 inference
measured images     = 5.000 COCO val image
repetitions         = 3–5 tam run
```

İlk model load / graph compile süresini steady-state inference'a karıştırmayın.

Ayrı ölçün:

```text
model load time
first inference
steady-state inference
```

Realtime application açısından ana sonuç steady-state olacaktır.

Latency için yalnızca mean vermeyin:

```text
mean
median / P50
P90
P95
P99
standard deviation
95% confidence interval
```

raporlayın.

Özellikle thermal throttling ve Linux scheduler kaynaklı latency spikes, ortalamada kaybolabilir.

### FPS tanımı

Single-stream için:

\[
FPS =
\frac{N}
{T_{inference,total}}
\]

Sadece:

\[
FPS = \frac{1}{\text{single-image mean latency}}
\]

hesabına güvenmek yerine bütün inference loop'un toplam süresinden throughput hesaplamak daha temizdir.

Hem:

```text
Model-only FPS
E2E FPS
```

raporlayın.

### Güç ölçümü

**Birinci kaynak olarak harici güç ölçer kullanın.**

Sebebi basit: Raspberry Pi ve Jetson'ın internal power telemetry yöntemleri aynı fiziksel noktayı ölçmeyebilir. Dolayısıyla yalnızca cihazların kendi internal telemetry sonuçlarını doğrudan birbirine eşdeğer kabul etmek doğru olmayabilir.

Her iki cihazın besleme girişine uygun:

```text
USB-C PD power meter
veya
inline DC power analyzer
veya
laboratuvar güç kaynağı + akım logger
```

kullanın.

Yüklediğiniz FPGA/GPU çalışmalarında da external power meter/multimeter ve Jetson `tegrastats` kullanılmıştır. fileciteturn0file2 fileciteturn0file4

En az üç güç değeri saklayın:

\[
P_{idle}
\]

\[
P_{load}
\]

\[
P_{dynamic} = P_{load} - P_{idle}
\]

Ana enerji verimliliği:

\[
FPS/W =
\frac{FPS}{P_{load}}
\]

ve daha fiziksel olarak anlaşılır:

\[
J/frame =
\frac{P_{load}}{FPS}
\]

olmalıdır.

Aslında:

\[
FPS/W = frames/J
\]

olduğu için `J/frame` sonucu daima ekleyin. Enerji tezlerinde çok daha yorumlanabilir bir metriktir.

### Jetson'da tegrastats

Jetson'da:

```bash
tegrastats \
    --interval 100 \
    --logfile tegrastats.log
```

kullanabilirsiniz.

Güncel NVIDIA dokümantasyonuna göre `tegrastats`, RAM/işlemci istatistiklerinin yanında power rail'ler için anlık ve ortalama mW değerlerini de raporlar. citeturn12search0

Dolayısıyla Jetson'da:

```text
external input Watt   ← ana tez sonucu
tegrastats rails      ← açıklayıcı ikinci ölçüm
GPU utilization
RAM
temperature
clock
```

birlikte toplanmalıdır.

### Raspberry Pi

Raspberry Pi'de:

```bash
htop
pidstat -u -r -p <PID> 1
/usr/bin/time -v ./benchmark
```

ile CPU ve peak RSS alınabilir.

Sıcaklık ve throttling ayrıca izlenmelidir.

Ana güç ölçümü dış güç analizöründen gelmelidir.

### Ölçüm tablosu

Her deney otomatik olarak şu CSV satırını üretmeli:

```text
run_id
device
device_mode
model_variant
backend
precision
input_size
batch
threads
map5095
ap50
latency_mean_ms
latency_p50_ms
latency_p95_ms
latency_p99_ms
fps
power_idle_w
power_load_w
power_dynamic_w
fps_per_watt
joule_per_frame
peak_ram_mb
temperature_start
temperature_end
```

Bu CSV'yi **tek source of truth** yapın. Tezdeki tablolar buradan otomatik üretilsin.

### En önemli teknik darboğazlar

| Darboğaz | Belirti | Çözüm |
|---|---|---|
| Unsupported ONNX op | export/compile fail | Search space'i ortak op setiyle sınırla |
| Dynamic shape | TensorRT/ncnn tarafında değişken performans veya build sorunları | Benchmark için fixed shape |
| NMS uyumsuzluğu | cihazlar arası farklı çıktı | Stage-1'de ortak post-processing |
| INT8 accuracy loss | mAP hızlı düşer | QAT + calibration analizi |
| Unstructured pruning hız getirmiyor | sparsity yüksek, FPS aynı | structured pruning |
| TensorRT engine incompatibility | plan load fail | engine'i hedef Jetson'da build et |
| Thermal throttling | FPS zamanla düşüyor | aktif cooling + sıcaklık log |
| Python overhead | küçük modelde latency şişiyor | C++ benchmark harness |
| Pre/post processing bottleneck | network hızlı, E2E yavaş | preprocess/model/post ayrı ölç |
| Calibration bias | belirli sınıflarda AP düşüyor | representative calibration set |
| NAS proxy bias | proxy winner full COCO'da kötü | top-k adayları full retrain et |

Özellikle **Python overhead** hafif modellerde ciddi metodolojik problem yaratabilir. YOLOX-Nano gibi küçük bir modelde Python preprocessing + NumPy + NMS süresi model inference süresine yaklaşabilir. Bu nedenle nihai performans benchmark'ı için mümkün olduğunca:

```text
C++ ONNX Runtime
C++ ncnn
C++ TensorRT
```

kullanmak daha güvenilir; Python'ı geliştirme ve doğruluk testinde tutmak daha iyidir.

## Akademik karşılaştırma matrisi, tez çıktısı ve çalışma planı

### Önce model optimizasyonunu donanımdan bağımsız gösterin

İlk ana tablo:

**Model Evrimi / Ablation**

| Model | Params M | MAC/GFLOPs | Model MB | Pruning % | Precision | mAP50:95 | AP50 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| YOLOX-Nano baseline | | | | 0 | FP32 | | | | | |
| + NAS | | | | 0 | FP32 | | | | | |
| + NAS + prune | | | | 20–30 | FP32 | | | | | |
| + PTQ | | | | | INT8 | | | | | |
| + QAT | | | | | INT8 | | | | | |

Bu tablo şunu kanıtlar:

> Modelin kendi mimari/veri başarımı ne kadar değişti?

### Sonra Stage-1 ortak çalışma ortamını gösterin

| Cihaz | Ortak model | Runtime | Precision | P50 ms | P95 ms | FPS | Load W | FPS/W | J/frame | Peak RAM |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Raspberry Pi | Final common | ORT CPU | FP32 | | | | | | | |
| Jetson Orin Nano | Final common | ORT CPU | FP32 | | | | | | | |
| Raspberry Pi | Final common | ORT CPU | INT8 | | | | | | | |
| Jetson Orin Nano | Final common | ORT CPU | INT8 | | | | | | | |

Bu tablo **portability ve host CPU farkını** gösterir. Jetson GPU kullanılmadığı için “CPU vs GPU” sonucu olarak yorumlanmaz.

### Sonra Stage-2 hardware-specific sonucu gösterin

| Cihaz | Backend | Precision | mAP | P50 ms | P95 ms | FPS | W | FPS/W | J/frame | RAM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pi 5 | ncnn | FP32 | | | | | | | | |
| Pi 5 | ncnn | INT8 | | | | | | | | |
| Orin Nano | TensorRT | FP16 | | | | | | | | |
| Orin Nano | TensorRT | INT8 | | | | | | | | |

TensorRT'nin PyTorch/ONNX modeline göre çok ciddi hız kazancı sağlayabilmesi yüklediğiniz Jetson çalışmasında da görülüyor; örneğin bazı düşük-FLOP mobil mimarilerde optimizer sonrası çok yüksek inference speedup raporlanmıştır. fileciteturn0file3 Bununla birlikte geçmiş çalışmaların sonuçlarını doğrudan sizin Orin Nano sonuçlarınız gibi kullanmamalısınız; Jetson Nano, Orin Nano ve TensorRT sürümleri farklı sistemlerdir.

### En önemli tablo: optimizer kazancı

| Cihaz | Stage-1 FPS | Stage-2 FPS | Speedup | Stage-1 FPS/W | Stage-2 FPS/W | Efficiency gain | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raspberry Pi | | | | | | | |
| Jetson Orin Nano | | | | | | | |

Burada:

\[
Speedup =
\frac{FPS_{Stage2}}
{FPS_{Stage1}}
\]

\[
EfficiencyGain =
\frac{FPS/W_{Stage2}}
{FPS/W_{Stage1}}
\]

\[
\Delta mAP =
mAP_{Stage2} - mAP_{canonical}
\]

hesaplanır.

Bu tablo tezin en kuvvetli tablosu olacaktır.

### Pareto grafikleri

Tezde en az şu iki scatter plot'u kullanın:

```text
X = FPS/W
Y = mAP
```

ve

```text
X = latency
Y = mAP
bubble size = power
```

Böylece “en hızlı cihaz” gibi tek-boyutlu bir sonuçtan kaçınırsınız.

Sonuçları önceden “Jetson daha hızlıdır” veya “Pi daha verimlidir” şeklinde varsaymayın. İki platform için aynı metrikleri ölçüp Pareto eğrisini sonuçların kendisinin belirlemesine izin verin. Örneğin bir platform daha yüksek mutlak FPS verirken diğeri belirli bir çalışma noktasında daha düşük toplam güç tüketimine sahip olabilir; hangi sonucun oluşacağı deneysel olarak gösterilmelidir.

Yüklediğiniz FPGA/GPU çalışmaları da performans, güç verimliliği ve geliştirme kolaylığının farklı platformlarda farklı şekilde üstün olabileceğini gösteriyor. fileciteturn0file2

### Geliştirme kolaylığı da nitel metrik olarak eklenebilir

Sayısallaştırması zor ama faydalı ek tablo:

| Kriter | Pi | Jetson |
|---|---|---|
| Ortam kurulum zorluğu | | |
| Model dönüşüm adımı | | |
| Unsupported op problemi | | |
| Compile süresi | | |
| Debug kolaylığı | | |
| Dokümantasyon olgunluğu | | |
| C++ entegrasyonu | | |

Bu, yüklediğiniz FPGA/GPU karşılaştırma literatürüyle de uyumludur; çalışmalar yalnızca FPS değil, geliştirme kolaylığını da donanım seçiminin parçası olarak inceliyor. fileciteturn0file2

### Tezin özgün katkısını böyle yazın

Savunmada kullanabileceğiniz güçlü bir akademik formulasyon:

> **Bu çalışmanın özgün katkısı, permissive açık kaynaklı gerçek zamanlı bir nesne tespit modelinin uç cihaz kısıtlarına göre çok amaçlı donanım-duyarlı mimari arama, yapısal budama ve INT8 kuantizasyon yoluyla yeniden optimize edilmesi; elde edilen tek ortak mimarinin Raspberry Pi ve NVIDIA Jetson Orin Nano üzerinde önce ortak bir CPU çalışma zamanı altında, ardından her platformun doğal optimize dağıtım yolu kullanılarak sistematik biçimde değerlendirilmesidir. Raspberry Pi tarafında ARM CPU/ncnn, Jetson tarafında GPU/TensorRT kullanılarak doğruluk, gecikme, enerji tüketimi, bellek kullanımı ve enerji başına throughput arasındaki Pareto ilişkisi ile platforma özel optimizasyonun ortak modele kattığı marjinal kazanç ölçülmektedir.**

Bu, mevcut öneri formundaki “NAS, budama ve kuantizasyonu bütüncül biçimde ele alma” iddiasıyla uyumludur fakat “sıfırdan yeni detector icat edildi” şeklinde savunulması zor bir iddiada bulunmaz. fileciteturn0file1

### Sürekli öğrenme pipeline'da nereye oturur?

Sürekli öğrenmeyi ana deney zincirine sokmayın. `yontem.txt` de zaten teorik seviyede tutulmasını öneriyor. fileciteturn0file0

Doğru vizyon:

```text
Edge inference
     ↓
Low-confidence / hard samples
     ↓
Local buffer
     ↓
Privacy-filtered sample transfer
     ↓
Central retraining
     ↓
Replay / distillation / continual-learning strategy
     ↓
NAS/pruning configuration korunarak fine-tune
     ↓
QAT / re-quantization
     ↓
Regression test
     ↓
new ONNX version
     ↓
device compiler
     ↓
OTA deployment
```

Yani cihazın sürekli kendi üzerinde full YOLO training yapmasını önermeyin. Teorik bölümde üç problem yeterlidir:

```text
concept drift
catastrophic forgetting
safe/rollback-capable model update
```

ve çözüm vizyonu olarak replay buffer, knowledge distillation ve periyodik merkezi yeniden eğitim tartışılabilir.

### Uygulanabilir çalışma takvimi

**İlk blok — Reprodüksiyon ve altyapı**

```text
COCO hazırlanması
YOLOX / RTMDet / SSDLite baseline
lisans ve artifact manifest'i
YOLOX-Nano seçimi
PyTorch ↔ ONNX regression
```

Çıktı: `baseline.csv`, canonical training environment.

**İkinci blok — Hardware-aware NAS**

```text
operator compatibility matrix
search space
COCO proxy subset
50–100 candidate
Pareto ranking
top-5 full evaluation
winner full training
```

Çıktı: `nas_results.csv`, Pareto plot, seçilen architecture config.

**Üçüncü blok — Pruning**

```text
sensitivity analysis
%10 / %20 / %30 / %40 structured
fine-tuning
latency validation
final pruning ratio
```

Çıktı: pruning ablation.

**Dördüncü blok — Quantization**

```text
PTQ calibration
INT8 validation
QAT gerekiyorsa fine-tuning
QDQ ONNX
```

Çıktı: FP32/PTQ/QAT mAP tablosu.

**Beşinci blok — Stage-1**

```text
aynı ONNX
ORT CPU
Pi / Jetson
FP32
INT8 secondary
```

Çıktı: common-runtime benchmark.

**Altıncı blok — Stage-2**

```text
Pi      → ncnn
Jetson  → TensorRT
```

Çıktı: maksimum optimize benchmark.

**Yedinci blok — İstatistik ve tez**

```text
95% CI
Pareto plots
ablation
Stage2/Stage1 speedup
energy analysis
continuous learning discussion
reproducibility appendix
```

### Nihai teknoloji yığını

Tezi gereksiz araç kalabalığına sokmadan kullanacağım stack şu olurdu:

| Katman | Tercih |
|---|---|
| Ana model | **YOLOX-Nano** |
| Araştırma alternatifi | RTMDet-Tiny / MMDetection |
| Veri seti | **COCO 2017** |
| Training | **PyTorch** |
| NAS orchestration | **NNI + kendi PyTorch search code'unuz** |
| Structured pruning | **Torch-Pruning / DepGraph** |
| PTQ referans | **ONNX Runtime quantization** |
| QAT | **torchao** |
| Ortak interchange | **ONNX** |
| Stage-1 runtime | **ONNX Runtime CPU EP** |
| Raspberry Pi Stage-2 | **ncnn INT8 / ARM NEON** |
| Jetson Stage-2 | **TensorRT FP16 + explicit-QDQ INT8** |
| Jetson telemetry | **tegrastats** |
| Sistem telemetry | `pidstat`, `htop`, `/usr/bin/time -v` |
| Ana enerji ölçümü | **harici inline power meter** |
| Sonuç saklama | CSV/Parquet + Git commit + SHA256 |

**Neural Magic/SparseML'i ana stack'e koymazdım**, çünkü proje 2025'te arşivlendi. citeturn5search1  
**YOLOv8'i ana model yapmazdım**, çünkü AGPL sizin permissive-lisans şartınızla gereksiz çatışma yaratıyor. citeturn4search0  
**TFLite'ı canonical interchange yapmazdım**, çünkü PyTorch başlangıçlı Raspberry Pi + Jetson deneyinde ONNX çok daha doğal ortak noktadır; ek bir TensorFlow/TFLite dönüşüm zinciri karşılaştırmaya gereksiz dönüştürme değişkenleri ekler.  
**Unstructured pruning'i ana hızlandırma yöntemi yapmazdım**; structured channel/filter pruning fiziksel graph küçülmesi verdiği için ONNX Runtime, ncnn ve TensorRT yollarına daha taşınabilir. citeturn5search7turn13academia43  
**Stage-1'i “CPU-GPU raw comparison” diye adlandırmazdım**; Stage-1'de iki cihaz da ONNX Runtime `CPUExecutionProvider` ile çalıştığı için bu aşama Jetson GPU performansını ölçmez. Stage-1 “common-runtime CPU portability baseline”, Stage-2 ise “platform-specialized deployment benchmark” olarak adlandırılmalıdır. Jetson GPU avantajı yalnızca Stage-2'de TensorRT ile ölçülmelidir.

Bu şekilde çalışma, öneri formunda vaat edilen NAS + budama + kuantizasyon bütünleşmesini gerçekten uygular; aynı zamanda model farklılıkları ile donanım farklılıklarını birbirine karıştırmadan Raspberry Pi'nin CPU tabanlı ve Jetson Orin Nano'nun GPU hızlandırmalı dağıtım yollarının **mAP–latency–energy** Pareto dengesini ölçülebilir ve savunulabilir bir deney tasarımına dönüştürür. fileciteturn0file1