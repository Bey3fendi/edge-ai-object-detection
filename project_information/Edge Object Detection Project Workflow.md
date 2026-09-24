# Uç Yapay Zekâ Ortamları İçin Nesne Tespit Modeli: Somut İş Akışı ve Dağıtım Rehberi

Bu rapor, nesne tespiti (object detection) modellerinin bulut tabanlı devasa ekran kartlarından koparılıp, Raspberry Pi 5 ve NVIDIA Jetson Orin Nano gibi kısıtlı kaynaklara sahip uç (edge) donanımlara aktarılması sürecini standartlaştıran kesin ve somut bir mühendislik iş akışıdır. Karmaşık terminolojiler ve soyut akademik kavramlar yerine, doğrudan klavyede yazılacak komutları, çalıştırılacak Python betiklerini ve bu işlemlerin arka planındaki fiziksel gerçeklikleri bir araya getirmektedir. Her bir adım; "Ne yapılıyor?", "Neden yapılıyor?" ve "Yapıldıktan sonra sistemde ne değişiyor?" sorularına net yanıtlar verecek şekilde yapılandırılmıştır.

## Birinci Aşama: Donanım Altyapısının Sağlanması ve Fiziksel Hazırlık

Derin öğrenme modelleri, üzerinde çalıştıkları donanımın termal ve elektriksel sınırlarına doğrudan bağlıdır. Yazılımsal optimizasyonlara başlamadan önce fiziksel çalışma ortamının kararlı hale getirilmesi zorunludur.

### Raspberry Pi 5 Konfigürasyonu

Raspberry Pi 5, 2.4 GHz hızında çalışan 64-bit dört çekirdekli Arm Cortex-A76 işlemciye (BCM2712 SoC), çekirdek başına 512 KB L2 önbelleğe ve 2 MB paylaşımlı L3 önbelleğe sahiptir. Önceki nesle kıyasla CPU performansında 2 ila 3 kat artış sunar. Türkiye pazarında Samm Market ve Robotistan gibi resmi distribütörler üzerinden 8 GB RAM kapasiteli versiyonu ve aktif soğutucu (Active Cooler) modülü ile tedarik edilebilir.

**Ne Yapılıyor:** Cihaza 64-bit Raspberry Pi OS (veya Ubuntu 22.04) kurulur. Aktif soğutucu pinlere bağlanır ve işletim sisteminde CPU güç yöneticisi (governor) performansa sabitlenir. Uçbirim (terminal) üzerinden aşağıdaki komut çalıştırılır:

```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Neden Yapılıyor:** Uç cihazlarda nesne tespiti sırasında işlemci %100 yüke biner. Aktif soğutma olmazsa, işlemci aşırı ısınmayı önlemek için frekansını düşürür (thermal throttling). Bu durum, modelin gecikme (latency) testlerinde yanlış ve dalgalı sonuçlar vermesine neden olur.

**Ne Değişti:** Cihaz, enerji tasarrufu modundan çıkarak sürekli maksimum saat hızında çalışmaya başladı. Test sonuçları artık donanımın gerçek üst sınırını yansıtacaktır.

### NVIDIA Jetson Orin Nano Konfigürasyonu

NVIDIA Jetson Orin Nano, Ampere mimarisine sahip bir GPU ve 6 çekirdekli ARM CPU içerir. Tensor çekirdekleri barındırması sayesinde, düşük hassasiyetli (FP16 ve INT8) matris çarpım işlemlerini donanımsal düzeyde hızlandırır. Bu cihaz da Samm Market ve Robotistan üzerinden Geliştirici Kiti (Developer Kit) formatında temin edilebilir.

**Ne Yapılıyor:** Cihazın güç tüketim profili maksimum değere (15W veya MAXN modu) kilitlenir ve tüm fanlar ile saat hızları tepe noktaya çıkarılır.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

**Neden Yapılıyor:** Jetson cihazları varsayılan olarak dinamik güç yönetimi ile gelir. GPU frekansının işlem yüküne göre inip çıkması, modelin her karesi (frame) için farklı işlem süreleri ölçülmesine yol açar. Bilimsel ve tutarlı bir kıyaslama yapabilmek için donanım kilitlenmelidir.

**Ne Değişti:** GPU, CPU ve bellek (RAM) saat hızları donanımın izin verdiği maksimum değerlere sabitlendi. Gecikme ölçümlerindeki sapmalar (varyans) minimuma indirildi.

## İkinci Aşama: Çalışma Ortamının ve Dizin Yapısının Kurulması

Her şeyin birbirine karıştığı bir klasör yapısı, ilerleyen aşamalarda hatalı model ağırlıklarının kullanılmasına yol açar. Bu nedenle kesin ve standart bir dizin hiyerarşisi oluşturulmalı ve bu yapı bir Git deposu (repository) olarak versiyon kontrolüne alınmalıdır.

**Ne Yapılıyor:** Kullanıcı ana dizininde (home directory) uçtan uca tüm süreci barındıracak klasörler oluşturulur ve proje resmi bir Git reposu olarak başlatılır.

```bash
mkdir -p ~/edge_ai_workspace/artifacts/{pytorch,onnx,ncnn,tensorrt}
mkdir -p ~/edge_ai_workspace/datasets
mkdir -p ~/edge_ai_workspace/logs
mkdir -p ~/edge_ai_workspace/scripts
cd ~/edge_ai_workspace

# Ana projeyi bir Git reposu olarak başlatın
git init
echo "# Edge AI Object Detection Pipeline" > README.md

# Ağırlıkları ve logları repoya yüklememek için .gitignore oluşturun
echo "datasets/" > .gitignore
echo "artifacts/" >> .gitignore
echo "logs/" >> .gitignore

# İlk (initial) commit işlemini gerçekleştirin
git add .
git commit -m "feat: init workspace structure and git repo"
```

**Neden Yapılıyor:** Süreç boyunca PyTorch ağırlıkları (.pth), ONNX grafikleri (.onnx), ncnn dosyaları (.param, .bin) ve TensorRT motorları (.plan) üretilecektir. Her bir formatın kendi dizininde izolasyonu ve projenin bir versiyon kontrol sistemine (Git) bağlanması, geçmişe dönük hataların ayıklanmasını (debugging) kolaylaştırır.

**Ne Değişti:** Sistematik ve versiyonlanabilir bir çalışma alanı yaratıldı. İlerleyen tüm komutlar bu ana dizin (~/edge_ai_workspace) temel alınarak yürütülecektir.

## Üçüncü Aşama: Model Seçimi ve YOLOX Depolama Alanının Klonlanması

Nesne tespiti alanında YOLO ailesi sürekli bir evrim içerisindedir. YOLOv1'den başlayarak YOLOv6, v7, v8, v9, v10, v11, v12 ve v26'ya kadar uzanan gelişim sürecinde; her yeni sürüm farklı omurga (backbone) yapıları, dikkat mekanizmaları (attention mechanisms) ve kayıp fonksiyonları (loss functions) ile hem hızı hem de doğruluğu artırmayı hedeflemiştir. Örneğin, YOLOv9 Programlanabilir Gradyan Bilgisi (PGI) ve GELAN mimarisini sunarken, YOLOv10 NMS (Non-Maximum Suppression) adımını ortadan kaldıran birebir atama (one-to-one assignment) stratejisini getirmiştir. YOLOv12 ise gecikme odaklı yapılandırılmış, YOLO26 ise CPU üzerinde daha yüksek hızlara ulaşabilmiştir.

Buna rağmen, bu mühendislik iş akışı için **YOLOX-Nano** modeli seçilmiştir. YOLOX, çapa kutularından bağımsız (anchor-free) bir mimari olup, ayrıştırılmış başlık (decoupled head) ve gelişmiş etiket atama (SimOTA) stratejileri kullanarak YOLO serisinde bir kırılma noktası yaratmıştır. Ayrıca, yaklaşık 0.91 milyon parametre ve 1.08 GFLOPs işlem yükü ile inanılmaz derecede hafiftir. Apache-2.0 lisansı altında sunulması ve ONNX, TensorRT, ncnn gibi dağıtım yollarını resmi olarak desteklemesi, modeli yasal ve teknik kısıtlamalar olmadan bükmemize (NAS, Pruning) olanak tanır.

**Ne Yapılıyor:** Resmi YOLOX kod deposu sisteme düz bir klonlama ile değil, oluşturduğumuz ana repoya bir alt modül (submodule) olarak eklenir. Temel bağımlılıklar ve ileride işimize yarayacak ekstralar yüklenerek sistem geliştirici modunda kurulur.

```bash
# YOLOX'u ana depomuza alt modül olarak ekliyoruz
git submodule add https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX

# Temel gereksinimleri kuruyoruz
pip3 install -U pip
pip3 install -r requirements.txt

# Ekstralar: ONNX, ONNX Runtime ve mimari görselleştirme (Netron) araçlarının kurulumu
pip3 install onnx onnx-simplifier onnxruntime netron

# YOLOX'u geliştirici (editable) modunda kuruyoruz
pip3 install -v -e .

# YOLOX eklemesini ana repoya commit ediyoruz
cd ~/edge_ai_workspace
git add .gitmodules YOLOX/
git commit -m "feat: add YOLOX submodule and install extra dependencies"
```

**Neden Yapılıyor:** YOLOX'u salt bir kütüphane olarak indirmek yerine kaynak kodlarını submodule olarak ekleyip -e (editable) bayrağı ile kurmak, modelin iç katmanlarına müdahale etmemizi sağlar. Python dosyalarında (örneğin export kodlarında) yapacağımız herhangi bir düzenleme, sistemi yeniden kurmaya gerek kalmadan anında geçerli olacaktır. Alt modül mantığı sayesinde ise YOLOX'un orijinal versiyonunu güncelleyebilir veya kendi yaptığımız değişiklikleri ana projemizin geçmişinde temiz bir şekilde tutabiliriz.

**Ne Değişti:** Çalışma alanımızda YOLOX mimarisini eğitebileceğimiz, test edebileceğimiz ve dışa aktarabileceğimiz tam donanımlı ve versiyon kontrollü bir PyTorch ortamı ayağa kalkmış oldu.

### Referans Ağırlıkların İndirilmesi ve Temel Test (Smoke Test)

**Ne Yapılıyor:** Megvii'nin resmi GitHub sürümlerinden YOLOX-Nano'nun önceden eğitilmiş ağırlıkları indirilir ve örnek bir görsel üzerinde çıkarım (inference) yapılır.

```bash
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -O ../artifacts/pytorch/yolox_nano.pth

python3 YOLOX/tools/demo.py image -n yolox-nano -c artifacts/pytorch/yolox_nano.pth --path YOLOX/assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 416 --save_result --device cpu
```

**Neden Yapılıyor:** Geliştirme sürecinin daha ilk adımında kod tabanının, bağımlılıkların (OpenCV, PyTorch) ve model ağırlıklarının uyum içinde çalıştığından emin olmak (smoke test) gerekir.

**Ne Değişti:** YOLOX_outputs/ klasöründe üzerinde tespit kutuları olan bir dog.jpg görseli oluştu. Sistemin fabrikadan çıktığı haliyle eksiksiz çalıştığı kanıtlandı.

## Dördüncü Aşama: Donanıma Duyarlı Mimari Arama (Hardware-Aware NAS)

İndirdiğimiz YOLOX-Nano modeli her ne kadar küçük olsa da, bulut GPU'ları düşünülerek standart kanal genişlikleri ile tasarlanmıştır. Uç donanımlardaki L2/L3 önbellek (cache) sınırlarına uyması için model mimarisinin yeniden yapılandırılması gerekir. Geleneksel modellerin Floating Point Operations (FLOPs) değerlerini küçültmeye odaklanan yaklaşımlar, gerçek dünyada çoğu zaman gecikmeyi (latency) düzeltmez; çünkü işlemcinin veriyi bellekten getirme süresi (memory access cost) genellikle hesaplamadan daha uzun sürer. Bu nedenle Donanıma Duyarlı Mimari Arama (Hardware-Aware NAS) süreci başlatılır.

**Ne Yapılıyor:** Bu işlem bulut ve uç cihazın birlikte kullanıldığı **hibrit (karma) bir yapıda** yürütülür:

- **Bulut/İş İstasyonu Ortamı (Örn: Google Colab, Yerel GPU'lu PC):** Arama uzayının belirlenmesi, aday mimarilerin oluşturulması ve ağır donanım gerektiren eğitim işlemleri burada yapılır. PyTorch ve Microsoft NNI gibi bir hiperparametre optimizasyon aracı kullanılarak, omurga katman genişlikleri, boyun (neck) genişlikleri ve girdi çözünürlükleri bir arama uzayına sokulur. Arama süresini aylardan günlere indirmek için COCO veri setinin tamamı yerine 10.000 görselden oluşan, sınıf dağılımı korunmuş bir vekil (proxy) alt küme kullanılır. NNI arama algoritması her aday modeli bu alt küme üzerinde 10 epoch boyunca eğitir. Uç cihazlarda doğrudan eğitim yapmak (backpropagation) termal ve kısıtlı kaynaklar sebebiyle imkansız olduğundan ağır iş yükü bulutta kalır.

- **Uç Ortam (Raspberry Pi 5 / Jetson Orin Nano):** Uç cihazlar sadece donanıma duyarlı "fiziksel gecikme ölçümü" (hardware-aware measurement) aşaması için kullanılır. Bulutta üretilip anında ONNX formatına çevrilen aday model uç cihaza gönderilir. Uç cihaz bu modeli çalıştırarak gerçek gecikme süresini (milisaniye) ölçer. Bulunan bu donanım metriği tekrar bulut ortamındaki NAS algoritmasına geri beslenerek (feedback) bir sonraki arama döngüsünün daha iyi optimize edilmesi sağlanır.

**Neden Yapılıyor:** En az FLOPs değerine sahip olan değil, fiziksel donanımın bellek yollarına ve vektör işlemcilerine en çok uyan mimariyi bulmak hedeflenir. Gecikme süresi ve ortalama hassasiyet (mAP) değerleri üzerinden bir Pareto cephesi (Pareto front) oluşturulur.

**Ne Değişti:** Sadece Raspberry Pi 5 ve Jetson Orin Nano'nun mimari kısıtlarına hitap eden, sıfırdan eğitilmiş yeni bir YOLOX konfigürasyon dosyası (nas_optimized.py) ve buna ait bir ağırlık dosyası (nas_optimized.pth) elde edildi. Bu model artık "Genel Geçer" bir model değil, "Donanım Odaklı" bir varyanttır.

## Beşinci Aşama: Bağımlılık Grafiği Destekli Yapısal Budama (Structured Pruning)

NAS süreci ile tasarlanan donanıma özel ağ, hala nesne tespiti için öğrenmemiş, ölü veya gereksiz (sıfıra yakın) ağırlıklar barındırıyor olabilir. Bu ağırlıkları maskelemek (sıfır yapmak) tensör boyutlarını küçültmediği için CPU ve GPU üzerinde bir hız kazandırmaz. Gerçek bir hızlanma için kanalların fiziksel olarak matrislerden silinmesi gerekir. Buna "Yapısal Budama" (Structured Pruning) denir.

Ancak bir evrişim (convolution) katmanının 64 olan çıkış kanalını 32'ye düşürürseniz, bu katmanı takip eden Batch Normalization (BN) katmanının ve bir sonraki evrişim katmanının giriş kanallarını da 32'ye düşürmek zorundasınız. YOLOX gibi karmaşık, atlamalı bağlantılara (skip connections) sahip ağlarda bu bağımlılıkları manuel yönetmek imkansızdır.

**Ne Yapılıyor:** Modelin iç yapısındaki düğüm noktalarını analiz etmek ve budama işlemini güvenle yapmak için Torch-Pruning kütüphanesi devreye sokulur. Bu kütüphane DepGraph (Bağımlılık Grafiği) adlı yapıyı kullanarak, bir kanal silindiğinde zincirleme olarak hangi katmanların silinmesi gerektiğini otomatik tespit eder.

Uçbirimde kütüphaneyi kuruyoruz:

```bash
pip install torch-pruning==1.1.9
```

Çalışma alanında prune_yolox.py adında bir dosya oluşturup şu Python kodunu çalıştırıyoruz:

```python
import torch
import torch_pruning as tp
from yolox.exp import get_exp

# 1. NAS Optimizasyonlu Modeli Yükle
exp = get_exp(exp_file="exps/default/nas_optimized.py")
model = exp.get_model().eval()
model.load_state_dict(torch.load("../artifacts/pytorch/nas_optimized.pth"))

# 2. Bağımlılık Grafiğini (DepGraph) İnşa Et
# Modele 416x416 boyutunda sahte bir tensor verilerek grafik rotası çizdirilir
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 416, 416))

# 3. L1 Norm Stratejisi Belirle
# Ağırlıklarının mutlak değeri en küçük (en önemsiz) kanallar hedeflenir
imp = tp.importance.MagnitudeImportance(p=1)

# 4. YOLO Tespit Başlıklarını (Heads) Budamadan Koru
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d) and m.out_channels == 85: # COCO (80) + bbox(4) + obj(1)
        ignored_layers.append(m)

# 5. Budayıcıyı (Pruner) Tanımla ve Çalıştır
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs=torch.randn(1, 3, 416, 416),
    importance=imp,
    ch_sparsity=0.3, # Ağdaki kanalların %30'unu sil
    ignored_layers=ignored_layers,
)
pruner.step()
print("Model kanalları %30 oranında fiziksel olarak silindi.")

# 6. SADECE AĞIRLIKLARI DEĞİL, TÜM MODELİ KAYDET
# Katman boyutları değiştiği için state_dict yerine model objesi kaydedilmelidir.
torch.save(model, '../artifacts/pytorch/nas_pruned_model.pth')
```

**Neden Yapılıyor:** Ağdaki düğüm sayıları azaldığında, işlemcinin (ALU) yapması gereken Çarpma-Biriktirme (MAC) operasyonları doğrudan %30-40 civarında düşer. Modelin boyutu RAM üzerinde çok daha az yer kaplar.

**Ne Değişti:** Orijinalinden çok daha ince, hızlı ve hafif bir nas_pruned_model.pth objesi üretildi. Ancak kanalların silinmesi modelin kafasını karıştırıp doğruluğunu (mAP) düşürdüğü için, bu model tüm COCO eğitim seti ile düşük öğrenme hızında (örneğin 1/10 oranında) 20 epoch boyunca yeniden eğitilir (Fine-Tuning). İnce ayar sonrasında hızlanmış ve doğruluğunu geri kazanmış nihai FP32 (32-bit kayan nokta) PyTorch modeli hazır hale gelir.

## Altıncı Aşama: Evrensel Formata Geçiş (ONNX Dönüşümü)

PyTorch, araştırma ve geliştirme için mükemmeldir ancak C++ tabanlı donanım hızlandırıcılarında çalıştırılması son derece hantal ve zordur. Bu noktada model, Açık Sinir Ağı Değişimi (Open Neural Network Exchange - ONNX) formatına aktarılarak çerçeveler arası bağımlılıktan kurtarılır.

**Ne Yapılıyor:** YOLOX'un sunduğu dışa aktarma betiği kullanılarak, budanmış ve ince ayarı yapılmış model ONNX'e dönüştürülür. Dönüşüm sırasında giriş tensörünün boyutu sabit (fixed shape) olarak tanımlanır (örneğin 1x3x416x416) ve dinamik eksenlerden (dynamic axes) kaçınılır.

```bash
python3 tools/export_onnx.py \
-n yolox-nano \
-c ../artifacts/pytorch/nas_pruned_finetuned.pth \
--output-name ../artifacts/onnx/yolox_fp32.onnx \
--no-onnxsim
```

**Neden Yapılıyor:** Cihaz üzerinde çalışacak derleyiciler (TensorRT ve ncnn) dinamik şekilleri desteklese de, sabit boyutlu bir tensör grafiği, donanımın önbellek ve register atamalarını çalışma zamanından önce optimize etmesine imkan verir. Bu da inference hızını doğrudan artırır.

**Ne Değişti:** PyTorch kütüphanesine olan bağımlılık sona erdi. Artık sadece ağırlıkları ve katmanların birbirine bağlanma şemasını içeren evrensel bir yolox_fp32.onnx grafik dosyamız var.

## Yedinci Aşama: Statik Kuantizasyon (Static PTQ) ve QDQ Manipülasyonu

Uç cihazlardaki işlemciler (Özellikle Jetson'un Tensor Core'ları ve Raspberry Pi'nin NEON birimleri), 8-bit tamsayı (INT8) matematiğinde, 32-bit kayan nokta (FP32) matematiğine göre teorik olarak 2 ila 4 kat daha hızlıdır ve çok daha az enerji tüketirler. Ancak bir modeli kaba kuvvetle INT8'e çevirmek, değerlerin \[-128, 127\] veya \[0, 255\] aralığına sıkışması nedeniyle ağın kör olmasına (doğruluğun çökmesine) yol açar. Bu sorunu çözmek için ONNX Runtime ile Kalibrasyonlu Statik Kuantizasyon (Static Post-Training Quantization - PTQ) yapılmalıdır.

### Kuantizasyon Teorisi ve QDQ Formatı

Kuantizasyon işlemi şu matematiksel formüle dayanır: $\mathrm{Değer}_{FP32} = \mathrm{Ölçek} \times (\mathrm{Değer}_{INT8} - \mathrm{Sıfır\ Noktası})$. Statik kuantizasyon sırasında modele bir "Kalibrasyon Veri Seti" verilerek, her bir katmandan geçen aktivasyon verilerinin sınırları (Data_Range_Max, Data_Range_Min) ölçülür. Bu sayede her bir tensör için ideal Ölçek (Scale) ve Sıfır Noktası (Zero Point) parametreleri hesaplanır.

Model grafiğini manipüle ederken, doğrudan Kuantize Operatörler (QOperator formatı) kullanmak yerine **QDQ (Quantize and DeQuantize)** formatı seçilmelidir. QDQ formatında, orijinal FP32 operatörlere dokunulmaz; bunun yerine tensörlerin arasına QuantizeLinear (Q) ve DeQuantizeLinear (DQ) isimli sahte düğümler eklenir. TensorRT gibi donanıma özel arka uçlar, bu Q ve DQ düğümlerini gördüklerinde ağı optimize etmek için muazzam bir hareket alanına (graph fusion) sahip olurlar.

**Ne Yapılıyor:** onnxruntime.quantization kütüphanesi kullanılarak statik QDQ kuantizasyonu uygulanır.

```python
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

# Kalibrasyon veri sağlayıcısı (COCO train setinden sabit 500 görseli okur)
class MyCOCOCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.enum_data = iter(self.image_paths)

    def get_next(self):
        img_path = next(self.enum_data, None)
        if img_path is None:
            return None
        # Resmi 1x3x416x416 FP32 tensöre dönüştüren YOLOX pre-process fonksiyonu
        input_data = yolox_preprocess(img_path)
        return {"images": input_data}

calibrator = MyCOCOCalibrationReader(load_calibration_images())

# Statik Kuantizasyon İşlemi
quantize_static(
    model_input="../artifacts/onnx/yolox_fp32.onnx",
    model_output="../artifacts/onnx/yolox_int8_qdq.onnx",
    calibration_data_reader=calibrator,
    quant_format=QuantFormat.QDQ, # TensorRT/ncnn için kritik format
    activation_type=QuantType.QUInt8, # Aktivasyonlar için 8-bit işaretsiz tamsayı
    weight_type=QuantType.QInt8, # Ağırlıklar için 8-bit işaretli tamsayı
    per_channel=False # Başlangıç için tensör bazlı ölçekleme
)
print("QDQ formatında statik INT8 kuantizasyonu tamamlandı.")
```

**Neden Yapılıyor:** Görüntü işleme sırasında veriler bir katmandan diğerine akarken anlık olarak INT8 uzayına indirgenip, hesaplanıp tekrar FP32 uzayına çıkarılarak aşırı hızlı ve az güç tüketen bir donanım hattı (pipeline) yaratmak hedeflenmektedir.

**Ne Değişti:** Orijinal model boyutuna yakın ama içine yüzlerce QuantizeLinear ve DeQuantizeLinear düğümü ile Scale/Zero_Point sabitleri eklenmiş yeni bir yolox_int8_qdq.onnx dosyası oluşturuldu.

*Not:* Bu işlemden sonra modelin mAP (Ortalama Hassasiyet) değeri hızla değerlendirilmelidir (Accuracy Gate). Eğer kayıp kabul edilemeyecek kadar büyükse, torchao kullanılarak model PyTorch seviyesindeyken Kuantizasyon Farkındalıklı Eğitime (Quantization-Aware Training - QAT) sokularak ağırlıkların INT8 formatına uygun şekilde yeniden eğitilmesi sağlanmalıdır.

## Sekizinci Aşama (Stage-1 Benchmark): Donanımdan Bağımsız Ortak CPU Karşılaştırması

Test ortamlarında yapılan en büyük hata, yazılımdan kaynaklanan hız artışını doğrudan donanımın kendisine mal etmektir. Bunu engellemek için, iki cihaz da öncelikle tamamen aynı yazılım altyapısı ile sadece Merkezi İşlem Birimi (CPU) üzerinden test edilmelidir.

**Ne Yapılıyor:** Geliştirilen ONNX modeli, hem Raspberry Pi 5 hem de Jetson Orin Nano üzerinde ONNX Runtime CPU Execution Provider (CPU EP) kullanılarak çalıştırılır. Jetson cihazındaki devasa GPU ve CUDA çekirdekleri bu testte kasıtlı olarak devre dışı bırakılır.

```python
import onnxruntime as ort
import numpy as np
import time

# Sadece CPU Sağlayıcısı aktif edilerek donanım hızlandırıcılar izole edilir
session = ort.InferenceSession(
    "../artifacts/onnx/yolox_fp32.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 416, 416).astype(np.float32)

# Isınma Turları (Bellek Tahsisatının Stabilizasyonu)
for _ in range(50):
    session.run(None, {input_name: dummy_input})

# Ölçüm Döngüsü (Sadece Model Gecikmesi, Post-Process Hariç)
start = time.perf_counter()
for _ in range(1000):
    session.run(None, {input_name: dummy_input})
end = time.perf_counter()

print(f"Stage-1 Ortalama Gecikme: {((end - start) * 1000) / 1000:.2f} ms")
```

**Neden Yapılıyor:** Raspberry Pi'nin Cortex-A76 çekirdekleri ile Jetson Orin Nano'nun ARM çekirdekleri arasındaki saf CPU taşıma kapasitesi (portability baseline) ölçülür.

**Ne Değişti:** Elimizde her iki cihaz için de donanımsal optimizasyonların sıfır olduğu bir referans performans tablosu (baseline metrics) oluştu.

## Dokuzuncu Aşama (Stage-2 Benchmark): Platforma Özel Maksimum Optimizasyon

Ortak CPU referansı alındıktan sonra, cihazların kendi sınırlarını çizebilmeleri için üreticilerin sağladığı özel donanım hızlandırıcı kütüphanelere geçiş yapılır. Model ağırlıkları ve mantıksal mimari tamamen aynıdır; değişen tek şey hesaplamayı yöneten derleyici (compiler) mantığıdır.

### 9.1 Raspberry Pi 5 Optimizasyonu: ncnn

Raspberry Pi 5 üzerindeki ARM Cortex çekirdekleri GPU barındırmasa da, içlerinde NEON isimli güçlü bir vektör işlem komut seti bulundurur. Bu mimariyi sömürmek için Tencent tarafından açık kaynak kodlu olarak geliştirilen, üçüncü taraf kütüphane bağımlılığı olmayan (no dependencies) ve uç cihazlar için optimize edilmiş ncnn çerçevesi kullanılır.

ONNX'ten ncnn'e dönüşüm yaparken, uyumsuz operatör hatalarını ve dinamik eksen sorunlarını önlemek için aracı bir çevirici olan pnnx (PyTorch to Neural Network Exchange) modülü kullanılır.

**Ne Yapılıyor:** Raspberry Pi üzerinde uçbirim açılarak doğrudan TorchScript üzerinden ncnn dönüşümü sağlanır.

```bash
pip install pnnx

# PyTorch (pt) formatına çıkar (YOLOX ortamı gerektirir)
python3 tools/export_torchscript.py -n yolox-nano -c ../artifacts/pytorch/nas_pruned_finetuned.pth --output-name ../artifacts/ncnn/yolox_ts.pt

# PNNX ile ncnn param (grafik) ve bin (ağırlık) dosyalarına dönüştür
pnnx ../artifacts/ncnn/yolox_ts.pt inputshape=[1,3,416,416]
```

**Neden Yapılıyor:** pnnx kullanmak, geleneksel ONNX dönüştürücüsünün (onnx2ncnn) takıldığı operatör (Slice, Concat) hatalarını bertaraf eder. **Ne Değişti:** Model, Raspberry Pi 5'in 4 çekirdeğine paralel iş parçacıkları (threads) atayarak %100 CPU ve NEON verimliliği ile çalışabilecek .param ve .bin formatlarına çevrildi.

### 9.2 Jetson Orin Nano Optimizasyonu: TensorRT

NVIDIA'nın Jetson sistemlerindeki muazzam başarısı salt donanımdan değil, TensorRT derleyicisinden gelir. TensorRT, sağlanan modeli alır, Jetson GPU'sunun mimari özelliklerine göre katmanları birleştirir (layer fusion), uygun çekirdek optimizasyonlarını (kernel auto-tuning) seçer ve sadece o cihaza özel çalışan bir motor (.plan) dosyası üretir.

**Ne Yapılıyor:** QDQ formatında hazırlanan INT8 ONNX modeli hedef Jetson Orin cihazında derlenir. *(Dikkat: Motor dosyaları farklı bir bilgisayarda derlenip Jetson'a taşınamaz, doğrudan hedef cihazda oluşturulmalıdır.)*

```bash
/usr/src/tensorrt/bin/trtexec \
--onnx=../artifacts/onnx/yolox_int8_qdq.onnx \
--saveEngine=../artifacts/tensorrt/yolox_int8.plan \
--shapes=images:1x3x416x416 \
--int8 \
--warmUp=3000 \
--duration=60
```

**Neden Yapılıyor:** Modeldeki FP32 yükü, GPU'daki Tensor Çekirdeklerinde koşacak INT8 operasyonlarına dönüştürülmektedir. --int8 parametresi ve QDQ formatı birleştiğinde, TensorRT hangi katmanların hassasiyetinden ödün verilmeyeceğini (Örn: YOLO tespit başlıkları) çok iyi anlar ve performansı uç noktaya taşır. **Ne Değişti:** Jetson üzerinde çalışan yolox_int8.plan motoru oluşturuldu. Uçbirim çıktısında, cihazın sağladığı "Ortalama Gecikme" (Mean Latency) ve "Kare Sayısı/Saniye" (Throughput/FPS) raporu otomatik olarak görüntülendi. Hızlanma faktörü (Speedup), Stage-1 CPU ölçümüne göre 10 kata kadar varan farklılıklar gösterebilir.

## Onuncu Aşama: Güç Ölçümü, Telemetri ve Pareto Analizi

Uç yapay zekâda asıl başarı kriteri, sistemin en yüksek FPS'yi alması değil; harcanan birim enerji (Watt veya Joule) başına en fazla kareyi (frame) işleyebilmesidir.

**Ne Yapılıyor:** İki cihazın donanımsal telemetri yöntemleri birbirinden farklıdır. En doğru sonuç için dışarıdan (harici) bir güç ölçüm metodu kullanılmalıdır.

- **Raspberry Pi 5:** Type-C besleme hattına donanımsal bir USB-C Güç Analizörü (Power Meter) takılır. Sistem boşta (idle) iken ve inference (load) sırasında iken amperaj farkları ölçülerek $P_{dynamic}$ (Dinamik Güç Tüketimi) bulunur. Arka planda /usr/bin/time -v ve htop ile Pik RAM kullanımı ölçülür.

- **Jetson Orin Nano:** Güç girişine takılan cihazın yanı sıra, NVIDIA'nın dahili aracı olan tegrastats ile güç hatlarındaki (rail) milisaniyelik tüketim log (kayıt) dosyasına dökülür:

  ```bash
  tegrastats --interval 100 --logfile ../logs/tegrastats_load.log
  ```

**Neden Yapılıyor:** Bir sistem saniyede 60 kare (FPS) işlerken 15 Watt güç tüketiyorsa, saniyede 30 kare işleyip 5 Watt tüketen bir sisteme göre "Enerji Verimliliği" açısından daha başarısızdır. Gerçek dünya uygulamalarında (bataryalı İHA'lar veya güneş enerjili kameralar) asıl metrik budur.

**Ne Değişti:** Elde edilen telemetri verileri sayesinde, hız ve güç tüketimi üzerinden matematiksel bir "Pareto Eğrisi" (Pareto Front) çizilmesi mümkün hale geldi.

### Çıktıların Karşılaştırmalı Tablosu

Tüm aşamalardan elde edilen veriler, bilimsel bir raporlama standardına (CSV) oturtulmalıdır. Tablo yapısı aşağıdaki formatta sunulmalıdır:

| **Cihaz**        | **Optimizasyon Aşaması** | **Arka Uç (Runtime)** | **Model Tipi** | **Input Boyutu** | **Ortalama Gecikme (ms)** | **Througput (FPS)** | **Yük Gücü (Watt)** | **Verimlilik (FPS/W)** | **Enerji (Joule/Frame)** |
|------------------|--------------------------|-----------------------|----------------|------------------|---------------------------|---------------------|---------------------|------------------------|--------------------------|
| RPi 5            | Stage-1                  | ORT CPU               | FP32           | 416x416          | *Ölçüm 1*                 | *F1*                | *W1*                | *F1/W1*                | *W1/F1*                  |
| Jetson Orin Nano | Stage-1                  | ORT CPU               | FP32           | 416x416          | *Ölçüm 2*                 | *F2*                | *W2*                | *F2/W2*                | *W2/F2*                  |
| RPi 5            | Stage-2                  | ncnn                  | INT8           | 416x416          | *Ölçüm 3*                 | *F3*                | *W3*                | *F3/W3*                | *W3/F3*                  |
| Jetson Orin Nano | Stage-2                  | TensorRT              | INT8 (QDQ)     | 416x416          | *Ölçüm 4*                 | *F4*                | *W4*                | *F4/W4*                | *W4/F4*                  |

Bu tablo aracılığıyla hesaplanan *Marjinal Verimlilik* formülü: $\mathrm{Speedup} = \frac{FPS_{Stage2}}{FPS_{Stage1}}$ şeklindedir. Böylece TensorRT veya ncnn kütüphanelerinin donanım üzerine eklediği salt yazılım/derleyici katkısı matematiksel olarak kanıtlanmış olur.

## On Birinci Aşama: Sürekli Öğrenme ve Canlı Sistem Entegrasyonu Vizyonu

Yukarıdaki 10 adımlık somut iş akışı tamamlandığında, elinizde donanıma duyarlı, budanmış, kuantize edilmiş ve hedef cihaza göre derlenmiş olağanüstü hızlı bir YOLOX-Nano modeli bulunur.

Ancak uç cihazlar, zaman içinde değişen saha koşullarına maruz kalır (Concept Drift - Kavram Kayması). Görüntü işleme esnasında, güvenilirlik skoru (confidence score) düşük olan görseller (low-confidence samples) Raspberry Pi veya Jetson cihazı üzerindeki yerel bir bellekte (buffer) toplanmalıdır. Bu veriler düzenli aralıklarla merkezi bulut sistemine aktarılmalı; model, NAS ve budama konfigürasyonu (mimari şeması) korunarak bu yeni verilerle merkezi sunucuda ince ayar (fine-tuning) işlemine tabi tutulmalıdır. Geliştirilmiş ve doğruluğu artırılmış model; yeniden ONNX formatına alınmalı, QDQ kuantizasyonundan geçmeli ve uzaktan güncelleme (OTA Deployment) aracılığıyla uç cihaza gönderilmelidir. Cihazın kendi kısıtlı işlemcisi üzerinde YOLOX eğitimini baştan başlatmak termal ve donanımsal olarak imkansız olduğundan, uç cihazlar veri toplayıcı ve çıkarım (inference) aracı olarak çalışmalı, eğitim yükü (training) merkezde bırakılmalıdır.

Bu rehberde sunulan iş akışı, laboratuvar ortamındaki prototiplerin gerçek saha donanımlarına (Edge AI) minimum performans kaybı ve maksimum enerji verimliliği ile aktarılabilmesi için somut, yenilenebilir ve teknik olarak güncellenebilir bir standart belirlemiştir.

#### Alıntılanan çalışmalar
