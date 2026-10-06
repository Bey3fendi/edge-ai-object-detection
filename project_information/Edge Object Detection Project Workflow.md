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

> **Uygulama (2026-10-04):** Çalışma alanı, mevcut GitHub deposu `Bey3fendi/edge-ai-object-detection` (public) klonlanarak Windows'ta `C:\projects\edge_ai_workspace` konumunda tutulur. Obsidian vault'unun **dışındadır**: vault içindeki hafıza dizini YOLOX'un `.md` dosyalarını da tarıyordu ve notlar git deposuna karışmamalı. Belgedeki klasörlere ek olarak `notebooks/` (Colab notebook'ları, `NN_asama_konu.ipynb` adlandırması) ve `results/` (ölçüm CSV'leri ve değerlendirme logları, git'te izlenir) vardır. Bu belgenin asıl kopyası vault'tadır; depodaki `project_information/` kopyası vault'tan eşitlenir.

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

> **Sürüm sabitleme (2026-10-04):** YOLOX submodule'ü Furkan'ın fork'u `Bey3fendi/YOLOX` (dal `edge-ai-thesis`) üzerinden `3ec62638119e272203c7672e0aaf770e20d3f7b0` commit'ine sabitlidir. Fork, upstream `6ddff48`'in üzerine iki düzeltme ekler: CPU'da değerlendirme desteği (`tools/eval.py`) ve güncel `torch.onnx.export` API'si (`tools/export_onnx.py`). Eğitim Colab'da yapılacağı için bağımlılıklar notebook içinde kurulur.

**Ne Değişti:** Çalışma alanımızda YOLOX mimarisini eğitebileceğimiz, test edebileceğimiz ve dışa aktarabileceğimiz tam donanımlı ve versiyon kontrollü bir PyTorch ortamı ayağa kalkmış oldu.

### Referans Ağırlıkların İndirilmesi ve Temel Test (Smoke Test)

**Ne Yapılıyor:** Megvii'nin resmi GitHub sürümlerinden YOLOX-Nano'nun önceden eğitilmiş ağırlıkları indirilir ve örnek bir görsel üzerinde çıkarım (inference) yapılır.

```bash
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -O ../artifacts/pytorch/yolox_nano.pth

python3 YOLOX/tools/demo.py image -n yolox-nano -c artifacts/pytorch/yolox_nano.pth --path YOLOX/assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 416 --save_result --device cpu
```

**Neden Yapılıyor:** Geliştirme sürecinin daha ilk adımında kod tabanının, bağımlılıkların (OpenCV, PyTorch) ve model ağırlıklarının uyum içinde çalıştığından emin olmak (smoke test) gerekir.

**Ne Değişti:** YOLOX_outputs/ klasöründe üzerinde tespit kutuları olan bir dog.jpg görseli oluştu. Sistemin fabrikadan çıktığı haliyle eksiksiz çalıştığı kanıtlandı.

## Dördüncü Aşama: Donanıma Duyarlı Makro-Mimari Araması (Hardware-Aware NAS)

Geleneksel modellerin Floating Point Operations (FLOPs) değerlerini küçültmeye odaklanan yaklaşımlar, gerçek dünyada çoğu zaman gecikmeyi (latency) düzeltmez; çünkü işlemcinin veriyi bellekten getirme süresi (memory access cost) ve işlemlerin donanımdaki maliyeti (ör. SiLU aktivasyonu) FLOPs'a yansımaz. Bu nedenle mimari seçimi, hedef donanımda **ölçülen gecikmeye** göre yapılır.

> **Kapsam kararı (Karar 5, 2026-10-03):** Klasik NAS uygulanmaz. Klasik NAS'ta her aday sıfırdan eğitilir; YOLOX'un COCO'daki tam eğitimi ~300 epoch'tur ve tek bir tam eğitim bile Colab Pro bütçesini (~400 kredi) aşar (kaba tahmin: 100-150 A100 saati). PASCAL VOC üzerinde klasik NAS da değerlendirildi; sıfırdan baseline gerektirdiği, COCO tabanlı kabul eşiklerini (Karar 2) geçersiz kıldığı ve tezin sorusunu değiştirdiği için reddedildi. Bunun yerine, önceden eğitilmiş ağırlıklarla uyumlu **ayrık bir makro-mimari araması** yapılır. Tezde yöntem "donanıma duyarlı makro-mimari araması" olarak adlandırılır.

**Görev ayrımı:** Bu aşama **mimariyi** (işlemler, derinlik, çözünürlük) belirler; **kanal genişliğine dokunmaz**. Genişlik, Beşinci Aşama'daki yapısal budamanın işidir. Böylece iki yöntem çakışmaz ve ablasyonda etkileri ayrı ölçülebilir.

**Arama uzayı** (tüm boyutlar önceden eğitilmiş YOLOX-Nano ağırlıklarıyla uyumludur):

| Boyut | Seçenekler | Ağırlık aktarımı |
| --- | --- | --- |
| Giriş çözünürlüğü | 320 / 416 / 512 | Ağırlıklar değişmez |
| Aktivasyon | SiLU / ReLU / HardSwish | Tensör boyutları aynı, doğrudan yüklenir |
| CSP blok derinliği | mevcut / bir blok eksik (backbone dark3 ve dark4: 3 → 2; son Bottleneck çıkarılır, tek bloklu aşamalara dokunulmaz) | Kalan bloklar aynen yüklenir |
| Depthwise kernel boyutu | 3 / 5 (tüm depthwise katmanlar) | 3×3 ağırlık 5×5'in merkezine konur, kenarlar sıfır: eğitim öncesi çıktı orijinalle birebir aynıdır (2026-10-05 doğrulandı) |

**Ne Yapılıyor:** Bu işlem bulut ve uç cihazın birlikte kullanıldığı **hibrit bir yapıda** yürütülür:

- **Bulut (Google Colab Pro):** Her aday, önceden eğitilmiş YOLOX-Nano ağırlıklarından başlatılır ve 10.000 görsellik, sınıf dağılımı korunmuş COCO alt kümesinde **10 epoch** eğitilir. mAP, COCO val2017 üzerinde mAP@[.5:.95] olarak ölçülür (Karar 2).
- **10k alt küme yöntemi (2026-10-04):** COCO train2017'den sınıf dağılımı korunarak (stratified) seçilir: anotasyonlu her görsel, içerdiği en nadir sınıfa (train2017 örnek sayısına göre) atanır ve bu katmanlardan orantılı (largest remainder yuvarlama) örnek çekilir; anotasyonsuz görseller dışarıda kalır. Sabit seed yoktur; seçilen görsel ID listesi sürümlenir (`results/pilot/subset_10k_image_ids.json`, Drive `edge_ai/data/coco_train2017_10k_stratified.zip`), alt küme bu listeden yeniden kurulur. Kontrol: 10.000 görsel, 73.757 örnek, 80 sınıfın tamamı mevcut; sınıf başına örnek payının tam train2017'den sapması ortalama 0,05, en fazla 0,54 puan.
- **Eğitim tarifi (2026-10-04):** Tüm adaylar ve kontrol koşusu için aynıdır: 10 epoch, batch 64, fp16, çoklu ölçek açık; **son 2 epoch mosaic'siz** (L1 kaybı açık), değerlendirme bu iki epoch'un sonunda yapılır. YOLOX'ta mosaic'siz epoch sayısı `no_aug_epochs + 1` olduğu için ayar `no_aug_epochs = 1`'dir; varsayılan 15 kullanılırsa 10 epoch'un tamamı mosaic'siz geçer.
- **Aday sayısı (2026-10-04):** İlk turda arama uzayının **tam ızgarası** (3 × 3 × 2 × 2 = 36 aday) değerlendirilir; örnekleme yapılmaz.
- **Çözünürlük yalnız değerlendirmede (Furkan, 2026-10-05):** YOLOX-Nano zaten 320–640 arası çok ölçekli eğitildiği için çözünürlük bir dağıtım ayarı olarak ele alınır: **12 mimari** (aktivasyon × derinlik × kernel) eğitilir, her biri val2017'de 320, 416 ve 512'de değerlendirilir → 36 aday. Eğitim maliyeti ≈ üçte bire iner (≈ 4,5 saat, ≈ 7 CU).
- **Aday mAP'i (Furkan, 2026-10-05):** Son epoch (EMA) ağırlığı, ayrı **fp32** `eval.py` ile (batch 64, conf 0,001; P0 baseline protokolü). Eğitim içi fp16 değerlendirme kapalıdır; iki değerlendirmeden iyisini seçmek adaylara şans avantajı vereceği için kullanılmaz.
- **Isınma (Furkan, 2026-10-05):** `warmup_epochs = 1` (YOLOX varsayılanı 5, 10 epoch'luk koşunun yarısı olurdu; YOLOX'un COCO→VOC fine-tune örneği 1 kullanır). Öğrenme hızı **0,001 / 64 görsel** (cos, min_lr_ratio 0,05; Furkan, 2026-10-06). İlk deneme (`nas_r1`) YOLOX varsayılanı 0,01 ile koşuldu; kontrol 416'da mAP 25,8 → 20,4 düştü ve erken kontrol (eşik 23,0) döngüyü durdurdu → tur `nas_r1b` olarak yeniden açıldı, kontrol önce tek başına koşulur.
- **Kontrol koşusu (2026-10-04):** Değişmemiş YOLOX-Nano aynı tarifle (10k alt küme, 10 epoch) eğitilir. Adayların mAP kaybı orijinal 25,8 yerine bu kontrole göre hesaplanır; böylece mimarinin etkisi kısa fine-tune'un etkisinden ayrılır. Kontrol, ızgaradaki `416 · SiLU · mevcut · k3` adayıyla aynı koşudur; **gürültüyü** (aynı tarifle tekrar eğitimde mAP'nin kendiliğinden oynaması) ölçmek için **ikinci kez** koşulur (Furkan, 2026-10-05).
- **Uygulama ve kayıt (2026-10-05):** `notebooks/02_makro_mimari_arama.ipynb`, `exps/nas/nano_macro.py` (model cerrahisi; YOLOX fork'una dokunulmaz), `scripts/nas_search.py`. Kayıtlar `results/nas_r1b/` (başarısız ilk deneme `nas_r1/`): `manifest.json` (tarif, ızgara, commit ve hash'ler), `runs/<koşu>/` (config, durum, log), `candidates.csv`. Koşular kesintiye dayanıklıdır (epoch başına Drive'a atomik checkpoint, kaldığı yerden devam).
- **Uç cihaz:** Aday ONNX'e çevrilir ve **gecikme ölçütü olarak Raspberry Pi 5 üzerinde ONNX Runtime CPU** ile ölçülür (Stage-1 ayarı; Karar 3 protokolü). Jetson Orin Nano gecikmesi de ölçülür ve raporlanır, ancak seçime girmez.
- **Seçim kuralı:** Kontrol koşusuna göre **mAP kaybı ≤ 1,0 puan** olan adaylar arasında RPi 5'te en hızlı olan seçilir. İki cihaz için **tek mimari** seçilir.
- **Durdurma kuralı:** En fazla 2 tur. Bir tur, (mAP, RPi gecikmesi) Pareto cephesini iyileştirmezse arama durur.
- **Seçilen mimari** tam COCO'da ~20-30 epoch fine-tune edilir (aktivasyon değiştiyse 10k alt küme yeterli olmayabilir).

**İlk somut adım (bütçe kalibrasyonu):** Aramaya başlamadan önce Colab'da 1 epoch'luk pilot eğitim yapılarak gerçek epoch süresi ve kredi tüketimi ölçülür; tüm eğitim bütçesi bu ölçümle hesaplanır.

**Pilot sonucu (2026-10-04, Colab L4, ~1,54 CU/saat; `notebooks/00_pilot_1epoch_sure_olcumu.ipynb`, `results/pilot/`):** 10k alt kümede 1 epoch (157 iterasyon, batch 64, fp16) 214 sn (0,092 CU); bunun ≈ 200 sn'si cuDNN'in 11 giriş boyutunun her biri için ilk görüldüğünde yaptığı tek seferlik algoritma aramasıdır, kararlı epoch ≈ 69 sn. val2017 değerlendirmesi (fp32, batch 64) 88 sn (0,037 CU); önceden eğitilmiş Nano bu ortamda mAP 25,8 / AP50 41,4 verdi (baseline birebir tekrar üretildi). **Bütçe tahmini** (ölçümlerden türetilmiştir): aday başına ≈ 0,45 CU (≈ 18 dk); 36 adaylık tur ≈ 16 CU, 2 tur ≤ ≈ 33 CU; tam COCO fine-tune (~25 epoch) ≈ 9 CU; budama + QAT ≈ 15–30 CU; toplam ≈ 60–75 CU, ~400 kredilik Colab Pro bütçesinin çok altında.

**Neden Yapılıyor:** En az FLOPs değerine sahip olan değil, hedef donanımda en hızlı çalışan ve doğruluğu bütçe içinde kalan mimariyi bulmak hedeflenir. mAP ve gecikme üzerinden bir Pareto cephesi oluşturulur.

**Ne Değişti:** Önceden eğitilmiş ağırlıklardan türetilmiş, fine-tune edilmiş yeni bir YOLOX konfigürasyon dosyası (`nas_optimized.py`) ve ağırlık dosyası (`nas_optimized.pth`) elde edildi. Bu model artık "genel geçer" değil, "donanım odaklı" bir varyanttır.

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
# YOLOX'ta başlık tek bir 85 kanallı katman DEĞİLDİR: her ölçek için ayrı
# cls_preds (num_classes), reg_preds (4) ve obj_preds (1) konvolüsyonları vardır
# (yolox/models/yolo_head.py). Bu yüzden doğrudan bu modüller korunur.
ignored_layers = []
for preds in (model.head.cls_preds, model.head.reg_preds, model.head.obj_preds):
    ignored_layers.extend(preds)

# 5. Budayıcıyı (Pruner) Tanımla ve Çalıştır
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs=torch.randn(1, 3, 416, 416),
    importance=imp,
    ch_sparsity=0.3, # Örnek değer; gerçek oran aşağıdaki taramayla seçilir (Karar 5)
    ignored_layers=ignored_layers,
)
pruner.step()
print("Kanallar seçilen oranda fiziksel olarak silindi.")

# 6. SADECE AĞIRLIKLARI DEĞİL, TÜM MODELİ KAYDET
# Katman boyutları değiştiği için state_dict yerine model objesi kaydedilmelidir.
torch.save(model, '../artifacts/pytorch/nas_pruned_model.pth')
```

**Neden Yapılıyor:** Ağdaki düğüm sayıları azaldığında, işlemcinin (ALU) yapması gereken Çarpma-Biriktirme (MAC) operasyonları budama oranıyla orantılı olarak azalır (gerçek gecikme kazancı cihazda ölçülür). Modelin boyutu RAM üzerinde çok daha az yer kaplar.

**Ne Değişti:** Orijinalinden çok daha ince, hızlı ve hafif bir nas_pruned_model.pth objesi üretildi. Ancak kanalların silinmesi modelin kafasını karıştırıp doğruluğunu (mAP) düşürdüğü için, bu model tüm COCO eğitim seti ile düşük öğrenme hızında (örneğin 1/10 oranında) 20 epoch boyunca yeniden eğitilir (Fine-Tuning). İnce ayar sonrasında hızlanmış ve doğruluğunu geri kazanmış nihai FP32 (32-bit kayan nokta) PyTorch modeli hazır hale gelir.

**Budama oranının seçimi (Karar 5):** Sabit %30 kullanılmaz. YOLOX-Nano zaten depthwise konvolüsyonlarla küçültülmüş, 0,91M parametrelik bir model olduğu için yüksek oranlar doğruluğu hızla düşürebilir. Oranlar **%10 / %20 / %30 / %40** olarak kısa bir fine-tune ile taranır. Budama bütçesi içinde kalan (**mAP kaybı ≤ 1,0 puan**, budama öncesi modele göre) en yüksek oran seçilir. Tam COCO, düşük öğrenme hızı, 20 epoch fine-tune yalnız seçilen oranla yapılır.

**mAP bütçesi (Karar 2 ve 5):** Referans YOLOX-Nano val2017 mAP@[.5:.95] = 25,8 (416). Aşama başına sınırlar NAS ≤ 1,0 · budama ≤ 1,0 · INT8 ≤ 2,0 puan; orijinal baseline'a göre toplam kayıp ≤ 4,0 puan (alt sınır ≈ 21,8).

**Ablasyon:** Her yöntemin katkısını ayırmak için dört model ayrı ayrı ölçülür: baseline · yalnız NAS · yalnız budama · NAS + budama.

## Altıncı Aşama: Evrensel Formata Geçiş (ONNX Dönüşümü)

PyTorch, araştırma ve geliştirme için mükemmeldir ancak C++ tabanlı donanım hızlandırıcılarında çalıştırılması son derece hantal ve zordur. Bu noktada model, Açık Sinir Ağı Değişimi (Open Neural Network Exchange - ONNX) formatına aktarılarak çerçeveler arası bağımlılıktan kurtarılır.

**Ne Yapılıyor:** YOLOX'un sunduğu dışa aktarma betiği kullanılarak, budanmış ve ince ayarı yapılmış model ONNX'e dönüştürülür. Dönüşüm sırasında giriş tensörünün boyutu sabit (fixed shape) olarak tanımlanır (örneğin 1x3x416x416) ve dinamik eksenlerden (dynamic axes) kaçınılır.

> **Not:** YOLOX'un `tools/export_onnx.py` betiği modeli `-n yolox-nano` (veya `-f exp_dosyası`) tanımından kurup `state_dict` yükler. Budanmış modelde katman boyutları değiştiği ve model `torch.save(model)` ile **bütün nesne** olarak kaydedildiği için bu betik doğrudan çalışmaz (boyut uyuşmazlığı). Bu yüzden kaydedilen model nesnesi doğrudan dışa aktarılır; betiğin varsayılanları (giriş adı `images`, çıkış adı `output`, `decode_in_inference=False`) korunur.

```python
import numpy as np
import onnxruntime as ort
import torch

# Budanmış + fine-tune edilmiş TAM model nesnesi (state_dict değil)
model = torch.load("../artifacts/pytorch/nas_pruned_finetuned.pth", weights_only=False).eval()
model.head.decode_in_inference = False  # export_onnx.py ile aynı davranış

dummy = torch.randn(1, 3, 416, 416)  # NAS'ın seçtiği çözünürlük kullanılır
torch.onnx.export(model, dummy, "../artifacts/onnx/yolox_fp32.onnx",
                  input_names=["images"], output_names=["output"],
                  opset_version=13)  # sabit (2026-10-04): per-channel QDQ için gereken en düşük opset

# Eşdeğerlik kontrolü: PyTorch ve ONNX Runtime çıktıları aynı olmalı
with torch.no_grad():
    ref = model(dummy).numpy()
out = ort.InferenceSession("../artifacts/onnx/yolox_fp32.onnx",
                           providers=["CPUExecutionProvider"]).run(None, {"images": dummy.numpy()})[0]
print("maks. mutlak fark:", np.abs(ref - out).max())
```

**Canonical model (Karar 1):** `yolox_fp32.onnx` tezin asıl referansıdır. INT8 model (`yolox_int8_qdq.onnx`) bundan türetilir ve ayrıca ölçülür.

**ONNX opset (2026-10-04): 13.** NAS adayları (N3) dahil tüm ONNX dışa aktarımlarında kullanılır. Per-channel (eksen bazlı) `QuantizeLinear`/`DequantizeLinear` ilk kez opset 13'te geldiği için per-channel PTQ, yeniden dışa aktarma gerekmeden aynı canonical FP32 dosyadan türetilebilir. ORT, TensorRT ve pnnx desteği cihaz smoke testinde (B2) doğrulanır.

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
    quant_format=QuantFormat.QDQ, # TensorRT explicit quantization için gerekli (ncnn QDQ'yu desteklemez, bkz. 9.1)
    activation_type=QuantType.QUInt8, # Aktivasyonlar için 8-bit işaretsiz tamsayı
    weight_type=QuantType.QInt8, # Ağırlıklar için 8-bit işaretli tamsayı
    per_channel=False # Başlangıç için tensör bazlı ölçekleme
)
print("QDQ formatında statik INT8 kuantizasyonu tamamlandı.")
```

**Neden Yapılıyor:** Görüntü işleme sırasında veriler bir katmandan diğerine akarken anlık olarak INT8 uzayına indirgenip, hesaplanıp tekrar FP32 uzayına çıkarılarak aşırı hızlı ve az güç tüketen bir donanım hattı (pipeline) yaratmak hedeflenmektedir.

**Ne Değişti:** Orijinal model boyutuna yakın ama içine yüzlerce QuantizeLinear ve DeQuantizeLinear düğümü ile Scale/Zero_Point sabitleri eklenmiş yeni bir yolox_int8_qdq.onnx dosyası oluşturuldu.

**Doğruluk kapısı (Accuracy Gate, Karar 2):**

- **Ölçüt:** COCO val2017 mAP@[.5:.95] (ana) ve mAP@.5 (yardımcı); FP32 ve INT8 modellerde aynı ön/son işlem ve NMS ayarları.
- **Kabul:** INT8 modelin `yolox_fp32.onnx`'e göre kaybı **≤ 2,0 puan** (mutlak).
- **Eşik aşılırsa:** Kuantizasyon Farkındalıklı Eğitim (QAT), **en fazla 2 deneme × 10 epoch**.
- **QAT de yetmezse:** Mixed precision (hassas katmanlar, özellikle tespit başlığı, FP16/FP32'de bırakılır); o da yetmezse sonuç olduğu gibi raporlanır.
- **Toplam bütçe:** Orijinal YOLOX-Nano'ya göre NAS + budama + INT8 toplam kaybı ≤ 4,0 puan.

*Açık noktalar (henüz karar verilmedi):* (1) YOLOX-Nano depthwise konvolüsyon kullandığı için tensör bazlı (`per_channel=False`) ağırlık kuantizasyonu doğruluğu belirgin düşürebilir; `per_channel=True` ile karşılaştırılması önerilir. (2) QAT için kullanılacak araç (ör. torchao) ve QAT modelinin QDQ ONNX'e nasıl dışa aktarılacağı henüz belirlenmedi.

## Sekizinci Aşama (Stage-1 Benchmark): Donanımdan Bağımsız Ortak CPU Karşılaştırması

Test ortamlarında yapılan en büyük hata, yazılımdan kaynaklanan hız artışını doğrudan donanımın kendisine mal etmektir. Bunu engellemek için, iki cihaz da öncelikle tamamen aynı yazılım altyapısı ile sadece Merkezi İşlem Birimi (CPU) üzerinden test edilmelidir.

**Ne Yapılıyor:** Geliştirilen ONNX modeli, hem Raspberry Pi 5 hem de Jetson Orin Nano üzerinde ONNX Runtime CPU Execution Provider (CPU EP) kullanılarak çalıştırılır. Jetson cihazındaki devasa GPU ve CUDA çekirdekleri bu testte kasıtlı olarak devre dışı bırakılır.

Stage-1'de iki model ölçülür (Karar 1): **ana veri** `yolox_fp32.onnx`, **ek veri** `yolox_int8_qdq.onnx`. Ölçüm protokolü Karar 3'e göredir: 100 çıkarım ısınma atılır, ardından 60 saniyelik kararlı ölçüm **3 kez** tekrarlanır ve ortalama ± standart sapma raporlanır. İş parçacığı sayısı iki cihazda da **4**'tür (2026-10-04; RPi 5'in çekirdek sayısı, Orin Nano'da 6 çekirdeğin 4'ü). NAS gecikme ölçümü (N4) de aynı ayarı kullanır.

```python
import time
import numpy as np
import onnxruntime as ort

WARMUP, DURATION_S, REPEATS, THREADS = 100, 60, 3, 4

def bench(model_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS  # iki cihazda aynı değer
    # Sadece CPU Sağlayıcısı aktif edilerek donanım hızlandırıcılar izole edilir
    session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.randn(1, 3, 416, 416).astype(np.float32)  # seçilen çözünürlük

    for _ in range(WARMUP):  # ısınma: bellek tahsisatının stabilizasyonu
        session.run(None, {name: x})

    runs = []
    for _ in range(REPEATS):  # her tekrar 60 sn; güç ölçer aynı pencereyi kaydeder
        lat, t_end = [], time.perf_counter() + DURATION_S
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            session.run(None, {name: x})  # yalnız model (post-process hariç)
            lat.append((time.perf_counter() - t0) * 1000)
        runs.append((np.mean(lat), 1000 / np.mean(lat)))
    ms, fps = np.array(runs).T
    print(f"{model_path}: {ms.mean():.2f} ± {ms.std():.2f} ms, {fps.mean():.1f} FPS (yalnız model)")

for path in ("../artifacts/onnx/yolox_fp32.onnx", "../artifacts/onnx/yolox_int8_qdq.onnx"):
    bench(path)
```

**Uçtan uca FPS (Karar 3):** Yukarıdaki ölçüm yalnız model penceresidir. Ayrıca gerçek görüntülerle ön işlem + model + NMS süresini kapsayan uçtan uca FPS ayrı bir sütun olarak ölçülür.

**Neden Yapılıyor:** Raspberry Pi'nin Cortex-A76 çekirdekleri ile Jetson Orin Nano'nun ARM çekirdekleri arasındaki saf CPU taşıma kapasitesi (portability baseline) ölçülür.

**Ne Değişti:** Elimizde her iki cihaz için de donanımsal optimizasyonların sıfır olduğu bir referans performans tablosu (baseline metrics) oluştu.

## Dokuzuncu Aşama (Stage-2 Benchmark): Platforma Özel Maksimum Optimizasyon

Ortak CPU referansı alındıktan sonra, cihazların kendi sınırlarını çizebilmeleri için üreticilerin sağladığı özel donanım hızlandırıcı kütüphanelere geçiş yapılır. Tüm Stage-2 modelleri aynı FP32 ağırlıklardan (canonical `yolox_fp32.onnx` ile aynı checkpoint) türetilir ve mantıksal mimari aynıdır. **Ancak INT8 modeller aynı değildir:** Jetson'daki TensorRT INT8, Stage-1'de ölçülen `yolox_int8_qdq.onnx`'in kuantizasyon parametrelerini kullanırken, Raspberry Pi'deki ncnn INT8 kendi araçlarıyla ayrıca kalibre edilir (bkz. 9.1). Bu nedenle her INT8 modelin mAP'i hedef cihazda ayrıca ölçülür ve Karar 2'deki eşik (FP32'ye göre ΔmAP ≤ 2,0 puan) her birine ayrı uygulanır.

**Stage-2 benchmark satırları:** RPi 5 → ncnn FP32 (dönüşüm doğrulaması) ve ncnn INT8; Jetson Orin Nano → TensorRT FP16 ve TensorRT INT8 (QDQ).

### 9.1 Raspberry Pi 5 Optimizasyonu: ncnn

Raspberry Pi 5 üzerindeki ARM Cortex çekirdekleri GPU barındırmasa da, içlerinde NEON isimli güçlü bir vektör işlem komut seti bulundurur. Bu mimariyi sömürmek için Tencent tarafından açık kaynak kodlu olarak geliştirilen, üçüncü taraf kütüphane bağımlılığı olmayan (no dependencies) ve uç cihazlar için optimize edilmiş ncnn çerçevesi kullanılır.

ncnn'e dönüşümde, uyumsuz operatör hatalarını ve dinamik eksen sorunlarını önlemek için pnnx (PyTorch Neural Network Exchange) kullanılır. Bu yol ONNX'ten değil, aynı FP32 checkpoint'ten üretilen TorchScript'ten başlar.

> **Neden QDQ ONNX modeli ncnn'de kullanılmıyor? (araştırma, 2026-10-03)** ncnn'in iki ONNX dönüştürücüsü de ONNX `QuantizeLinear` / `DequantizeLinear` düğümlerini desteklemez. ncnn kaynak kodunda (master, commit `9f9d4ec`, son sürüm `20260526`) `tools/onnx/onnx2ncnn.cpp` bu düğümler için `"QuantizeLinear not supported yet!"` uyarısı basar ve tanımadığı düğümü geçersiz bir katman olarak yazar. pnnx'in ONNX yolunda (`tools/pnnx/src`) bu düğümleri işleyen bir pass yoktur. pnnx'in ncnn arka ucu (`pass_ncnn`) PyTorch quantized modüllerini de INT8 katmana çevirmez. Aynı sorun ncnn issue #5183'te 2023'ten beri açıktır; 2025-11 tarihli yorum da desteğin hâlâ olmadığını teyit eder. Sonuç olarak ncnn'de INT8'in resmi yolu, FP32 modelden `ncnn2table` + `ncnn2int8` ile yapılan kendi PTQ'sudur.

**Ne Yapılıyor:** Raspberry Pi üzerinde uçbirim açılarak doğrudan TorchScript üzerinden ncnn dönüşümü sağlanır.

```bash
pip install pnnx

# PyTorch (pt) formatına çıkar (YOLOX ortamı gerektirir)
python3 tools/export_torchscript.py -n yolox-nano -c ../artifacts/pytorch/nas_pruned_finetuned.pth --output-name ../artifacts/ncnn/yolox_ts.pt

# PNNX ile ncnn param (grafik) ve bin (ağırlık) dosyalarına dönüştür
pnnx ../artifacts/ncnn/yolox_ts.pt inputshape=[1,3,416,416]
```

**Neden Yapılıyor:** pnnx kullanmak, geleneksel ONNX dönüştürücüsünün (onnx2ncnn) takıldığı operatör (Slice, Concat) hatalarını bertaraf eder. **Ne Değişti:** Model, Raspberry Pi 5'in 4 çekirdeğine paralel iş parçacıkları (threads) atayarak %100 CPU ve NEON verimliliği ile çalışabilecek .param ve .bin formatlarına çevrildi. Bu FP32 ncnn modeli ayrı bir satır olarak ölçülür: mAP'i ONNX FP32 ile (ölçüm toleransı içinde) aynı çıkmalıdır, aksi halde dönüşüm hatalıdır.

#### 9.1.1 ncnn INT8: kendi kalibrasyonu (ncnn2table + ncnn2int8)

**Ne Yapılıyor:** FP32 ncnn modeli, ncnn'in kendi PTQ araçlarıyla INT8'e çevrilir. Kalibrasyon için **ONNX PTQ'da kullanılan 500 COCO görselinin aynısı** kullanılır. Böylece iki INT8 modelin kalibrasyon verisi ortak olur.

```bash
# pnnx çıktısı: yolox_ts.ncnn.param / yolox_ts.ncnn.bin (pnnx ile üretildiği için ncnnoptimize adımı atlanır)
find ../data/calib_500/ -type f > calib_list.txt

# Kalibrasyon tablosu. mean/norm/pixel değerleri YOLOX'un çıkarım ön işlemesiyle BİREBİR aynı olmalı.
# Kurulu YOLOX sürümünün ValTransform'u kontrol edilip doldurulacak (doğrulanmadı).
ncnn2table yolox_ts.ncnn.param yolox_ts.ncnn.bin calib_list.txt yolox.table \
  mean=[<yolox_mean>] norm=[<yolox_norm>] shape=[416,416,3] pixel=BGR thread=4 method=kl

# INT8 modeli üret
ncnn2int8 yolox_ts.ncnn.param yolox_ts.ncnn.bin yolox_int8.param yolox_int8.bin yolox.table
```

**Bilinmesi gerekenler:** `ncnn2int8` yalnızca Convolution, ConvolutionDepthWise, InnerProduct/Gemm ve birkaç dizi/dikkat katmanını INT8'e çevirir; diğer katmanlar FP32 kalır. ncnn belgeleri kalibrasyon için 5000'den fazla görsel önerir. 500 görsel, ONNX PTQ ile ortak veri kullanmak için bilinçli bir seçimdir ve tezde gerekçesiyle belirtilmelidir. INT8 ncnn modelinin mAP'i RPi üzerinde ayrıca ölçülür ve aynı ΔmAP ≤ 2,0 eşiği uygulanır. Tezde, RPi INT8 modelinin QDQ modelinin aynısı olmadığı, aynı FP32 ağırlıklar ve aynı kalibrasyon verisiyle ayrıca kuantize edildiği açıkça yazılmalıdır.

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

**Ek satır: TensorRT FP16.** INT8'in FP16'ya göre kazancını göstermek için canonical FP32 ONNX modeli FP16 motoruna da derlenir:

```bash
/usr/src/tensorrt/bin/trtexec \
--onnx=../artifacts/onnx/yolox_fp32.onnx \
--saveEngine=../artifacts/tensorrt/yolox_fp16.plan \
--shapes=images:1x3x416x416 \
--fp16 \
--warmUp=3000 \
--duration=60
```

**Karar (2026-10-03):** TensorRT INT8, TensorRT'nin kendi kalibratörüyle değil, **QDQ modelinden (explicit quantization)** üretilir. Böylece Karar 2'deki doğruluk kontrolü (PTQ/QAT) ile Jetson'da çalışan kuantizasyon aynı parametreleri kullanır. Kurulu TensorRT sürümünde implicit (kalibratörlü) INT8'in durumu ayrıca kontrol edilmelidir. **Risk:** ONNX Runtime'ın yerleştirdiği QDQ düğümleri TensorRT için en uygun yerleşim olmayabilir; bazı katmanlar FP32'de kalıp hız beklenenden düşük çıkabilir. Motorun mAP'i ve hızı Jetson'da ayrıca ölçülür. (`trtexec` hız ölçümü yalnız model penceresidir; uçtan uca FPS Karar 3'e göre ayrı ölçülür.)

**Neden Yapılıyor:** Modeldeki FP32 yükü, GPU'daki Tensor Çekirdeklerinde koşacak INT8 operasyonlarına dönüştürülmektedir. QDQ formatında hangi katmanların INT8'de çalışacağını TensorRT değil, QDQ düğümlerinin yerleşimi belirler. QDQ içermeyen katmanlar (ör. hassasiyet için dışarıda bırakılan tespit başlıkları) daha yüksek hassasiyette kalır. **Ne Değişti:** Jetson üzerinde çalışan yolox_int8.plan motoru oluşturuldu. Uçbirim çıktısında, cihazın sağladığı "Ortalama Gecikme" (Mean Latency) ve "Kare Sayısı/Saniye" (Throughput/FPS) raporu otomatik olarak görüntülendi. Hızlanma faktörü (Speedup), Stage-1 CPU ölçümüne göre 10 kata kadar varan farklılıklar gösterebilir.

## Onuncu Aşama: Güç Ölçümü, Telemetri ve Pareto Analizi

Uç yapay zekâda asıl başarı kriteri, sistemin en yüksek FPS'yi alması değil; harcanan birim enerji (Watt veya Joule) başına en fazla kareyi (frame) işleyebilmesidir.

**Ne Yapılıyor:** İki cihazın donanımsal telemetri yöntemleri birbirinden farklıdır. En doğru sonuç için dışarıdan (harici) bir güç ölçüm metodu kullanılmalıdır.

**Ölçüm protokolü (Karar 3):**

- **Ana güç ölçümü (iki cihaz):** Besleme girişine takılan USB-C güç ölçer ile **toplam kart gücü**. Cihazlar arası karşılaştırmada yalnız bu değer kullanılır. Güç ölçerin örnekleme hızı kaydedilir.
- **Idle güç:** Her ölçüm oturumunda, çıkarım başlamadan önce ayrıca ölçülür.
- **Enerji iki türlü raporlanır:**
  - $J/frame_{toplam} = W_{çıkarım} / FPS$ (cihazın gerçekte harcadığı)
  - $J/frame_{artımsal} = (W_{çıkarım} - W_{idle}) / FPS$ (yalnız çıkarımın payı; eski adıyla $P_{dynamic}$)
- **Pencere:** 100 çıkarım ısınma atılır → 60 sn kararlı ölçüm × 3 tekrar, ortalama ± std. Güç ve FPS **aynı pencerede** ölçülür.
- **Sabit tutulanlar:** Jetson güç modu (`nvpmodel`) ve `jetson_clocks` durumu, RPi CPU governor, soğutma (fan) koşulları. Ekran vb. çevre birimleri bağlı değil.
- **RAM:** Pik RAM (MB) `/usr/bin/time -v` ile ölçülür.
- **Jetson ek telemetrisi:** tegrastats, kart içi güç hatlarını (rail) **ayrı bir sütun** olarak kaydeder. USB-C ölçerle aynı şeyi ölçmediği için cihazlar arası karşılaştırmaya girmez:

  ```bash
  tegrastats --interval 100 --logfile ../logs/tegrastats_load.log
  ```

**Neden Yapılıyor:** Bir sistem saniyede 60 kare (FPS) işlerken 15 Watt güç tüketiyorsa, saniyede 30 kare işleyip 5 Watt tüketen bir sisteme göre "Enerji Verimliliği" açısından daha başarısızdır. Gerçek dünya uygulamalarında (bataryalı İHA'lar veya güneş enerjili kameralar) asıl metrik budur.

**Ne Değişti:** Elde edilen telemetri verileri sayesinde, hız ve güç tüketimi üzerinden matematiksel bir "Pareto Eğrisi" (Pareto Front) çizilmesi mümkün hale geldi.

### Çıktıların Karşılaştırmalı Tablosu

Tüm aşamalardan elde edilen veriler, bilimsel bir raporlama standardına (CSV) oturtulmalıdır. Her satır bir **cihaz × aşama × arka uç × hassasiyet × giriş boyutu × ölçüm** birleşimidir. Aşağıdaki değerler semboliktir; hiçbiri ölçülmüş değildir. 416×416 örnek değerdir; gerçek değer NAS'ın seçtiği çözünürlüktür.

| Cihaz | Aşama | Arka uç | Model tipi | Giriş | mAP | Gecikme (ms) | FPS (model) | FPS (uçtan uca) | Güç (W) | Idle (W) | FPS/W | J/frame (toplam) | J/frame (artımsal) | RAM (MB) | tegrastats (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RPi 5 | Stage-1 | ORT CPU | FP32 | 416×416 | mAP1 | Ölçüm 1 | F1 | E1 | W1 | I1 | F1/W1 | W1/F1 | (W1−I1)/F1 | R1 | — |
| Jetson Orin Nano | Stage-1 | ORT CPU | FP32 | 416×416 | mAP2 | Ölçüm 2 | F2 | E2 | W2 | I2 | F2/W2 | W2/F2 | (W2−I2)/F2 | R2 | T2 |
| RPi 5 | Stage-1 | ORT CPU | INT8 (QDQ) — ek | 416×416 | mAP1b | Ölçüm 1b | F1b | E1b | W1b | I1b | F1b/W1b | W1b/F1b | (W1b−I1b)/F1b | R1b | — |
| Jetson Orin Nano | Stage-1 | ORT CPU | INT8 (QDQ) — ek | 416×416 | mAP2b | Ölçüm 2b | F2b | E2b | W2b | I2b | F2b/W2b | W2b/F2b | (W2b−I2b)/F2b | R2b | T2b |
| RPi 5 | Stage-2 | ncnn | FP32 (dönüşüm doğrulaması) | 416×416 | mAP3a | Ölçüm 3a | F3a | E3a | W3a | I3a | F3a/W3a | W3a/F3a | (W3a−I3a)/F3a | R3a | — |
| RPi 5 | Stage-2 | ncnn | INT8 (ncnn2int8) | 416×416 | mAP3 | Ölçüm 3 | F3 | E3 | W3 | I3 | F3/W3 | W3/F3 | (W3−I3)/F3 | R3 | — |
| Jetson Orin Nano | Stage-2 | TensorRT | FP16 | 416×416 | mAP4a | Ölçüm 4a | F4a | E4a | W4a | I4a | F4a/W4a | W4a/F4a | (W4a−I4a)/F4a | R4a | T4a |
| Jetson Orin Nano | Stage-2 | TensorRT | INT8 (QDQ) | 416×416 | mAP4 | Ölçüm 4 | F4 | E4 | W4 | I4 | F4/W4 | W4/F4 | (W4−I4)/F4 | R4 | T4 |

Bu tablo aracılığıyla hesaplanan *Marjinal Verimlilik* formülü: $\mathrm{Speedup} = \frac{FPS_{Stage2}}{FPS_{Stage1}}$ şeklindedir. Salt yazılım/derleyici katkısını göstermek için **aynı cihazda ve aynı hassasiyette** karşılaştırılmalıdır (ör. RPi: ncnn FP32 ÷ ORT FP32). Hassasiyet de değişiyorsa (ör. TensorRT FP16 ÷ ORT FP32) hız artışı derleyici ile hassasiyetin birleşik etkisidir ve öyle raporlanmalıdır. Stage-2'de amaç cihazları birbiriyle yarıştırmak değil, her yöntemin kendi cihazındaki etkisini belgelemektir (Karar 5).

## On Birinci Aşama: Sürekli Öğrenme ve Canlı Sistem Entegrasyonu Vizyonu

Yukarıdaki 10 adımlık somut iş akışı tamamlandığında, elinizde donanıma duyarlı, budanmış, kuantize edilmiş ve hedef cihaza göre derlenmiş olağanüstü hızlı bir YOLOX-Nano modeli bulunur.

Ancak uç cihazlar, zaman içinde değişen saha koşullarına maruz kalır (Concept Drift - Kavram Kayması). Görüntü işleme esnasında, güvenilirlik skoru (confidence score) düşük olan görseller (low-confidence samples) Raspberry Pi veya Jetson cihazı üzerindeki yerel bir bellekte (buffer) toplanmalıdır. Bu veriler düzenli aralıklarla merkezi bulut sistemine aktarılmalı; model, NAS ve budama konfigürasyonu (mimari şeması) korunarak bu yeni verilerle merkezi sunucuda ince ayar (fine-tuning) işlemine tabi tutulmalıdır. Geliştirilmiş ve doğruluğu artırılmış model; yeniden ONNX formatına alınmalı, QDQ kuantizasyonundan geçmeli ve uzaktan güncelleme (OTA Deployment) aracılığıyla uç cihaza gönderilmelidir. Cihazın kendi kısıtlı işlemcisi üzerinde YOLOX eğitimini baştan başlatmak termal ve donanımsal olarak imkansız olduğundan, uç cihazlar veri toplayıcı ve çıkarım (inference) aracı olarak çalışmalı, eğitim yükü (training) merkezde bırakılmalıdır.

Bu rehberde sunulan iş akışı, laboratuvar ortamındaki prototiplerin gerçek saha donanımlarına (Edge AI) minimum performans kaybı ve maksimum enerji verimliliği ile aktarılabilmesi için somut, yenilenebilir ve teknik olarak güncellenebilir bir standart belirlemiştir.

#### Alıntılanan çalışmalar
