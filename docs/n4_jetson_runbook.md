# N4 Jetson Orin Nano gecikme olcumu - adim adim

N4 = aday mimarilerin cihazda ONNX Runtime CPU gecikme olcumu (Jetson on olcumu; RPi 5 gelince ayni akis).
Script: `scripts/n4_latency.py`. Jetson'da yalniz numpy + onnxruntime gerekir (torch gerekmez).

---------------------------------------------------------------------------------------------------

## 1) Jetson ayarlari ve komutlari

### 1a. Sistem bilgisi (ciktiyi ajana gonder)

    cat /etc/nv_tegra_release
    python3 --version
    sudo nvpmodel -q --verbose | head -20
    grep -n "POWER_MODEL" /etc/nvpmodel.conf

Son komut guc modlarini ve ID'lerini listeler.

### 1b. Guc modu (nvpmodel)

Mod ID'si JetPack surumune gore degisir:
- JetPack 6.2: genelde 0 = 15W, 1 = 7W, 2 = MAXN_SUPER
- Eski surumler: en yuksek mod 0 = 15W

Listedeki EN YUKSEK modu sec:

    sudo nvpmodel -m <ID>
    sudo nvpmodel -q

Reboot isterse YES de.

### 1c. Saatleri sabitle + fan tam devir

jetson_clocks = CPU/GPU/bellek saatlerini en yuksege kilitler (olcumde frekans oynamaz).
Fan tam devir = isinmadan dolayi yavaslama (thermal throttling) olmaz.

    sudo jetson_clocks
    sudo jetson_clocks --fan
    sudo jetson_clocks --show | tee ~/jetson_clocks_show.txt

DIKKAT: Reboot sonrasi jetson_clocks sifirlanir (nvpmodel kalir). Script bunu fark etmez.
Cihaz yeniden baslarsa devam etmeden once `sudo jetson_clocks` ve `sudo jetson_clocks --fan` TEKRAR calistir.

### 1d. Arka plan yukunu azalt (onerilir, SSH ile bagliysan)

    sudo systemctl isolate multi-user.target
    uptime

uptime'daki yuk ortalamasi ~0 olmali (script 1,0 ustunde uyari verir).
Masaustunu geri acmak icin: `sudo systemctl isolate graphical.target`

### 1e. Python ortami

    sudo apt install -y python3-venv tmux unzip git
    python3 -m venv ~/n4env
    conda deactivate
    source ~/n4env/bin/activate
    pip install --upgrade pip
    pip install numpy onnxruntime
    python -c "import onnxruntime as o, numpy as n; print(o.__version__, n.__version__, o.get_available_providers())"

Son satirin ciktisinda `CPUExecutionProvider` gorunmeli.

Bu Jetson'da (2026-10-12): python3 = conda base'in Python 3.13.9'u; venv onun uzerine kurulu, calisiyor.
Yeni bir terminal/tmux penceresinde conda base kendiliginden acilir. Her seferinde:

    conda deactivate
    source ~/n4env/bin/activate
    which python3

`which python3` ciktisi `~/n4env/bin/python3` olmali (/home/iqmine/n4env/bin/python3).

---------------------------------------------------------------------------------------------------

## 2) Hangi dosyalar + calistirma akisi + flag'ler

### 2a. Jetson'a IKI sey lazim

(1) REPO - script, yanindaki nas_search.py ve nas_export.py'yi kullanir; tek dosya yetmez.
Bu Jetson'da repo yolu: `~/edge_ai_tez/edge-ai-object-detection` (zaten klonlu, yalniz guncelle):

    cd ~/edge_ai_tez/edge-ai-object-detection
    git pull
    git log --oneline -1

Ilk kez kuruluyorsa: `mkdir -p ~/edge_ai_tez && cd ~/edge_ai_tez && git clone https://github.com/Bey3fendi/edge-ai-object-detection.git`

Commit 3794a47 veya daha yeni olmali.

(2) PAKET - Drive: `edge_ai/artifacts/onnx/nas_r1b_n4_onnx.zip` (66,6 MB). Windows'tan (PowerShell):

    scp nas_r1b_n4_onnx.zip iqmine@<jetson-ip>:~/

Jetson'da:

    cd ~
    sha256sum nas_r1b_n4_onnx.zip
    unzip nas_r1b_n4_onnx.zip
    cd nas_r1b_n4_onnx
    sha256sum -c SHA256SUMS
    ls inputs/

Kontrol:
- zip sha256 `e2429322` ile baslamali (`faeebfcc` ile baslayan ESKI ve gecersiz)
- `sha256sum -c` her satirda OK
- inputs/ icinde: input_s320.npy, input_s416.npy, input_s512.npy

### 2b. tmux (SSH koparsa olcum durmasin)

- Yeni oturum: `tmux new -s n4`
- Oturumdan cik (olcum surer): Ctrl-b, sonra d
- Geri don: `tmux attach -t n4`

### 2c. ADIM 1 - Smoke test (10 isinma + 5 sn x 1, en fazla ~6 dk)

    conda deactivate
    source ~/n4env/bin/activate
    which python3
    cd ~/edge_ai_tez/edge-ai-object-detection
    OUT=results/nas_r1b/n4_latency
    mkdir -p $OUT
    python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4_onnx --device jetson_orin_nano_cpu --smoke --tegrastats 2>&1 | tee $OUT/smoke.log

Beklenen: 19 satirin hepsi isaretli (✔) ve sonda `19/19 aday olculdu`.
Hata cikarsa tam olcume GECME, ciktiyi ajana gonder.

### 2d. ADIM 2 - Tam olcum (Karar 3: 100 isinma, 60 sn x 3, 4 thread; ~62 dk)

    python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4_onnx --device jetson_orin_nano_cpu --tegrastats 2>&1 | tee -a $OUT/full.log
    cp ~/jetson_clocks_show.txt $OUT/jetson_orin_nano_cpu/

### 2e. Flag'ler

| Flag | Anlami |
|---|---|
| `--pkg ~/nas_r1b_n4_onnx` | zip'ten cikan klasor |
| `--device jetson_orin_nano_cpu` | sonuc klasorunun adi; smoke ve tam olcumde AYNI olmali |
| `--smoke` | kisa deneme; ayri `_smoke` klasorune yazar |
| `--tegrastats` | Jetson'un kendi guc olcumu (VDD_IN) ayri sutun; onerilir. RPi karsilastirmasina girmez, asil guc USB-C olcerle (Karar 3) |
| `--only id1,id2` | yalniz secilen adaylar; normalde gerekmez |

### 2f. Kesinti olursa

Ayni komutu tekrar calistir: olculmus adaylar atlanir, kaldigi yerden devam eder.
Olcum sirasinda nvpmodel/governor degisirse script bilerek durur (karisik olcum olmasin).
Olcum surerken Jetson'da baska is calistirma.

---------------------------------------------------------------------------------------------------

## 3) Ciktilar ve ajana ulastirma

### 3a. Ciktilar nereye yazilir (repo icinde, otomatik)

    results/nas_r1b/n4_latency/
        smoke.log
        full.log
        jetson_orin_nano_cpu_smoke/
            manifest.json
            runs/<aday>.json      (19 dosya)
            latency.csv
        jetson_orin_nano_cpu/
            manifest.json
            runs/<aday>.json      (19 dosya)
            latency.csv
            jetson_clocks_show.txt

### 3b. Dosyalarin icerigi

- manifest.json: protokol, ORT/numpy surumleri, nvpmodel + governor, L4T, kart modeli, paket SHA'lari, repo commit'i
- runs/<aday>.json: her tekrar icin ortalama/medyan/p90/p99/min/max ms, FPS, bas/son sicaklik, CPU MHz, tegrastats mW, RAM tepe
- latency.csv: aday basina 1 satir ozet: mAP, delta (kontrole gore), GFLOPs, gecikme ort +- std, p99, FPS, RAM, max sicaklik, VDD_IN

Konsolda her aday icin ornek satir:

    ✔ 7/19 silu_dfull_k3_s416   48.12 ± 0.31 ms ·  20.8 FPS · p99 49.9 · 52.1 °C

### 3c. Ajana ulastirma

YOL A (onerilen): klasoru Windows'ta ayni yola kopyala (PowerShell):

    scp -r iqmine@<jetson-ip>:~/edge_ai_tez/edge-ai-object-detection/results/nas_r1b/n4_latency C:\projects\edge_ai_workspace\results\nas_r1b\

Sonra ajana "attim" de; oradan okur, commit/push birlikte yapilir. Jetson'da GitHub girisi gerekmez.

YOL B: Jetson'dan commit + push, ajan pull eder (Jetson'da GitHub kimlik bilgisi gerekir).

---------------------------------------------------------------------------------------------------

## 4) N4b - ReLU k3 gecikmesi (2. tur oncesi, Furkan 2026-10-12)

Amac: 02d'deki uzun egitimden ONCE relu_dfull_k3'un gecikmesini olcmek. Kontrol (silu_dfull_k3) ayni
oturumda yeniden olculur (anchor): ReLU/SiLU orani oturumlar arasi kaymadan bagimsiz olur.
6 aday (relu_dfull_k3 + silu_dfull_k3, 320/416/512). Tam olcum ~20 dk.

### 4a. Colab: paketi uret (notebook 02c, CPU runtime yeterli)

`notebooks/02c_relu_k3_onnx_disa_aktarma.ipynb` hucre 1-9. Sonunda Drive'da:
`edge_ai/artifacts/onnx/nas_r1b_n4b_onnx.zip` (hucre 9 sha256'yi yazar; not al).
Hucre 9 ayrica gecikme girdilerinin N4'tekilerle ayni oldugunu denetler ("N4 girdileriyle ayni: True").

### 4b. Jetson hazirligi

Cihaz yeniden basladiysa saatleri tekrar sabitle (1c), sonra:

    sudo nvpmodel -q
    sudo jetson_clocks
    sudo jetson_clocks --fan
    sudo systemctl isolate multi-user.target
    conda deactivate
    source ~/n4env/bin/activate
    cd ~/edge_ai_tez/edge-ai-object-detection
    git pull
    git log --oneline -1

`nvpmodel -q` MAXN_SUPER (2) gostermeli (N4 ile ayni mod).

Paketi gonder (Windows PowerShell):

    scp .\nas_r1b_n4b_onnx.zip iqmine@<jetson-ip>:~/

Jetson'da:

    cd ~
    sha256sum nas_r1b_n4b_onnx.zip
    rm -rf nas_r1b_n4b_onnx
    unzip nas_r1b_n4b_onnx.zip
    cd nas_r1b_n4b_onnx && sha256sum -c SHA256SUMS && ls inputs/

Beklenen: zip sha256 02c hucre 9'un yazdigiyla ayni; 9 satir OK (6 ONNX + 3 npy).

### 4c. Olcum (--out ile ayri klasor; N4 sonuclarina dokunmaz)

    cd ~/edge_ai_tez/edge-ai-object-detection
    OUT=results/nas_r1b/n4b_latency
    mkdir -p $OUT
    python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4b_onnx --device jetson_orin_nano_cpu --out $OUT --smoke --tegrastats 2>&1 | tee $OUT/smoke.log
    python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4b_onnx --device jetson_orin_nano_cpu --out $OUT --tegrastats 2>&1 | tee -a $OUT/full.log

Smoke 6/6 olculduyse tam olcume gec. Yuk ortalamasi uyarisi artik yalniz 5,0 ustunde cikar
(onceki ~4,0 uyarilari, bir onceki adayin 4 thread'lik olcumuydu).

### 4d. Ciktilari ajana ulastir

    scp -r iqmine@<jetson-ip>:~/edge_ai_tez/edge-ai-object-detection/results/nas_r1b/n4b_latency C:\projects\edge_ai_workspace\results\nas_r1b\

Sonra ajana "attim" de. Beklenti: ReLU k3, ayni cozunurlukteki SiLU k3'ten ~%20 hizli (N4'te ReLU k5 /
SiLU k5 = %19-21). Kazanc beklenen duzeydeyse notebook 02d (tam COCO, iki kol) baslatilir.
