# LUT (gecikme tablosu) Jetson Orin Nano olcumu - adim adim

Gercek NAS (Karar 6), notebook 03b'nin girdisi. Script: `scripts/latency_lut.py`.
Jetson'da yalniz numpy + onnxruntime gerekir (torch gerekmez). N4 ortami (`~/n4env`) aynen kullanilir.

---------------------------------------------------------------------------------------------------

## 0) Ne, neden?

**Ne:** Super agin arama uzayindaki her parcanin (her depthwise katman x kernel 3/5/7 x 7 cozunurluk, her ek
CSP blogu, sabit govde) Jetson CPU'sunda tek tek gecikmesi olculur. Bu tablo (LUT = look-up table) sayesinde
03b'deki evrimsel arama, ~1000 alt agin her birini cihazda olcmeden gecikmesini toplama ile tahmin eder:

    T(alt ag) = a * [ sabit(res) + toplam_aktif_DW( dw(res, katman, k) - dw(res, katman, 3) ) + toplam_ek_blok(res) ] + b

**Neden dogrulama da var:** Mikro modellerin toplami CPU'da tam modeli birebir vermez (cagri yuku, onbellek).
Bu yuzden pakette ayrica 48 tam alt ag var (Nano x 7 cozunurluk + en buyuk + 40 rastgele). Bunlar da olculur;
LUT tahmini ile karsilastirilir (Spearman siralama uyumu rho, yuzde hata) ve a, b dogrusal duzeltmesi bulunur.

**Kapi (onceden kayitli, 03b manifesti):** rho >= 0,9 degilse arama BASLAMAZ, konusulur.
(PC'de deneme: rho 0,95, ortalama hata %3,9 - Jetson'da ayrica olculmeli.)

**Protokol (LUT icin kisaltilmis; secilen modellerin RAPORLANAN gecikmesi ayrica Karar 3 ile, 03c'de olculur):**
ORT CPUExecutionProvider, intra_op 4 thread, inter_op 1, sirali, ORT_ENABLE_ALL; mikro model: 30 isinma +
1 sn x 3 tekrar; tam model: 50 isinma + 3 sn x 3 tekrar; deger = tekrar medyanlarinin ortalamasi. Ogeler
tohumlu karisik sirayla olculur (isinan cihazin yavaslamasi tek bir gruba yuklenmez). Girdi rastgele tensor
(gecikme agirliga/icerige bagli degil; N4 gibi gercek goruntu gerekmez).

Paket: 713 oge = 658 mikro ONNX (630 DW + 28 blok) + 55 tam ONNX (7 sabit + 48 dogrulama), opset 13.

---------------------------------------------------------------------------------------------------

## 1) Jetson hazirligi (N4 ile ayni ayarlar)

Cihaz yeniden basladiysa saatleri tekrar sabitle:

    sudo nvpmodel -q
    sudo jetson_clocks
    sudo jetson_clocks --fan
    sudo jetson_clocks --show | tee ~/jetson_clocks_show_lut.txt
    sudo systemctl isolate multi-user.target
    uptime

Kontrol:
- `nvpmodel -q` MAXN_SUPER (2) gostermeli (N4/N4b ile ayni mod)
- uptime yuk ortalamasi ~0 olmali

Python ortami:

    conda deactivate
    source ~/n4env/bin/activate
    which python3

`which python3` ciktisi `~/n4env/bin/python3` olmali.

Repo:

    cd ~/edge_ai_tez/edge-ai-object-detection
    git pull
    git log --oneline -1

Commit `docs: LUT Jetson runbook` (bu dosyanin commit'i) veya daha yeni olmali.

---------------------------------------------------------------------------------------------------

## 2) Paket

Windows'ta paket: `C:\projects\edge_ai_workspace\artifacts\lut\nas_super1_lut.zip` (183 MB, git'te yok).
Windows'tan (PowerShell):

    scp C:\projects\edge_ai_workspace\artifacts\lut\nas_super1_lut.zip iqmine@<jetson-ip>:~/

Jetson'da:

    cd ~
    sha256sum nas_super1_lut.zip
    rm -rf nas_super1_lut
    unzip -q nas_super1_lut.zip
    ls nas_super1_lut

Kontrol:
- zip sha256 `deb18637ae7522c7a37d575a643a3443c531b9bad7b7656b69d68e6449a134d5`
- klasorde: `lut_index.json`, `dw/`, `block/`, `full/`
- Ayrica script olcumden once 713 dosyanin hepsinin sha256'sini kendisi dogrular ("paket dogrulandi: 713 dosya").

---------------------------------------------------------------------------------------------------

## 3) Olcum

tmux (SSH koparsa olcum durmasin):

    tmux new -s lut

(cik: Ctrl-b, sonra d · geri don: `tmux attach -t lut`)

### 3a. ADIM 1 - Smoke test (12 oge, 5 isinma + 0,3 sn; ~1 dk)

    conda deactivate
    source ~/n4env/bin/activate
    cd ~/edge_ai_tez/edge-ai-object-detection
    python3 scripts/latency_lut.py measure --pkg ~/nas_super1_lut --device jetson_orin_nano_cpu --smoke 2>&1 | tee ~/lut_smoke.log

Beklenen:

    paket dogrulandi: 713 dosya sha256 OK · {'cpu_governor': [...], 'cpu_count': 6, 'nvpmodel': '...MAXN_SUPER...'}
    [12/12] ... sn · son <oge> <ms> ms

Smoke ayri klasore yazar (`measured/jetson_orin_nano_cpu_smoke/`); tam olcumu etkilemez.
Hata cikarsa tam olcume GECME, ciktiyi ajana gonder.

### 3b. ADIM 2 - Tam olcum (~45-60 dk; PC'de 37 dk surdu)

    python3 scripts/latency_lut.py measure --pkg ~/nas_super1_lut --device jetson_orin_nano_cpu 2>&1 | tee ~/lut_full.log
    cp ~/jetson_clocks_show_lut.txt ~/nas_super1_lut/measured/jetson_orin_nano_cpu/

Konsolda her 20 ogede bir ilerleme satiri:

    [140/713] 512 sn · son dw|448|backbone.backbone.dark3.1.m.1.conv2.dconv.conv|5 0.0912 ms

Bitince son satir `[713/713] ...`.

### 3c. Flag'ler

| Flag | Anlami |
|---|---|
| `measure` | olcum alt komutu (build/fit ajanin bilgisayarinda) |
| `--pkg ~/nas_super1_lut` | zip'ten cikan klasor |
| `--device jetson_orin_nano_cpu` | sonuc klasorunun adi; smoke ve tam olcumde ayni ad (smoke `_smoke` ekler) |
| `--smoke` | kisa deneme, ayri klasor |

### 3d. Kesinti olursa

Ayni komutu tekrar calistir: olculmus ogeler (ayni sha256) atlanir, kaldigi yerden devam eder
(ara kayit her 20 ogede bir). Olcum ortasinda nvpmodel/governor degisirse script bilerek durur.
Cihaz yeniden basladiysa once 1) adimindaki jetson_clocks komutlarini tekrar calistir.
Olcum surerken Jetson'da baska is calistirma.

---------------------------------------------------------------------------------------------------

## 4) Ciktilar ve ajana ulastirma

### 4a. Ciktilar (paket klasorunun icinde)

    ~/nas_super1_lut/measured/
        jetson_orin_nano_cpu_smoke/
            session.json
            measurements.json      (12 oge)
        jetson_orin_nano_cpu/
            session.json           cihaz bilgisi, L4T, nvpmodel + governor, ORT/numpy surumleri, protokol, baslangic/bitis
            measurements.json      713 oge: ms, 3 tekrarin medyanlari, sha256, olcum ani sicakligi
            jetson_clocks_show_lut.txt

Loglar: `~/lut_smoke.log`, `~/lut_full.log`.

### 4b. Ajana ulastirma (yalniz kucuk JSON'lar; 183 MB'lik paket geri gelmez)

Windows'tan (PowerShell):

    mkdir C:\projects\edge_ai_workspace\results\nas_super1\lut -Force
    scp -r iqmine@<jetson-ip>:~/nas_super1_lut/measured C:\projects\edge_ai_workspace\results\nas_super1\lut\
    scp iqmine@<jetson-ip>:~/lut_smoke.log iqmine@<jetson-ip>:~/lut_full.log C:\projects\edge_ai_workspace\results\nas_super1\lut\

Sonra ajana "attim" de.

### 4c. Ajanin yapacagi (bilgi icin)

1. `latency_lut.py fit` (PC'de, torch gerekir): mikro olcumlerden LUT'u kurar, 48 tam alt agla dogrular,
   a, b duzeltmesini bulur → `lut.json` + `lut_validation.json` (rho, yuzde hata, satir satir tahmin/olcum).
2. Kapi: rho >= 0,9 mi? Sonucu sana gosterir.
3. `results/nas_super1/lut/jetson_orin_nano_cpu/` altina koyup commit/push eder; 03b bu dosyayi repodan
   okur (manifestine sha256'sini yazar).

RPi 5 gelince ayni akis `--device rpi5_cpu` ile tekrarlanir (seçim olcutu RPi; Karar 5).
