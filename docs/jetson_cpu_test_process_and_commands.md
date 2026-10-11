PS C:\Users\24040401110\Downloads> scp nas_r1b_n4_onnx.zip iqmine@192.168.1.130:~/
iqmine@192.168.1.130's password:
Permission denied, please try again.
iqmine@192.168.1.130's password:
nas_r1b_n4_onnx.zip                                                                   100%   63MB  11.4MB/s   00:05
PS C:\Users\24040401110\Downloads>

(n4env) iqmine@iqmine-jetson:~$     cd ~
(n4env) iqmine@iqmine-jetson:~$     sha256sum nas_r1b_n4_onnx.zip
-c SHA256SUMS
    ls inputs/e2429322e480394a0396c4dbad11354bbcaa4b5c7cfb6236ae748b4b2b47cfa8  nas_r1b_n4_onnx.zip
(n4env) iqmine@iqmine-jetson:~$     unzip nas_r1b_n4_onnx.zip
Archive:  nas_r1b_n4_onnx.zip
  inflating: nas_r1b_n4_onnx/hardswish_dfull_k3_s320.onnx
  inflating: nas_r1b_n4_onnx/silu_dfull_k3_s320.onnx
  inflating: nas_r1b_n4_onnx/silu_dfull_k5_s320.onnx
  inflating: nas_r1b_n4_onnx/silu_dreduced_k5_s320.onnx
  inflating: nas_r1b_n4_onnx/hardswish_dfull_k5_s416.onnx
  inflating: nas_r1b_n4_onnx/silu_dfull_k3_s416.onnx
  inflating: nas_r1b_n4_onnx/silu_dfull_k5_s416.onnx
  inflating: nas_r1b_n4_onnx/silu_dreduced_k5_s416.onnx
  inflating: nas_r1b_n4_onnx/hardswish_dfull_k3_s512.onnx
  inflating: nas_r1b_n4_onnx/hardswish_dfull_k5_s512.onnx
  inflating: nas_r1b_n4_onnx/silu_dfull_k3_s512.onnx
  inflating: nas_r1b_n4_onnx/silu_dfull_k5_s512.onnx
  inflating: nas_r1b_n4_onnx/silu_dreduced_k5_s512.onnx
  inflating: nas_r1b_n4_onnx/hardswish_dfull_k5_s320.onnx
  inflating: nas_r1b_n4_onnx/silu_dreduced_k3_s320.onnx
  inflating: nas_r1b_n4_onnx/hardswish_dfull_k3_s416.onnx
  inflating: nas_r1b_n4_onnx/relu_dfull_k5_s320.onnx
  inflating: nas_r1b_n4_onnx/relu_dfull_k5_s416.onnx
  inflating: nas_r1b_n4_onnx/relu_dfull_k5_s512.onnx
  inflating: nas_r1b_n4_onnx/inputs/input_s320.npy
  inflating: nas_r1b_n4_onnx/inputs/input_s416.npy
  inflating: nas_r1b_n4_onnx/inputs/input_s512.npy
  inflating: nas_r1b_n4_onnx/inputs/inputs.json
  inflating: nas_r1b_n4_onnx/n4_candidates.csv
  inflating: nas_r1b_n4_onnx/exports.csv
  inflating: nas_r1b_n4_onnx/manifest.json
  inflating: nas_r1b_n4_onnx/SHA256SUMS
(n4env) iqmine@iqmine-jetson:~$     cd nas_r1b_n4_onnx
(n4env) iqmine@iqmine-jetson:~/nas_r1b_n4_onnx$     sha256sum -c SHA256SUMS
hardswish_dfull_k3_s320.onnx: OK
silu_dfull_k3_s320.onnx: OK
silu_dfull_k5_s320.onnx: OK
silu_dreduced_k5_s320.onnx: OK
hardswish_dfull_k5_s416.onnx: OK
silu_dfull_k3_s416.onnx: OK
silu_dfull_k5_s416.onnx: OK
silu_dreduced_k5_s416.onnx: OK
hardswish_dfull_k3_s512.onnx: OK
hardswish_dfull_k5_s512.onnx: OK
silu_dfull_k3_s512.onnx: OK
silu_dfull_k5_s512.onnx: OK
silu_dreduced_k5_s512.onnx: OK
hardswish_dfull_k5_s320.onnx: OK
silu_dreduced_k3_s320.onnx: OK
hardswish_dfull_k3_s416.onnx: OK
relu_dfull_k5_s320.onnx: OK
relu_dfull_k5_s416.onnx: OK
relu_dfull_k5_s512.onnx: OK
inputs/input_s320.npy: OK
inputs/input_s416.npy: OK
inputs/input_s512.npy: OK
(n4env) iqmine@iqmine-jetson:~/nas_r1b_n4_onnx$     ls inputs/
input_s320.npy  input_s416.npy  input_s512.npy  inputs.json
(n4env) iqmine@iqmine-jetson:~/nas_r1b_n4_onnx$

(base) iqmine@iqmine-jetson:~/nas_r1b_n4_onnx$ conda deactivate
iqmine@iqmine-jetson:~/nas_r1b_n4_onnx$ source ~/n4env/bin/activate
(n4env) iqmine@iqmine-jetson:~/nas_r1b_n4_onnx$ which python3
/home/iqmine/n4env/bin/python3
(n4env) iqmine@iqmine-jetson:~/nas_r1b_n4_onnx$ cd ~/edge_ai_tez/edge-ai-object-detection
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ ls
AGENTS.md  exps  notebooks  project_information  README.md  results  scripts  YOLOX
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ OUT=results/nas_r1b/n4_latency
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ mkdir -p $OUT

(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4_onnx --device jetson_orin_nano_cpu --smoke --tegrastats 2>&1 | tee $OUT/smoke.log
jetson_orin_nano_cpu: 19 aday, sıra tohumu 0, en çok ≈6 dk · ayarlar {'cpu_governor': ['schedutil'], 'cpu_count': 6, 'nvpmodel': 'NV Power Mode: MAXN_SUPER\n2'} · yük (0.82177734375, 0.9169921875, 0.45654296875) · sıcaklık {'cpu-thermal': 38.656, 'gpu-thermal': 39.406, 'soc0-thermal': 38.843, 'soc1-thermal': 38.062, 'soc2-thermal': 39.031, 'tj-thermal': 39.406}
✔ 1/19 hardswish_dfull_k5_s512       48.22 ± 0.00 ms ·   20.7 FPS · p99 48.51 · 40.843 °C
  uyarı: 1 dk yük ortalaması 1.08 (arka planda iş var mı?)
✔ 2/19 silu_dfull_k5_s512            48.97 ± 0.00 ms ·   20.4 FPS · p99 49.26 · 40.843 °C
  uyarı: 1 dk yük ortalaması 1.31 (arka planda iş var mı?)
✔ 3/19 silu_dreduced_k3_s320         18.21 ± 0.00 ms ·   54.9 FPS · p99 18.34 · 40.906 °C
  uyarı: 1 dk yük ortalaması 1.93 (arka planda iş var mı?)
✔ 4/19 relu_dfull_k5_s512            39.80 ± 0.00 ms ·   25.1 FPS · p99 40.10 · 41.281 °C
  uyarı: 1 dk yük ortalaması 2.09 (arka planda iş var mı?)
✔ 5/19 hardswish_dfull_k3_s320       18.66 ± 0.00 ms ·   53.6 FPS · p99 18.75 · 41.562 °C
  uyarı: 1 dk yük ortalaması 2.17 (arka planda iş var mı?)
✔ 6/19 relu_dfull_k5_s320            17.62 ± 0.00 ms ·   56.7 FPS · p99 17.80 · 41.656 °C
  uyarı: 1 dk yük ortalaması 2.31 (arka planda iş var mı?)
✔ 7/19 silu_dfull_k3_s512            42.31 ± 0.00 ms ·   23.6 FPS · p99 42.51 · 41.843 °C
  uyarı: 1 dk yük ortalaması 2.45 (arka planda iş var mı?)
✔ 8/19 silu_dfull_k5_s320            22.15 ± 0.00 ms ·   45.1 FPS · p99 22.40 · 41.843 °C
  uyarı: 1 dk yük ortalaması 2.57 (arka planda iş var mı?)
✔ 9/19 silu_dreduced_k5_s320         21.20 ± 0.00 ms ·   47.1 FPS · p99 21.41 · 42.093 °C
  uyarı: 1 dk yük ortalaması 2.69 (arka planda iş var mı?)
✔ 10/19 silu_dfull_k3_s416            29.74 ± 0.00 ms ·   33.6 FPS · p99 30.04 · 42.156 °C
  uyarı: 1 dk yük ortalaması 2.79 (arka planda iş var mı?)
✔ 11/19 relu_dfull_k5_s416            27.96 ± 0.00 ms ·   35.8 FPS · p99 28.20 · 42.343 °C
  uyarı: 1 dk yük ortalaması 2.89 (arka planda iş var mı?)
✔ 12/19 hardswish_dfull_k5_s416       34.10 ± 0.00 ms ·   29.3 FPS · p99 34.22 · 42.437 °C
  uyarı: 1 dk yük ortalaması 3.06 (arka planda iş var mı?)
✔ 13/19 silu_dfull_k5_s416            34.58 ± 0.00 ms ·   28.9 FPS · p99 34.92 · 42.593 °C
  uyarı: 1 dk yük ortalaması 3.14 (arka planda iş var mı?)
✔ 14/19 silu_dreduced_k5_s416         33.37 ± 0.00 ms ·   30.0 FPS · p99 33.68 · 42.531 °C
  uyarı: 1 dk yük ortalaması 3.21 (arka planda iş var mı?)
✔ 15/19 hardswish_dfull_k3_s416       29.03 ± 0.00 ms ·   34.4 FPS · p99 29.19 · 42.843 °C
  uyarı: 1 dk yük ortalaması 3.27 (arka planda iş var mı?)
✔ 16/19 hardswish_dfull_k3_s512       41.15 ± 0.00 ms ·   24.3 FPS · p99 41.56 · 42.843 °C
  uyarı: 1 dk yük ortalaması 3.33 (arka planda iş var mı?)
✔ 17/19 silu_dfull_k3_s320            18.82 ± 0.00 ms ·   53.1 FPS · p99 19.02 · 42.687 °C
  uyarı: 1 dk yük ortalaması 3.38 (arka planda iş var mı?)
✔ 18/19 hardswish_dfull_k5_s320       22.05 ± 0.00 ms ·   45.3 FPS · p99 22.33 · 42.843 °C
  uyarı: 1 dk yük ortalaması 3.43 (arka planda iş var mı?)
✔ 19/19 silu_dreduced_k5_s512         47.39 ± 0.00 ms ·   21.1 FPS · p99 47.73 · 42.812 °C
19/19 aday ölçüldü → /home/iqmine/edge_ai_tez/edge-ai-object-detection/scripts/../results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/latency.csv

(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4_onnx --device jetson_orin_nano_cpu --tegrastats 2>&1 | tee -a $OUT/full.log
jetson_orin_nano_cpu: 19 aday, sıra tohumu 0, en çok ≈62 dk · ayarlar {'cpu_governor': ['schedutil'], 'cpu_count': 6, 'nvpmodel': 'NV Power Mode: MAXN_SUPER\n2'} · yük (1.57275390625, 1.59423828125, 0.8330078125) · sıcaklık {'cpu-thermal': 38.843, 'gpu-thermal': 39.687, 'soc0-thermal': 38.937, 'soc1-thermal': 38.531, 'soc2-thermal': 39.312, 'tj-thermal': 39.687}
  uyarı: 1 dk yük ortalaması 1.57 (arka planda iş var mı?)
✔ 1/19 hardswish_dfull_k5_s512       48.07 ± 0.09 ms ·   20.8 FPS · p99 48.52 · 43.312 °C
  uyarı: 1 dk yük ortalaması 3.89 (arka planda iş var mı?)
✔ 2/19 silu_dfull_k5_s512            49.06 ± 0.01 ms ·   20.4 FPS · p99 49.35 · 43.656 °C
  uyarı: 1 dk yük ortalaması 4.05 (arka planda iş var mı?)
✔ 3/19 silu_dreduced_k3_s320         18.38 ± 0.03 ms ·   54.4 FPS · p99 18.59 · 43.25 °C
  uyarı: 1 dk yük ortalaması 4.03 (arka planda iş var mı?)
✔ 4/19 relu_dfull_k5_s512            39.63 ± 0.01 ms ·   25.2 FPS · p99 39.95 · 43.281 °C
  uyarı: 1 dk yük ortalaması 4.00 (arka planda iş var mı?)
✔ 5/19 hardswish_dfull_k3_s320       18.81 ± 0.04 ms ·   53.1 FPS · p99 19.06 · 43.437 °C
  uyarı: 1 dk yük ortalaması 4.03 (arka planda iş var mı?)
✔ 6/19 relu_dfull_k5_s320            17.48 ± 0.03 ms ·   57.2 FPS · p99 17.71 · 43.093 °C
  uyarı: 1 dk yük ortalaması 4.02 (arka planda iş var mı?)
✔ 7/19 silu_dfull_k3_s512            42.44 ± 0.08 ms ·   23.6 FPS · p99 42.80 · 43.031 °C
  uyarı: 1 dk yük ortalaması 4.00 (arka planda iş var mı?)
✔ 8/19 silu_dfull_k5_s320            22.19 ± 0.02 ms ·   45.1 FPS · p99 22.46 · 42.937 °C
  uyarı: 1 dk yük ortalaması 3.99 (arka planda iş var mı?)
✔ 9/19 silu_dreduced_k5_s320         21.23 ± 0.01 ms ·   47.1 FPS · p99 21.48 · 43.406 °C
  uyarı: 1 dk yük ortalaması 4.24 (arka planda iş var mı?)
✔ 10/19 silu_dfull_k3_s416            29.70 ± 0.05 ms ·   33.7 FPS · p99 30.07 · 43.093 °C
  uyarı: 1 dk yük ortalaması 4.02 (arka planda iş var mı?)
✔ 11/19 relu_dfull_k5_s416            27.70 ± 0.00 ms ·   36.1 FPS · p99 28.01 · 43.156 °C
  uyarı: 1 dk yük ortalaması 4.02 (arka planda iş var mı?)
✔ 12/19 hardswish_dfull_k5_s416       34.12 ± 0.07 ms ·   29.3 FPS · p99 34.42 · 43.437 °C
  uyarı: 1 dk yük ortalaması 4.00 (arka planda iş var mı?)
✔ 13/19 silu_dfull_k5_s416            34.58 ± 0.07 ms ·   28.9 FPS · p99 34.93 · 43.281 °C
  uyarı: 1 dk yük ortalaması 4.06 (arka planda iş var mı?)
✔ 14/19 silu_dreduced_k5_s416         33.62 ± 0.04 ms ·   29.7 FPS · p99 34.03 · 43.218 °C
  uyarı: 1 dk yük ortalaması 4.01 (arka planda iş var mı?)
✔ 15/19 hardswish_dfull_k3_s416       28.85 ± 0.00 ms ·   34.6 FPS · p99 29.17 · 43.625 °C
  uyarı: 1 dk yük ortalaması 4.00 (arka planda iş var mı?)
✔ 16/19 hardswish_dfull_k3_s512       41.32 ± 0.12 ms ·   24.2 FPS · p99 41.73 · 43.562 °C
  uyarı: 1 dk yük ortalaması 4.00 (arka planda iş var mı?)
✔ 17/19 silu_dfull_k3_s320            18.77 ± 0.00 ms ·   53.2 FPS · p99 18.91 · 43.187 °C
  uyarı: 1 dk yük ortalaması 4.00 (arka planda iş var mı?)
✔ 18/19 hardswish_dfull_k5_s320       21.86 ± 0.10 ms ·   45.7 FPS · p99 22.06 · 43.281 °C
  uyarı: 1 dk yük ortalaması 4.08 (arka planda iş var mı?)
✔ 19/19 silu_dreduced_k5_s512         47.80 ± 0.01 ms ·   20.9 FPS · p99 48.02 · 43.156 °C
19/19 aday ölçüldü → /home/iqmine/edge_ai_tez/edge-ai-object-detection/scripts/../results/nas_r1b/n4_latency/jetson_orin_nano_cpu/latency.csv