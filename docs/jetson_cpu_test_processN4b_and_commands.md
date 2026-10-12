iqmine@iqmine-jetson:~$ sudo nvpmodel -q
[sudo] password for iqmine:
NV Power Mode: MAXN_SUPER
2
iqmine@iqmine-jetson:~$ sudo jetson_clocks
iqmine@iqmine-jetson:~$ sudo jetson_clocks --fan
iqmine@iqmine-jetson:~$ sudo systemctl isolate multi-user.target
iqmine@iqmine-jetson:~$ sudo jetson_clocks
    sudo jetson_clocks --faniqmine@iqmine-jetson:~$
iqmine@iqmine-jetson:~$ sudo jetson_clocks
iqmine@iqmine-jetson:~$ sudo jetson_clocks --fan
iqmine@iqmine-jetson:~$ conda deactivate
iqmine@iqmine-jetson:~$ source ~/n4env/bin/activate
(n4env) iqmine@iqmine-jetson:~$ cd ~/edge_ai_tez/edge-ai-object-detection
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ git pull
remote: Enumerating objects: 106, done.
remote: Counting objects: 100% (106/106), done.
remote: Compressing objects: 100% (34/34), done.
remote: Total 97 (delta 63), reused 96 (delta 62), pack-reused 0 (from 0)
Unpacking objects: 100% (97/97), 74.77 KiB | 1.02 MiB/s, done.
From https://github.com/Bey3fendi/edge-ai-object-detection
   5f2a93a..34b326e  main       -> origin/main
Updating 5f2a93a..34b326e
error: The following untracked working tree files would be overwritten by merge:
        results/nas_r1b/n4_latency/full.log
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/latency.csv
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/manifest.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k3_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/latency.csv
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/manifest.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s512.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k3_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s320.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s416.json
        results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s512.json
        results/nas_r1b/n4_latency/smoke.log
Please move or remove them before you merge.
Aborting
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ mkdir -p ~/n4_yedek && mv results/nas_r1b/n4_latency ~/n4_yedek/
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ git pull
Updating 5f2a93a..34b326e
Fast-forward
 docs/jetson_cpu_test_process_and_commands.md                                            | 160 +++++++++++++++++++
 docs/n4_jetson_runbook.md                                                               | 247 ++++++++++++++++++++++++++++
 exps/nas/nano_macro_long.py                                                             |  77 +++++++++
 notebooks/02c_relu_k3_onnx_disa_aktarma.ipynb                                           | 304 +++++++++++++++++++++++++++++++++++
 notebooks/02d_nas_tur2_relu_tam_coco.ipynb                                              | 542 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k3_s320.json                         |  88 ++++++++++
 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k3_s416.json                         |  88 ++++++++++
 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k3_s512.json                         |  88 ++++++++++
 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k5_s320.json                         |  88 ++++++++++
 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k5_s416.json                         |  88 ++++++++++
 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k5_s512.json                         |  88 ++++++++++
 results/nas_r1b/n3_onnx/candidates/relu_dfull_k5_s320.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/relu_dfull_k5_s416.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/relu_dfull_k5_s512.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dfull_k3_s320.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dfull_k3_s416.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dfull_k3_s512.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dfull_k5_s320.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dfull_k5_s416.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dfull_k5_s512.json                              |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k3_s320.json                           |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k5_s320.json                           |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k5_s416.json                           |  87 ++++++++++
 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k5_s512.json                           |  87 ++++++++++
 results/nas_r1b/n3_onnx/exports.csv                                                     |  20 +++
 results/nas_r1b/n3_onnx/manifest.json                                                   | 155 ++++++++++++++++++
 results/nas_r1b/n4_latency/full.log                                                     |  40 +++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/jetson_system_info.md                   | 108 +++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/latency.csv                             |  20 +++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/manifest.json                           | 142 ++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s320.json       | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s416.json       | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s512.json       | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s320.json       | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s416.json       | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s512.json       | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s320.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s416.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s512.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s320.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s416.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s512.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s320.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s416.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s512.json            | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k3_s320.json         | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s320.json         | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s416.json         | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s512.json         | 157 ++++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/latency.csv                       |  20 +++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/manifest.json                     | 142 ++++++++++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s320.json |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s416.json |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s512.json |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s320.json |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s416.json |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s512.json |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s320.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s416.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s512.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s320.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s416.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s512.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s320.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s416.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s512.json      |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k3_s320.json   |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s320.json   |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s416.json   |  73 +++++++++
 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s512.json   |  73 +++++++++
 results/nas_r1b/n4_latency/smoke.log                                                    |  39 +++++
 results/nas_r1b/n4b_candidates.csv                                                      |   7 +
 scripts/n4_latency.py                                                                   |   5 +-
 scripts/nas_export.py                                                                   |  26 ++-
 scripts/nas_long.py                                                                     |  84 ++++++++++
 75 files changed, 8164 insertions(+), 3 deletions(-)
 create mode 100644 docs/jetson_cpu_test_process_and_commands.md
 create mode 100644 docs/n4_jetson_runbook.md
 create mode 100644 exps/nas/nano_macro_long.py
 create mode 100644 notebooks/02c_relu_k3_onnx_disa_aktarma.ipynb
 create mode 100644 notebooks/02d_nas_tur2_relu_tam_coco.ipynb
 create mode 100644 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k3_s320.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k3_s416.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k3_s512.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/hardswish_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/relu_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/relu_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/relu_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dfull_k3_s320.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dfull_k3_s416.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dfull_k3_s512.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k3_s320.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k5_s320.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k5_s416.json
 create mode 100644 results/nas_r1b/n3_onnx/candidates/silu_dreduced_k5_s512.json
 create mode 100644 results/nas_r1b/n3_onnx/exports.csv
 create mode 100644 results/nas_r1b/n3_onnx/manifest.json
 create mode 100644 results/nas_r1b/n4_latency/full.log
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/jetson_system_info.md
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/latency.csv
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/manifest.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k3_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/hardswish_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/relu_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k3_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k3_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu/runs/silu_dreduced_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/latency.csv
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/manifest.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k3_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/hardswish_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/relu_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k3_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dfull_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k3_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s320.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s416.json
 create mode 100644 results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/runs/silu_dreduced_k5_s512.json
 create mode 100644 results/nas_r1b/n4_latency/smoke.log
 create mode 100644 results/nas_r1b/n4b_candidates.csv
 create mode 100644 scripts/nas_long.py
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ diff -r ~/n4_yedek/n4_latency results/nas_r1b/n4_latency && echo "AYNI"
Only in results/nas_r1b/n4_latency/jetson_orin_nano_cpu: jetson_system_info.md
diff -r /home/iqmine/n4_yedek/n4_latency/jetson_orin_nano_cpu/latency.csv results/nas_r1b/n4_latency/jetson_orin_nano_cpu/latency.csv
1,20c1,20
< candidate_id,tier,selectable,arch_id,act,depth,dw_k,resolution,map5095,delta_vs_control,gflops,latency_ms_mean,latency_ms_std,latency_ms_median,latency_ms_p90,latency_ms_p99,fps_model,n_inferences,rss_peak_mb,temp_c_max,tegra_vdd_in_mw,onnx_sha256,measured_utc
< hardswish_dfull_k3_s320,budget,True,hardswish_dfull_k3,hardswish,full,3,320,20.5,-0.9,0.66,18.807,0.04,18.792,18.887,19.058,53.15,9569,86.5,43.437,9353.1,6d56b7aecd3cdb199a6675a1486b70a0fc7d6fae0ba2f2faca440b7bbf736985,2026-10-11T22:33:19+00:00
< silu_dfull_k3_s320,budget,True,silu_dfull_k3,silu,full,3,320,21.4,0.0,0.66,18.775,0.003,18.771,18.819,18.912,53.25,9587,77.5,43.187,9172.7,627ae4a457d4ca962526d94b45897022c42688b74c01fbefd0ba77ad74033819,2026-10-11T23:09:59+00:00
< silu_dfull_k5_s320,budget,True,silu_dfull_k5,silu,full,5,320,21.6,0.2,0.7,22.186,0.02,22.177,22.272,22.465,45.06,8112,80.3,42.937,9166.2,e1a7f7781e202b3db68bdcfce9c45306c5eaae1df05a35dfc0c197e7e38c5443,2026-10-11T22:42:28+00:00
< silu_dreduced_k5_s320,budget,True,silu_dreduced_k5,silu,reduced,5,320,20.7,-0.7,0.68,21.235,0.015,21.23,21.281,21.476,47.08,8475,77.6,43.406,9180.0,9e2ee05ce4bf6ca6a7a93562c872b2d722558309a82595d32b51dcf93e899199,2026-10-11T22:45:31+00:00
< hardswish_dfull_k5_s416,budget,True,hardswish_dfull_k5,hardswish,full,5,416,23.5,-1.0,1.18,34.116,0.069,34.096,34.215,34.421,29.3,5276,87.4,43.437,9404,b45c9cb4e179d27c642e17ffbacd29f987b773d01582bb916a77d83e3b094c1a,2026-10-11T22:54:41+00:00
< silu_dfull_k3_s416,budget,True,silu_dfull_k3,silu,full,3,416,24.5,0.0,1.11,29.698,0.053,29.677,29.786,30.065,33.66,6061,83.5,43.093,9322.6,858d4b45de60c4b63f5e82ed5e40387e5a79fc9fcfa0d7ecaab74bab6dbbf8d5,2026-10-11T22:48:34+00:00
< silu_dfull_k5_s416,budget,True,silu_dfull_k5,silu,full,5,416,24.9,0.4,1.18,34.579,0.067,34.569,34.627,34.926,28.91,5205,91.8,43.281,9270.8,7741ccc056c2aa36c9ff3e1c6d1f1388743749995441c507f189f48d3458dd1e,2026-10-11T22:57:45+00:00
< silu_dreduced_k5_s416,budget,True,silu_dreduced_k5,silu,reduced,5,416,23.5,-1.0,1.15,33.617,0.044,33.578,33.759,34.031,29.74,5355,87.5,43.218,9270.8,3d8113122db8aaf4d63c4185d104cb5dac5afe19610a1e9e72948b48b697600c,2026-10-11T23:00:49+00:00
< hardswish_dfull_k3_s512,budget,True,hardswish_dfull_k3,hardswish,full,3,512,24.7,-1.0,1.69,41.323,0.125,41.31,41.469,41.732,24.2,4357,103.6,43.562,9561.1,72896e3d1db65c19966a1353a531c81544f6a45eea56945bf74844e1a844d788,2026-10-11T23:06:57+00:00
< hardswish_dfull_k5_s512,budget,True,hardswish_dfull_k5,hardswish,full,5,512,25.0,-0.7,1.79,48.066,0.088,48.045,48.255,48.523,20.8,3745,101.4,43.312,9484.5,ee2f16002e2fb3b1688c8a54221e6219f55a05b1a1766e00a45f368d206f7b2d,2026-10-11T22:21:04+00:00
< silu_dfull_k3_s512,budget,True,silu_dfull_k3,silu,full,3,512,25.7,0.0,1.69,42.439,0.082,42.425,42.559,42.803,23.56,4242,97.3,43.031,9340.9,996412201ec2bcd37e7bc4a772235a4697a194cf800a88bdbfc5eec8e2858402,2026-10-11T22:39:26+00:00
< silu_dfull_k5_s512,budget,True,silu_dfull_k5,silu,full,5,512,26.2,0.5,1.79,49.064,0.008,49.057,49.146,49.355,20.38,3669,96.7,43.656,9319.3,052b1df9da19f84e7cb5133134d3b01dfac2193ede44e74b912d709e43ee8e01,2026-10-11T22:24:10+00:00
< silu_dreduced_k5_s512,budget,True,silu_dreduced_k5,silu,reduced,5,512,24.7,-1.0,1.74,47.802,0.009,47.798,47.903,48.025,20.92,3766,99.7,43.156,9285.4,8872e2cf6cc824650fdd854ed3ea3c3c01117e19e34781adb650d3e063568a32,2026-10-11T23:16:07+00:00
< hardswish_dfull_k5_s320,noise_margin,False,hardswish_dfull_k5,hardswish,full,5,320,20.3,-1.1,0.7,21.858,0.099,21.852,21.93,22.06,45.74,8234,86.2,43.281,9292.5,9d0a662230737f756839cff88c5e45b43b7444e80935cd17117468ed950c92b5,2026-10-11T23:13:02+00:00
< silu_dreduced_k3_s320,noise_margin,False,silu_dreduced_k3,silu,reduced,3,320,20.3,-1.1,0.64,18.382,0.028,18.381,18.434,18.587,54.38,9790,79.5,43.25,9205,97e0f5bc15d70982622695d4b0fd65b30211bb847f0ae0bb8ea5e3ecf1115593,2026-10-11T22:27:12+00:00
< hardswish_dfull_k3_s416,noise_margin,False,hardswish_dfull_k3,hardswish,full,3,416,23.3,-1.2,1.11,28.848,0.003,28.843,28.895,29.166,34.65,6240,86.5,43.625,9443.8,aa5810cc163bd498130557ac7229b45121659140635dbc658dc900624e476f6b,2026-10-11T23:03:52+00:00
< relu_dfull_k5_s320,reference,False,relu_dfull_k5,relu,full,5,320,17.1,-4.3,0.7,17.477,0.025,17.463,17.552,17.711,57.2,10297,78.4,43.093,9166,f2346d79d8eaf106213406bc2b4d9b9bb34285976dbc1408f3310efda2588a04,2026-10-11T22:36:21+00:00
< relu_dfull_k5_s416,reference,False,relu_dfull_k5,relu,full,5,416,19.6,-4.9,1.18,27.702,0.003,27.696,27.743,28.01,36.09,6497,87.6,43.156,9269.1,992453e1e9a2a8de9e02f29d3cb252c6389bbe47064fee91185a5d9705ab961a,2026-10-11T22:51:37+00:00
< relu_dfull_k5_s512,reference,False,relu_dfull_k5,relu,full,5,512,21.3,-4.4,1.79,39.632,0.014,39.623,39.727,39.945,25.23,4543,110.1,43.281,9284.4,16bc53527e66e2931f42eac1ae1d549f27a39797700143adb3970f95788ca857,2026-10-11T22:30:16+00:00
---
> candidate_id,tier,selectable,arch_id,act,depth,dw_k,resolution,map5095,delta_vs_control,gflops,latency_ms_mean,latency_ms_std,latency_ms_median,latency_ms_p90,latency_ms_p99,fps_model,n_inferences,rss_peak_mb,temp_c_max,tegra_vdd_in_mw,onnx_sha256,measured_utc
> hardswish_dfull_k3_s320,budget,True,hardswish_dfull_k3,hardswish,full,3,320,20.5,-0.9,0.66,18.807,0.04,18.792,18.887,19.058,53.15,9569,86.5,43.437,9353.1,6d56b7aecd3cdb199a6675a1486b70a0fc7d6fae0ba2f2faca440b7bbf736985,2026-10-11T22:33:19+00:00
> silu_dfull_k3_s320,budget,True,silu_dfull_k3,silu,full,3,320,21.4,0.0,0.66,18.775,0.003,18.771,18.819,18.912,53.25,9587,77.5,43.187,9172.7,627ae4a457d4ca962526d94b45897022c42688b74c01fbefd0ba77ad74033819,2026-10-11T23:09:59+00:00
> silu_dfull_k5_s320,budget,True,silu_dfull_k5,silu,full,5,320,21.6,0.2,0.7,22.186,0.02,22.177,22.272,22.465,45.06,8112,80.3,42.937,9166.2,e1a7f7781e202b3db68bdcfce9c45306c5eaae1df05a35dfc0c197e7e38c5443,2026-10-11T22:42:28+00:00
> silu_dreduced_k5_s320,budget,True,silu_dreduced_k5,silu,reduced,5,320,20.7,-0.7,0.68,21.235,0.015,21.23,21.281,21.476,47.08,8475,77.6,43.406,9180.0,9e2ee05ce4bf6ca6a7a93562c872b2d722558309a82595d32b51dcf93e899199,2026-10-11T22:45:31+00:00
> hardswish_dfull_k5_s416,budget,True,hardswish_dfull_k5,hardswish,full,5,416,23.5,-1.0,1.18,34.116,0.069,34.096,34.215,34.421,29.3,5276,87.4,43.437,9404,b45c9cb4e179d27c642e17ffbacd29f987b773d01582bb916a77d83e3b094c1a,2026-10-11T22:54:41+00:00
> silu_dfull_k3_s416,budget,True,silu_dfull_k3,silu,full,3,416,24.5,0.0,1.11,29.698,0.053,29.677,29.786,30.065,33.66,6061,83.5,43.093,9322.6,858d4b45de60c4b63f5e82ed5e40387e5a79fc9fcfa0d7ecaab74bab6dbbf8d5,2026-10-11T22:48:34+00:00
> silu_dfull_k5_s416,budget,True,silu_dfull_k5,silu,full,5,416,24.9,0.4,1.18,34.579,0.067,34.569,34.627,34.926,28.91,5205,91.8,43.281,9270.8,7741ccc056c2aa36c9ff3e1c6d1f1388743749995441c507f189f48d3458dd1e,2026-10-11T22:57:45+00:00
> silu_dreduced_k5_s416,budget,True,silu_dreduced_k5,silu,reduced,5,416,23.5,-1.0,1.15,33.617,0.044,33.578,33.759,34.031,29.74,5355,87.5,43.218,9270.8,3d8113122db8aaf4d63c4185d104cb5dac5afe19610a1e9e72948b48b697600c,2026-10-11T23:00:49+00:00
> hardswish_dfull_k3_s512,budget,True,hardswish_dfull_k3,hardswish,full,3,512,24.7,-1.0,1.69,41.323,0.125,41.31,41.469,41.732,24.2,4357,103.6,43.562,9561.1,72896e3d1db65c19966a1353a531c81544f6a45eea56945bf74844e1a844d788,2026-10-11T23:06:57+00:00
> hardswish_dfull_k5_s512,budget,True,hardswish_dfull_k5,hardswish,full,5,512,25.0,-0.7,1.79,48.066,0.088,48.045,48.255,48.523,20.8,3745,101.4,43.312,9484.5,ee2f16002e2fb3b1688c8a54221e6219f55a05b1a1766e00a45f368d206f7b2d,2026-10-11T22:21:04+00:00
> silu_dfull_k3_s512,budget,True,silu_dfull_k3,silu,full,3,512,25.7,0.0,1.69,42.439,0.082,42.425,42.559,42.803,23.56,4242,97.3,43.031,9340.9,996412201ec2bcd37e7bc4a772235a4697a194cf800a88bdbfc5eec8e2858402,2026-10-11T22:39:26+00:00
> silu_dfull_k5_s512,budget,True,silu_dfull_k5,silu,full,5,512,26.2,0.5,1.79,49.064,0.008,49.057,49.146,49.355,20.38,3669,96.7,43.656,9319.3,052b1df9da19f84e7cb5133134d3b01dfac2193ede44e74b912d709e43ee8e01,2026-10-11T22:24:10+00:00
> silu_dreduced_k5_s512,budget,True,silu_dreduced_k5,silu,reduced,5,512,24.7,-1.0,1.74,47.802,0.009,47.798,47.903,48.025,20.92,3766,99.7,43.156,9285.4,8872e2cf6cc824650fdd854ed3ea3c3c01117e19e34781adb650d3e063568a32,2026-10-11T23:16:07+00:00
> hardswish_dfull_k5_s320,noise_margin,False,hardswish_dfull_k5,hardswish,full,5,320,20.3,-1.1,0.7,21.858,0.099,21.852,21.93,22.06,45.74,8234,86.2,43.281,9292.5,9d0a662230737f756839cff88c5e45b43b7444e80935cd17117468ed950c92b5,2026-10-11T23:13:02+00:00
> silu_dreduced_k3_s320,noise_margin,False,silu_dreduced_k3,silu,reduced,3,320,20.3,-1.1,0.64,18.382,0.028,18.381,18.434,18.587,54.38,9790,79.5,43.25,9205,97e0f5bc15d70982622695d4b0fd65b30211bb847f0ae0bb8ea5e3ecf1115593,2026-10-11T22:27:12+00:00
> hardswish_dfull_k3_s416,noise_margin,False,hardswish_dfull_k3,hardswish,full,3,416,23.3,-1.2,1.11,28.848,0.003,28.843,28.895,29.166,34.65,6240,86.5,43.625,9443.8,aa5810cc163bd498130557ac7229b45121659140635dbc658dc900624e476f6b,2026-10-11T23:03:52+00:00
> relu_dfull_k5_s320,reference,False,relu_dfull_k5,relu,full,5,320,17.1,-4.3,0.7,17.477,0.025,17.463,17.552,17.711,57.2,10297,78.4,43.093,9166,f2346d79d8eaf106213406bc2b4d9b9bb34285976dbc1408f3310efda2588a04,2026-10-11T22:36:21+00:00
> relu_dfull_k5_s416,reference,False,relu_dfull_k5,relu,full,5,416,19.6,-4.9,1.18,27.702,0.003,27.696,27.743,28.01,36.09,6497,87.6,43.156,9269.1,992453e1e9a2a8de9e02f29d3cb252c6389bbe47064fee91185a5d9705ab961a,2026-10-11T22:51:37+00:00
> relu_dfull_k5_s512,reference,False,relu_dfull_k5,relu,full,5,512,21.3,-4.4,1.79,39.632,0.014,39.623,39.727,39.945,25.23,4543,110.1,43.281,9284.4,16bc53527e66e2931f42eac1ae1d549f27a39797700143adb3970f95788ca857,2026-10-11T22:30:16+00:00
diff -r /home/iqmine/n4_yedek/n4_latency/jetson_orin_nano_cpu_smoke/latency.csv results/nas_r1b/n4_latency/jetson_orin_nano_cpu_smoke/latency.csv
1,20c1,20
< candidate_id,tier,selectable,arch_id,act,depth,dw_k,resolution,map5095,delta_vs_control,gflops,latency_ms_mean,latency_ms_std,latency_ms_median,latency_ms_p90,latency_ms_p99,fps_model,n_inferences,rss_peak_mb,temp_c_max,tegra_vdd_in_mw,onnx_sha256,measured_utc
< hardswish_dfull_k3_s320,budget,True,hardswish_dfull_k3,hardswish,full,3,320,20.5,-0.9,0.66,18.658,0.0,18.654,18.718,18.752,53.58,268,83.8,41.562,9284.4,6d56b7aecd3cdb199a6675a1486b70a0fc7d6fae0ba2f2faca440b7bbf736985,2026-10-11T22:15:29+00:00
< silu_dfull_k3_s320,budget,True,silu_dfull_k3,silu,full,3,320,21.4,0.0,0.66,18.82,0.0,18.815,18.861,19.017,53.11,266,76.2,42.687,9143.9,627ae4a457d4ca962526d94b45897022c42688b74c01fbefd0ba77ad74033819,2026-10-11T22:16:38+00:00
< silu_dfull_k5_s320,budget,True,silu_dfull_k5,silu,full,5,320,21.6,0.2,0.7,22.152,0.0,22.143,22.199,22.398,45.13,226,76.3,41.843,9126.1,e1a7f7781e202b3db68bdcfce9c45306c5eaae1df05a35dfc0c197e7e38c5443,2026-10-11T22:15:46+00:00
< silu_dreduced_k5_s320,budget,True,silu_dreduced_k5,silu,reduced,5,320,20.7,-0.7,0.68,21.205,0.0,21.198,21.262,21.405,47.14,236,79.8,42.093,9099.6,9e2ee05ce4bf6ca6a7a93562c872b2d722558309a82595d32b51dcf93e899199,2026-10-11T22:15:52+00:00
< hardswish_dfull_k5_s416,budget,True,hardswish_dfull_k5,hardswish,full,5,416,23.5,-1.0,1.18,34.099,0.0,34.101,34.148,34.218,29.32,147,85.4,42.437,9377.4,b45c9cb4e179d27c642e17ffbacd29f987b773d01582bb916a77d83e3b094c1a,2026-10-11T22:16:09+00:00
< silu_dfull_k3_s416,budget,True,silu_dfull_k3,silu,full,3,416,24.5,0.0,1.11,29.741,0.0,29.735,29.782,30.038,33.61,169,85.4,42.156,9284.6,858d4b45de60c4b63f5e82ed5e40387e5a79fc9fcfa0d7ecaab74bab6dbbf8d5,2026-10-11T22:15:58+00:00
< silu_dfull_k5_s416,budget,True,silu_dfull_k5,silu,full,5,416,24.9,0.4,1.18,34.575,0.0,34.564,34.619,34.925,28.91,145,86.7,42.593,9227.3,7741ccc056c2aa36c9ff3e1c6d1f1388743749995441c507f189f48d3458dd1e,2026-10-11T22:16:15+00:00
< silu_dreduced_k5_s416,budget,True,silu_dreduced_k5,silu,reduced,5,416,23.5,-1.0,1.15,33.37,0.0,33.363,33.408,33.678,29.96,150,83.1,42.531,9236.2,3d8113122db8aaf4d63c4185d104cb5dac5afe19610a1e9e72948b48b697600c,2026-10-11T22:16:20+00:00
< hardswish_dfull_k3_s512,budget,True,hardswish_dfull_k3,hardswish,full,3,512,24.7,-1.0,1.69,41.152,0.0,41.139,41.253,41.561,24.3,122,99.8,42.843,9522.8,72896e3d1db65c19966a1353a531c81544f6a45eea56945bf74844e1a844d788,2026-10-11T22:16:32+00:00
< hardswish_dfull_k5_s512,budget,True,hardswish_dfull_k5,hardswish,full,5,512,25.0,-0.7,1.79,48.216,0.0,48.196,48.329,48.513,20.74,104,101.9,40.843,9434.4,ee2f16002e2fb3b1688c8a54221e6219f55a05b1a1766e00a45f368d206f7b2d,2026-10-11T22:15:06+00:00
< silu_dfull_k3_s512,budget,True,silu_dfull_k3,silu,full,3,512,25.7,0.0,1.69,42.312,0.0,42.306,42.401,42.505,23.63,119,95.9,41.843,9306.4,996412201ec2bcd37e7bc4a772235a4697a194cf800a88bdbfc5eec8e2858402,2026-10-11T22:15:41+00:00
< silu_dfull_k5_s512,budget,True,silu_dfull_k5,silu,full,5,512,26.2,0.5,1.79,48.971,0.0,48.961,49.046,49.256,20.42,103,93.6,40.843,9262.9,052b1df9da19f84e7cb5133134d3b01dfac2193ede44e74b912d709e43ee8e01,2026-10-11T22:15:12+00:00
< silu_dreduced_k5_s512,budget,True,silu_dreduced_k5,silu,reduced,5,512,24.7,-1.0,1.74,47.386,0.0,47.376,47.455,47.729,21.1,106,96.4,42.812,9289.1,8872e2cf6cc824650fdd854ed3ea3c3c01117e19e34781adb650d3e063568a32,2026-10-11T22:16:49+00:00
< hardswish_dfull_k5_s320,noise_margin,False,hardswish_dfull_k5,hardswish,full,5,320,20.3,-1.1,0.7,22.05,0.0,22.043,22.101,22.328,45.34,227,82.3,42.843,9258.3,9d0a662230737f756839cff88c5e45b43b7444e80935cd17117468ed950c92b5,2026-10-11T22:16:43+00:00
< silu_dreduced_k3_s320,noise_margin,False,silu_dreduced_k3,silu,reduced,3,320,20.3,-1.1,0.64,18.215,0.0,18.213,18.264,18.345,54.88,275,77.7,40.906,9126.1,97e0f5bc15d70982622695d4b0fd65b30211bb847f0ae0bb8ea5e3ecf1115593,2026-10-11T22:15:18+00:00
< hardswish_dfull_k3_s416,noise_margin,False,hardswish_dfull_k3,hardswish,full,3,416,23.3,-1.2,1.11,29.032,0.0,29.025,29.082,29.194,34.43,173,84.9,42.843,9438.9,aa5810cc163bd498130557ac7229b45121659140635dbc658dc900624e476f6b,2026-10-11T22:16:26+00:00
< relu_dfull_k5_s320,reference,False,relu_dfull_k5,relu,full,5,320,17.1,-4.3,0.7,17.625,0.0,17.619,17.68,17.801,56.72,284,77.8,41.656,9130.6,f2346d79d8eaf106213406bc2b4d9b9bb34285976dbc1408f3310efda2588a04,2026-10-11T22:15:35+00:00
< relu_dfull_k5_s416,reference,False,relu_dfull_k5,relu,full,5,416,19.6,-4.9,1.18,27.965,0.0,27.959,28.017,28.205,35.75,179,81.7,42.343,9222.9,992453e1e9a2a8de9e02f29d3cb252c6389bbe47064fee91185a5d9705ab961a,2026-10-11T22:16:03+00:00
< relu_dfull_k5_s512,reference,False,relu_dfull_k5,relu,full,5,512,21.3,-4.4,1.79,39.804,0.0,39.792,39.895,40.105,25.12,126,93.5,41.281,9245.1,16bc53527e66e2931f42eac1ae1d549f27a39797700143adb3970f95788ca857,2026-10-11T22:15:24+00:00
---
> candidate_id,tier,selectable,arch_id,act,depth,dw_k,resolution,map5095,delta_vs_control,gflops,latency_ms_mean,latency_ms_std,latency_ms_median,latency_ms_p90,latency_ms_p99,fps_model,n_inferences,rss_peak_mb,temp_c_max,tegra_vdd_in_mw,onnx_sha256,measured_utc
> hardswish_dfull_k3_s320,budget,True,hardswish_dfull_k3,hardswish,full,3,320,20.5,-0.9,0.66,18.658,0.0,18.654,18.718,18.752,53.58,268,83.8,41.562,9284.4,6d56b7aecd3cdb199a6675a1486b70a0fc7d6fae0ba2f2faca440b7bbf736985,2026-10-11T22:15:29+00:00
> silu_dfull_k3_s320,budget,True,silu_dfull_k3,silu,full,3,320,21.4,0.0,0.66,18.82,0.0,18.815,18.861,19.017,53.11,266,76.2,42.687,9143.9,627ae4a457d4ca962526d94b45897022c42688b74c01fbefd0ba77ad74033819,2026-10-11T22:16:38+00:00
> silu_dfull_k5_s320,budget,True,silu_dfull_k5,silu,full,5,320,21.6,0.2,0.7,22.152,0.0,22.143,22.199,22.398,45.13,226,76.3,41.843,9126.1,e1a7f7781e202b3db68bdcfce9c45306c5eaae1df05a35dfc0c197e7e38c5443,2026-10-11T22:15:46+00:00
> silu_dreduced_k5_s320,budget,True,silu_dreduced_k5,silu,reduced,5,320,20.7,-0.7,0.68,21.205,0.0,21.198,21.262,21.405,47.14,236,79.8,42.093,9099.6,9e2ee05ce4bf6ca6a7a93562c872b2d722558309a82595d32b51dcf93e899199,2026-10-11T22:15:52+00:00
> hardswish_dfull_k5_s416,budget,True,hardswish_dfull_k5,hardswish,full,5,416,23.5,-1.0,1.18,34.099,0.0,34.101,34.148,34.218,29.32,147,85.4,42.437,9377.4,b45c9cb4e179d27c642e17ffbacd29f987b773d01582bb916a77d83e3b094c1a,2026-10-11T22:16:09+00:00
> silu_dfull_k3_s416,budget,True,silu_dfull_k3,silu,full,3,416,24.5,0.0,1.11,29.741,0.0,29.735,29.782,30.038,33.61,169,85.4,42.156,9284.6,858d4b45de60c4b63f5e82ed5e40387e5a79fc9fcfa0d7ecaab74bab6dbbf8d5,2026-10-11T22:15:58+00:00
> silu_dfull_k5_s416,budget,True,silu_dfull_k5,silu,full,5,416,24.9,0.4,1.18,34.575,0.0,34.564,34.619,34.925,28.91,145,86.7,42.593,9227.3,7741ccc056c2aa36c9ff3e1c6d1f1388743749995441c507f189f48d3458dd1e,2026-10-11T22:16:15+00:00
> silu_dreduced_k5_s416,budget,True,silu_dreduced_k5,silu,reduced,5,416,23.5,-1.0,1.15,33.37,0.0,33.363,33.408,33.678,29.96,150,83.1,42.531,9236.2,3d8113122db8aaf4d63c4185d104cb5dac5afe19610a1e9e72948b48b697600c,2026-10-11T22:16:20+00:00
> hardswish_dfull_k3_s512,budget,True,hardswish_dfull_k3,hardswish,full,3,512,24.7,-1.0,1.69,41.152,0.0,41.139,41.253,41.561,24.3,122,99.8,42.843,9522.8,72896e3d1db65c19966a1353a531c81544f6a45eea56945bf74844e1a844d788,2026-10-11T22:16:32+00:00
> hardswish_dfull_k5_s512,budget,True,hardswish_dfull_k5,hardswish,full,5,512,25.0,-0.7,1.79,48.216,0.0,48.196,48.329,48.513,20.74,104,101.9,40.843,9434.4,ee2f16002e2fb3b1688c8a54221e6219f55a05b1a1766e00a45f368d206f7b2d,2026-10-11T22:15:06+00:00
> silu_dfull_k3_s512,budget,True,silu_dfull_k3,silu,full,3,512,25.7,0.0,1.69,42.312,0.0,42.306,42.401,42.505,23.63,119,95.9,41.843,9306.4,996412201ec2bcd37e7bc4a772235a4697a194cf800a88bdbfc5eec8e2858402,2026-10-11T22:15:41+00:00
> silu_dfull_k5_s512,budget,True,silu_dfull_k5,silu,full,5,512,26.2,0.5,1.79,48.971,0.0,48.961,49.046,49.256,20.42,103,93.6,40.843,9262.9,052b1df9da19f84e7cb5133134d3b01dfac2193ede44e74b912d709e43ee8e01,2026-10-11T22:15:12+00:00
> silu_dreduced_k5_s512,budget,True,silu_dreduced_k5,silu,reduced,5,512,24.7,-1.0,1.74,47.386,0.0,47.376,47.455,47.729,21.1,106,96.4,42.812,9289.1,8872e2cf6cc824650fdd854ed3ea3c3c01117e19e34781adb650d3e063568a32,2026-10-11T22:16:49+00:00
> hardswish_dfull_k5_s320,noise_margin,False,hardswish_dfull_k5,hardswish,full,5,320,20.3,-1.1,0.7,22.05,0.0,22.043,22.101,22.328,45.34,227,82.3,42.843,9258.3,9d0a662230737f756839cff88c5e45b43b7444e80935cd17117468ed950c92b5,2026-10-11T22:16:43+00:00
> silu_dreduced_k3_s320,noise_margin,False,silu_dreduced_k3,silu,reduced,3,320,20.3,-1.1,0.64,18.215,0.0,18.213,18.264,18.345,54.88,275,77.7,40.906,9126.1,97e0f5bc15d70982622695d4b0fd65b30211bb847f0ae0bb8ea5e3ecf1115593,2026-10-11T22:15:18+00:00
> hardswish_dfull_k3_s416,noise_margin,False,hardswish_dfull_k3,hardswish,full,3,416,23.3,-1.2,1.11,29.032,0.0,29.025,29.082,29.194,34.43,173,84.9,42.843,9438.9,aa5810cc163bd498130557ac7229b45121659140635dbc658dc900624e476f6b,2026-10-11T22:16:26+00:00
> relu_dfull_k5_s320,reference,False,relu_dfull_k5,relu,full,5,320,17.1,-4.3,0.7,17.625,0.0,17.619,17.68,17.801,56.72,284,77.8,41.656,9130.6,f2346d79d8eaf106213406bc2b4d9b9bb34285976dbc1408f3310efda2588a04,2026-10-11T22:15:35+00:00
> relu_dfull_k5_s416,reference,False,relu_dfull_k5,relu,full,5,416,19.6,-4.9,1.18,27.965,0.0,27.959,28.017,28.205,35.75,179,81.7,42.343,9222.9,992453e1e9a2a8de9e02f29d3cb252c6389bbe47064fee91185a5d9705ab961a,2026-10-11T22:16:03+00:00
> relu_dfull_k5_s512,reference,False,relu_dfull_k5,relu,full,5,512,21.3,-4.4,1.79,39.804,0.0,39.792,39.895,40.105,25.12,126,93.5,41.281,9245.1,16bc53527e66e2931f42eac1ae1d549f27a39797700143adb3970f95788ca857,2026-10-11T22:15:24+00:00
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ diff -r --strip-trailing-cr ~/n4_yedek/n4_latency results/nas_r1b/n4_latency && echo "AYNI"
Only in results/nas_r1b/n4_latency/jetson_orin_nano_cpu: jetson_system_info.md
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ rm -rf ~/n4_yedek
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ cd ~
(n4env) iqmine@iqmine-jetson:~$ sha256sum nas_r1b_n4b_onnx.zip
3d687508349a9ae1a71af12f8e95b2412f3467293395a7f1f2346ac10bfa4c02  nas_r1b_n4b_onnx.zip
(n4env) iqmine@iqmine-jetson:~$ rm -rf nas_r1b_n4b_onnx
(n4env) iqmine@iqmine-jetson:~$ unzip nas_r1b_n4b_onnx.zip
Archive:  nas_r1b_n4b_onnx.zip
  inflating: nas_r1b_n4b_onnx/relu_dfull_k3_s320.onnx
  inflating: nas_r1b_n4b_onnx/relu_dfull_k3_s416.onnx
  inflating: nas_r1b_n4b_onnx/relu_dfull_k3_s512.onnx
  inflating: nas_r1b_n4b_onnx/silu_dfull_k3_s320.onnx
  inflating: nas_r1b_n4b_onnx/silu_dfull_k3_s416.onnx
  inflating: nas_r1b_n4b_onnx/silu_dfull_k3_s512.onnx
  inflating: nas_r1b_n4b_onnx/inputs/input_s320.npy
  inflating: nas_r1b_n4b_onnx/inputs/input_s416.npy
  inflating: nas_r1b_n4b_onnx/inputs/input_s512.npy
  inflating: nas_r1b_n4b_onnx/inputs/inputs.json
  inflating: nas_r1b_n4b_onnx/n4_candidates.csv
  inflating: nas_r1b_n4b_onnx/exports.csv
  inflating: nas_r1b_n4b_onnx/manifest.json
  inflating: nas_r1b_n4b_onnx/SHA256SUMS
(n4env) iqmine@iqmine-jetson:~$ cd nas_r1b_n4b_onnx && sha256sum -c SHA256SUMS && ls inputs/
relu_dfull_k3_s320.onnx: OK
relu_dfull_k3_s416.onnx: OK
relu_dfull_k3_s512.onnx: OK
silu_dfull_k3_s320.onnx: OK
silu_dfull_k3_s416.onnx: OK
silu_dfull_k3_s512.onnx: OK
inputs/input_s320.npy: OK
inputs/input_s416.npy: OK
inputs/input_s512.npy: OK
input_s320.npy  input_s416.npy  input_s512.npy  inputs.json
(n4env) iqmine@iqmine-jetson:~/nas_r1b_n4b_onnx$ cd ~/edge_ai_tez/edge-ai-object-detection
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ OUT=results/nas_r1b/n4b_latency
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ mkdir -p $OUT
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4b_onnx --device jetson_orin_nano_cpu --out $OUT --smoke --tegrastats 2>&1 | tee $OUT/smoke.log
jetson_orin_nano_cpu: 6 aday, sıra tohumu 0, en çok ≈2 dk · ayarlar {'cpu_governor': ['schedutil'], 'cpu_count': 6, 'nvpmodel': 'NV Power Mode: MAXN_SUPER\n2'} · yük (0.0, 0.095703125, 0.08203125) · sıcaklık {'cpu-thermal': 38.625, 'gpu-thermal': 39.5, 'soc0-thermal': 38.718, 'soc1-thermal': 38.062, 'soc2-thermal': 39.031, 'tj-thermal': 39.5}
✔ 1/6 silu_dfull_k3_s416            29.57 ± 0.00 ms ·   33.8 FPS · p99 29.71 · 40.531 °C
✔ 2/6 relu_dfull_k3_s512            33.16 ± 0.00 ms ·   30.1 FPS · p99 33.36 · 41.0 °C
✔ 3/6 relu_dfull_k3_s416            22.63 ± 0.00 ms ·   44.2 FPS · p99 22.77 · 41.531 °C
✔ 4/6 relu_dfull_k3_s320            14.32 ± 0.00 ms ·   69.8 FPS · p99 14.43 · 41.531 °C
✔ 5/6 silu_dfull_k3_s512            42.20 ± 0.00 ms ·   23.7 FPS · p99 42.48 · 41.718 °C
✔ 6/6 silu_dfull_k3_s320            19.01 ± 0.00 ms ·   52.6 FPS · p99 19.16 · 41.937 °C
6/6 aday ölçüldü → results/nas_r1b/n4b_latency/jetson_orin_nano_cpu_smoke/latency.csv
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4b_onnx --device jetson_orin_nano_cpu --out $OUT --tegrastats 2>&1 | tee -a $OUT/full.log
jetson_orin_nano_cpu: 6 aday, sıra tohumu 0, en çok ≈20 dk · ayarlar {'cpu_governor': ['schedutil'], 'cpu_count': 6, 'nvpmodel': 'NV Power Mode: MAXN_SUPER\n2'} · yük (0.2802734375, 0.3623046875, 0.1962890625) · sıcaklık {'cpu-thermal': 38.875, 'gpu-thermal': 39.718, 'soc0-thermal': 39.0, 'soc1-thermal': 38.375, 'soc2-thermal': 39.281, 'tj-thermal': 39.718}
✔ 1/6 silu_dfull_k3_s416            29.70 ± 0.16 ms ·   33.7 FPS · p99 30.03 · 44.125 °C
✔ 2/6 relu_dfull_k3_s512            33.23 ± 0.15 ms ·   30.1 FPS · p99 34.36 · 44.781 °C
✔ 3/6 relu_dfull_k3_s416            22.54 ± 0.03 ms ·   44.4 FPS · p99 22.68 · 44.656 °C
✔ 4/6 relu_dfull_k3_s320            14.31 ± 0.05 ms ·   69.8 FPS · p99 14.62 · 44.468 °C
✔ 5/6 silu_dfull_k3_s512            42.28 ± 0.01 ms ·   23.6 FPS · p99 42.56 · 44.5 °C
✔ 6/6 silu_dfull_k3_s320            18.83 ± 0.01 ms ·   53.1 FPS · p99 19.08 · 44.906 °C
6/6 aday ölçüldü → results/nas_r1b/n4b_latency/jetson_orin_nano_cpu/latency.csv
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$