# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia

python3 --version
Python 3.13.9

(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ sudo nvpmodel -q --verbose | head -20
[sudo] password for iqmine:
NVPM VERB: Config file: /etc/nvpmodel.conf
NVPM VERB: parsing done for /etc/nvpmodel.conf
NVPM VERB: Current mode: NV Power Mode: MAXN_SUPER
2
NVPM VERB: PARAM CPU_ONLINE: ARG CORE_0: PATH /sys/devices/system/cpu/cpu0/online: REAL_VAL: 1 CONF_VAL: 1
NVPM VERB: PARAM CPU_ONLINE: ARG CORE_1: PATH /sys/devices/system/cpu/cpu1/online: REAL_VAL: 1 CONF_VAL: 1
NVPM VERB: PARAM CPU_ONLINE: ARG CORE_2: PATH /sys/devices/system/cpu/cpu2/online: REAL_VAL: 1 CONF_VAL: 1
NVPM VERB: PARAM CPU_ONLINE: ARG CORE_3: PATH /sys/devices/system/cpu/cpu3/online: REAL_VAL: 1 CONF_VAL: 1
NVPM VERB: PARAM CPU_ONLINE: ARG CORE_4: PATH /sys/devices/system/cpu/cpu4/online: REAL_VAL: 1 CONF_VAL: 1
NVPM VERB: PARAM CPU_ONLINE: ARG CORE_5: PATH /sys/devices/system/cpu/cpu5/online: REAL_VAL: 1 CONF_VAL: 1
NVPM VERB: PARAM FBP_POWER_GATING: ARG FBP_PG_MASK: PATH /sys/devices/platform/gpu.0/fbp_pg_mask: REAL_VAL: 2 CONF_VAL: 2
NVPM VERB: PARAM TPC_POWER_GATING: ARG TPC_PG_MASK: PATH /sys/devices/platform/gpu.0/tpc_pg_mask: REAL_VAL: 240 CONF_VAL: 240
NVPM VERB: PARAM GPU_POWER_CONTROL_ENABLE: ARG GPU_PWR_CNTL_EN: PATH /sys/devices/platform/gpu.0/power/control: REAL_VAL: auto CONF_VAL: on
NVPM VERB: PARAM CPU_A78_0: ARG MIN_FREQ: PATH /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq: REAL_VAL: 729600 CONF_VAL: 729600
NVPM VERB: PARAM CPU_A78_0: ARG MAX_FREQ: PATH /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq: REAL_VAL: 1728000 CONF_VAL: 9223372036854775807
NVPM VERB: PARAM CPU_A78_1: ARG MIN_FREQ: PATH /sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq: REAL_VAL: 729600 CONF_VAL: 729600
NVPM VERB: PARAM CPU_A78_1: ARG MAX_FREQ: PATH /sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq: REAL_VAL: 1728000 CONF_VAL: 9223372036854775807
NVPM VERB: PARAM CPU_A78_2: ARG MIN_FREQ: PATH /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq: REAL_VAL: 729600 CONF_VAL: 729600
NVPM VERB: PARAM CPU_A78_2: ARG MAX_FREQ: PATH /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq: REAL_VAL: 1728000 CONF_VAL: 9223372036854775807
NVPM VERB: PARAM CPU_A78_3: ARG MIN_FREQ: PATH /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq: REAL_VAL: 729600 CONF_VAL: 729600

(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ grep -n "POWER_MODEL" /etc/nvpmodel.conf
21:# < POWER_MODEL ID=id_num NAME=mode_name >
26:# This starts a section of POWER_MODEL configurations, followed by
32:# This file must contain at least one POWER_MODEL section.
130:# POWER_MODEL DEFINITIONS #
134:< POWER_MODEL ID=0 NAME=15W >
161:< POWER_MODEL ID=1 NAME=25W >
188:< POWER_MODEL ID=2 NAME=MAXN_SUPER >
215:< POWER_MODEL ID=3 NAME=7W >
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$

(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ sudo nvpmodel -m 2
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ sudo nvpmodel -q
NV Power Mode: MAXN_SUPER
2
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$

(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ sudo jetson_clocks
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ sudo jetson_clocks --fan
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ sudo jetson_clocks --show | tee ~/jetson_clocks_show.txt
SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
Online CPUs: 0-5
cpu0:  Online=1 Governor=schedutil MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=0 c7=0
cpu1:  Online=1 Governor=schedutil MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=0 c7=0
cpu2:  Online=1 Governor=schedutil MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=0 c7=0
cpu3:  Online=1 Governor=schedutil MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=0 c7=0
cpu4:  Online=1 Governor=schedutil MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=0 c7=0
cpu5:  Online=1 Governor=schedutil MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=0 c7=0
GPU MinFreq=1020000000 MaxFreq=1020000000 CurrentFreq=1020000000
Active GPU TPCs: 4
EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=3199000000 FreqOverride=1
FAN Dynamic Speed Control=disabled hwmon0_pwm1=255
NV Power Mode: MAXN_SUPER
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$

(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ sudo systemctl isolate multi-user.target
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ uptime
 00:49:25 up 42 min,  2 users,  load average: 0,00, 0,04, 0,32

 (base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ python3 -m venv ~/n4env
(base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ source ~/n4env/bin/activate
(n4env) (base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ pip install --upgrade pip
Requirement already satisfied: pip in /home/iqmine/n4env/lib/python3.13/site-packages (25.2)
Collecting pip
  Downloading pip-26.2.1-py3-none-any.whl.metadata (4.6 kB)
Downloading pip-26.2.1-py3-none-any.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 9.5 MB/s  0:00:00
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 25.2
    Uninstalling pip-25.2:
      Successfully uninstalled pip-25.2
Successfully installed pip-26.2.1
(n4env) (base) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ conda deactivate
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ pip install numpy onnxruntime
Collecting numpy
  Downloading numpy-2.5.4-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl.metadata (6.6 kB)
Collecting onnxruntime
  Downloading onnxruntime-1.31.0-cp313-cp313-manylinux_2_28_aarch64.whl.metadata (5.7 kB)
Collecting flatbuffers (from onnxruntime)
  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
Collecting packaging (from onnxruntime)
  Downloading packaging-26.3-py3-none-any.whl.metadata (3.5 kB)
Collecting protobuf>=4.25.8 (from onnxruntime)
  Downloading protobuf-7.36.2-cp310-abi3-manylinux2014_aarch64.whl.metadata (595 bytes)
Downloading numpy-2.5.4-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl (15.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.7/15.7 MB 8.1 MB/s  0:00:02
Downloading onnxruntime-1.31.0-cp313-cp313-manylinux_2_28_aarch64.whl (21.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.4/21.4 MB 8.2 MB/s  0:00:02
Downloading protobuf-7.36.2-cp310-abi3-manylinux2014_aarch64.whl (344 kB)
Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
Downloading packaging-26.3-py3-none-any.whl (129 kB)
Installing collected packages: flatbuffers, protobuf, packaging, numpy, onnxruntime
Successfully installed flatbuffers-25.12.19 numpy-2.5.4 onnxruntime-1.31.0 packaging-26.3 protobuf-7.36.2
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$ python -c "import onnxruntime as o, numpy as n; print(o.__version__, n.__version__, o.get_available_providers())"
1.31.0 2.5.4 ['AzureExecutionProvider', 'CPUExecutionProvider']
(n4env) iqmine@iqmine-jetson:~/edge_ai_tez/edge-ai-object-detection$