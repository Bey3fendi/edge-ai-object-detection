---
description: Universal engineering standards for Edge AI object detection models.
alwaysOn: true
---

# Edge AI Engineering Standards

1. **Hardware-Aware Metric Priority**: Hardware memory access cost (MAC) dominates runtime latency. Never treat FLOPs minimization as an exact proxy for latency reduction.
2. **Submodule Discipline**: All source code changes to YOLOX must be committed cleanly inside the `YOLOX/` git submodule. Never overwrite original source files without editable installation (`pip install -e .`).
3. **Execution Isolation**: Prohibit model backpropagation and full COCO training on the physical Raspberry Pi 5 or Jetson Orin Nano boards. Heavy training loops must run on local workstations or cloud GPU clusters (mostly Google Colab).
4. **Thermal Throttling Guard**: Before every edge benchmark run, verify that CPU governor is set to `performance` on Raspberry Pi 5 and Jetson is configured with `sudo nvpmodel -m 0` and `sudo jetson_clocks`.5. **Environment Awareness & Validation**: Always explicitly identify the current execution environment (using `uname -a`, `/etc/os-release`, or system info) at the start of tasks to distinguish between Jetson Orin Nano, Raspberry Pi 5, or a PC/Laptop workstation. Never assume the platform. Restrict model training tasks strictly to workstations or Google Colab.
