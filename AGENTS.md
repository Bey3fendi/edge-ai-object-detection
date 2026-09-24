# AGENTS.md - Edge AI Object Detection Constitution

## 1. Project Philosophy & Core Claim
- **Core Claim**: This thesis does NOT invent a detection architecture from scratch. It develops an edge-optimized, hardware-aware variant of the permissive open-source **YOLOX-Nano** model (Apache-2.0) via NAS, structured channel pruning, and INT8 quantization.
- **Hardware Scope**: Strictly limited to **Raspberry Pi 5** (ARM Cortex-A76 CPU) and **NVIDIA Jetson Orin Nano** (Ampere GPU + ARM CPU).
- **Continual Learning**: Explored strictly at the theoretical level (concept drift, replay buffers, knowledge distillation); never train or run backpropagation directly on edge devices.

## 2. Golden Evaluation Metrics
- **Accuracy**: $mAP_{50:95}$ evaluated on the **full COCO val2017** dataset (5,000 images). Do not report proxy dataset metrics as final academic findings.
- **Latency & Throughput**: Steady-state inference latency ($P50$, $P95$, $P99$ in ms) and throughput ($FPS$) across 5,000 images with a minimum 100-frame warmup.
- **Energy & Efficiency**: Physical load power ($P_{load}$ in Watts), Dynamic Power ($P_{dynamic} = P_{load} - P_{idle}$), $FPS/W$ (efficiency), and $J/frame$ (energy per frame).

## 3. Methodological Non-Negotiables
- **No Unstructured Pruning**: Only structured channel/filter pruning (via Torch-Pruning DepGraph) is permitted. Unstructured weight-zeroing does not translate to edge hardware acceleration.
- **Static Input Tensor**: Fixed shape ($1 \times 3 \times H \times W$, default $1 \times 3 \times 416 \times 416$). No dynamic axes in deployment artifacts.
- **ONNX QDQ Format**: All INT8 quantization must export QuantizeLinear (Q) and DeQuantizeLinear (DQ) nodes for runtime compiler fusion.
- **Two-Stage Benchmarking Isolation**:
  - **Stage 1 (Portability Baseline)**: Identical canonical ONNX model running under `ONNX Runtime CPUExecutionProvider` on both Raspberry Pi 5 and Jetson Orin Nano. (Do NOT refer to Stage 1 as "CPU vs GPU").
  - **Stage 2 (Hardware-Specific Max Optimization)**: ncnn (ARM NEON / INT8) on Raspberry Pi 5; TensorRT engine (FP16 / INT8 QDQ) on Jetson Orin Nano built directly on the target hardware.
- **Single Source of Truth**: All benchmark runs must append structured rows to `logs/benchmark_results.csv`.