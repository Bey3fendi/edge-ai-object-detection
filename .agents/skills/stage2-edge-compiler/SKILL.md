---
name: stage2-edge-compiler
description: Compiles optimized platform-specific runtime engines (ncnn on Raspberry Pi 5 or TensorRT on NVIDIA Jetson Orin Nano). Trigger this skill when generating hardware-accelerated runtime engines for Stage-2 benchmarks.
---

# Stage-2 Edge Platform Compiler Skill

### Branch A: Raspberry Pi 5 (ncnn + ARM NEON)
1. Convert the PyTorch model checkpoint to TorchScript via `tools/export_torchscript.py`.
2. Execute `pnnx` to generate ncnn format:
   ```bash
   pnnx artifacts/ncnn/yolox_ts.pt inputshape=[1,3,416,416]
3. Generate INT8 calibration tables with ncnn2table and compile with ncnn2int8.
4. Output files: artifacts/ncnn/yolox.param and artifacts/ncnn/yolox.bin.

### Branch B: NVIDIA Jetson Orin Nano (TensorRT)

1. Verify execution is running directly on the Jetson hardware (TensorRT engines are non-portable across architectures).
2. Execute the TensorRT compiler CLI tool:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
  --onnx=artifacts/onnx/yolox_int8_qdq.onnx \
  --saveEngine=artifacts/tensorrt/yolox_int8.plan \
  --shapes=images:1x3x416x416 \
  --int8 \
  --warmUp=3000 \
  --duration=60
3. Verify serialization output at artifacts/tensorrt/yolox_int8.plan.