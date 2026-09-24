---
name: static-qdq-quantization
description: Applies calibrated static post-training quantization (PTQ) in QDQ format to ONNX models using ONNX Runtime. Trigger this skill when converting FP32 ONNX graphs to INT8 precision for TensorRT or ncnn deployment.
---

# Static INT8 QDQ Quantization Skill

When invoked:
1. Load calibration parameters from `.agents/skills.config.yaml` (`dataset.calibration_size`).
2. Build an `onnxruntime.quantization.CalibrationDataReader` loading representative sample images from `datasets/coco/train2017` (never from `val2017`).
3. Apply static quantization using `quantize_static`:
   - `quant_format = QuantFormat.QDQ`
   - `activation_type = QuantType.QUInt8`
   - `weight_type = QuantType.QInt8`
   - `per_channel = False` (initial baseline)
4. Save the artifact to `artifacts/onnx/yolox_int8_qdq.onnx`.
5. Run the **Accuracy Gate**:
   - Evaluate $mAP_{50:95}$ on COCO val2017.
   - If $\Delta mAP = mAP_{FP32} - mAP_{INT8} > 1.0$, notify the user and trigger Quantization-Aware Training (QAT) via `torchao` before deploying.