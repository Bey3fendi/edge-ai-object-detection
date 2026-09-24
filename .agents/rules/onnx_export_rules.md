---
description: Enforces strict shape and operator requirements during ONNX conversion.
globs: ["**/*export*.py", "**/*onnx*.py"]
---

# ONNX Export Constraints

- **Static Tensors Only**: Do not use `dynamic_axes` in `torch.onnx.export`. Set shape explicitly to `[1, 3, 416, 416]` (or the resolution specified in `.agents/skills.config.yaml`).
- **Opset Version**: Pin the ONNX opset to 13 or 17 to ensure full QDQ (`QuantizeLinear`/`DequantizeLinear`) operator support.
- **Verification Routine**: Every exported model must immediately pass:
  ```python
  import onnx
  onnx_model = onnx.load(model_path)
  onnx.checker.check_model(onnx_model)