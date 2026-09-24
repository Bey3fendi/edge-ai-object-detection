### /deploy-pipeline

Run the full end-to-end model export and quantization pipeline:

1. **Step 1: Export Canonical ONNX**
   Run YOLOX export using parameters from `.agents/skills.config.yaml`:
   ```bash
   python3 YOLOX/tools/export_onnx.py \
     -n yolox-nano \
     -c artifacts/pytorch/nas_pruned_finetuned.pth \
     --output-name artifacts/onnx/yolox_fp32.onnx \
     --no-onnxsim

2. **Step 2: Verify ONNX Model Integrity**
    Verify the graph using onnx.checker and confirm shape is strictly fixed to [1, 3, 416, 416].

3. **Step 3: Execute Static QDQ Quantization**
    Trigger the static-qdq-quantization skill using 500 representative calibration samples.

4. **Step 4: Accuracy Validation**
    Evaluate $mAP_{50:95}$ on COCO val2017. If degradation exceeds 1.0 AP, halt and recommend QAT fine-tuning.