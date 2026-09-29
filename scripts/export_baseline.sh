#!/bin/bash
set -e

echo "Starting YOLOX baseline export to ONNX..."
cd /home/furkan/project/TEZ/implementation

echo "Creating artifacts/onnx directory..."
mkdir -p artifacts/onnx

echo "Exporting ONNX model..."
/home/furkan/miniconda3/envs/edge_yolox_env/bin/python YOLOX/tools/export_onnx.py \
    --output-name artifacts/onnx/yolox_nano_baseline_fp32.onnx \
    -n yolox-nano \
    -c artifacts/pytorch/yolox_nano.pth \
    --opset 13 \
    --no-onnxsim

echo "Export completed successfully!"
ls -lh artifacts/onnx/yolox_nano_baseline_fp32.onnx
