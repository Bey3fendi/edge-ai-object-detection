#!/bin/bash
set -e

PROJECT_DIR="/home/furkan/project/TEZ/implementation"

echo "Creating project directory if it doesn't exist: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Initializing YOLOX submodule..."
if [ ! -d "YOLOX" ] || [ -z "$(ls -A YOLOX)" ]; then
    git submodule update --init --recursive
else
    echo "YOLOX submodule already exists."
fi

echo "Creating artifacts directories..."
mkdir -p artifacts/pytorch

echo "Downloading yolox_nano.pth weights..."
if [ ! -f "artifacts/pytorch/yolox_nano.pth" ]; then
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -O artifacts/pytorch/yolox_nano.pth
else
    echo "yolox_nano.pth already exists."
fi

echo "Setting up conda environment 'edge_yolox_env'..."
source /home/furkan/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "edge_yolox_env"; then
    conda create -y -n edge_yolox_env python=3.10
fi

conda activate edge_yolox_env

echo "Installing required packages..."
# Installing standard PyTorch packages and the rest via pip
pip install torch torchvision torchaudio onnx onnxruntime pytest nni torch-pruning

echo "Environment setup complete!"
