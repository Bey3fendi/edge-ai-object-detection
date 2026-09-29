#!/bin/bash
set -e
cd /home/furkan/project/TEZ/implementation/datasets/coco
echo "Downloading val2017 images..."
wget -c http://images.cocodataset.org/zips/val2017.zip
echo "Downloading annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Extracting..."
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

echo "Done!"
