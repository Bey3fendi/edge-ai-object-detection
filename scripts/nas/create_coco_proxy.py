import json
import random
import os
import argparse

def create_proxy(input_json, output_json, num_samples=10000):
    if not os.path.exists(input_json):
        print(f"Error: {input_json} does not exist. Please ensure COCO dataset is downloaded.")
        return
        
    print(f"Loading {input_json}...")
    with open(input_json, 'r') as f:
        coco = json.load(f)
        
    images = coco['images']
    annotations = coco['annotations']
    
    print(f"Original images: {len(images)}")
    
    if len(images) <= num_samples:
        print("Dataset is already smaller or equal to requested samples. No sub-sampling needed.")
        return

    # Sample 10,000 images randomly (with fixed seed for reproducibility)
    random.seed(42)
    sampled_images = random.sample(images, num_samples)
    sampled_image_ids = {img['id'] for img in sampled_images}
    
    # Filter annotations
    sampled_annotations = [ann for ann in annotations if ann['image_id'] in sampled_image_ids]
    
    proxy_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': sampled_images,
        'annotations': sampled_annotations
    }
    
    print(f"Sampled images: {len(proxy_coco['images'])}")
    print(f"Sampled annotations: {len(proxy_coco['annotations'])}")
    
    with open(output_json, 'w') as f:
        json.dump(proxy_coco, f)
    print(f"Saved proxy dataset to {output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a 10k proxy subset of COCO.")
    parser.add_argument('--input', type=str, default='../../datasets/coco/annotations/instances_train2017.json')
    parser.add_argument('--output', type=str, default='../../datasets/coco/annotations/instances_proxy10k.json')
    parser.add_argument('--samples', type=int, default=10000)
    
    args = parser.parse_args()
    create_proxy(args.input, args.output, args.samples)
