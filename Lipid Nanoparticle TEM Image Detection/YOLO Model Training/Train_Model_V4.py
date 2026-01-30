import os
import shutil
import random
import yaml
import sys
import datetime
import traceback
import argparse
from ultralytics import YOLO

# --- CONFIGURATION ---
SOURCE_DIR = "/scratch/jbs263/LipidDetectionDataset/dataset_1"
DATASET_DIR = "/scratch/jbs263/lnp_yolo_staging_area" 
MODEL_WEIGHTS = "yolo11n.pt"

# [UPDATED] Descriptive Class Names (Must match the 0-4 order of your generator)
CLASSES = [
    "Solid_LNP",               # ID 0
    "Unilamellar_Vesicle",     # ID 1 (ulv)
    "Multilamellar_Vesicle",   # ID 2 (mlv)
    "Multivesicular_Liposome", # ID 3 (mvl)
    "Bleb"                     # ID 4
]

def setup_dataset():
    print(f"--- Organizing Data from '{SOURCE_DIR}' ---")
    
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    if os.path.exists(DATASET_DIR):
        print(f"Cleaning old staging area {DATASET_DIR}...")
        shutil.rmtree(DATASET_DIR)
    
    # Create YOLO structure
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            os.makedirs(f"{DATASET_DIR}/{split}/{kind}", exist_ok=True)
            
    img_path = os.path.join(SOURCE_DIR, "images")
    if not os.path.exists(img_path):
         img_path = SOURCE_DIR

    print(f"Scanning for images in: {img_path}")
    all_images = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)
    
    if not all_images:
        raise ValueError("No images found! Check source directory.")

    # 80/20 Split
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    print(f"Found {len(all_images)} images. Split: {len(train_imgs)} Train, {len(val_imgs)} Val.")

    def copy_files(file_list, split_name):
        for img_name in file_list:
            # Copy Image
            src_img = os.path.join(img_path, img_name)
            dst_img = os.path.join(DATASET_DIR, split_name, "images", img_name)
            shutil.copy(src_img, dst_img)
            
            # Copy Label
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(SOURCE_DIR, "labels", label_name)
            dst_lbl = os.path.join(DATASET_DIR, split_name, "labels", label_name)
            
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                open(dst_lbl, 'w').close()

    print("Copying Training data...")
    copy_files(train_imgs, 'train')
    print("Copying Validation data...")
    copy_files(val_imgs, 'val')
    
    # Create YAML with DESCRIPTIVE NAMES
    yaml_content = {
        'path': os.path.abspath(DATASET_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    yaml_path = os.path.join(DATASET_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
        
    return yaml_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--name', type=str, default='lnp_run')
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    print(f"========== PROCESS START: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    print(f"Configuration: Epochs={args.epochs}, Classes={CLASSES}")
    
    try:
        yaml_config = setup_dataset()

        if yaml_config:
            print(f"--- Loading weights and starting Training ---")
            
            try:
                model = YOLO(MODEL_WEIGHTS)
            except Exception:
                model = YOLO("yolo11n.pt")

            model.train(
                data=yaml_config,
                epochs=args.epochs,       
                imgsz=1024,
                batch=128,          
                workers=16,        
                patience=25,      
                name=args.name,    
                device=[0,1,2,3],  
                exist_ok=True,
                amp=True          
            )
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {traceback.format_exc()}")
        sys.exit(1) 
        
    finally:
        end_time = datetime.datetime.now()
        print(f"\n========== PROCESS END: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
