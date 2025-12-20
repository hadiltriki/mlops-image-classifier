"""Module de prétraitement des images"""
import os
from pathlib import Path
from PIL import Image
import yaml

def preprocess_images(raw_dir="data/raw", processed_dir="data/processed", img_size=(224, 224)):
    """
    Redimensionne toutes les images
    
    Args:
        raw_dir: Dossier source
        processed_dir: Dossier destination
        img_size: Taille cible
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    count = 0
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    
                    rel_path = os.path.relpath(root, raw_dir)
                    save_dir = os.path.join(processed_dir, rel_path)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    img.save(os.path.join(save_dir, file))
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Processed {count} images...")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    print(f"Preprocessing complete: {count} images")

if __name__ == "__main__":
    preprocess_images()