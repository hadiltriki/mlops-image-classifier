"""Module de chargement des données"""
import os
import pandas as pd
from pathlib import Path

CLASSES = ["bebe", "enfant", "femme", "homme"]

def parse_utkface_filename(filename):
    """
    Parse UTKFace filename: [age]_[gender]_[race]_[date].jpg
    
    Returns:
        dict: Informations extraites
    """
    try:
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])  # 0=male, 1=female
        
        # Catégorisation
        if age <= 2:
            category = 0  # bebe
        elif age <= 12:
            category = 1  # enfant
        elif gender == 1:
            category = 2  # femme
        else:
            category = 3  # homme
        
        return {
            'age': age,
            'gender': gender,
            'category': category,
            'label': CLASSES[category]
        }
    except:
        return None

def load_dataset(data_dir, max_samples=None):
    """
    Charge le dataset UTKFace
    
    Args:
        data_dir: Chemin vers les données
        max_samples: Nombre max d'images
    
    Returns:
        pd.DataFrame: Dataset avec métadonnées
    """
    image_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    data = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        info = parse_utkface_filename(filename)
        
        if info:
            info['path'] = img_path
            data.append(info)
    
    df = pd.DataFrame(data)
    print(f" Dataset loaded: {len(df)} images")
    print(f" Distribution:\n{df['label'].value_counts()}")
    
    return df

if __name__ == "__main__":
    df = load_dataset("data/raw", max_samples=100)
    print(df.head())