"""
Module de chargement du dataset UTKFace
"""
import os
import pandas as pd
from pathlib import Path

def parse_utkface_filename(filename):
    """
    Parse le nom de fichier UTKFace
    Format: [age]_[gender]_[race]_[date&time].jpg
    
    Args:
        filename (str): Nom du fichier
    
    Returns:
        dict: {age, gender, race}
    """
    try:
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        
        return {
            'age': age,
            'gender': gender,
            'race': race
        }
    except:
        return None

def classify_image(age, gender):
    """
    Classifie l'image en 4 catégories
    
    Args:
        age (int): Âge de la personne
        gender (int): 0=homme, 1=femme
    
    Returns:
        str: Classe (bebe, enfant, femme, homme)
    """
    if age <= 2:
        return 'bebe'
    elif age <= 12:
        return 'enfant'
    elif gender == 1:
        return 'femme'
    else:
        return 'homme'

def load_utkface_dataset(data_path='data/raw/UTKFace'):
    """
    Charge le dataset UTKFace et crée un DataFrame
    
    Args:
        data_path (str): Chemin vers le dossier UTKFace
    
    Returns:
        pd.DataFrame: DataFrame avec colonnes [path, age, gender, race, label]
    """
    data = []
    data_path = Path(data_path)
    
    # Parcourir tous les fichiers
    for img_file in data_path.glob('*.jpg'):
        filename = img_file.name
        
        # Parser le nom
        parsed = parse_utkface_filename(filename)
        if parsed is None:
            continue
        
        # Classifier
        label = classify_image(parsed['age'], parsed['gender'])
        
        # Ajouter au dataset
        data.append({
            'path': str(img_file),
            'age': parsed['age'],
            'gender': parsed['gender'],
            'race': parsed['race'],
            'label': label
        })
    
    # Créer DataFrame
    df = pd.DataFrame(data)
    
    print(f"✅ Dataset chargé: {len(df)} images")
    print(f"\n📊 Distribution des classes:")
    print(df['label'].value_counts())
    
    return df

if __name__ == "__main__":
    # Test
    df = load_utkface_dataset()