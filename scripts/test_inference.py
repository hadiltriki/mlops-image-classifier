"""
Script de test pour l'inférence
"""

import sys
from pathlib import Path
import random

# Ajouter le répertoire racine au path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.models.inference import ImageClassifier

def parse_filename(filename):
    """Parse le nom de fichier pour obtenir la vraie classe"""
    parts = filename.split('_')
    try:
        age = int(parts[0])
        gender = int(parts[1])
        
        if age <= 2:
            return 'bebe', age, gender
        elif age <= 12:
            return 'enfant', age, gender
        elif gender == 1:
            return 'femme', age, gender
        else:
            return 'homme', age, gender
    except:
        return None, None, None

def test_single_image():
    """Test sur une image aléatoire"""
    classifier = ImageClassifier()
    
    # Chercher une image de test
    data_path = Path('data/raw/UTKFace')
    
    if not data_path.exists():
        print("❌ Dataset non trouvé")
        return
    
    # Chercher seulement les fichiers au bon format
    image_files = [
        f for f in data_path.glob('*.jpg') 
        if len(f.stem.split('_')) >= 4 and not '.chip' in f.name
    ]
    
    if not image_files:
        print("❌ Aucune image trouvée")
        return
    
    # 🎲 SÉLECTION ALÉATOIRE
    test_image = random.choice(image_files)
    
    # Parser le nom pour connaître la vraie classe
    expected_class, age, gender = parse_filename(test_image.stem)
    
    print(f"📸 Test sur: {test_image.name}")
    if expected_class:
        print(f"👤 Info: Age={age}, Gender={gender}")
        print(f"📋 Classe attendue: {expected_class}\n")
    
    # Prédiction
    result = classifier.predict(str(test_image))
    
    # Afficher
    print(f"🎯 Classe prédite: {result['predicted_class']}")
    print(f"📊 Confiance: {result['confidence']:.2f}%")
    
    # Vérifier si correct
    if expected_class:
        if result['predicted_class'] == expected_class:
            print("✅ PRÉDICTION CORRECTE")
        else:
            print("❌ PRÉDICTION INCORRECTE")
    
    print(f"\n📈 Probabilités:")
    
    for cls, prob in sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], 
                            reverse=True):
        bar = '█' * int(prob/5)
        print(f"  {cls:8s}: {prob:6.2f}%  {bar}")

def test_batch(n=5):
    """Test sur plusieurs images aléatoires"""
    classifier = ImageClassifier()
    
    data_path = Path('data/raw/UTKFace')
    if not data_path.exists():
        return
    
    # Chercher toutes les images valides
    all_images = [
        f for f in data_path.glob('*.jpg') 
        if len(f.stem.split('_')) >= 4 and not '.chip' in f.name
    ]
    
    if len(all_images) < n:
        print(f"⚠️ Seulement {len(all_images)} images disponibles")
        n = len(all_images)
    
    # 🎲 SÉLECTION ALÉATOIRE de n images
    image_files = random.sample(all_images, n)
    image_paths = [str(img) for img in image_files]
    
    print(f"\n🔄 Test batch sur {len(image_paths)} images aléatoires...\n")
    
    results = classifier.predict_batch(image_paths)
    
    correct = 0
    total = 0
    
    for i, result in enumerate(results, 1):
        img_name = Path(result['image_path']).name
        
        # Parser pour obtenir la vraie classe
        expected_class, age, gender = parse_filename(Path(result['image_path']).stem)
        
        is_correct = ""
        if expected_class:
            total += 1
            if result['predicted_class'] == expected_class:
                correct += 1
                is_correct = " ✅"
            else:
                is_correct = " ❌"
        
        print(f"{i}. {img_name}")
        if expected_class:
            print(f"   Attendu: {expected_class} (age={age}, gender={gender})")
        print(f"   Prédit:  {result['predicted_class']} ({result['confidence']:.1f}%){is_correct}")
        print()
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"📊 Accuracy: {correct}/{total} = {accuracy:.1f}%")

def test_multiple_random(n_tests=10):
    """Test sur plusieurs images aléatoires individuelles"""
    classifier = ImageClassifier()
    
    data_path = Path('data/raw/UTKFace')
    if not data_path.exists():
        print("❌ Dataset non trouvé")
        return
    
    # Chercher toutes les images valides
    all_images = [
        f for f in data_path.glob('*.jpg') 
        if len(f.stem.split('_')) >= 4 and not '.chip' in f.name
    ]
    
    if len(all_images) < n_tests:
        n_tests = len(all_images)
    
    # 🎲 SÉLECTION ALÉATOIRE
    test_images = random.sample(all_images, n_tests)
    
    print(f"🎲 Test sur {n_tests} images aléatoires\n")
    
    correct = 0
    total = 0
    
    for i, test_image in enumerate(test_images, 1):
        expected_class, age, gender = parse_filename(test_image.stem)
        
        if not expected_class:
            continue
        
        result = classifier.predict(str(test_image))
        
        is_correct = result['predicted_class'] == expected_class
        total += 1
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        
        print(f"{status} {i}. {test_image.name}")
        print(f"     Attendu: {expected_class:8s} (age={age:3d}, gender={gender})")
        print(f"     Prédit:  {result['predicted_class']:8s} ({result['confidence']:.1f}%)")
        print()
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n📊 RÉSULTAT FINAL")
        print(f"   Correct    : {correct}/{total}")
        print(f"   Accuracy   : {accuracy:.1f}%")

if __name__ == "__main__":
    print("="*60)
    print("TEST INFÉRENCE - IMAGE UNIQUE ALÉATOIRE")
    print("="*60)
    test_single_image()
    
    print("\n" + "="*60)
    print("TEST INFÉRENCE - BATCH ALÉATOIRE (5 images)")
    print("="*60)
    test_batch(n=5)
    
    print("\n" + "="*60)
    print("TEST INFÉRENCE - ÉCHANTILLON ALÉATOIRE (10 images)")
    print("="*60)
    test_multiple_random(n_tests=10)