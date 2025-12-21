"""
Script de démo pour MLflow monitoring
"""


import sys
import os
from pathlib import Path



# Maintenant on peut importer depuis src
from src.utils.mlflow_logger import init_mlflow, log_prediction, log_batch_predictions, log_model_info
from src.data.load_data import  classify_image
from src.models.inference import ImageClassifier

import random
# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def demo_single_predictions():
    """Démo de logging de prédictions individuelles"""
    print("=" * 60)
    print("DÉMO MLFLOW - PRÉDICTIONS INDIVIDUELLES")
    print("=" * 60)
    
    # Initialiser MLflow
    init_mlflow()
    print("✅ MLflow initialisé")
    
    # Charger le classifier
    classifier = ImageClassifier()
    print("✅ Modèle chargé")
    
    # Dataset
    data_path = Path('data/raw/UTKFace')
    if not data_path.exists():
        print(f"❌ Dataset non trouvé: {data_path}")
        return
    
    images = [f for f in data_path.glob('*.jpg') if len(f.stem.split('_')) >= 4]
    
    if len(images) == 0:
        print(f"❌ Aucune image trouvée dans {data_path}")
        return
    
    # Prendre 5 images aléatoires
    num_samples = min(5, len(images))
    sample_images = random.sample(images, num_samples)
    
    print(f"\n📊 Logging de {len(sample_images)} prédictions dans MLflow...\n")
    
    for i, img_path in enumerate(sample_images, 1):
        try:
            # Parser le nom pour obtenir la vraie classe
            parts = img_path.stem.split('_')
            age = int(parts[0])
            gender = int(parts[1])
            actual_class = classify_image(age, gender)
            
            # Prédiction
            result = classifier.predict(str(img_path))
            
            # Logger dans MLflow
            log_prediction(
                image_path=str(img_path),
                predicted_class=result['predicted_class'],
                confidence=result['confidence'] / 100,  # Convertir en 0-1
                probabilities={k: v/100 for k, v in result['probabilities'].items()},
                actual_class=actual_class
            )
            
            # Afficher
            is_correct = "✅" if result['predicted_class'] == actual_class else "❌"
            print(f"{i}. {img_path.name}")
            print(f"   Prédit: {result['predicted_class']} | Réel: {actual_class} {is_correct}")
            print(f"   Confiance: {result['confidence']:.1f}%\n")
        
        except Exception as e:
            print(f"❌ Erreur sur {img_path.name}: {e}")
            continue
    
    print("✅ Prédictions loggées dans MLflow !")
    print(f"📊 Ouvrir: http://localhost:5000")

def demo_batch_prediction():
    """Démo de logging de batch"""
    print("\n" + "=" * 60)
    print("DÉMO MLFLOW - BATCH PRÉDICTIONS")
    print("=" * 60)
    
    init_mlflow()
    classifier = ImageClassifier()
    
    # Dataset
    data_path = Path('data/raw/UTKFace')
    if not data_path.exists():
        print(f"❌ Dataset non trouvé: {data_path}")
        return
    
    images = [f for f in data_path.glob('*.jpg') if len(f.stem.split('_')) >= 4]
    
    if len(images) == 0:
        print(f"❌ Aucune image trouvée")
        return
    
    # Batch de 10 images
    num_samples = min(10, len(images))
    sample_images = random.sample(images, num_samples)
    
    print(f"\n📊 Logging d'un batch de {len(sample_images)} prédictions...\n")
    
    predictions = []
    
    for img_path in sample_images:
        try:
            # Parser
            parts = img_path.stem.split('_')
            age = int(parts[0])
            gender = int(parts[1])
            actual_class = classify_image(age, gender)
            
            # Prédiction
            result = classifier.predict(str(img_path))
            
            predictions.append({
                'image_path': str(img_path),
                'predicted_class': result['predicted_class'],
                'actual_class': actual_class,
                'confidence': result['confidence'] / 100
            })
        except Exception as e:
            print(f"⚠️ Erreur sur {img_path.name}: {e}")
            continue
    
    if len(predictions) == 0:
        print("❌ Aucune prédiction réussie")
        return
    
    # Logger le batch
    log_batch_predictions(predictions)
    
    # Stats
    correct = sum(1 for p in predictions if p['predicted_class'] == p['actual_class'])
    accuracy = (correct / len(predictions)) * 100
    avg_conf = (sum(p['confidence'] for p in predictions) / len(predictions)) * 100
    
    print(f"✅ Batch loggé dans MLflow !")
    print(f"📊 Accuracy: {accuracy:.1f}%")
    print(f"📊 Confiance moyenne: {avg_conf:.1f}%")
    print(f"📊 Ouvrir: http://localhost:5000")

def demo_model_info():
    """Démo de logging des infos du modèle"""
    print("\n" + "=" * 60)
    print("DÉMO MLFLOW - INFO MODÈLE")
    print("=" * 60)
    
    init_mlflow()
    
    log_model_info(
        model_path="models/artifacts/best_model.pth",
        accuracy=0.716,  # 71.6%
        dataset_size=23708
    )
    
    print("✅ Infos du modèle loggées dans MLflow !")
    print(f"📊 Ouvrir: http://localhost:5000")

if __name__ == "__main__":
    print("🚀 DÉMO MONITORING MLFLOW\n")
    
    try:
        # 1. Prédictions individuelles
        demo_single_predictions()
        
        # 2. Batch
        demo_batch_prediction()
        
        # 3. Infos modèle
        demo_model_info()
        
        print("\n" + "=" * 60)
        print("✅ DÉMO TERMINÉE")
        print("=" * 60)
        print("\n📊 Ouvrir MLflow UI: http://localhost:5000")
        print("🔍 Explore les runs, métriques et artéfacts !")
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()