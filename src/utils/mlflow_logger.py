"""
MLflow logging pour le monitoring des prédictions
"""
import mlflow
import mlflow.pytorch
from datetime import datetime
from pathlib import Path

# Configuration MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "image_classification_production"

def init_mlflow():
    """Initialiser MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_prediction(image_path: str, predicted_class: str, confidence: float, 
                  probabilities: dict, actual_class: str = None):
    """
    Logger une prédiction dans MLflow
    
    Args:
        image_path: Chemin de l'image
        predicted_class: Classe prédite
        confidence: Niveau de confiance
        probabilities: Dict des probabilités pour chaque classe
        actual_class: Vraie classe (optionnel)
    """
    with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log des paramètres
        mlflow.log_param("image", Path(image_path).name)
        mlflow.log_param("predicted_class", predicted_class)
        
        if actual_class:
            mlflow.log_param("actual_class", actual_class)
            mlflow.log_param("correct", predicted_class == actual_class)
        
        # Log des métriques
        mlflow.log_metric("confidence", confidence)
        
        # Log des probabilités pour chaque classe
        for class_name, prob in probabilities.items():
            mlflow.log_metric(f"prob_{class_name}", prob)
        
        # Log de l'image (optionnel mais impressionnant)
        mlflow.log_artifact(image_path, "input_images")

def log_batch_predictions(predictions: list):
    """
    Logger un batch de prédictions
    
    Args:
        predictions: Liste de dicts avec les prédictions
    """
    with mlflow.start_run(run_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Calculer les stats du batch
        total = len(predictions)
        confidences = [p['confidence'] for p in predictions]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Si on a les vraies classes
        if all('actual_class' in p for p in predictions):
            correct = sum(1 for p in predictions if p['predicted_class'] == p['actual_class'])
            accuracy = correct / total
            mlflow.log_metric("batch_accuracy", accuracy)
            mlflow.log_metric("correct_predictions", correct)
        
        # Log des métriques du batch
        mlflow.log_metric("batch_size", total)
        mlflow.log_metric("avg_confidence", avg_confidence)
        mlflow.log_metric("max_confidence", max(confidences))
        mlflow.log_metric("min_confidence", min(confidences))
        
        # Distribution des classes prédites
        class_counts = {}
        for p in predictions:
            cls = p['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in class_counts.items():
            mlflow.log_metric(f"count_{cls}", count)

def log_model_info(model_path: str, accuracy: float, dataset_size: int):
    """
    Logger les infos du modèle
    
    Args:
        model_path: Chemin du modèle
        accuracy: Accuracy sur le dataset de validation
        dataset_size: Taille du dataset
    """
    with mlflow.start_run(run_name=f"model_info_{datetime.now().strftime('%Y%m%d')}"):
        # Paramètres du modèle
        mlflow.log_param("model_architecture", "ResNet50")
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("dataset_size", dataset_size)
        mlflow.log_param("num_classes", 4)
        mlflow.log_param("classes", "bebe,enfant,femme,homme")
        
        # Métriques
        mlflow.log_metric("validation_accuracy", accuracy)