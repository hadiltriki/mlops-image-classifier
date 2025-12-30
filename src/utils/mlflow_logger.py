"""
MLflow logging pour le monitoring des prédictions
Compatible local + Docker
"""
import os
import mlflow
import mlflow.pytorch
from datetime import datetime
from pathlib import Path


# ============================================
# CONFIGURATION MLFLOW (DOCKER FRIENDLY)
# ============================================

# Par défaut: stockage local (pas besoin de serveur)
# Pour utiliser un serveur: export MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "./mlruns"  # Stockage local par défaut
)

EXPERIMENT_NAME = "image_classification_production"

# ============================================
# INITIALISATION
# ============================================

def init_mlflow():
    """Initialiser MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Indiquer le mode utilisé
    if MLFLOW_TRACKING_URI.startswith("http"):
        print(f"📊 MLflow - Mode serveur: {MLFLOW_TRACKING_URI}")
    else:
        print(f"📊 MLflow - Mode local: {MLFLOW_TRACKING_URI}")

# ============================================
# LOG PRÉDICTION UNIQUE
# ============================================

def log_prediction(
    image_path: str,
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    actual_class: str = None
):
    """
    Logger une prédiction dans MLflow
    
    Args:
        image_path: Chemin de l'image
        predicted_class: Classe prédite
        confidence: Confiance (0-1)
        probabilities: Dict {classe: probabilité}
        actual_class: Classe réelle (optionnel)
    """
    with mlflow.start_run(
        run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Paramètres
        mlflow.log_param("image", Path(image_path).name)
        mlflow.log_param("predicted_class", predicted_class)
        
        if actual_class is not None:
            mlflow.log_param("actual_class", actual_class)
            mlflow.log_param("correct", predicted_class == actual_class)
        
        # Métriques
        mlflow.log_metric("confidence", confidence)
        
        for class_name, prob in probabilities.items():
            mlflow.log_metric(f"prob_{class_name}", prob)
        
        # Artifact (image)
        if Path(image_path).exists():
            mlflow.log_artifact(image_path, artifact_path="input_images")

# ============================================
# LOG BATCH DE PRÉDICTIONS
# ============================================

def log_batch_predictions(predictions: list):
    """
    Logger un batch de prédictions
    
    Args:
        predictions: Liste de dicts avec:
            - predicted_class
            - confidence
            - actual_class (optionnel)
    """
    with mlflow.start_run(
        run_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        total = len(predictions)
        confidences = [p["confidence"] for p in predictions]
        
        # Métriques de base
        mlflow.log_metric("batch_size", total)
        mlflow.log_metric("avg_confidence", sum(confidences) / total)
        mlflow.log_metric("max_confidence", max(confidences))
        mlflow.log_metric("min_confidence", min(confidences))
        
        # Accuracy si on a les vraies classes
        if all("actual_class" in p for p in predictions):
            correct = sum(
                1 for p in predictions
                if p["predicted_class"] == p["actual_class"]
            )
            mlflow.log_metric("batch_accuracy", correct / total)
            mlflow.log_metric("correct_predictions", correct)
        
        # Distribution des classes
        class_counts = {}
        for p in predictions:
            cls = p["predicted_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in class_counts.items():
            mlflow.log_metric(f"count_{cls}", count)

# ============================================
# LOG INFOS MODÈLE
# ============================================

def log_model_info(model_path: str, accuracy: float, dataset_size: int):
    """
    Logger les informations du modèle
    
    Args:
        model_path: Chemin du modèle
        accuracy: Accuracy sur validation (0-1)
        dataset_size: Taille du dataset
    """
    with mlflow.start_run(
        run_name=f"model_info_{datetime.now().strftime('%Y%m%d')}"
    ):
        # Paramètres du modèle
        mlflow.log_param("model_architecture", "ResNet50")
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("dataset_size", dataset_size)
        mlflow.log_param("num_classes", 4)
        mlflow.log_param("classes", "bebe,enfant,femme,homme")
        
        # Métrique de performance
        mlflow.log_metric("validation_accuracy", accuracy)
def log_and_register_model(model, model_name="image_classifier"):
    # IMPORTANT : définir l'experiment AVANT start_run
    mlflow.set_experiment("image_classification_production")

    with mlflow.start_run(run_name="register_model"):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

