"""
Utilitaires MLflow pour le tracking des prédictions
Compatible local + Docker
"""

import os
import mlflow
import mlflow.pytorch
from datetime import datetime

# ============================================
# CONFIGURATION MLFLOW (DOCKER FRIENDLY)
# ============================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

EXPERIMENT_NAME = "image_classification_production"

# ============================================
# INITIALISATION
# ============================================

def init_mlflow():
    """Initialiser MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

# ============================================
# LOG PRÉDICTION
# ============================================

def log_prediction(prediction_data: dict):
    """
    Logger une prédiction dans MLflow

    prediction_data attendu :
    {
        "predicted_class": str,
        "confidence": float,
        "probabilities": dict
    }
    """
    with mlflow.start_run(
        run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        mlflow.log_param(
            "predicted_class",
            prediction_data["predicted_class"]
        )

        mlflow.log_metric(
            "confidence",
            prediction_data["confidence"]
        )

        for cls, prob in prediction_data["probabilities"].items():
            mlflow.log_metric(f"prob_{cls}", prob)
