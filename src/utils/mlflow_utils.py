"""
Utilitaires MLflow pour tracking
"""
import mlflow
import mlflow.pytorch
from datetime import datetime

def log_prediction(prediction_data):
    """Log une prédiction dans MLflow"""
    with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("predicted_class", prediction_data['predicted_class'])
        mlflow.log_metric("confidence", prediction_data['confidence'])
        
        for cls, prob in prediction_data['probabilities'].items():
            mlflow.log_metric(f"prob_{cls}", prob)