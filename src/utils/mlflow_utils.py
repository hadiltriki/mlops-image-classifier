"""Utilitaires MLflow"""
import mlflow
import os
from pathlib import Path

def setup_mlflow(tracking_uri="./mlruns", experiment_name="image-classifier"):
    """
    Configure MLflow local
    
    Args:
        tracking_uri: Chemin local pour MLflow
        experiment_name: Nom de l'expérience
    """
    tracking_path = Path(tracking_uri).absolute()
    mlflow.set_tracking_uri(f"file://{tracking_path}")
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow configuré")
    print(f"📂 Tracking: {mlflow.get_tracking_uri()}")
    print(f"🧪 Experiment: {experiment_name}")
    
    return mlflow

def log_metrics_only(params, metrics):
    """
    Log seulement params et metrics
    
    Args:
        params: Dict des hyperparamètres
        metrics: Dict des métriques
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Run logged: {run_id}")
        
        return run_id

if __name__ == "__main__":
    mlf = setup_mlflow()
    log_metrics_only(
        params={"test": "value"},
        metrics={"accuracy": 0.95}
    )