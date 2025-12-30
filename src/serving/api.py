"""
API FastAPI pour servir le modèle de classification
Avec monitoring MLflow (production ready)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import io
import logging
import os
from PIL import Image

from src.models.inference import ImageClassifier
from src.utils.mlflow_logger import init_mlflow, log_prediction

# ============================================
# LOGGING
# ============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="MLOps Image Classifier API",
    description="API de classification d'images (bébé, enfant, femme, homme)",
    version="1.0.0"
)

classifier = None

# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialisation MLflow + chargement du modèle"""
    global classifier
    try:
        # Initialiser MLflow
        init_mlflow()
        logger.info("✅ MLflow initialisé")

        # Charger le modèle
        model_path = os.getenv(
            "MODEL_PATH",
            "models/artifacts/best_model.pth"
        )

        classifier = ImageClassifier(
            model_path=model_path,
            config_path="configs/model_config.yaml"
        )

        logger.info("✅ Modèle chargé avec succès")

    except Exception as e:
        logger.error(f"❌ Erreur au démarrage API: {e}")
        raise

# ============================================
# PYDANTIC MODELS
# ============================================

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# ============================================
# ROUTES
# ============================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "MLOps Image Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

# ============================================
# PREDICTION SINGLE
# ============================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):

    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Utilisez JPG ou PNG"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        temp_path = "/tmp/input_image.jpg"
        image.save(temp_path)

        result = classifier.predict(temp_path)

        # 🔥 LOG MLFLOW
        log_prediction(
            image_path=temp_path,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"] / 100,
            probabilities={k: v / 100 for k, v in result["probabilities"].items()}
        )

        logger.info(
            f"Prédiction: {result['predicted_class']} "
            f"({result['confidence']:.2f}%)"
        )

        return result

    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# PREDICTION BATCH
# ============================================

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):

    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images par batch"
        )

    results = []

    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            temp_path = f"/tmp/input_image_{idx}.jpg"
            image.save(temp_path)

            result = classifier.predict(temp_path)
            results.append(result)

            # 🔥 LOG MLFLOW (par image)
            log_prediction(
                image_path=temp_path,
                predicted_class=result["predicted_class"],
                confidence=result["confidence"] / 100,
                probabilities={k: v / 100 for k, v in result["probabilities"].items()}
            )

        except Exception as e:
            logger.error(f"Erreur image {idx}: {e}")
            results.append({
                "predicted_class": "error",
                "confidence": 0.0,
                "probabilities": {}
            })

    return results

# ============================================
# INFO
# ============================================

@app.get("/classes", tags=["Info"])
async def get_classes():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return {
        "classes": classifier.classes,
        "num_classes": classifier.num_classes
    }

# ============================================
# EXCEPTION HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur"}
    )
