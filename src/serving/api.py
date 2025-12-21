"""
API FastAPI pour servir le modèle de classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import io
from PIL import Image
import logging

from src.models.inference import ImageClassifier

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser l'application FastAPI
app = FastAPI(
    title="MLOps Image Classifier API",
    description="API de classification d'images (bébé, enfant, femme, homme)",
    version="1.0.0"
)

# Initialiser le classifier (au démarrage de l'app)
classifier = None

@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    global classifier
    try:
        classifier = ImageClassifier(
            model_path='models/artifacts/best_model.pth',
            config_path='configs/model_config.yaml'
        )
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        raise

# ============================================
# MODELS PYDANTIC
# ============================================

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    model_loaded: bool

# ============================================
# ENDPOINTS
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine"""
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Prédire la classe d'une image uploadée
    
    Args:
        file: Image (JPG, PNG)
    
    Returns:
        PredictionResponse avec classe, confiance, probabilités
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Vérifier type de fichier
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Utilisez JPG ou PNG"
        )
    
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Sauvegarder temporairement
        temp_path = "/tmp/temp_image.jpg"
        image.save(temp_path)
        
        # Prédiction
        result = classifier.predict(temp_path)
        
        logger.info(f"Prédiction: {result['predicted_class']} ({result['confidence']:.2f}%)")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Prédire sur plusieurs images
    
    Args:
        files: Liste d'images
    
    Returns:
        Liste de PredictionResponse
    """
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
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            temp_path = f"/tmp/temp_image_{idx}.jpg"
            image.save(temp_path)
            
            result = classifier.predict(temp_path)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Erreur image {idx}: {e}")
            results.append({
                "predicted_class": "error",
                "confidence": 0.0,
                "probabilities": {}
            })
    
    return results

@app.get("/classes", tags=["Info"])
async def get_classes():
    """Retourner les classes disponibles"""
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
    """Handler pour HTTPException"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler pour exceptions générales"""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur"}
    )