"""
Test de l'API
"""
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health check"""
    response = requests.get(f"{API_URL}/health")
    print(f"✅ Health: {response.json()}")

def test_predict():
    """Test prédiction"""
    # Prendre une image
    img_path = Path("data/raw/UTKFace")
    images = list(img_path.glob("*.jpg"))
    
    if not images:
        print("❌ Pas d'images trouvées")
        return
    
    test_image = images[0]
    
    with open(test_image, "rb") as f:
        files = {"file": (test_image.name, f, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📸 Image: {test_image.name}")
        print(f"🎯 Prédiction: {result['predicted_class']}")
        print(f"📊 Confiance: {result['confidence']:.2f}%")
    else:
        print(f"❌ Erreur: {response.status_code}")

if __name__ == "__main__":
    print("="*50)
    print("TEST API")
    print("="*50)
    test_health()
    test_predict()