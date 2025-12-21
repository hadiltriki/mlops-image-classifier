"""
Module d'inférence pour faire des prédictions
"""
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from pathlib import Path
import yaml
from typing import Dict, List

from src.data.preprocessing import get_val_transform

class ImageClassifier:
    """Classe pour charger le modèle et faire des prédictions"""
    
    def __init__(self, model_path: str = 'models/artifacts/best_model.pth', 
                 config_path: str = 'configs/model_config.yaml'):
        """
        Initialise le classifier
        
        Args:
            model_path: Chemin vers le modèle .pth
            config_path: Chemin vers la config YAML
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {self.device}")
        
        # Charger config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['model']['classes']
        self.num_classes = self.config['model']['num_classes']
        
        # Créer et charger le modèle
        self.model = self._load_model(model_path)
        
        # Transformations
        img_size = self.config['preprocessing']['image_size']
        self.transform = get_val_transform(img_size=img_size)
        
        print(f"✅ Modèle chargé: {model_path}")
        print(f"📊 Classes: {self.classes}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Charge le modèle ResNet50"""
        # Créer architecture
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Charger les poids
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extraire state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Nettoyer les clés (enlever préfixe "resnet.")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('resnet.', '')
            new_state_dict[new_key] = value
        
        # Charger le state_dict nettoyé
        model.load_state_dict(new_state_dict, strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_path: str) -> Dict:
        """
        Fait une prédiction sur une image
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            dict avec predicted_class, confidence, probabilities
        """
        # Charger et transformer image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        # Résultats
        predicted_class = self.classes[predicted_idx.item()]
        confidence_pct = confidence.item() * 100
        
        probs_dict = {
            self.classes[i]: probabilities[0][i].item() * 100
            for i in range(self.num_classes)
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_pct,
            'probabilities': probs_dict
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Prédictions sur plusieurs images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            result['image_path'] = img_path
            results.append(result)
        return results

if __name__ == "__main__":
    # Test
    classifier = ImageClassifier()
    print(" Classifier initialisé")