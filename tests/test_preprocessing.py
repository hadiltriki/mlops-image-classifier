"""
Tests unitaires pour le preprocessing
"""
import pytest
import torch
from PIL import Image
import numpy as np
from src.data.preprocessing import get_train_transform, get_val_transform

class TestTransforms:
    """Tests pour les transformations d'images"""
    
    @pytest.fixture
    def sample_image(self):
        """Créer une image de test"""
        # Créer une image RGB 100x100
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def test_train_transform_output_shape(self, sample_image):
        """Test que train_transform retourne la bonne forme"""
        transform = get_train_transform(img_size=224)
        output = transform(sample_image)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)  # (C, H, W)
    
    def test_val_transform_output_shape(self, sample_image):
        """Test que val_transform retourne la bonne forme"""
        transform = get_val_transform(img_size=224)
        output = transform(sample_image)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
    
    def test_transform_output_range(self, sample_image):
        """Test que les valeurs sont normalisées"""
        transform = get_val_transform()
        output = transform(sample_image)
        
        # Après normalisation, les valeurs peuvent être négatives ou positives
        # mais devraient être dans une plage raisonnable
        assert output.min() >= -5.0
        assert output.max() <= 5.0
    
    def test_different_image_sizes(self, sample_image):
        """Test avec différentes tailles d'images"""
        for size in [128, 224, 256]:
            transform = get_val_transform(img_size=size)
            output = transform(sample_image)
            assert output.shape == (3, size, size)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])