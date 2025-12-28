"""
Tests unitaires pour le module de chargement de données
"""
import pytest
import pandas as pd
from pathlib import Path
from src.data.load_data import (
    parse_utkface_filename,
    classify_image,
    load_utkface_dataset
)

class TestParseFilename:
    """Tests pour la fonction parse_utkface_filename"""
    
    def test_parse_valid_filename(self):
        """Test parsing d'un nom de fichier valide"""
        filename = "25_0_2_20170116174525125.jpg"
        result = parse_utkface_filename(filename)
        
        assert result is not None
        assert result['age'] == 25
        assert result['gender'] == 0
        assert result['race'] == 2
    
    def test_parse_invalid_filename(self):
        """Test parsing d'un nom de fichier invalide"""
        filename = "invalid_filename.jpg"
        result = parse_utkface_filename(filename)
        
        assert result is None
    
    def test_parse_baby_filename(self):
        """Test parsing pour un bébé"""
        filename = "1_1_0_20170116174525125.jpg"
        result = parse_utkface_filename(filename)
        
        assert result['age'] == 1
        assert result['gender'] == 1

class TestClassifyImage:
    """Tests pour la fonction classify_image"""
    
    def test_classify_baby(self):
        """Test classification bébé (0-2 ans)"""
        assert classify_image(age=0, gender=0) == 'bebe'
        assert classify_image(age=1, gender=1) == 'bebe'
        assert classify_image(age=2, gender=0) == 'bebe'
    
    def test_classify_child(self):
        """Test classification enfant (3-12 ans)"""
        assert classify_image(age=3, gender=0) == 'enfant'
        assert classify_image(age=8, gender=1) == 'enfant'
        assert classify_image(age=12, gender=0) == 'enfant'
    
    def test_classify_woman(self):
        """Test classification femme"""
        assert classify_image(age=25, gender=1) == 'femme'
        assert classify_image(age=50, gender=1) == 'femme'
    
    def test_classify_man(self):
        """Test classification homme"""
        assert classify_image(age=25, gender=0) == 'homme'
        assert classify_image(age=50, gender=0) == 'homme'
    
    def test_classify_edge_cases(self):
        """Test cas limites"""
        # Enfant de 13 ans
        assert classify_image(age=13, gender=0) == 'enfant'
        assert classify_image(age=13, gender=1) == 'enfant'

class TestLoadDataset:
    """Tests pour load_utkface_dataset"""
    
    @pytest.mark.skipif(
        not Path('data/raw/UTKFace').exists(),
        reason="Dataset UTKFace non disponible"
    )
    def test_load_dataset_returns_dataframe(self):
        """Test que load_dataset retourne bien un DataFrame"""
        df = load_utkface_dataset('data/raw/UTKFace')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    @pytest.mark.skipif(
        not Path('data/raw/UTKFace').exists(),
        reason="Dataset UTKFace non disponible"
    )
    def test_dataframe_has_correct_columns(self):
        """Test que le DataFrame a les bonnes colonnes"""
        df = load_utkface_dataset('data/raw/UTKFace')
        
        expected_columns = ['path', 'age', 'gender', 'race', 'label']
        assert list(df.columns) == expected_columns
    
    @pytest.mark.skipif(
        not Path('data/raw/UTKFace').exists(),
        reason="Dataset UTKFace non disponible"
    )
    def test_labels_are_valid(self):
        """Test que les labels sont valides"""
        df = load_utkface_dataset('data/raw/UTKFace')
        
        valid_labels = {'bebe', 'enfant', 'femme', 'homme'}
        unique_labels = set(df['label'].unique())
        
        assert unique_labels.issubset(valid_labels)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])