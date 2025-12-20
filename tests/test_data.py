"""Tests pour le module data"""
import pytest
from src.data.load_data import parse_utkface_filename

def test_parse_baby():
    """Test parsing bébé"""
    result = parse_utkface_filename("1_0_0_20170101.jpg")
    assert result['category'] == 0
    assert result['label'] == "bebe"

def test_parse_child():
    """Test parsing enfant"""
    result = parse_utkface_filename("10_1_2_20170101.jpg")
    assert result['category'] == 1
    assert result['label'] == "enfant"

def test_parse_woman():
    """Test parsing femme"""
    result = parse_utkface_filename("25_1_0_20170101.jpg")
    assert result['category'] == 2
    assert result['label'] == "femme"

def test_parse_man():
    """Test parsing homme"""
    result = parse_utkface_filename("30_0_1_20170101.jpg")
    assert result['category'] == 3
    assert result['label'] == "homme"

def test_invalid_filename():
    """Test fichier invalide"""
    result = parse_utkface_filename("invalid.jpg")
    assert result is None