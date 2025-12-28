"""
Module de preprocessing des images
Transformations identiques à celles utilisées dans Kaggle
"""
from torchvision import transforms


def get_inference_transform(img_size=224):
    """
    Transformations pour l'inférence (identiques à Kaggle)
    
    Args:
        img_size (int): Taille des images (224 par défaut)
    
    Returns:
        torchvision.transforms.Compose: Pipeline de transformations
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])


# Alias pour compatibilité avec différents noms
get_val_transform = get_inference_transform
get_test_transform = get_inference_transform