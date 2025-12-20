"""
Module de preprocessing des images
"""
from torchvision import transforms

def get_train_transform(img_size=224):
    """
    Transformations pour l'entraînement (avec augmentation)
    
    Args:
        img_size (int): Taille des images
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transform(img_size=224):
    """
    Transformations pour la validation (sans augmentation)
    
    Args:
        img_size (int): Taille des images
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])