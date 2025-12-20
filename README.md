# 🎯 MLOps Image Classifier

Projet MLOps - Classification d'images avec détection d'émotions

## 📋 Description

Système de classification d'images en 4 catégories (bébé, enfant, femme, homme) avec détection d'émotions et génération de captions.

### Objectifs
- Classification démographique (4 classes)
- Détection d'émotions (7 émotions)
- Génération de captions intelligentes
- Pipeline MLOps complet

## 🏗️ Architecture
```
mlops-image-classifier/
├── configs/              # Configurations YAML
├── data/                 # Datasets (géré par DVC)
├── docker/              # Dockerfiles
├── models/              # Modèles (géré par DVC)
├── notebooks/           # Notebooks Jupyter
├── src/                 # Code source
├── tests/              # Tests unitaires
└── requirements.txt    # Dépendances
```

## 🚀 Installation

### Prérequis
- Python 3.12
- Git
- DVC

### Setup
```bash
# Cloner le repository
git clone https://github.com/hadiltriki/mlops-image-classifier.git
cd mlops-image-classifier

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données depuis DagsHub
dvc pull
```

## 📊 Dataset

- **Source :** UTKFace (Kaggle)
- **Taille :** 23,000 images
- **Utilisé :** 5,000 images (échantillon)
- **Split :** 80% train / 20% validation

## 🤖 Modèle

- **Architecture :** ResNet50 (from scratch)
- **Classes :** bébé, enfant, femme, homme
- **Accuracy :** 71.6%
- **Émotions :** 7 émotions détectées (FER model)

## 📈 Résultats

| Classe  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Bébé    | 0.886     | 0.484  | 0.626    |
| Enfant  | 0.441     | 0.417  | 0.429    |
| Femme   | 0.704     | 0.683  | 0.693    |
| Homme   | 0.751     | 0.825  | 0.786    |

## 🛠️ Technologies

- **ML :** PyTorch, TensorFlow
- **Versioning :** Git, DVC
- **Storage :** DagsHub
- **Deployment :** Docker, FastAPI (à venir)
- **CI/CD :** GitHub Actions (à venir)
- **Monitoring :** MLflow (à venir)

