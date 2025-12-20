cat > README.md << 'EOF'
# MLOps Image Classifier

Classification d'images (bébé/enfant/femme/homme) avec génération de captions.

##  Dataset
- **Source** : UTKFace (Kaggle)
- **Classes** : bébé (0-2 ans), enfant (3-12 ans), femme, homme
- **Taille** : 5,000 images pour training
- **Stockage** : DVC + Google Drive

##  Architecture MLOps
```
┌─────────────────┐      ┌──────────────┐      ┌─────────────┐
│ Kaggle Training │─────▶│ Google Drive │◀────▶│   GitHub    │
│   (GPU T4)      │      │     (DVC)    │      │   (Code)    │
└─────────────────┘      └──────────────┘      └─────────────┘
         │                                              │
         │                                              │
         ▼                                              ▼
┌─────────────────┐                          ┌─────────────────┐
│  FastAPI + Docker│                          │  CI/CD Pipeline │
│   (Production)   │                          │ (GitHub Actions)│
└─────────────────┘                          └─────────────────┘
```

## Stack Technique

- **ML** : ResNet50 (classification) + BLIP (captioning)
- **Framework** : PyTorch + Transformers
- **Tracking** : MLflow (local)
- **Versioning** : Git + DVC (Google Drive)
- **API** : FastAPI
- **CI/CD** : GitHub Actions
- **Deployment** : Docker
- **Monitoring** : Prometheus + Grafana

##  Quick Start

### 1. Clone et Setup
```bash
# Cloner le repo
git clone https://github.com/USERNAME/mlops-image-classifier.git
cd mlops-image-classifier

# Installer les dépendances
pip install -r requirements.txt

# Récupérer les données depuis Google Drive
dvc pull
```

### 2. Training (Kaggle)

1. Ouvrir le notebook : `notebooks/exploratory/kaggle_training.ipynb`
2. Importer dans Kaggle
3. Ajouter dataset UTKFace
4. Activer GPU T4
5. Run all cells

### 3. Local Development
```bash
# Lancer MLflow UI
mlflow ui

# Lancer les tests
pytest tests/

# Prétraiter les données
python src/data/preprocessing.py
```

### 4. API
```bash
# Lancer l'API
cd src/serving
uvicorn app:app --reload

# Tester
curl http://localhost:8000/health
```

## Structure du Projet
```
mlops-image-classifier/
├── configs/              # Configurations YAML
├── data/                 # Datasets (géré par DVC)
│   ├── raw/             # Données brutes
│   └── processed/       # Données prétraitées
├── docker/              # Dockerfiles
├── models/              # Modèles entraînés (géré par DVC)
├── notebooks/           # Notebooks Jupyter
├── src/                 # Code source
│   ├── data/           # Chargement et preprocessing
│   ├── models/         # Training et évaluation
│   ├── serving/        # API FastAPI
│   └── utils/          # Utilitaires
├── tests/              # Tests unitaires
├── dvc.yaml            # Pipeline DVC
└── requirements.txt    # Dépendances Python
```

## Gestion du Déséquilibre

Le dataset présente un déséquilibre naturel (7:1 homme/bébé).

**Solution implémentée :**
- Class weights dans CrossEntropyLoss
- Impact équilibré : 1000.0 pour toutes les classes
- F1-Score équilibré attendu

## Métriques Attendues

| Classe | Samples | F1-Score |
|--------|---------|----------|
| Bébé   | 258     | 0.70-0.80|
| Enfant | 286     | 0.70-0.80|
| Femme  | 1,627   | 0.75-0.85|
| Homme  | 1,829   | 0.75-0.85|

**Accuracy globale** : 70-75%  
**F1-Score moyen** : 0.75-0.80

## Workflow MLOps

1. **Training** : Kaggle (GPU gratuit)
2. **Versioning** : DVC → Google Drive
3. **Code** : GitHub
4. **Tests** : pytest + GitHub Actions
5. **Deployment** : Docker + FastAPI
6. **Monitoring** : Prometheus + Grafana

##  Équipe

- Hadil Triki
- Salma Louati
- Imen Bourassine

##  Timeline

-  Semaine 1 : Setup + Configuration
-  Semaine 2 : Training + EDA
-  Semaine 3-4 : Pipeline + Docker
- Semaine 5-6 : API + CI/CD
- Semaine 7-8 : Monitoring + Présentation

##  License

Projet académique - CESI École d'Ingénieurs 2024-2025
EOF