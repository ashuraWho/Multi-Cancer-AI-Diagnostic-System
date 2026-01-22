"""
Configurazione centralizzata per il progetto di classificazione Multi-Cancro.
"""
import os
from pathlib import Path

# Percorsi base
# Si assume che lo script venga eseguito dalla root del progetto 'multi_cancer_ai'
# O che la cartella 'Multi Cancer' sia allo stesso livello di 'multi_cancer_ai' o dentro 'archive'
BASE_DIR = Path(__file__).resolve().parent.parent
# Path del dataset: /Users/ashura/Desktop/archive/Multi Cancer/Multi Cancer
# Modificare questo percorso se il dataset si trova altrove
DATASET_PATH = BASE_DIR.parent / "Multi Cancer" / "Multi Cancer"

# Percorsi di output
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Assicura che le directory esistano
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parametri Immagini
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 3
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

# Iperparametri Training
BATCH_SIZE = 16 # Reduced to avoid OOM
EPOCHS = 30  # Increased for >99% accuracy
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
SEED = 42

# Mappatura delle Classi (Folder Name -> Readable Name)
# Questa mappa aiuta a normalizzare i nomi per i report
# Nota: Le chiavi devono corrispondere esattamente ai nomi delle sottocartelle
CLASS_MAPPING = {
    'ALL': 'Leucemia Linfoblastica Acuta',
    'Brain Cancer': 'Cancro al Cervello',
    'Breast Cancer': 'Cancro al Seno',
    'Cervical Cancer': 'Cancro alla Cervice',
    'Kidney Cancer': 'Cancro al Rene',
    'Lung and Colon Cancer': 'Cancro a Polmone e Colon',
    'Lymphoma': 'Linfoma',
    'Oral Cancer': 'Cancro Orale'
}
