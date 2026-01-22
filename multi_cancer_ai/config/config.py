"""
Questo modulo gestisce la configurazione centralizzata per l'intero progetto Multi-Cancer AI.
Contiene tutte le costanti, i percorsi dei file, i parametri del modello e le impostazioni di training.
Centralizzare la configurazione permette di modificare facilmente i parametri sperimentali (es. Batch Size, Learning Rate)
senza dover cercare in tutto il codice.
"""

# Importiamo 'os' per interagire con il sistema operativo (es. variabili d'ambiente)
import os

# Importiamo 'Path' da 'pathlib' per una gestione dei percorsi file orientata agli oggetti e indipendente dal SO (Windows/Mac/Linux)
from pathlib import Path

# ==============================================================================
# 1. PERCORSI DEL FILE SYSTEM (FILE SYSTEM PATHS)
# ==============================================================================

# Definiamo la directory di base del progetto.
# __file__ è il percorso di questo file (config.py).
# .resolve() ottiene il percorso assoluto.
# .parent risale alla cartella 'config'.
# .parent.parent risale alla cartella 'multi_cancer_ai' (root del pacchetto Python).
BASE_DIR = Path(__file__).resolve().parent.parent

# Definiamo il percorso dove risiede il dataset delle immagini.
# Si assume che la cartella 'Multi Cancer' sia parallela alla cartella del codice o in una posizione specifica.
# BASE_DIR.parent ci porta fuori da 'multi_cancer_ai', ipotizzando che il dataset sia nella root del repository o workspace.
# Percorso atteso: .../Multi-Cancer-AI-Diagnostic-System/Multi Cancer/Multi Cancer
DATASET_PATH = BASE_DIR.parent / "Multi Cancer" / "Multi Cancer"

# Definiamo dove salvare i modelli addestrati (.h5, .tflite).
MODELS_DIR = BASE_DIR / "models"

# Definiamo dove salvare i log di training (per TensorBoard o CSV).
LOGS_DIR = BASE_DIR / "logs"

# Definiamo dove salvare i risultati grafici (grafici accuracy/loss, matrici di confusione).
RESULTS_DIR = BASE_DIR / "results"

# Definiamo dove salvare i dati etichettati dall'utente (per Active Learning).
USER_DATA_DIR = BASE_DIR / "user_data"

# ==============================================================================
# 2. INIZIALIZZAZIONE DIRECTORY (DIRECTORY INITIALIZATION)
# ==============================================================================

# Creiamo le directory definite sopra se non esistono già.
# parents=True crea anche le cartelle genitore se mancanti (es. mkdir -p).
# exist_ok=True non genera errore se la cartella esiste già.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 3. PARAMETRI IMMAGINI (IMAGE PARAMETERS)
# ==============================================================================

# Larghezza dell'immagine in input alla rete neurale (in pixel).
IMG_WIDTH = 224

# Altezza dell'immagine in input alla rete neurale (in pixel).
IMG_HEIGHT = 224

# Tupla che rappresenta le dimensioni spaziali (Width, Height). Usata spesso da Keras.
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Numero di canali colore dell'immagine. 3 indica RGB (Red, Green, Blue).
CHANNELS = 3

# Shape completa del tensore di input: (224, 224, 3).
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

# ==============================================================================
# 4. IPERPARAMETRI DI TRAINING (TRAINING HYPERPARAMETERS)
# ==============================================================================

# Dimensione del batch (Batch Size): numero di immagini processate contemporaneamente dalla GPU/CPU.
# 16 è un valore conservativo per evitare errori di memoria (OOM) su macchine standard.
BATCH_SIZE = 16 

# Numero di epoche (Epochs): quante volte l'intero dataset viene visto dal modello durante il training.
# 30 è un buon compromesso tra tempo di training e convergenza a >99% di accuratezza.
EPOCHS = 30 

# Learning Rate (Tasso di Apprendimento): quanto velocemente il modello aggiorna i propri pesi.
# 1e-4 (0.0001) è un valore standard per il Fine-Tuning di modelli pre-addestrati come EfficientNet.
LEARNING_RATE = 1e-4

# Split di Validazione: percentuale dei dati di training riservata per la validazione (20%).
# Usiamo 0.2, quindi l'80% sarà per il training e il 20% per il test/validazione.
VALIDATION_SPLIT = 0.2

# Seed casuale per la riproducibilità degli esperimenti.
# Fissare il seed assicura che lo split train/val sia sempre lo stesso ad ogni esecuzione.
SEED = 42

# ==============================================================================
# 5. MAPPATURA CLASSI (CLASS MAPPING)
# ==============================================================================

# Dizionario che mappa i nomi delle cartelle del dataset (chiavi) in nomi leggibili per l'uomo (valori).
# Questo è fondamentale per visualizzare output comprensibili nell'interfaccia utente.
# ATTENZIONE: Le chiavi DEVONO corrispondere ESATTAMENTE ai nomi delle sottocartelle nel dataset.
CLASS_MAPPING = {
    'ALL': 'Leucemia Linfoblastica Acuta',         # Acute Lymphoblastic Leukemia
    'Brain Cancer': 'Cancro al Cervello',          # Generico (spesso Glioma/Meningioma)
    'Breast Cancer': 'Cancro al Seno',             # Carcinoma mammario
    'Cervical Cancer': 'Cancro alla Cervice',      # Carcinoma cervicale
    'Kidney Cancer': 'Cancro al Rene',             # Spesso Carcinoma a cellule renali
    'Lung and Colon Cancer': 'Cancro a Polmone e Colon', # Categoria mista (dataset legacy)
    'Lymphoma': 'Linfoma',                         # Tumore del sistema linfatico
    'Oral Cancer': 'Cancro Orale'                  # Carcinoma del cavo orale
}
