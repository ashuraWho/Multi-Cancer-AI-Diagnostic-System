# Multi-Cancer AI Diagnostic System

> **Sistema di Diagnostica Istopatologica Assistita da Intelligenza Artificiale**  
> *Enterprise-Grade Deep Learning Solution per Classificazione Multi-Cancro*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)](https://www.tensorflow.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP%208-blueviolet)](https://pep8.org/)

---

## 📑 Indice

- [Panoramica](#-panoramica)
- [Caratteristiche Principali](#-caratteristiche-principali)
- [Architettura del Sistema](#-architettura-del-sistema)
- [Installazione](#-installazione)
- [Quick Start](#-quick-start)
- [Pipeline di Training](#-pipeline-di-training)
- [Inference e Desktop App](#-inference-e-desktop-app)
- [Performance e Benchmark](#-performance-e-benchmark)
- [Riproducibilità](#-riproducibilità)
- [Troubleshooting](#-troubleshooting)
- [Contribuire](#-contribuire)
- [Disclaimer Medico](#-disclaimer-medico)
- [Licenza](#-licenza)

---

## 🌟 Panoramica

Il **Multi-Cancer AI Diagnostic System** è una soluzione completa di Computer Vision per la classificazione automatica di immagini istopatologiche in 8 categorie di cancro. Basato su **EfficientNetV2** e **Transfer Learning**, il sistema fornisce diagnosi assistite con interpretabilità tramite **Grad-CAM** e supporto per **Active Learning** per miglioramento continuo.

### Casi d'Uso

- **Ricerca Medica**: Analisi rapida di grandi volumi di immagini istologiche
- **Supporto Diagnostico**: Assistenza ai patologi nella classificazione preliminare
- **Educazione**: Tool didattico per studenti di medicina e patologia
- **Sviluppo**: Base per sistemi più complessi (WSI analysis, multi-scale detection)

### Classi Supportate

1. **ALL** (Leucemia Linfoblastica Acuta)
2. **Brain Cancer** (Tumori Cerebrali: Gliomi, Meningiomi)
3. **Breast Cancer** (Carcinoma Mammario)
4. **Cervical Cancer** (Carcinoma della Cervice)
5. **Kidney Cancer** (Carcinoma Renale)
6. **Lung and Colon Cancer** (Adenocarcinomi e Carcinomi Squamosi)
7. **Lymphoma** (Linfomi del Sistema Linfatico)
8. **Oral Cancer** (Carcinoma del Cavo Orale)

---

## ✨ Caratteristiche Principali

### 🧠 Deep Learning Avanzato
- **Architettura**: EfficientNetV2-B0 (Google AI) con Transfer Learning da ImageNet
- **Mixed Precision Training**: FP16 per ottimizzazione GPU (2x speedup, 50% meno memoria)
- **Data Augmentation**: Rotazione, zoom, flip per robustezza
- **Early Stopping & LR Scheduling**: Prevenzione overfitting e convergenza ottimale

### 🔍 Explainable AI (XAI)
- **Grad-CAM**: Visualizzazione heatmap delle regioni critiche per la decisione
- **Confidence Scores**: Probabilità per classe con visualizzazione interattiva
- **Confusion Matrix**: Analisi dettagliata degli errori di classificazione

### 🔄 Active Learning
- **Online Fine-Tuning**: Aggiornamento modello in tempo reale su correzioni utente
- **Offline Data Collection**: Accumulo di immagini etichettate per training completo
- **Human-in-the-Loop**: Integrazione feedback medico nel ciclo di apprendimento

### 📱 Deployment Ready
- **TFLite Export**: Modello ottimizzato per mobile/embedded (<20MB con FP16)
- **Desktop GUI**: Applicazione CustomTkinter cross-platform (Windows/Mac/Linux)
- **API-Ready**: Architettura modulare per integrazione REST/GraphQL

---

## 🏗️ Architettura del Sistema

### Overview Architetturale

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Dataset (Multi Cancer/)                                    │
│       │                                                     │
│       ├─► Data Loader (tf.data)                             │
│       │   ├─ Split Train/Val (80/20)                        │
│       │   ├─ Data Augmentation (Training)                   │
│       │   └─ Prefetching & Parallel I/O                     │
│       │                                                     │
│       ├─► Model Builder                                     │
│       │   ├─ EfficientNetV2-B0 (Frozen, ImageNet)           │
│       │   └─ Custom Head (Trainable)                        │
│       │       ├─ GlobalAveragePooling2D                     │
│       │       ├─ BatchNorm + Dropout                        │
│       │       └─ Dense(8, Softmax)                          │
│       │                                                     │
│       ├─► Trainer                                           │
│       │   ├─ ModelCheckpoint (best val_accuracy)            │
│       │   ├─ EarlyStopping (patience=10)                    │
│       │   ├─ ReduceLROnPlateau (factor=0.2)                 │
│       │   └─ CSVLogger (metrics tracking)                   │
│       │                                                     │
│       └─► Evaluator                                         │
│           ├─ Classification Report (Precision/Recall/F1)    │
│           ├─ Confusion Matrix                               │
│           └─ Grad-CAM Heatmaps                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Image (224x224x3)                                    │
│       │                                                     │
│       ├─► Preprocessing                                     │
│       │   ├─ Resize to (224, 224)                           │
│       │   └─ Normalize [0, 255] → [0, 1]                    │
│       │                                                     │
│       ├─► Model Inference                                   │
│       │   ├─ EfficientNetV2 Feature Extraction              │
│       │   └─ Classification Head                            │
│       │       └─ Softmax → Probabilities [8]                │
│       │                                                     │
│       ├─► Post-Processing                                   │
│       │   ├─ Top-K Predictions                              │
│       │   ├─ Confidence Scores                              │
│       │   └─ Class Name Mapping                             │
│       │                                                     │
│       └─► Optional: Grad-CAM                                │
│           ├─ Gradient Computation                           │
│           ├─ Heatmap Generation                             │
│           └─ Overlay Visualization                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Stack Tecnologico

| Componente | Tecnologia | Versione |
|------------|------------|----------|
| **Deep Learning** | TensorFlow / Keras | 2.16+ |
| **Architettura** | EfficientNetV2-B0 | Pre-trained (ImageNet) |
| **Data Pipeline** | tf.data | Native TensorFlow |
| **GUI** | CustomTkinter | 5.0+ |
| **Image Processing** | PIL / OpenCV | 9.0+ / 4.6+ |
| **Visualization** | Matplotlib / Seaborn | 3.5+ / 0.11+ |
| **Metrics** | scikit-learn | 1.0+ |

### Struttura del Codice

```
Multi-Cancer-AI-Diagnostic-System/
├── multi_cancer_ai/
│   ├── train_main.py              # 🎯 Entry point training
│   ├── desktop_app.py             # 🎯 Entry point GUI
│   │
│   ├── config/
│   │   └── config.py              # ⚙️ Configurazione centralizzata
│   │
│   ├── src/
│   │   ├── model.py               # 🧠 Architettura EfficientNetV2
│   │   ├── data_loader.py         # 🚚 Pipeline tf.data + augmentation
│   │   ├── trainer.py             # 🏋️ Training loop + callbacks
│   │   ├── evaluator.py           # 📊 Metrics + Grad-CAM
│   │   ├── active_trainer.py      # 🔄 Active Learning
│   │   ├── utils.py               # 🛠️ Logging + plotting
│   │   ├── localization.py        # 🌐 i18n (IT/EN)
│   │   └── knowledge_base.py      # 📚 Medical glossary
│   │
│   ├── tools/
│   │   ├── import_kaggle.py       # 📥 Dataset downloader
│   │   ├── finetune.py            # 🔧 Incremental training
│   │   └── inspect_dataset.py     # 🔍 Dataset inspector
│   │
│   ├── models/                    # 💾 Saved models (.h5, .tflite)
│   ├── results/                   # 📈 Training outputs (plots, CSV)
│   ├── logs/                      # 📝 Training logs
│   └── user_data/                 # 👤 Active Learning data
│
├── Multi Cancer/                  # 📦 Dataset (non committato)
│   └── Multi Cancer/
│       ├── ALL/
│       ├── Brain Cancer/
│       └── ...
│
├── .gitignore                     # 🚫 Git exclusions
├── requirements.txt               # 📋 Dependencies
└── README.md                      # 📖 This file
```

---

## 💻 Installazione

### Prerequisiti

- **Python**: 3.10 o superiore
- **Sistema Operativo**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)
- **RAM**: Minimo 8GB (16GB consigliato per training)
- **GPU**: Opzionale ma consigliata (NVIDIA con CUDA 11.8+ per training veloce)
- **Spazio Disco**: ~10GB per dataset + modelli

### Setup Step-by-Step

#### 1. Clone del Repository

```bash
git clone https://github.com/ashuraWho/Multi-Cancer-AI-Diagnostic-System.git
cd Multi-Cancer-AI-Diagnostic-System
```

#### 2. Creazione Ambiente Virtuale

```bash
# Creazione venv
python3 -m venv venv

# Attivazione (macOS/Linux)
source venv/bin/activate

# Attivazione (Windows)
venv\Scripts\activate
```

#### 3. Installazione Dipendenze

```bash
pip install --upgrade pip
pip install -r multi_cancer_ai/requirements.txt
```

**Nota**: Se hai una GPU NVIDIA, installa TensorFlow con supporto GPU:

```bash
pip install tensorflow[and-cuda]
```

#### 4. Configurazione Dataset

**Opzione A: Path Default**
```bash
# Scarica il dataset da Kaggle e posizionalo in:
Multi Cancer/Multi Cancer/
```

**Opzione B: Path Personalizzato**
```bash
# Configura variabile d'ambiente
export MULTI_CANCER_DATASET_PATH="/path/to/your/dataset/Multi Cancer"
```

**Struttura Dataset Attesa:**
```
Multi Cancer/
└── Multi Cancer/
    ├── ALL/
    │   ├── image_001.jpg
    │   └── ...
    ├── Brain Cancer/
    ├── Breast Cancer/
    └── ...
```

#### 5. Verifica Installazione

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "from multi_cancer_ai.config import config; print(f'Config loaded: {config.DATASET_PATH}')"
```

---

## 🚀 Quick Start

### Training Completo

```bash
# Avvia training end-to-end
python -m multi_cancer_ai.train_main

# Output atteso:
# - models/best_model.h5 (modello migliore)
# - models/model_optimized.tflite (versione mobile)
# - results/training_history.png (grafici)
# - results/confusion_matrix.png (matrice confusione)
# - results/training_log.csv (metriche per epoca)
```

### Inference con Desktop App

```bash
# Avvia GUI desktop
python -m multi_cancer_ai.desktop_app

# Funzionalità:
# - Upload immagine e classificazione
# - Visualizzazione Grad-CAM heatmap
# - Active Learning (correzione predizioni)
```

### Inference Programmabile

```python
import tensorflow as tf
from PIL import Image
import numpy as np
from multi_cancer_ai.config import config

# Carica modello
model = tf.keras.models.load_model("multi_cancer_ai/models/best_model.h5")

# Preprocess immagine
img = Image.open("path/to/image.jpg")
img = img.resize(config.IMG_SIZE)
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Inference
predictions = model.predict(img_batch)[0]
class_idx = np.argmax(predictions)
confidence = predictions[class_idx]

print(f"Classe: {class_idx}, Confidence: {confidence:.2%}")
```

---

## 🎓 Pipeline di Training

### Fasi del Training

1. **Data Loading** (`data_loader.py`)
   - Caricamento immagini da disco con `tf.keras.utils.image_dataset_from_directory`
   - Split automatico train/validation (80/20) con seed fisso per riproducibilità
   - Data augmentation su training set: `RandomRotation(0.2)`, `RandomZoom(0.2)`, `RandomFlip("horizontal")`
   - Normalizzazione pixel: `[0, 255] → [0, 1]`
   - Pipeline ottimizzata: `prefetch(buffer_size=AUTOTUNE)`, parallel I/O

2. **Model Building** (`model.py`)
   - Backbone: EfficientNetV2-B0 pre-trained su ImageNet (frozen)
   - Custom Head:
     ```
     GlobalAveragePooling2D()
     → BatchNormalization()
     → Dropout(0.3)
     → Dense(512, ReLU)
     → BatchNormalization()
     → Dropout(0.3)
     → Dense(num_classes, Softmax, dtype=float32)
     ```
   - Compilazione: Adam(lr=1e-4), CategoricalCrossentropy, metrics=[accuracy, precision, recall]

3. **Training Loop** (`trainer.py`)
   - Callbacks:
     - **ModelCheckpoint**: Salva `best_model.h5` quando `val_accuracy` migliora
     - **EarlyStopping**: Ferma se `val_loss` non migliora per 10 epoche (restore_best_weights=True)
     - **ReduceLROnPlateau**: Riduce LR di 5x se `val_loss` si blocca per 5 epoche (min_lr=1e-6)
     - **CSVLogger**: Salva tutte le metriche in `training_log.csv`

4. **Evaluation** (`evaluator.py`)
   - Predizioni su validation set completo
   - Classification Report (Precision, Recall, F1 per classe)
   - Confusion Matrix visualizzata con Seaborn
   - Grad-CAM per interpretabilità (opzionale)

5. **Export** (`train_main.py`)
   - Salvataggio modello finale
   - Conversione TFLite con quantizzazione FP16 (dimezza dimensione)

### Iperparametri Default

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `IMG_SIZE` | (224, 224) | Dimensione input immagini |
| `BATCH_SIZE` | 16 | Batch size (ridurre se OOM) |
| `EPOCHS` | 30 | Epoche massime (early stopping può fermare prima) |
| `LEARNING_RATE` | 1e-4 | Learning rate iniziale |
| `VALIDATION_SPLIT` | 0.2 | Percentuale dati per validation |
| `SEED` | 42 | Seed per riproducibilità |

### Modifica Iperparametri

Edita `multi_cancer_ai/config/config.py`:

```python
BATCH_SIZE = 32  # Aumenta se hai più GPU memory
EPOCHS = 50      # Più epoche per dataset grandi
LEARNING_RATE = 5e-5  # LR più basso per fine-tuning conservativo
```

---

## 🔬 Inference e Desktop App

### Desktop GUI

L'applicazione desktop (`desktop_app.py`) fornisce:

- **Diagnosis Tab**: Upload immagine, classificazione, visualizzazione probabilità
- **Grad-CAM Visualization**: Heatmap sovrapposta all'immagine originale
- **Active Training Tab**: Correzione predizioni errate e fine-tuning online
- **Dataset Info**: Struttura dataset e glossario medico

**Avvio:**
```bash
python -m multi_cancer_ai.desktop_app
```

**Features:**
- Supporto multi-lingua (IT/EN)
- Tema chiaro/scuro
- Validazione input (formato, dimensione file)
- Logging errori strutturato

### API Programmabile

Per integrazione in pipeline personalizzate:

```python
from multi_cancer_ai.src.evaluator import Evaluator, GradCAM
from multi_cancer_ai.src.model import build_model

# Carica modello
model = tf.keras.models.load_model("models/best_model.h5")

# Grad-CAM
grad_cam = GradCAM(model)
heatmap = grad_cam.compute_heatmap(img_array, class_idx=2)
overlay = grad_cam.overlay_heatmap(heatmap, original_img, alpha=0.4)
```

---

## 📊 Performance e Benchmark

### Metriche Attese

Su dataset Multi Cancer (129K+ immagini, 8 classi):

| Metrica | Valore Atteso | Note |
|---------|---------------|------|
| **Accuracy** | >95% | Su validation set |
| **Precision (macro)** | >0.90 | Media precision per classe |
| **Recall (macro)** | >0.90 | Media recall per classe |
| **F1-Score (macro)** | >0.90 | Media armonica precision/recall |
| **Inference Time** | <100ms | CPU (single image, batch=1) |
| **Inference Time (GPU)** | <20ms | NVIDIA RTX 3080 |

### Confusion Matrix

Le classi più confuse tipicamente:
- **Lung & Colon Cancer**: Entrambi adenocarcinomi, texture simili
- **Brain Cancer subtypes**: Glioma vs Meningioma (se non separati nel dataset)

### Ottimizzazioni Performance

1. **Mixed Precision (FP16)**: 2x speedup su GPU moderne
2. **tf.data Prefetching**: Elimina I/O bottleneck
3. **Batch Size Tuning**: Aumenta se GPU memory lo permette
4. **TFLite Quantization**: FP16 → 50% dimensione, velocità simile

---

## 🔬 Riproducibilità

### Seed e Determinismo

Il sistema usa seed fisso (`SEED=42`) per:
- Split train/validation riproducibile
- Inizializzazione pesi (se non usando pre-trained)
- Data augmentation (se applicabile)

**Nota**: TensorFlow non è completamente deterministico su GPU per default. Per riproducibilità assoluta:

```python
# In train_main.py, prima di importare tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

### Versioning Modelli

I modelli vengono salvati con naming fisso:
- `best_model.h5`: Modello migliore (val_accuracy)
- `final_model.h5`: Modello finale (ultima epoca)
- `model_optimized.tflite`: Versione mobile

**Raccomandazione**: Aggiungi timestamp per versioning:

```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"models/best_model_{timestamp}.h5")
```

### Logging e Tracciabilità

Tutti i training generano:
- **Log file**: `logs/TrainingPipeline_<timestamp>.log` (completo con stack trace)
- **CSV metrics**: `results/training_log.csv` (metriche per epoca, Excel-compatibile)
- **Plots**: `results/training_history.png`, `results/confusion_matrix.png`

---

## 🔧 Troubleshooting

### Errori Comuni

#### 1. `FileNotFoundError: Dataset path non trovato`

**Causa**: Dataset non presente o path errato.

**Soluzione**:
```bash
# Verifica path
ls -la "Multi Cancer/Multi Cancer/"

# Oppure configura env var
export MULTI_CANCER_DATASET_PATH="/path/to/dataset"
```

#### 2. `OutOfMemoryError (OOM)`

**Causa**: Batch size troppo grande o dataset troppo grande per GPU memory.

**Soluzione**:
```python
# In config/config.py
BATCH_SIZE = 8  # Riduci da 16 a 8 o 4

# Oppure disabilita caching in data_loader.py (già fatto)
# ds = ds.cache()  # Commentato per evitare OOM
```

#### 3. `NameError: name 'np' is not defined` in evaluator.py

**Causa**: Import numpy mancante (già fixato, ma se ricompare).

**Soluzione**: Verifica che `evaluator.py` contenga `import numpy as np` all'inizio.

#### 4. Training Loss non scende / Accuracy bloccata

**Possibili cause**:
- Learning rate troppo alto (riduci a 5e-5)
- Dataset sbilanciato (usa class weights)
- Overfitting (aumenta dropout o riduci epoche)

**Debug**:
```python
# Verifica distribuzione classi
from multi_cancer_ai.tools.inspect_dataset import get_structure_string
print(get_structure_string())
```

#### 5. macOS SegFault con Tkinter + TensorFlow

**Causa**: Conflitto TensorFlow Metal/GPU con Tkinter.

**Soluzione**: Già gestito in `desktop_app.py` (disabilita GPU):
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

#### 6. TFLite Conversion Fallisce

**Causa**: Modello con operazioni non supportate da TFLite.

**Soluzione**: Usa solo operazioni TFLite-compatibili o salta la conversione (opzionale).

### Debug Mode

Per logging dettagliato:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Per profiling performance:

```python
# In train_main.py
import tensorflow as tf
tf.profiler.experimental.start('logs/profile')
# ... training ...
tf.profiler.experimental.stop()
```

---

## 🤝 Contribuire

### Setup Sviluppo

```bash
# Clone repo
git clone https://github.com/ashuraWho/Multi-Cancer-AI-Diagnostic-System.git
cd Multi-Cancer-AI-Diagnostic-System

# Crea branch feature
git checkout -b feature/your-feature-name

# Installa in modalità sviluppo (se aggiungi setup.py)
pip install -e .
```

### Code Style

- **PEP 8**: Segui Python style guide
- **Type Hints**: Aggiungi type hints a tutte le funzioni
- **Docstrings**: Google/NumPy style (già implementato)
- **Logging**: Usa `logging` invece di `print()`

### Testing

```bash
# Unit tests (se aggiunti)
pytest tests/

# Linting
flake8 multi_cancer_ai/
black multi_cancer_ai/  # Formattazione automatica
```

### Pull Request Process

1. Fork del repository
2. Crea feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Apri Pull Request

### Aree di Contribuzione

- 🐛 **Bug Fixes**: Correzione errori esistenti
- ✨ **Features**: Nuove funzionalità (es. supporto WSI, API REST)
- 📚 **Documentation**: Miglioramento README, docstring, tutorial
- 🧪 **Testing**: Aggiunta unit test, integration test
- ⚡ **Performance**: Ottimizzazioni velocità/memoria
- 🌐 **i18n**: Traduzioni aggiuntive (FR, DE, ES, etc.)

---

## ⚖️ Disclaimer Medico

> **⚠️ ATTENZIONE: SOLO PER SCOPI DI RICERCA E DIDATTICI**

Questo software è un **CDSS (Clinical Decision Support System)** sperimentale e **NON** è un dispositivo medico approvato.

### Limitazioni

1. **Non Approvato**: Non possiede marcatura CE, approvazione FDA, o certificazioni mediche
2. **Solo Ricerca**: Inteso per ricerca accademica e didattica, non per uso clinico reale
3. **Supervisione Obbligatoria**: Qualsiasi output deve essere verificato da un patologo umano qualificato
4. **Nessuna Garanzia**: L'AI può sbagliare - non sostituisce la diagnosi medica professionale

### Responsabilità

L'autore e i contributori declinano ogni responsabilità per:
- Uso in ambito clinico reale senza supervisione medica
- Conseguenze derivanti da decisioni basate esclusivamente sull'output del sistema
- Perdite di dati o malfunzionamenti del software

### Raccomandazioni

- **Sempre verificare** le predizioni con analisi patologica tradizionale
- **Non usare** come unico strumento diagnostico
- **Consultare** sempre un medico specialista per diagnosi definitive
- **Rispettare** le normative locali sull'uso di AI in ambito medico

---

## 📞 Contatti e Supporto

- **Issues**: [GitHub Issues](https://github.com/ashuraWho/Multi-Cancer-AI-Diagnostic-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ashuraWho/Multi-Cancer-AI-Diagnostic-System/discussions)
- **Email**: emanueleanzellotti@gmail.com

---

## 🙏 Ringraziamenti

- **Dataset**: [Multi Cancer Dataset su Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer)
- **Architettura**: EfficientNetV2 (Google AI Research)
- **Framework**: TensorFlow / Keras team
- **Community**: Tutti i contributori e tester

---

**⭐ Se questo progetto ti è utile, considera di lasciare una stella su GitHub!**

---

*Ultimo aggiornamento: Gennaio 2026*
