# Multi-Cancer AI Diagnostic System 🏥

Sistema avanzato di Intelligenza Artificiale per la classificazione e la diagnosi assistita di 8 tipologie di tumori tramite analisi di immagini mediche.
Il progetto utilizza Deep Learning (EfficientNetV2), Data Augmentation avanzata e tecniche di Explainable AI (Grad-CAM) per fornire predizioni accurate e interpretabili.

## 🌟 Caratteristiche Principali

- **Supporto Multi-Classe**: Classificazione di 8 tipi di cancro (Leucemia, Cervello, Seno, Cervice, Rene, Polmone/Colon, Linfoma, Orale) e 26 sottoclassi.
- **Deep Learning SOTA**: Utilizza `EfficientNetV2B3` pre-addestrato su ImageNet e fine-tunato.
- **Desktop App Offline**: Applicazione nativa velocissima (CustomTkinter + TFLite) che non richiede internet.
- **Active Learning (v2.0)**: Sistema di auto-apprendimento che permette all'utente di correggere le diagnosi e migliorare il modello in tempo reale.
- **Smart Fallback**: Passaggio automatico tra TFLite (velocità) e Keras (potenza) in base alla compatibilità del sistema.
- **Explainability**: Integrazione di **Grad-CAM** (versione training) per visualizzare le aree determinanti.
- **Architettura Modulare**: Codice strutturato in moduli (Config, Data, Model, Trainer, Evaluator, DesktopApp).

## 📂 Struttura del Progetto

```
multi_cancer_ai/
├── config/             # Configurazione centralizzata
├── src/
│   ├── active_trainer.py   # [NEW] Logica di Active Learning
│   ├── data_loader.py      # Pipeline dati
│   ├── model.py            # Architettura EfficientNetV2
│   ├── trainer.py          # Training Loop
│   └── evaluator.py        # Reportistica
├── models/             # Modelli .h5 e .tflite
├── user_data/          # [NEW] Dati salvati dall'utente (correzioni)
├── desktop_app.py      # [NEW] Applicazione Desktop (GUI)
├── train_main.py       # Script di addestramento
├── run_desktop.sh      # Launcher App
└── requirements.txt    # Dipendenze
```

## 🚀 Installazione

1. **Prerequisiti**: Python 3.9+ (Consigliato 3.10/3.11).
2. **Installazione Dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Utilizzo

### 1. Avvio Applicazione Desktop (Consigliato)
Per lanciare l'interfaccia diagnostica offline:

```bash
./run_desktop.sh
```
*Include la nuova Home, Diagnosi e Active Training.*

### 2. Addestramento del Modello
Per ri-addestrare il modello da zero:

```bash
python train_main.py
```
Il browser si aprirà automaticamente all'indirizzo locale (es. `http://localhost:8501`).

## 📊 Mapping Classi
Il sistema riconosce le seguenti categorie principali:
- **ALL**: Acute Lymphoblastic Leukemia
- **Brain Cancer**: Tumori cerebrali (Glioma, Meningioma, Pituitary)
- **Breast Cancer**: Carcinoma mammario (Benign, Malignant)
- **Cervical Cancer**: Cancro della cervice
- **Kidney Cancer**: Cancro renale
- **Lung and Colon Cancer**: Adenocarcinomi polmonari e del colon
- **Lymphoma**: Linfomi (CLL, FL, MCL)
- **Oral Cancer**: Carcinoma orale

## ⚠️ Disclaimer Medico
Questo software è sviluppato esclusivamente per fini di ricerca e dimostrativi. **NON** è un dispositivo medico diagnostico certificato. I risultati forniti dall'AI devono essere sempre verificati da personale medico qualificato.

---
*Progetto sviluppato con ❤️ e Python.*
