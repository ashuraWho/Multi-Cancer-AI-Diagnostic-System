# 🩺 Multi-Cancer AI Diagnostic System

> **Sistema di Diagnostica Istopatologica Assistita da Intelligenza Artificiale**  
> *Una soluzione Enterprise-Grade basata su TensorFlow 2.x, EfficientNetV2 e Active Learning.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-orange)
![Type](https://img.shields.io/badge/AI-Computer%20Vision-purple)
![Domain](https://img.shields.io/badge/MedTech-Oncology-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Indice dei Contenuti

1.  [Introduzione e Visione](#-introduzione-e-visione)
2.  [Contesto Medico e Scientifico](#-contesto-medico-e-scientifico)
3.  [Architettura Tecnologica](#-architettura-tecnologica)
4.  [Funzionalità Chiave](#-funzionalità-chiave)
5.  [Struttura del Progetto](#-struttura-del-progetto)
6.  [Guida all'Installazione](#-guida-allinstallazione)
7.  [Manuale Operativo (Usage)](#-manuale-operativo-usage)
8.  [Interpretabilità (XAI) e Grad-CAM](#-interpretabilità-xai-e-grad-cam)
9.  [Disclaimer Legale](#-disclaimer-legale)

---

## 🌟 Introduzione e Visione

Il **Multi-Cancer AI Diagnostic System** nasce con un obiettivo ambizioso: democratizzare l'accesso a diagnosi istologiche rapide e accurate, fornendo ai patologi uno "secondo occhio" digitale instancabile.

### Per i Non-Tecnici
Immagina un assistente virtuale che ha studiato milioni di immagini di tessuti umani al microscopio. Quando un medico carica l'immagine di una biopsia, questo sistema la analizza pixel per pixel e suggerisce: "Attenzione, questo tessuto assomiglia al 99% a un Linfoma". Non sostituisce il medico, ma lo aiuta a non trascurare nulla e a lavorare più velocemente.

### Per gli Esperti (Tecnici/Medici)
Il sistema è una pipeline *End-to-End* di Computer Vision basata su **Reti Neurali Convoluzionali (CNN)**. Utilizza un approccio di **Transfer Learning** su architettura **EfficientNetV2**, ottimizzata con **Mixed Precision Training (FP16)** per massimizzare il throughput su GPU. Include un modulo di **Active Learning** per il miglioramento continuo del modello tramite feedback umano (Human-in-the-Loop) e tecniche di **Explainable AI (Grad-CAM)** per garantire la trasparenza decisionale (Black-Box transparency).

---

## 🔬 Contesto Medico e Scientifico

### Il Problema: L'Istopatologia
La diagnosi definitiva del cancro avviene quasi sempre tramite istopatologia: l'analisi al microscopio di un campione di tessuto (biopsia). È un processo:
*   **Complesso**: Richiede anni di specializzazione.
*   **Soggettivo**: Patologi diversi possono dare interpretazioni diverse su casi limite.
*   **Lento**: Richiede tempo prezioso.

### La Soluzione AI
Il nostro sistema lavora su "Patches" (ritagli quadrati) di immagini istologiche digitalizzate (**Whole Slide Images - WSI**). Classifica il tessuto in 8 categorie principali:

1.  **ALL (Leucemia Linfoblastica Acuta)**: Cancro del sangue che colpisce i linfoblasti nel midollo osseo.
2.  **Brain Cancer (Tumore Cerebrale)**: Rilevamento di Gliomi (cellule gliali) e Meningiomi.
3.  **Breast Cancer (Cancro al Seno)**: Carcinomi duttali invasivi (il tipo più comune).
4.  **Cervical Cancer (Cancro Cervice)**: Carcinomi a cellule squamose, spesso legati all'HPV.
5.  **Kidney Cancer (Cancro Rene)**: Carcinoma a cellule renali (RCC).
6.  **Lung & Colon**: Adenocarcinomi (ghiandolari) e Carcinomi Squamosi.
7.  **Lymphoma (Linfoma)**: Tumori del sistema linfatico (Linfociti B e T).
8.  **Oral Cancer (Cancro Orale)**: Carcinomi del cavo orale (lingua, gengive).

> **Nota**: Il sistema distingue non solo tra "Sano" e "Malato", ma identifica lo specifico organo e tipo di patologia basandosi sulle texture cellulari (nuclei, citoplasma, organizzazione tissutale).

> **Dataset Kaggle**: Multi Cancer Dataset [Link](https://www.kaggle.com/datasets/obulisainaren/multi-cancer)

---

## ⚙️ Architettura Tecnologica

Il cuore del sistema è una Rete Neurale profonda. Ecco le scelte ingegneristiche nel dettaglio:

### 1. Il Modello: EfficientNetV2 (Google AI)
Abbiamo scelto `EfficientNetV2-B0` invece di architetture classiche (come ResNet50 o VGG16) per tre motivi:
*   **Efficienza**: Minore numero di parametri, addestramento più veloce.
*   **Accuratezza**: Utilizza blocchi *Fused-MBConv* che catturano meglio le feature locali tipiche dei tessuti biologici.
*   **Training Speed**: Ottimizzato per le moderne TPU e GPU.

### 2. Trasferimento della Conoscenza (Transfer Learning)
Non partiamo da zero ("Tabula Rasa"). Il modello nasce già "imparato" sul dataset **ImageNet** (14 milioni di immagini generiche). 
*   **Fase 1 (Frozen)**: Congeliamo la "base" della rete. Usiamo le sue capacità visive generiche (riconoscere bordi, curve, texture) per estrarre caratteristiche dalle immagini mediche.
*   **Fase 2 (Fine-Tuning)**: Sostituiamo la "testa" della rete con nuovi strati neurali specifici per le nostre 8 classi di cancro e addestriamo solo quelli.

### 3. Pipeline Dati Ottimizzata (tf.data)
Il caricamento delle immagini è spesso il collo di bottiglia. Usiamo l'API `tf.data` per creare una pipeline asincrona:
*   **Parallelismo**: La CPU cariche e prepara le immagini *mentre* la GPU addestra il blocco precedente.
*   **Data Augmentation**: Creiamo artificialmente nuove immagini (rotazioni, zoom) *al volo* per rendere il modello robusto a variazioni.
*   **Prefetching**: Manteniamo sempre la GPU "sfamata" di dati.

---

## 🚀 Funzionalità Chiave

### 🔄 Active Learning (Apprendimento Attivo)
Il modello non è statico. Se l'AI sbaglia una diagnosi e un medico la corregge, il sistema può eseguire un **Micro-Training istantaneo** su quell'immagine specifica. Questo permette al modello di adattarsi a casi rari o specifici del laboratorio locale senza dover riaddestrare tutto da zero.

### 👁️ Explainable AI (XAI) con Grad-CAM
Uno dei problemi dell'AI è la "Scatola Nera" (Black Box): ti dice "è cancro" ma non *perché*.
Il nostro modulo **Grad-CAM** (Gradient-weighted Class Activation Mapping) genera una "mappa di calore" termica sovrapposta all'immagine originale.
*   **Rosso**: L'area che ha convinto l'AI (es. un nucleo cellulare irregolare).
*   **Blu**: Aree ignorate (es. sfondo vuoto).
Questo permette al medico di *fidarsi* (o meno) della diagnosi verificando se l'AI ha guardato la zona giusta.

### 📱 Ottimizzazione TFLite (Mobile Ready)
Alla fine del training, il modello viene compressa e convertita in formato `TFLite` con quantizzazione **Float16**.
*   **Risultato**: Un file leggero (<20MB) che può girare su smartphone, tablet o dispositivi embedded (Raspberry Pi) in un microscopio intelligente, senza bisogno di internet.

---

## 📂 Struttura del Progetto

Una panoramica per orientarsi nel codice `multi_cancer_ai/src/`:

| File | Ruolo | Descrizione |
| :--- | :--- | :--- |
| `train_main.py` | 🎬 Regista | Script principale. Coordina tutto: carica dati, crea modello, addestra, salva. |
| `config.py` | ⚙️ Impostazioni | Contiene tutti i parametri (Path del dataset, Learning Rate, Dimensioni Immagini). Cambia qui, cambia ovunque. |
| `model.py` | 🧠 Cervello | Definisce l'architettura della Rete Neurale (EfficientNetV2 + Custom Head). |
| `data_loader.py` | 🚚 Logistica | Gestisce il caricamento efficiente delle immagini dal disco alla memoria GPU. |
| `trainer.py` | 🏋️ Coach | Gestisce il ciclo di addestramento, i salvataggi automatici e lo stop anticipato (Early Stopping). |
| `evaluator.py` | 📊 Analista | Calcola le metriche (Precisione, Matrice di Confusione) e genera le mappe Grad-CAM. |
| `knowledge_base.py` | 📚 Enciclopedia | Contiene le definizioni mediche testuali mostrate all'utente. |

---

## 💻 Guida all'Installazione

### Prerequisiti
*   **Computer**: Mac, Windows o Linux. (GPU NVIDIA consigliata ma non obbligatoria).
*   **Python**: Versione 3.10 o successiva.

### Setup Rapido
1.  **Scarica il codice**:
    ```bash
    git clone https://github.com/ashuraWho/Multi-Cancer-AI-Diagnostic-System.git
    cd Multi-Cancer-AI-Diagnostic-System
    ```

2.  **Prepara l'ambiente virtuale** (Isola le librerie per non fare conflitti):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    # venv\Scripts\activate   # Windows
    ```

3.  **Installa le librerie**:
    ```bash
    pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python
    ```

---

## 🕹️ Manuale Operativo (Usage)

### 1. Preparazione Dataset
Assicurati che le tue immagini siano organizzate in cartelle così:
```
Multi Cancer/
  ├── ALL/
  ├── Breast Cancer/
  ├── ... (altre classi)
```
*Se il percorso è diverso, modifica `DATASET_PATH` in `multi_cancer_ai/config/config.py`.*

### 2. Avvio Training
Lancia il comando magico. Il sistema farà tutto da solo.
```bash
python3 -m multi_cancer_ai.train_main
```
Vedrai scorrere i log: "Fase 1: Caricamento...", "Fase 2: Training...".

### 3. Analisi Risultati
Al termine (dopo circa 30-60 minuti a seconda del PC), troverai nella cartella `results/`:
*   `training_history.png`: Grafico che mostra se il modello ha imparato bene.
*   `confusion_matrix.png`: Grafico che mostra quali cancri vengono confusi tra loro.
*   `training_log.csv`: I dati grezzi excel-compatibili.

Nella cartella `models/` troverai:
*   `best_model.h5`: Il modello "perfetto".
*   `model_optimized.tflite`: La versione "leggera" per l'app.

---

## ⚖️ Disclaimer Legale

> **⚠️ ATTENZIONE: SOLO PER SCOPI DI RICERCA E DIDATTICI.**

Questo software è un **CDSS (Clinical Decision Support System)** sperimentale.
1.  **Non è un Dispositivo Medico**: Non possiede marcatura CE o approvazione FDA.
2.  **Responsabilità**: L'autore declina ogni responsabilità per l'uso in ambito clinico reale.
3.  **Supervisione**: Qualsiasi output del sistema deve essere verificato da un patologo umano qualificato. L'AI può sbagliare.