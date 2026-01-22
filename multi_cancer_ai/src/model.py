"""
Definizione dell'architettura del modello di Deep Learning basata su EfficientNetV2.
Questo modulo costruisce la Rete Neurale Convoluzionale (CNN) utilizzando Transfer Learning.
"""

import tensorflow as tf

# Utilizziamo gli alias per accedere ai sottomoduli di Keras in modo pulito.
layers = tf.keras.layers
models = tf.keras.models

# Importiamo l'architettura EfficientNetV2B0 pre-addestrata.
# B0 è la variante più leggera e veloce ("Small"), ottima per bilanciare velocità e accuratezza.
EfficientNetV2B0 = tf.keras.applications.EfficientNetV2B0

from multi_cancer_ai.config import config

# ==============================================================================
# OTTIMIZZAZIONE PRESTAZIONI: MIXED PRECISON
# ==============================================================================
# La Mixed Precision usa float16 (16-bit) invece di float32 per i calcoli intermedi.
# Vantaggi:
# 1. Riduce l'uso della memoria video (VRAM) quasi della metà.
# 2. Raddoppia la velocità di calcolo su GPU moderne (NVIDIA serie 20xx e successive).
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[INFO] Mixed Precision (float16) abilitata per massimizzare le performance.")
except Exception as e:
    # Se la GPU non supporta FP16 o siamo su CPU, falliamo silenziosamente o logghiamo l'errore.
    print(f"[WARN] Impossibile abilitare Mixed Precision: {e}")


def build_model(num_classes):
    """
    Costruisce, assembla e compila il modello Keras per la classificazione multi-cancro.
    
    Architettura:
    - Input Layer: (224, 224, 3)
    - Backbone: EfficientNetV2B0 (ImageNet weights, frozen)
    - Head (Classificatore):
        - Global Average Pooling
        - Batch Normalization
        - Dropout
        - Dense (512)
        - Output Dense (num_classes, Softmax)
    
    Args:
        num_classes (int): Numero di classi di output (es. 8 per il dataset Multi Cancer).
        
    Returns:
        tf.keras.Model: Un'istanza di modello Keras compilata, pronta per .fit().
    """
    
    # -------------------------------------------------------------------------
    # 1. Definizione dell'Input
    # -------------------------------------------------------------------------
    # Definiamo il tensore di ingresso con la forma specificata nel config (224, 224, 3).
    inputs = layers.Input(shape=config.INPUT_SHAPE)

    # -------------------------------------------------------------------------
    # 2. Base Model (Feature Extractor)
    # -------------------------------------------------------------------------
    # Carichiamo EfficientNetV2B0 pre-addestrato su ImageNet.
    # include_top=False: Rimuove l'ultimo livello denso (che classifica le 1000 classi di ImageNet)
    # perchè vogliamo sostituirlo con il nostro classificatore personalizzato.
    # weights='imagenet': Carica i pesi appresi su milioni di immagini (Transfer Learning).
    # include_preprocessing=False: EfficientNetV2 si aspetta pixel [0, 255] se rescale è dentro il modello,
    # ma noi gestiamo la normalizzazione esternamente o nel data loader.
    # (Nota: EfficientNetV2 in Keras ha solitamente il preprocessing integrato, ma controlliamo data_loader).
    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        include_preprocessing=False 
    )
    
    # "Congeliamo" (Freeze) i pesi del modello base.
    # Questo impedisce che durante il primo training i pesi pre-addestrati vengano distrutti
    # dai gradienti elevati generati dai layer inizializzati casualmente (la testa).
    base_model.trainable = False

    # -------------------------------------------------------------------------
    # 3. Costruzione della Head di Classificazione (Top Layers)
    # -------------------------------------------------------------------------
    # Prendiamo l'output dell'ultimo livello convoluzionale di EfficientNet.
    # Shape attesa: (Batch, 7, 7, 1280) per B0.
    x = base_model.output
    
    # Global Average Pooling: Riduce le dimensioni spaziali facendo la media su H e W.
    # Trasforma (Batch, 7, 7, 1280) in un vettore (Batch, 1280).
    # Questo riduce drasticamente il numero di parametri rispetto al Flattening.
    x = layers.GlobalAveragePooling2D()(x)
    
    # Batch Normalization: Normalizza le attivazioni per stabilizzare il training.
    x = layers.BatchNormalization()(x)
    
    # Dropout (0.3): Spegne casualmente il 30% dei neuroni durante il training.
    # Serve a prevenire l'Overfitting (il modello memorizza i dati invece di imparare).
    x = layers.Dropout(0.3)(x)

    # Layer Denso Intermedio: Aggiunge capacità di apprendimento non lineare.
    # 512 neuroni con attivazione ReLU (Rectified Linear Unit).
    x = layers.Dense(512, activation='relu')(x)
    
    # Ancora Batch Normalization e Dropout per ulteriore regolarizzazione.
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Layer di Output: Il livello finale che produce le probabilità.
    # num_classes neuroni (uno per ogni tipo di cancro).
    # activation='softmax': Trasforma i numeri in uscita in probabilità che sommano a 1.
    # dtype='float32': Fondamentale quando si usa Mixed Precision. L'output deve tornare a 32-bit
    # per evitare instabilità numerica nel calcolo della loss (valori troppo piccoli in float16).
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name="predictions")(x)

    # -------------------------------------------------------------------------
    # 4. Creazione dell'Oggetto Modello
    # -------------------------------------------------------------------------
    # Creiamo il modello funzionale collegando Inputs e Outputs.
    model = models.Model(inputs=inputs, outputs=outputs, name="CancerClassifier_EffNetV2")

    # -------------------------------------------------------------------------
    # 5. Compilazione
    # -------------------------------------------------------------------------
    # Configuriamo il processo di apprendimento.
    
    # Optimizer Adam: Algoritmo di ottimizzazione standard ed efficace.
    # Learning Rate definito in config (1e-4).
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        # L'ottimizzatore che aggiornerà i pesi
        optimizer=optimizer,
        
        # Loss Function: Categorical Crossentropy è la funzione di costo standard per classificazione multi-classe
        # quando le etichette sono in formato one-hot encoding.
        loss='categorical_crossentropy', 
        
        # Metriche da monitorare durante il training.
        # Oltre all'Accuracy, monitoriamo Precision e Recall che sono cruciali in ambito medico.
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model
