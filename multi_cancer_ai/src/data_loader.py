"""
Modulo per la gestione dei dati (Data Loading) e Data Augmentation.
Crea pipeline di dati efficienti usando tf.data per massimizzare l'utilizzo della GPU.
"""

import tensorflow as tf
from multi_cancer_ai.config import config

def create_generators():
    """
    Crea e restituisce i dataset tf.data ottimizzati per Training e Validation.
    
    Questa funzione:
    1. Legge le immagini dalla directory su disco.
    2. Divide in Training e Validation set.
    3. Mappa i nomi delle classi ai numeri (Label Encoding).
    4. Applica Data Augmentation (solo su training) per aumentare la varietà dei dati.
    5. Applica Prefetching per caricare i dati in background mentre la GPU lavora.
    
    Returns:
        tuple: (train_ds, val_ds, class_indices)
               - train_ds: Dataset di training (infinito/ripetibile per epoche)
               - val_ds: Dataset di validazione
               - class_indices: Dizionario {NomeClasse: IndiceNumerico}
    """
    print(f"[INFO] Caricamento dati da directory: {config.DATASET_PATH}")

    # ==============================================================================
    # 1. Caricamento Dataset Raw (da disco)
    # ==============================================================================
    
    # tf.keras.utils.image_dataset_from_directory è una utility potente che:
    # - Scansiona ricorsivamente le cartelle.
    # - Inferisce le etichette dai nomi delle sottocartelle.
    # - Ridimensiona le immagini.
    # - Crea batch.
    
    # Setup Dataset di TRAINING
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH,                 # Cartella radice
        validation_split=config.VALIDATION_SPLIT, # 0.2 (20% per validation)
        subset="training",                   # Selezioniamo il sottoinsieme "training" (80%)
        seed=config.SEED,                    # Seed per riproducibilità dello split
        image_size=config.IMG_SIZE,          # Resize a (224, 224)
        batch_size=config.BATCH_SIZE,        # Batch size (16)
        label_mode='categorical',            # Converte le label in vettori one-hot (es. [0, 1, 0...])
        shuffle=True                         # Mescola le immagini (fondamentale per il training)
    )

    # Setup Dataset di VALIDATION
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",                 # Selezioniamo il sottoinsieme "validation" (20%)
        seed=config.SEED,                    # DEVE essere uguale al training per avere split coerente
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='categorical',
        shuffle=False                        # Non mescoliamo la validazione (utile per valutazioni stabili)
    )
    
    # ==============================================================================
    # 2. Estrazione Metadati Classi
    # ==============================================================================
    
    # Recuperiamo i nomi delle classi (ordinati alfabeticamente dalle cartelle)
    class_names = train_ds.class_names
    
    # Creiamo un dizionario di mapping {Nome: Indice}
    # Esempio: {'ALL': 0, 'Brain Cancer': 1, ...}
    # Questo serve all'Evaluator e al sistema di inferenza.
    class_indices = {name: i for i, name in enumerate(class_names)}
    
    # "Patchiamo" gli oggetti dataset per portarsi dietro questo dizionario.
    # Utile per compatibilità con codice legacy che si aspetta ImageDataGenerator.
    train_ds.class_indices = class_indices
    val_ds.class_indices = class_indices

    # ==============================================================================
    # 3. Data Augmentation (Aumento Dati Sintetico)
    # ==============================================================================
    
    # Definiamo una sequenza di trasformazioni casuali.
    # Le applichiamo SOLO al training set per prevenire l'overfitting.
    # Questo rende il modello robusto a variazioni di orientamento, zoom, ecc.
    data_augmentation = tf.keras.Sequential([
        # Scala i valori dei pixel da [0, 255] a [0, 1] (Normalizzazione).
        tf.keras.layers.Rescaling(1./255),
        
        # Ruota casualmente l'immagine fino al 20%.
        tf.keras.layers.RandomRotation(0.2),
        
        # Esegue uno zoom casuale (in o out) fino al 20%.
        tf.keras.layers.RandomZoom(0.2),
        
        # Ribalta l'immagine orizzontalmente (utile per istologia non orientata).
        tf.keras.layers.RandomFlip("horizontal"),
        
        # Nota: Non usiamo RandomContrast qui per evitare alterazioni di colore critiche per l'istologia.
    ])
    
    # Layer di sola normalizzazione per la validazione.
    # I dati di validazione NON devono essere ruotati o zoomati, solo normalizzati.
    rescaling_layer = tf.keras.layers.Rescaling(1./255)

    # ==============================================================================
    # 4. Pipeline Ottimizzata tf.data
    # ==============================================================================
    
    # AUTOTUNE permette a TensorFlow di decidere dinamicamente quante risorse CPU dedicare
    # al caricamento dati in parallelo.
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare_train(ds):
        """Pipeline per il training dataset."""
        # 1. Applica Data Augmentation (in parallelo grazie a num_parallel_calls)
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                   num_parallel_calls=AUTOTUNE)
        
        # 2. Caching (Commentato per evitare OOM su dataset grandi).
        # ds = ds.cache() 
        
        # 3. Prefetching: Carica il prossimo batch in CPU mentre la GPU elabora quello attuale.
        # Elimina i tempi morti di attesa I/O.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def prepare_val(ds):
        """Pipeline per il validation dataset."""
        # 1. Applica SOLO Normalizzazione (niente rotazioni!)
        ds = ds.map(lambda x, y: (rescaling_layer(x), y), 
                   num_parallel_calls=AUTOTUNE)
        
        # 2. Prefetching
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # Applichiamo le pipeline
    train_ds = prepare_train(train_ds)
    val_ds = prepare_val(val_ds)

    print(f"[INFO] Pipeline tf.data inizializzata correttamente.")
    print(f"[INFO] Classi rilevate: {class_names}")

    # Restituiamo i dataset pronti e il mapping delle classi
    return train_ds, val_ds, class_indices
