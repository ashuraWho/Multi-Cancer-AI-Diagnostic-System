"""
Modulo per la gestione dei dati, data augmentation e creazione dei dataset tf.data.
"""
import tensorflow as tf
from multi_cancer_ai.config import config

def create_generators():
    """
    Crea e restituisce i dataset tf.data per Training e Validation.
    
    Utilizza tf.data.Dataset per massimizzare le performance (prefetch, parallelismo).
    
    Returns:
        tuple: (train_ds, val_ds) - Dataset ottimizzati
    """
    print(f"[INFO] Caricamento dati da: {config.DATASET_PATH}")

    # 1. Caricamento Dataset (Training)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

    # 2. Caricamento Dataset (Validation)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Estraiamo i nomi delle classi PRIMA delle trasformazioni (utile per l'app e i log)
    class_names = train_ds.class_names
    # Creiamo un attributo 'class_indices' fake per retro-compatibilità con il codice esistente che si aspetta generatori
    # Creiamo un dizionario {NomeClasse: Indice}
    class_indices = {name: i for i, name in enumerate(class_names)}
    
    # Patching degli oggetti dataset per includere attributi utili (simulando ImageDataGenerator)
    train_ds.class_indices = class_indices
    val_ds.class_indices = class_indices

    # 3. Definizione Data Augmentation Layers
    # Usiamo i layer di Keras che sono GPU-accelerated
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255), # Normalizzazione
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        # tf.keras.layers.RandomContrast(0.2), # Opzionale
    ])
    
    rescaling_layer = tf.keras.layers.Rescaling(1./255) # Solo normalizzazione per validation

    # 4. Applicazione Trasformazioni e Ottimizzazione Performance
    # AUTOTUNE permette a TF di decidere dinamicamente il buffer ottimale
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare_train(ds):
        # Augmentation -> Prefetch (Removed cache to save RAM)
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                   num_parallel_calls=AUTOTUNE)
        # ds = ds.cache() # Removed: caused OOM (Killed: 9) on large datasets
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def prepare_val(ds):
        # Solo Rescaling -> Prefetch (Removed cache)
        ds = ds.map(lambda x, y: (rescaling_layer(x), y), 
                   num_parallel_calls=AUTOTUNE)
        # ds = ds.cache() # Removed: caused OOM
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = prepare_train(train_ds)
    val_ds = prepare_val(val_ds)

    print(f"[INFO] Pipeline tf.data inizializzata. Classi: {class_names}")

    # Return metadata explicitly
    return train_ds, val_ds, class_indices
