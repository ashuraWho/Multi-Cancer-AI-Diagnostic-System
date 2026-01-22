"""
Definizione dell'architettura del modello di Deep Learning basata su EfficientNetV2.
"""
import tensorflow as tf
import tensorflow as tf
# Use tf.keras accessors for robustness across TF versions
layers = tf.keras.layers
models = tf.keras.models
EfficientNetV2B0 = tf.keras.applications.EfficientNetV2B0
from multi_cancer_ai.config import config

# Abilita Mixed Precision (FP16) per performance migliori (2x velocità su GPU moderne)
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[INFO] Mixed Precision (float16) abilitata.")
except Exception as e:
    print(f"[WARN] Impossibile abilitare Mixed Precision: {e}")


def build_model(num_classes):
    """
    Costruisce e compila il modello per la classificazione.
    
    Strategy: Transfer Learning con EfficientNetV2B3.
    1. Base Model pre-addestrato su ImageNet (Feature Extractor).
    2. Head di classificazione personalizzata.
    
    Args:
        num_classes (int): Numero di classi di output (8 per questo progetto).
        
    Returns:
        tf.keras.Model: Modello compilato pronto per l'addestramento.
    """
    # 1. Definizione Input
    inputs = layers.Input(shape=config.INPUT_SHAPE)

    # 2. Base Model (EfficientNetV2B3)
    # include_top=False rimuove l'ultimo layer di classificazione (1000 classi ImageNet)
    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        include_preprocessing=False # Gestiamo noi il preprocessing o rescaling
    )
    
    # Congeliamo i pesi del base model inizialmente per non distruggere le feature apprese
    base_model.trainable = False

    # 3. Costruzione della Head di Classificazione
    x = base_model.output
    
    # Global Average Pooling riduce le dimensioni spaziali (H,W) -> 1 vettore
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  # Dropout per ridurre overfitting

    # Layer Denso Intermedio
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Layer di Output (Softmax per classificazione multi-classe)
    # IMPORTANTE: Con mixed_precision, l'output deve essere float32 per stabilità numerica
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name="predictions")(x)

    # 4. Creazione del Modello
    model = models.Model(inputs=inputs, outputs=outputs, name="CancerClassifier_EffNetV2")

    # 5. Compilazione
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # Loss per one-hot encoded labels
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
