"""
Modulo per la gestione del training del modello.
"""
import os
import tensorflow as tf
# Use tf.keras accessors for robustness
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
CSVLogger = tf.keras.callbacks.CSVLogger
from multi_cancer_ai.config import config

class Trainer:
    """
    Classe che incapsula la logica di addestramento.
    """
    def __init__(self, model, train_gen, val_gen, logger):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.logger = logger
        self.history = None

    def _get_callbacks(self):
        """Definisce i callback per il training."""
        
        # 1. Salva il modello solo quando la validation accuracy migliora
        checkpoint = ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "best_model.h5"),
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        # 2. Ferma il training se non c'è miglioramento per 'patience' epoche
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        # 3. Riduci il learning rate se il training stalla
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # 4. Log dei risultati su CSV
        csv_logger = CSVLogger(str(config.LOGS_DIR / "training_log.csv"))
        
        return [checkpoint, early_stop, reduce_lr, csv_logger]

    def train(self):
        """
        Esegue il training del modello.
        """
        self.logger.info("Inizio Training...")
        
        callbacks = self._get_callbacks()
        
        self.history = self.model.fit(
            self.train_gen,
            epochs=config.EPOCHS,
            validation_data=self.val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Training completato.")
        return self.history
    
    def save_final_model(self, filename="final_model.h5"):
        """Salva il modello finale su disco."""
        path = config.MODELS_DIR / filename
        self.model.save(str(path))
        self.logger.info(f"Modello finale salvato in: {path}")
