"""
Modulo per la gestione del ciclo di training del modello.

Incapsula la logica di addestramento, configurazione dei callback (checkpointing,
early stopping, learning rate scheduling) e salvataggio automatico del modello migliore.

Author: Multi-Cancer AI Team
License: MIT
"""

from typing import List, Optional
import logging
import tensorflow as tf

# Importiamo classi callback specifiche da Keras in modo esplicito per chiarezza.
# ModelCheckpoint: Salva il modello periodicamente.
# EarlyStopping: Interrompe il training se non ci sono miglioramenti.
# ReduceLROnPlateau: Riduci il learning rate se si raggiunge un plateau.
# CSVLogger: Salva le metriche in un file CSV.
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
CSVLogger = tf.keras.callbacks.CSVLogger

from multi_cancer_ai.config import config

class Trainer:
    """
    Classe Orchestratore del Training.
    
    Gestisce il ciclo di vita completo dell'addestramento:
    - Configurazione automatica dei callback (checkpoint, early stopping, LR scheduling)
    - Esecuzione del training loop (model.fit)
    - Salvataggio del modello migliore e dello storico delle metriche
    
    Attributes:
        model: Modello Keras compilato da addestrare.
        train_gen: Dataset di training (tf.data.Dataset).
        val_gen: Dataset di validazione (tf.data.Dataset).
        logger: Logger Python standard per tracciamento eventi.
        history: Storia del training (History object da model.fit), None fino al termine.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        train_gen: tf.data.Dataset,
        val_gen: tf.data.Dataset,
        logger: logging.Logger
    ) -> None:
        """
        Inizializza il Trainer con modello, dataset e logger.
        
        Args:
            model: Il modello Keras compilato da addestrare.
            train_gen: Dataset di training (con augmentation).
            val_gen: Dataset di validazione (solo normalizzazione).
            logger: Logger Python standard per registrare eventi e metriche.
        
        Raises:
            TypeError: Se model non è un'istanza di tf.keras.Model.
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"model deve essere tf.keras.Model, got {type(model)}")
        
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.logger = logger
        self.history: Optional[tf.keras.callbacks.History] = None
        
        # Log di stato iniziale
        self.logger.info(f"Trainer inizializzato. Epochs: {config.EPOCHS}, Batch Size: {config.BATCH_SIZE}")

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Configura e restituisce la lista dei callback di Keras.
        
        Callback configurati:
        1. ModelCheckpoint: Salva best_model.h5 quando val_accuracy migliora
        2. EarlyStopping: Ferma il training se val_loss non migliora per 10 epoche
        3. ReduceLROnPlateau: Riduce LR di 5x se val_loss si blocca per 5 epoche
        4. CSVLogger: Salva tutte le metriche in training_log.csv per analisi post-hoc
        
        Returns:
            Lista di oggetti Callback pronti per essere passati a model.fit().
        
        Note:
            - I callback operano sulla validation set per evitare overfitting
            - restore_best_weights=True in EarlyStopping garantisce il modello migliore
        """
        callbacks_list = []
        
        # 1. ModelCheckpoint
        # Salva i pesi del modello su disco ogni volta che la 'val_accuracy' migliora.
        # Questo garantisce di avere sempre la versione "migliore" salvata, non l'ultima.
        checkpoint = ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "best_model.h5"), # Percorso file (es. models/best_model.h5)
            monitor='val_accuracy',      # Metrica da monitorare (Accuracy su Validation)
            verbose=1,                   # Stampa messaggio quando salva
            save_best_only=True,         # True = sovrascrive solo se migliore del precedente
            mode='max'                   # 'max' perché vogliamo massimizzare l'accuracy
        )
        callbacks_list.append(checkpoint)
        
        # 2. EarlyStopping
        # Interrompe il training prematuramente se la loss di validazione smette di scendere.
        # Previene spreco di tempo e overfitting.
        early_stop = EarlyStopping(
            monitor='val_loss',          # Monitoriamo l'errore su validation
            patience=10,                 # Attendi 10 epoche di "non miglioramento" prima di fermarti
            verbose=1,
            restore_best_weights=True    # Importante: alla fine ripristina i pesi dell'epoca migliore
        )
        callbacks_list.append(early_stop)
        
        # 3. ReduceLROnPlateau
        # Se il modello si blocca in un minimo locale (loss non scende), riduce il learning rate
        # per fare passi più piccoli e "raffinare" la ricerca del minimo globale.
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,                  # Moltiplica il LR attuale per 0.2 (divide per 5)
            patience=5,                  # Dopo 5 epoche di stallo
            min_lr=1e-6,                 # Limite inferiore sotto cui non scendere
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # 4. CSVLogger
        # Salva lo storico delle metriche (loss, acc, val_loss, val_acc) in un file CSV.
        # Utile per analisi post-training (es. plottare curve in Excel/Python).
        csv_logger = CSVLogger(str(config.RESULTS_DIR / "training_log.csv"))
        callbacks_list.append(csv_logger)
        
        return callbacks_list

    def train(self) -> tf.keras.callbacks.History:
        """
        Avvia il processo di training effettivo (model.fit).
        
        Esegue il training loop completo con monitoraggio delle metriche,
        applicazione dei callback (checkpoint, early stopping, LR scheduling),
        e validazione periodica su validation set.
        
        Returns:
            History object contenente i valori delle metriche per ogni epoca:
            - history.history['loss']: Training loss per epoca
            - history.history['val_loss']: Validation loss per epoca
            - history.history['accuracy']: Training accuracy per epoca
            - history.history['val_accuracy']: Validation accuracy per epoca
            - E altre metriche configurate (precision, recall)
        
        Raises:
            RuntimeError: Se il training fallisce per errori di memoria o dati.
        
        Note:
            - Il training può terminare prima di config.EPOCHS se EarlyStopping interviene
            - Il modello migliore viene salvato automaticamente da ModelCheckpoint
            - Le metriche vengono loggate su file CSV e console in tempo reale
        """
        self.logger.info("=" * 60)
        self.logger.info("AVVIO TRAINING")
        self.logger.info("=" * 60)
        
        # Otteniamo i callback configurati
        callbacks = self._get_callbacks()
        
        # Chiamata bloccante a model.fit()
        # Questa funzione gestisce tutto il loop di training sulle epoche
        try:
            self.history = self.model.fit(
                self.train_gen,              # Dati di training (immagini + label)
                epochs=config.EPOCHS,        # Numero massimo di epoche
                validation_data=self.val_gen,# Dati di validazione per valutare le performance live
                callbacks=callbacks,         # Lista dei callback
                verbose=1                    # Mostra barra di progresso
            )
            
            # Estrai metriche finali per logging
            final_train_acc = self.history.history.get('accuracy', [0])[-1]
            final_val_acc = self.history.history.get('val_accuracy', [0])[-1]
            final_train_loss = self.history.history.get('loss', [float('inf')])[-1]
            final_val_loss = self.history.history.get('val_loss', [float('inf')])[-1]
            
            self.logger.info("=" * 60)
            self.logger.info("TRAINING COMPLETATO")
            self.logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
            self.logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
            self.logger.info(f"Final Training Loss: {final_train_loss:.4f}")
            self.logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Errore durante il training: {e}", exc_info=True)
            raise RuntimeError(f"Training fallito: {e}") from e
        
        return self.history
    
    def save_final_model(self, filename: str = "final_model.h5") -> None:
        """
        Salva forzatamente lo stato finale del modello al termine del training.
        
        Utile se EarlyStopping non è intervenuto o per avere l'ultimo stato assoluto
        (non necessariamente il migliore). Il modello "migliore" viene già salvato
        automaticamente da ModelCheckpoint durante il training.
        
        Args:
            filename: Nome del file di output (default: "final_model.h5").
                     Il file viene salvato in config.MODELS_DIR.
        
        Raises:
            IOError: Se il salvataggio fallisce per problemi di permessi o spazio disco.
        
        Note:
            - Questo metodo salva lo stato FINALE, non necessariamente il migliore
            - Per il modello migliore, usa best_model.h5 salvato da ModelCheckpoint
        """
        path = config.MODELS_DIR / filename
        try:
            self.model.save(str(path))
            self.logger.info(f"Modello finale salvato in: {path}")
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio del modello: {e}", exc_info=True)
            raise IOError(f"Impossibile salvare il modello: {e}") from e
