"""
Modulo per la gestione del ciclo di training del modello.
Incapsula la logica di addestramento, configurazione dei callback e salvataggio.
"""

import os
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
    Gestisce il ciclo di vita dell'addestramento: setup callback, esecuzione di model.fit, salvataggio.
    """
    
    def __init__(self, model, train_gen, val_gen, logger):
        """
        Inizializza il Trainer.
        
        Args:
            model (tf.keras.Model): Il modello compilato da addestrare.
            train_gen (tf.data.Dataset): Dataset di training.
            val_gen (tf.data.Dataset): Dataset di validazione.
            logger (logging.Logger): Logger per registrare eventi.
        """
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.logger = logger
        self.history = None # Conterrà lo storico delle metriche post-training
        
        # Log di stato iniziale
        self.logger.info(f"Trainer inizializzato. Epochs: {config.EPOCHS}")

    def _get_callbacks(self):
        """
        Configura e restituisce la lista dei callback di Keras.
        I callback sono funzioni eseguite in punti specifici del training (es. fine epoca).
        
        Returns:
            list: Lista di oggetti callback.
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
        csv_logger = CSVLogger(str(config.LOGS_DIR / "training_log.csv"))
        callbacks_list.append(csv_logger)
        
        return callbacks_list

    def train(self):
        """
        Avvia il processo di training effettivo (model.fit).
        
        Returns:
            History: Oggetto contenente i valori delle metriche per ogni epoca.
        """
        self.logger.info("Inizio Training...")
        
        # Otteniamo i callback configurati
        callbacks = self._get_callbacks()
        
        # Chiamata bloccante a model.fit()
        # Questa funzione gestisce tutto il loop di training sulle epoche
        self.history = self.model.fit(
            self.train_gen,              # Dati di training (immagini + label)
            epochs=config.EPOCHS,        # Numero massimo di epoche
            validation_data=self.val_gen,# Dati di validazione per valutare le performance live
            callbacks=callbacks,         # Lista dei callback
            verbose=1                    # Mostra barra di progresso
        )
        
        self.logger.info("Training completato.")
        return self.history
    
    def save_final_model(self, filename="final_model.h5"):
        """
        Salva forzatamente lo stato finale del modello al termine del training.
        Utile se EarlyStopping non è intervenuto o per avere l'ultimo stato assoluto.
        
        Args:
            filename (str): Nome del file di output.
        """
        path = config.MODELS_DIR / filename
        self.model.save(str(path))
        self.logger.info(f"Modello finale salvato in: {path}")
