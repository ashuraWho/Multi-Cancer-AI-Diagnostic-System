"""
Modulo per il "Training Attivo" (Active Learning) o Fine-Tuning online.

Questo modulo permette all'applicazione di "imparare" dai propri errori in tempo reale.
Quando un utente corregge una diagnosi errata, il sistema può:
1. Salvare l'immagine con l'etichetta corretta per futuri training completi (offline learning).
2. Eseguire un rapido fine-tuning su quella singola immagine per aggiornare immediatamente i pesi (online learning).

Author: Multi-Cancer AI Team
License: MIT
"""

from typing import Tuple, Optional
from pathlib import Path
import shutil
import datetime
import logging

from multi_cancer_ai.config import config

# Nota: TensorFlow e Numpy vengono importati localmente nei metodi per evitare
# conflitti di inizializzazione (SegFault) con Tkinter su macOS.


class ActiveTrainer:
    """
    Gestisce le operazioni di addestramento interattivo e salvataggio dati utente.
    
    Implementa un sistema di Active Learning dove l'utente può correggere predizioni errate
    e il modello viene aggiornato in tempo reale (online learning) o i dati vengono salvati
    per un training completo successivo (offline learning).
    
    Attributes:
        model_path: Percorso del modello base da caricare (default: best_model.h5).
        custom_model_path: Percorso dove salvare il modello personalizzato dopo fine-tuning.
        model: Modello Keras caricato in memoria (None fino a load_model_for_training()).
    """
    
    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Inizializza il Trainer Attivo.
        
        Args:
            model_path: Percorso specifico del modello da caricare.
                       Se None, usa il modello di default definito in config (best_model.h5).
        
        Note:
            - Il modello non viene caricato automaticamente (lazy loading)
            - Chiama load_model_for_training() prima di usare train_on_single_image()
        """
        # Se non viene fornito un path, usiamo quello di default (best_model.h5)
        self.model_path = model_path or (config.MODELS_DIR / "best_model.h5")
        
        # Percorso dove salvare il modello "personalizzato" dopo il fine-tuning.
        # Evitiamo di sovrascrivere direttamente il modello base originale per sicurezza.
        self.custom_model_path = config.MODELS_DIR / "best_model_custom.h5"
        
        # Variabile per mantenere il modello caricato in memoria.
        self.model = None
        
    def load_model_for_training(self) -> Tuple[bool, str]:
        """
        Carica il modello in memoria preparandolo per il training.
        
        Se esiste già una versione 'custom' (addestrata dall'utente), carica quella.
        Altrimenti carica il modello base originale. Il modello viene ricompilato
        con un learning rate molto basso (1e-5) per evitare catastrophic forgetting.
        
        Returns:
            Tuple (success, message):
                - success: True se il caricamento è riuscito, False altrimenti.
                - message: Messaggio descrittivo dello stato (errore o conferma).
        
        Note:
            - Il modello viene ricompilato con Adam(lr=1e-5) per fine-tuning conservativo
            - Se il caricamento fallisce, self.model rimane None
        """
        import tensorflow as tf # Lazy import
        import logging
        logger = logging.getLogger("MultiCancerAI")
        
        # Logica di selezione del file: preferisci il custom se esiste
        path_to_load = self.custom_model_path if self.custom_model_path.exists() else self.model_path
        logger.info(f"[ActiveTrainer] Loading model from: {path_to_load}")
        
        try:
            # Caricamento del modello Keras (.h5)
            self.model = tf.keras.models.load_model(str(path_to_load))
            
            # Ricompiliamo il modello. 
            # È necessario ricompilare per poter eseguire .fit().
            # Usiamo un Learning Rate MOLTO basso (1e-5) perché stiamo facendo fine-tuning
            # su pochissimi dati e non vogliamo distruggere i pesi già appresi (Catastrophic Forgetting).
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
            
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy', # Loss standard per classificazione multi-classe
                metrics=['accuracy']
            )
            return True, "Model loaded successfully"
            
        except Exception as e:
            # Gestione errori (es. file corrotto, path sbagliato)
            return False, f"Failed to load model: {e}"

    def train_on_single_image(
        self,
        img_array,
        label_index: int
    ) -> Tuple[bool, str]:
        """
        Esegue un passo di training (Fine-Tuning) su una SINGOLA immagine.
        
        Questo è un approccio di "Online Learning" dove il modello viene aggiornato
        immediatamente dopo una correzione utente. Utilizza un learning rate molto basso
        (1e-5) e poche epoche (5) per evitare overfitting su un singolo campione.
        
        ATTENZIONE: Addestrare su una singola immagine può causare instabilità o overfitting
        su quell'immagine specifica. Si usa qui a scopo dimostrativo o per correzioni rapide.
        Per risultati più stabili, accumula più correzioni e ri-addestra offline.
        
        Args:
            img_array: L'immagine pre-processata (shape: [224, 224, 3] o [1, 224, 224, 3]).
                      Valori devono essere in [0, 1] (float32).
            label_index: L'indice della classe CORRETTA (fornito dall'utente).
                        Deve essere in range [0, num_classes-1].
        
        Returns:
            Tuple (success, message):
                - success: True se il training è riuscito, False altrimenti.
                - message: Messaggio descrittivo con loss finale o errore.
        
        Raises:
            RuntimeError: Se il modello non è stato caricato (chiama load_model_for_training() prima).
            ValueError: Se label_index è fuori range o img_array ha shape non valida.
        
        Note:
            - Il modello viene salvato automaticamente in best_model_custom.h5 dopo il training
            - La loss viene monitorata per verificare che il modello stia imparando
            - verbose=0 per evitare output eccessivo in GUI
        """
        import tensorflow as tf # Lazy import
        import numpy as np
        import logging
        logger = logging.getLogger("MultiCancerAI")
        
        # Verifica preliminare: il modello deve essere caricato
        if self.model is None:
            return False, "Model not loaded. Call load_model_for_training() first."

        try:
            # Preparazione Dati (Input X)
            # Keras si aspetta sempre un batch di immagini, non un'immagine singola.
            # Quindi aggiungiamo una dimensione extra all'inizio: (224, 224, 3) -> (1, 224, 224, 3)
            if img_array.ndim == 3:
                X = np.expand_dims(img_array, axis=0)
            elif img_array.ndim == 4:
                X = img_array
            else:
                raise ValueError(f"img_array deve avere 3 o 4 dimensioni, got {img_array.ndim}")
            
            # Preparazione Etichetta (Target Y) - One-Hot Encoding manuale
            num_classes = len(config.CLASS_MAPPING) # Numero totale di classi (8)
            
            if label_index < 0 or label_index >= num_classes:
                raise ValueError(f"label_index {label_index} fuori range [0, {num_classes-1}]")
            
            y = np.zeros((1, num_classes), dtype=np.float32)  # Vettore di zeri
            y[0, label_index] = 1.0                           # Impostiamo a 1 l'indice corretto
            
            logger.info(f"[ActiveTrainer] Training on class index: {label_index} ({list(config.CLASS_MAPPING.keys())[label_index]})")
            
            # Eseguiamo il training (fit) per più epoche su questo singolo campione.
            # Aumentato a 10 epoche per dare più tempo al modello di apprendere.
            # Usiamo un batch size di 1 (singola immagine) e verbose=0 per pulizia.
            # NOTA: Training su singola immagine può causare overfitting, ma è accettabile
            # per active learning dove l'obiettivo è correggere rapidamente errori specifici.
            history = self.model.fit(X, y, epochs=10, batch_size=1, verbose=0)
            
            # Recuperiamo l'ultimo valore di loss per monitorare se sta imparando.
            loss = history.history['loss'][-1]
            
            # Salvataggio del modello aggiornato.
            # Sovrascriviamo (o creiamo) il file 'best_model_custom.h5'.
            logger.info(f"[ActiveTrainer] Saving updated model to {self.custom_model_path}")
            self.model.save(str(self.custom_model_path))
            
            return True, f"Training complete. Loss: {loss:.4f}"
            
        except Exception as e:
            logger.error(f"[ActiveTrainer] Error during training: {e}", exc_info=True)
            return False, f"Training failed: {str(e)}"

    def save_labeled_image(
        self,
        original_path: str,
        label_name: str
    ) -> bool:
        """
        Salva una copia dell'immagine etichettata dall'utente in una cartella specifica.
        
        Questo è l'approccio più sicuro: accumulare i dati corretti per ri-addestrare tutto il modello
        con calma in un secondo momento (Offline Learning), invece di modificare i pesi al volo.
        Le immagini salvate possono essere usate per un training completo futuro con più dati.
        
        Args:
            original_path: Path del file immagine originale caricato dall'utente.
                          Può essere Path object o stringa.
            label_name: Nome della classe corretta (es. "Lung Cancer").
                       Deve corrispondere a una chiave in config.CLASS_MAPPING.
        
        Returns:
            True se il salvataggio è riuscito, False altrimenti.
        
        Note:
            - Il file viene salvato in: user_data/labeled/<label_name>/user_corrected_<timestamp>_<original_name>
            - Il timestamp garantisce unicità del nome file
            - I metadati del file originale vengono preservati (shutil.copy2)
        
        Example:
            >>> trainer.save_labeled_image("/path/to/image.jpg", "Breast Cancer")
            >>> # Salva in: user_data/labeled/Breast Cancer/user_corrected_20240126_143022_image.jpg
        """
        import logging
        logger = logging.getLogger("MultiCancerAI")
        
        try:
            original_path = Path(original_path)
            
            if not original_path.exists():
                logger.error(f"[ActiveTrainer] File non trovato: {original_path}")
                return False
            
            # Creiamo un timestamp per rendere il nome file unico e evitare sovrascritture.
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Costruiamo il nuovo nome file: user_corrected_2023..._original.jpg
            filename = f"user_corrected_{timestamp}_{original_path.name}"
            
            # Definiamo la cartella di destinazione: user_data/labeled/<NomeClasse>/
            # Esempio: .../user_data/labeled/Lung Cancer/
            save_dir = config.USER_DATA_DIR / "labeled" / label_name
            
            # Creiamo la cartella se non esiste
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Definiamo il percorso completo di destinazione
            dest_path = save_dir / filename
            
            # Copiamo il file originale nella nuova destinazione, preservando i metadati (copy2).
            shutil.copy2(original_path, dest_path)
            
            logger.info(f"[ActiveTrainer] Image saved to {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"[ActiveTrainer] Failed to save image: {e}", exc_info=True)
            return False
