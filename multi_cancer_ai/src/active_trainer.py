"""
Modulo per il "Training Attivo" (Active Learning) o Fine-Tuning online.
Questo modulo permette all'applicazione di "imparare" dai propri errori in tempo reale.
Quando un utente corregge una diagnosi errata, il sistema può:
1. Salvare l'immagine con l'etichetta corretta per futuri training completi.
2. Eseguire un rapido fine-tuning su quella singola immagine per aggiornare immediatamente i pesi (sperimentale).
"""

# Imports
from pathlib import Path
import shutil
import datetime

from multi_cancer_ai.config import config # Importiamo la configurazione globale

# Nota: TensorFlow e Numpy vengono importati localmente nei metodi per evitare
# conflitti di inizializzazione (SegFault) con Tkinter su macOS.


class ActiveTrainer:
    """
    Gestisce le operazioni di addestramento interattivo e salvataggio dati utente.
    """
    
    def __init__(self, model_path=None):
        """
        Inizializza il Trainer Attivo.
        
        Args:
            model_path (Path, optional): Percorso specifico del modello da caricare.
                                         Se None, usa il modello di default definito in config.
        """
        # Se non viene fornito un path, usiamo quello di default (best_model.h5)
        self.model_path = model_path or (config.MODELS_DIR / "best_model.h5")
        
        # Percorso dove salvare il modello "personalizzato" dopo il fine-tuning.
        # Evitiamo di sovrascrivere direttamente il modello base originale per sicurezza.
        self.custom_model_path = config.MODELS_DIR / "best_model_custom.h5"
        
        # Variabile per mantenere il modello caricato in memoria.
        self.model = None
        
    def load_model_for_training(self):
        """
        Carica il modello in memoria preparandolo per il training.
        Se esiste già una versione 'custom' (addestrata dall'utente), carica quella.
        Altrimenti carica il modello base originale.
        
        Returns:
            tuple: (bool, str) -> (Successo?, Messaggio di stato)
        """
        import tensorflow as tf # Lazy import
        
        # Logica di selezione del file: preferisci il custom se esiste
        path_to_load = self.custom_model_path if self.custom_model_path.exists() else self.model_path
        print(f"[TRAINER] Loading model from: {path_to_load}")
        
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

    def train_on_single_image(self, img_array, label_index):
        """
        Esegue un passo di training (Fine-Tuning) su una SINGOLA immagine.
        Questo è un approccio di "Online Learning".
        
        ATTENZIONE: Addestrare su una singola immagine può causare instabilità o overfitting
        su quell'immagine specifica. Si usa qui a scopo dimostrativo o per correzioni rapide.
        
        Args:
            img_array (numpy.ndarray): L'immagine pre-processata (shape: 224, 224, 3).
            label_index (int): L'indice della classe CORRETTA (fornito dall'utente).
            
        Returns:
            tuple: (bool, str) -> (Successo?, Messaggio)
        """
        import tensorflow as tf # Lazy import
        import numpy as np
        
        # Verifica preliminare: il modello deve essere caricato
        if self.model is None:
            return False, "Model not loaded"

        try:
            # Preparazione Dati (Input X)
            # Keras si aspetta sempre un batch di immagini, non un'immagine singola.
            # Quindi aggiungiamo una dimensione extra all'inizio: (224, 224, 3) -> (1, 224, 224, 3)
            X = np.expand_dims(img_array, axis=0)
            
            # Preparazione Etichetta (Target Y) - One-Hot Encoding manuale
            num_classes = len(config.CLASS_MAPPING) # Numero totale di classi (8)
            y = np.zeros((1, num_classes))          # Vettore di zeri: [0, 0, 0, 0, 0, 0, 0, 0]
            y[0, label_index] = 1.0                 # Impostiamo a 1 l'indice corretto: [0, 0, 1, 0...]
            
            
            print(f"[TRAINER] Training on class index: {label_index}")
            
            # Eseguiamo il training (fit) per poche epoche (es. 5) su questo singolo campione.
            # verbose=0 nasconde l'output della console per pulizia.
            history = self.model.fit(X, y, epochs=5, verbose=0)
            
            # Recuperiamo l'ultimo valore di loss per monitorare se sta imparando.
            loss = history.history['loss'][-1]
            
            # Salvataggio del modello aggiornato.
            # Sovrascriviamo (o creiamo) il file 'best_model_custom.h5'.
            print(f"[TRAINER] Saving updated model to {self.custom_model_path}")
            self.model.save(str(self.custom_model_path))
            
            return True, f"Training complete. Loss: {loss:.4f}"
            
        except Exception as e:
            print(f"[TRAINER] Error during training: {e}")
            return False, str(e)

    def save_labeled_image(self, original_path, label_name):
        """
        Salva una copia dell'immagine etichettata dall'utente in una cartella specifica.
        Questo è l'approccio più sicuro: accumulare i dati corretti per ri-addestrare tutto il modello
        con calma in un secondo momento (Offline Learning), invece di modificare i pesi al volo.
        
        Args:
            original_path (str): Path del file immagine originale caricato dall'utente.
            label_name (str): Nome della classe corretta (es. "Lung Cancer").
            
        Returns:
            bool: True se il salvataggio è riuscito.
        """
        try:
            # Creiamo un timestamp per rendere il nome file unico e evitare sovrascritture.
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Costruiamo il nuovo nome file: user_corrected_2023..._original.jpg
            filename = f"user_corrected_{timestamp}_{Path(original_path).name}"
            
            # Definiamo la cartella di destinazione: user_data/labeled/<NomeClasse>/
            # Esempio: .../user_data/labeled/Lung Cancer/
            save_dir = config.USER_DATA_DIR / "labeled" / label_name
            
            # Creiamo la cartella se non esiste
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Definiamo il percorso completo di destinazione
            dest_path = save_dir / filename
            
            # Copiamo il file originale nella nuova destinazione, preservando i metadati (copy2).
            shutil.copy2(original_path, dest_path)
            
            print(f"[TRAINER] Image saved to {dest_path}")
            return True
            
        except Exception as e:
            print(f"[TRAINER] Failed to save image: {e}")
            return False
