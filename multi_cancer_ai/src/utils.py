"""
Modulo di utility per logging, gestione file e visualizzazioni grafiche ausiliarie.

Contiene funzioni trasversali usate da vari moduli del progetto:
- Configurazione logging strutturato (file + console)
- Generazione grafici training history (accuracy/loss curves)
- Utility per timestamp e formattazione

Author: Multi-Cancer AI Team
License: MIT
"""

from typing import Optional
from pathlib import Path
import logging
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per server/script
import matplotlib.pyplot as plt
from datetime import datetime

from multi_cancer_ai.config.config import LOGS_DIR

def setup_logging(name: str = "MultiCancerAI") -> logging.Logger:
    """
    Configura il sistema di logging per scrivere i messaggi sia sulla console (stdout)
    che su un file di testo persistente.
    
    Configurazione:
        - Livello: INFO (registra INFO, WARNING, ERROR, CRITICAL)
        - Formato: timestamp - logger_name - level - message
        - Output: Console (StreamHandler) + File (FileHandler con timestamp nel nome)
        - File log: logs/<name>_<timestamp>.log
    
    Args:
        name: Il nome del logger (default: "MultiCancerAI").
              Usa nomi diversi per distinguere log di training vs inference.
    
    Returns:
        Logger configurato e pronto all'uso. Se il logger esiste già, viene riutilizzato
        (singleton pattern per evitare duplicati).
    
    Example:
        >>> logger = setup_logging("TrainingPipeline")
        >>> logger.info("Training started")
        >>> # Log salvato in: logs/TrainingPipeline_20240126_143022.log
    """
    # Creiamo un timestamp attuale (es. 20231027_103000) per rendere unico il file di log.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Definiamo il percorso completo del file di log.
    # Esempio: .../logs/MultiCancerAI_20231027_103000.log
    log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    
    # Otteniamo l'istanza del logger richiesta.
    logger = logging.getLogger(name)
    
    # Impostiamo il livello base di logging a INFO.
    # Significa che verranno registrati messaggi di livello INFO, WARNING, ERROR, CRITICAL.
    # I messaggi di DEBUG verranno ignorati.
    logger.setLevel(logging.INFO)
    
    # Se il logger ha già degli handler (es. se richiamato più volte), li puliamo per evitare duplicati.
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Definiamo il formato dei messaggi di log.
    # Esempio: "2023-10-27 10:30:00 - MultiCancerAI - INFO - Messaggio di prova"
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- Handler Console (StreamHandler) ---
    # Questo handler invia i log allo standard output (il terminale).
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter) # Applichiamo il formato definito sopra
    logger.addHandler(console_handler)      # Aggiungiamo l'handler al logger
    
    # --- Handler File (FileHandler) ---
    # Questo handler scrive i log sul file specificato.
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)    # Applichiamo lo stesso formato
    logger.addHandler(file_handler)         # Aggiungiamo l'handler al logger
    
    # Restituiamo l'oggetto logger configurato al chiamante.
    return logger

def plot_training_history(
    history,
    save_path: Optional[Path] = None
) -> None:
    """
    Genera e salva i grafici dell'andamento del training (accuracy e loss curves).
    
    Crea una figura con due subplot side-by-side:
    1. Accuracy: Training Accuracy vs Validation Accuracy (linee blu e rosse)
    2. Loss: Training Loss vs Validation Loss (linee blu e rosse)
    
    Utile per:
    - Identificare overfitting (gap tra train e val)
    - Verificare convergenza (curve che si stabilizzano)
    - Debugging problemi di training (loss che non scende)
    
    Args:
        history: L'oggetto History restituito da model.fit() di Keras.
                Deve contenere almeno 'accuracy', 'val_accuracy', 'loss', 'val_loss' in history.history.
        save_path: Percorso opzionale dove salvare l'immagine PNG.
                  Se None, il grafico viene generato ma non salvato (utile per debug).
                  Può essere Path object o stringa.
    
    Raises:
        KeyError: Se history.history non contiene le chiavi attese (accuracy, loss, etc.).
        IOError: Se il salvataggio fallisce per problemi di permessi o spazio disco.
    
    Note:
        - La figura viene chiusa automaticamente dopo il salvataggio (plt.close())
        - Dimensione figura: 12x5 pollici per buona leggibilità
        - Formato output: PNG (lossless, adatto per report)
    
    Example:
        >>> history = model.fit(train_ds, validation_data=val_ds, epochs=30)
        >>> plot_training_history(history, save_path="results/training_history.png")
    """
    # Estraiamo i dati di Accuracy dal dizionario history.history.
    # .get() restituisce una lista vuota [] se la chiave non esiste (sicurezza).
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    
    # Estraiamo i dati di Loss (Errore).
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    # Creiamo un range di numeri per l'asse X (le epoche), da 0 a N-1.
    epochs = range(len(acc))
    
    # Creiamo una nuova figura Matplotlib con dimensione 12x5 pollici.
    plt.figure(figsize=(12, 5))
    
    # --- Subplot 1: Accuracy ---
    # (1 row, 2 columns, position 1)
    plt.subplot(1, 2, 1)
    
    # Plottiamo l'accuracy di training in blu ('b').
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    
    # Plottiamo l'accuracy di validation in rosso ('r').
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    
    # Impostiamo il titolo del grafico.
    plt.title('Training and Validation Accuracy')
    
    # Mostriamo la legenda per distinguere le linee.
    plt.legend()
    
    # --- Subplot 2: Loss ---
    # (1 row, 2 columns, position 2)
    plt.subplot(1, 2, 2)
    
    # Plottiamo la loss di training in blu.
    plt.plot(epochs, loss, 'b', label='Training Loss')
    
    # Plottiamo la loss di validation in rosso.
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    
    # Impostiamo il titolo.
    plt.title('Training and Validation Loss')
    
    # Mostriamo la legenda.
    plt.legend()
    
    # Se è stato specificato un percorso di salvataggio...
    if save_path:
        # Salviamo la figura su disco.
        plt.savefig(save_path)
        print(f"Grafico salvato in: {save_path}")
    
    # Chiudiamo la figura per liberare la memoria (importante in loop o script lunghi).
    plt.close()
