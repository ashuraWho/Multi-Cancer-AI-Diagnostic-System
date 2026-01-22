"""
Modulo di utility per logging, gestione file e visualizzazioni grafiche ausiliarie.
Contiene funzioni trasversali usate da vari moduli del progetto.
"""

# Importiamo 'os' per operazioni di sistema (anche se qui usiamo principalmente logging e matplotlib).
import os

# Importiamo il modulo standard 'logging' per gestire i log (info, warning, error).
import logging

# Importiamo 'matplotlib.pyplot' per generare i grafici delle performance.
import matplotlib.pyplot as plt

# Importiamo 'datetime' per generare timestamp da inserire nei nomi dei file.
from datetime import datetime

# Importiamo la costante LOGS_DIR dal nostro modulo di configurazione.
from multi_cancer_ai.config.config import LOGS_DIR

def setup_logging(name="MultiCancerAI"):
    """
    Configura il sistema di logging per scrivere i messaggi sia sulla console (stdout)
    che su un file di testo persistente. Questo è cruciale per il debugging e per tenere traccia
    degli esperimenti passati.
    
    Args:
        name (str): Il nome del logger (default: "MultiCancerAI").
        
    Returns:
        logging.Logger: Un'istanza di logger configurata e pronta all'uso.
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

def plot_training_history(history, save_path=None):
    """
    Genera, visualizza e facoltativamente salva su disco i grafici dell'andamento del training.
    Visualizza due subplot:
    1. Accuracy (Training vs Validation)
    2. Loss (Training vs Validation)
    
    Args:
        history: L'oggetto 'History' restituito dal metodo model.fit() di Keras. Contiene le metriche per ogni epoca.
        save_path (Path, optional): Il percorso completo dove salvare l'immagine del grafico. Se None, non salva.
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
