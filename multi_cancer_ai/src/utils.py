"""
Modulo di utility per logging, gestione file e visualizzazioni ausiliarie.
"""
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from multi_cancer_ai.config.config import LOGS_DIR

def setup_logging(name="MultiCancerAI"):
    """
    Configura il logger per scrivere sia su console che su file.
    
    Args:
        name (str): Nome del logger.
        
    Returns:
        logging.Logger: Istanza del logger configurata.
    """
    # Crea una directory di log se non esiste
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formattazione
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler File
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def plot_training_history(history, save_path=None):
    """
    Visualizza e salva i grafici di Accuracy e Loss.
    
    Args:
        history: Oggetto history restituito da model.fit()
        save_path (Path, optional): Percorso dove salvare l'immagine.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Grafico salvato in: {save_path}")
    
    plt.close()
