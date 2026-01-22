"""
Script principale (Entry Point) per l'addestramento del sistema Multi-Cancer AI.
Questo script orchestra l'intera pipeline di Machine Learning:
1. Setup e Logging.
2. Caricamento e Preparazione Dati (Data Pipeline).
3. Costruzione del Modello (Architecture).
4. Addestramento (Training Loop).
5. Valutazione (Evaluation & Metrics).
6. Esportazione per Mobile/Web (TFLite).
"""

import os
import tensorflow as tf

# Importiamo i moduli interni del progetto
from multi_cancer_ai.config import config
from multi_cancer_ai.src.utils import setup_logging, plot_training_history
from multi_cancer_ai.src.data_loader import create_generators
from multi_cancer_ai.src.model import build_model
from multi_cancer_ai.src.trainer import Trainer
from multi_cancer_ai.src.evaluator import Evaluator

def main():
    """
    Funzione principale che esegue sequenzialmente tutti gli step della pipeline.
    """
    
    # -------------------------------------------------------------------------
    # 1. SETUP E LOGGING
    # -------------------------------------------------------------------------
    # Inizializziamo il sistema di logging per tracciare ogni operazione su file e console.
    logger = setup_logging()
    logger.info("Avvio Pipeline di Training Multi-Cancer AI")
    logger.info(f"Configurazione Attiva -> Immagini: {config.IMG_SIZE}, Batch Size: {config.BATCH_SIZE}, Epochs: {config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. CARICAMENTO DATI (DATA LOADING)
    # -------------------------------------------------------------------------
    logger.info("Fase 1: Preparazione Dati e Pipeline tf.data...")
    try:
        # Chiamiamo il modulo data_loader per ottenere i generatori ottimizzati.
        # train_gen: Dataset di training (con augmentation).
        # val_gen: Dataset di validazione (solo rescaling).
        # class_indices: Mapping {NomeClasse: Indice}.
        train_gen, val_gen, class_indices = create_generators()
        
        # Calcoliamo il numero totale di classi (output neurons).
        num_classes = len(class_indices)
        logger.info(f"Numero classi trovate: {num_classes}")
        logger.info(f"Mapping Classi-Indici: {class_indices}")
        
    except Exception as e:
        # Se i dati non vengono caricati (es. path errato), interrompiamo tutto.
        logger.error(f"Errore critico nel caricamento dati: {e}")
        return

    # -------------------------------------------------------------------------
    # 3. COSTRUZIONE MODELLO (MODEL BUILDING)
    # -------------------------------------------------------------------------
    logger.info("Fase 2: Costruzione Architettura Modello (EfficientNetV2)...")
    try:
        # Costruiamo il modello Keras compilato.
        model = build_model(num_classes)
        
        # Stampiamo il sommario dell'architettura nel log (numero parametri, layer, ecc).
        model.summary(print_fn=logger.info)
        
    except Exception as e:
        logger.error(f"Errore nella creazione del modello: {e}")
        return

    # -------------------------------------------------------------------------
    # 4. TRAINING
    # -------------------------------------------------------------------------
    logger.info("Fase 3: Avvio Ciclo di Training...")
    
    # Istanziamo la classe Trainer che gestisce fit, callbacks e checkpointing.
    trainer = Trainer(model, train_gen, val_gen, logger)
    
    try:
        # Avviamo l'addestramento. history conterrà l'andamento di loss e accuracy.
        history = trainer.train()
        
        # Salviamo esplicitamente il modello finale (anche se i callback salvano il migliore).
        trainer.save_final_model()
        
    except Exception as e:
        logger.error(f"Errore durante il training: {e}")
        return

    # -------------------------------------------------------------------------
    # 5. VISUALIZZAZIONE STORICO (HISTORY PLOTTING)
    # -------------------------------------------------------------------------
    logger.info("Salvataggio grafici andamento training...")
    
    # Definiamo il percorso per salvare il grafico PNG (training_history.png).
    plot_path = config.RESULTS_DIR / "training_history.png"
    
    # Generiamo il grafico Curve di Apprendimento.
    plot_training_history(history, save_path=plot_path)

    # -------------------------------------------------------------------------
    # 6. VALUTAZIONE FINALE (EVALUATION)
    # -------------------------------------------------------------------------
    logger.info("Fase 4: Valutazione Approfondita su Validation Set...")
    
    # Istanziamo l'Evaluator.
    evaluator = Evaluator(model, val_gen, class_indices)
    
    # Otteniamo le etichette reali (y_true) e predette (y_pred) su tutto il validation set.
    y_true, y_pred, _ = evaluator.evaluate()
    
    # Generiamo il report testuale (Precision, Recall, F1).
    report = evaluator.generate_report(y_true, y_pred)
    logger.info("\n" + report)
    
    # Generiamo e salviamo la Matrice di Confusione.
    cm_path = config.RESULTS_DIR / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    logger.info(f"Matrice di confusione salvata in: {cm_path}")

    # -------------------------------------------------------------------------
    # 7. ESPORTAZIONE TFLITE (OPTIMIZATION)
    # -------------------------------------------------------------------------
    logger.info("Fase 5: Conversione e Ottimizzazione Modello per Mobile/Web (TFLite)...")
    try:
        # Creiamo un convertitore dal modello Keras.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # --- Ottimizzazione: Quantizzazione ---
        # Abilitiamo ottimizzazioni default (spesso int8 weights).
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convertiamo i pesi in float16. 
        # Questo dimezza la dimensione del modello (es. 100MB -> 50MB) 
        # mantenendo quasi intatta la precisione.
        converter.target_spec.supported_types = [tf.float16]
        
        # Eseguiamo la conversione.
        tflite_model = converter.convert()

        # Salviamo il file .tflite.
        tflite_path = config.MODELS_DIR / "model_optimized.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calcoliamo la dimensione in MB per logging.
        size_mb = len(tflite_model) / 1024 / 1024
        logger.info(f"Modello TFLite salvato in: {tflite_path}")
        logger.info(f"Dimensione TFLite: {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Errore durante la conversione TFLite: {e}")

    logger.info("Pipeline completata con successo.")

# Questo blocco assicura che il main() venga eseguito solo se lanciamo lo script direttamente,
# e non se lo importiamo come modulo.
if __name__ == "__main__":
    main()
