"""
Script principale (Entry Point) per l'addestramento del sistema Multi-Cancer AI.

Questo script orchestra l'intera pipeline di Machine Learning end-to-end:
1. Setup e Logging (configurazione ambiente e tracciamento eventi)
2. Caricamento e Preparazione Dati (Data Pipeline con tf.data)
3. Costruzione del Modello (Architecture EfficientNetV2 + custom head)
4. Addestramento (Training Loop con callbacks)
5. Valutazione (Evaluation & Metrics su validation set)
6. Esportazione per Mobile/Web (TFLite conversion con quantizzazione)

Author: Multi-Cancer AI Team
License: MIT

Usage:
    python -m multi_cancer_ai.train_main
    # oppure
    cd multi_cancer_ai && python train_main.py
"""

import sys
from pathlib import Path
import tensorflow as tf

# Importiamo i moduli interni del progetto
from multi_cancer_ai.config import config
from multi_cancer_ai.src.utils import setup_logging, plot_training_history
from multi_cancer_ai.src.data_loader import create_generators
from multi_cancer_ai.src.model import build_model
from multi_cancer_ai.src.trainer import Trainer
from multi_cancer_ai.src.evaluator import Evaluator

def main() -> int:
    """
    Funzione principale che esegue sequenzialmente tutti gli step della pipeline.
    
    Returns:
        Exit code: 0 se il training completa con successo, 1 in caso di errore.
                  Utile per integrazione in script bash/CI/CD.
    
    Raises:
        SystemExit: Se errori critici impediscono il completamento della pipeline.
    
    Note:
        - Tutti gli errori vengono loggati con stack trace completo
        - Il modello migliore viene sempre salvato anche in caso di early stopping
        - Le metriche vengono salvate in CSV per analisi post-hoc
    """
    
    # -------------------------------------------------------------------------
    # 1. SETUP E LOGGING
    # -------------------------------------------------------------------------
    # Inizializziamo il sistema di logging per tracciare ogni operazione su file e console.
    logger = setup_logging("TrainingPipeline")
    
    logger.info("=" * 80)
    logger.info("AVVIO PIPELINE DI TRAINING MULTI-CANCER AI")
    logger.info("=" * 80)
    logger.info(f"Configurazione Attiva:")
    logger.info(f"  - Immagini: {config.IMG_SIZE}")
    logger.info(f"  - Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  - Epochs: {config.EPOCHS}")
    logger.info(f"  - Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  - Validation Split: {config.VALIDATION_SPLIT}")
    logger.info(f"  - Dataset Path: {config.DATASET_PATH}")
    logger.info("=" * 80)
    
    # Verifica preliminare: dataset path esiste?
    if not config.DATASET_PATH.exists():
        logger.error(f"Dataset path non trovato: {config.DATASET_PATH}")
        logger.error("Verifica che il dataset sia presente o configura MULTI_CANCER_DATASET_PATH")
        return 1

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
        logger.info(f"✓ Dataset caricato con successo")
        logger.info(f"  - Numero classi: {num_classes}")
        logger.info(f"  - Mapping Classi-Indici: {class_indices}")
        
    except FileNotFoundError as e:
        logger.error(f"Errore critico: Dataset non trovato")
        logger.error(f"  Path cercato: {config.DATASET_PATH}")
        logger.error(f"  Soluzione: Verifica il path o configura MULTI_CANCER_DATASET_PATH")
        return 1
    except ValueError as e:
        logger.error(f"Errore critico: Dataset vuoto o struttura non valida")
        logger.error(f"  Dettagli: {e}")
        logger.error(f"  Soluzione: Verifica che il dataset contenga sottocartelle per ogni classe")
        return 1
    except Exception as e:
        # Se i dati non vengono caricati (es. path errato), interrompiamo tutto.
        logger.error(f"Errore critico nel caricamento dati: {e}", exc_info=True)
        return 1

    # -------------------------------------------------------------------------
    # 3. COSTRUZIONE MODELLO (MODEL BUILDING)
    # -------------------------------------------------------------------------
    logger.info("Fase 2: Costruzione Architettura Modello (EfficientNetV2)...")
    try:
        # Costruiamo il modello Keras compilato.
        model = build_model(num_classes)
        
        # Stampiamo il sommario dell'architettura nel log (numero parametri, layer, ecc).
        logger.info("Architettura Modello:")
        model.summary(print_fn=lambda x: logger.info(f"  {x}"))
        
        # Log info su mixed precision se abilitata
        try:
            policy = tf.keras.mixed_precision.global_policy()
            if policy.name == 'mixed_float16':
                logger.info("✓ Mixed Precision (FP16) abilitata per ottimizzazione GPU")
        except:
            pass
        
    except ValueError as e:
        logger.error(f"Errore nella creazione del modello: {e}")
        return 1
    except Exception as e:
        logger.error(f"Errore nella creazione del modello: {e}", exc_info=True)
        return 1

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
        
    except RuntimeError as e:
        logger.error(f"Errore durante il training: {e}", exc_info=True)
        logger.error("Possibili cause: OOM (Out of Memory), dataset corrotto, o errori di configurazione")
        return 1
    except KeyboardInterrupt:
        logger.warning("Training interrotto dall'utente (Ctrl+C)")
        logger.info("Salvataggio modello corrente...")
        trainer.save_final_model("interrupted_model.h5")
        return 1
    except Exception as e:
        logger.error(f"Errore inatteso durante il training: {e}", exc_info=True)
        return 1

    # -------------------------------------------------------------------------
    # 5. VISUALIZZAZIONE STORICO (HISTORY PLOTTING)
    # -------------------------------------------------------------------------
    logger.info("Fase 4: Salvataggio grafici andamento training...")
    
    # Definiamo il percorso per salvare il grafico PNG (training_history.png).
    plot_path = config.RESULTS_DIR / "training_history.png"
    
    try:
        # Generiamo il grafico Curve di Apprendimento.
        plot_training_history(history, save_path=plot_path)
        logger.info(f"✓ Grafico training history salvato in: {plot_path}")
    except Exception as e:
        logger.warning(f"Errore nel salvataggio del grafico: {e}")
        # Non blocchiamo la pipeline per un errore di visualizzazione

    # -------------------------------------------------------------------------
    # 6. VALUTAZIONE FINALE (EVALUATION)
    # -------------------------------------------------------------------------
    logger.info("Fase 5: Valutazione Approfondita su Validation Set...")
    
    try:
        # Istanziamo l'Evaluator.
        evaluator = Evaluator(model, val_gen, class_indices)
        
        # Otteniamo le etichette reali (y_true) e predette (y_pred) su tutto il validation set.
        y_true, y_pred, _ = evaluator.evaluate()
        
        # Generiamo il report testuale (Precision, Recall, F1).
        report = evaluator.generate_report(y_true, y_pred)
        logger.info("\n" + "=" * 80)
        logger.info("REPORT VALUTAZIONE FINALE")
        logger.info("=" * 80)
        logger.info(report)
        logger.info("=" * 80)
        
        # Generiamo e salviamo la Matrice di Confusione.
        cm_path = config.RESULTS_DIR / "confusion_matrix.png"
        evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        logger.info(f"✓ Matrice di confusione salvata in: {cm_path}")
        
    except Exception as e:
        logger.error(f"Errore durante la valutazione: {e}", exc_info=True)
        # Non blocchiamo la pipeline, ma logghiamo l'errore

    # -------------------------------------------------------------------------
    # 7. ESPORTAZIONE TFLITE (OPTIMIZATION)
    # -------------------------------------------------------------------------
    logger.info("Fase 6: Conversione e Ottimizzazione Modello per Mobile/Web (TFLite)...")
    try:
        # Carichiamo il modello migliore (salvato da ModelCheckpoint)
        best_model_path = config.MODELS_DIR / "best_model.h5"
        if best_model_path.exists():
            logger.info(f"Caricamento modello migliore da: {best_model_path}")
            model = tf.keras.models.load_model(str(best_model_path))
        else:
            logger.warning("best_model.h5 non trovato, uso modello corrente")
        
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
        logger.info(f"✓ Modello TFLite salvato in: {tflite_path}")
        logger.info(f"  Dimensione: {size_mb:.2f} MB (quantizzazione FP16)")
        
    except Exception as e:
        logger.error(f"Errore durante la conversione TFLite: {e}", exc_info=True)
        logger.warning("La conversione TFLite è opzionale, il training è comunque completato")

    # -------------------------------------------------------------------------
    # 8. RIEPILOGO FINALE
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETATA CON SUCCESSO")
    logger.info("=" * 80)
    logger.info(f"Modelli salvati in: {config.MODELS_DIR}")
    logger.info(f"Risultati salvati in: {config.RESULTS_DIR}")
    logger.info(f"Log salvato in: {config.LOGS_DIR}")
    logger.info("=" * 80)
    
    return 0

# Questo blocco assicura che il main() venga eseguito solo se lanciamo lo script direttamente,
# e non se lo importiamo come modulo.
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
