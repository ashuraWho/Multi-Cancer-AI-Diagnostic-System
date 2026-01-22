"""
Script principale per l'addestramento del sistema di classificazione Multi-Cancro.
"""
import os
import tensorflow as tf
from multi_cancer_ai.config import config
from multi_cancer_ai.src.utils import setup_logging, plot_training_history
from multi_cancer_ai.src.data_loader import create_generators
from multi_cancer_ai.src.model import build_model
from multi_cancer_ai.src.trainer import Trainer
from multi_cancer_ai.src.evaluator import Evaluator

def main():
    # 1. Setup Logging
    logger = setup_logging()
    logger.info("Avvio Pipeline di Training Multi-Cancer AI")
    logger.info(f"Configurazione: Immagini {config.IMG_SIZE}, Batch {config.BATCH_SIZE}, Epochs {config.EPOCHS}")

    # 2. Caricamento Dati
    logger.info("Fase 1: Preparazione Dati...")
    try:
        # Adjusted for tf.data: unpack 3 values
        train_gen, val_gen, class_indices = create_generators()
        num_classes = len(class_indices)
        logger.info(f"Classi trovate: {num_classes}")
        logger.info(f"Mapping Classi: {class_indices}")
    except Exception as e:
        logger.error(f"Errore nel caricamento dati: {e}")
        return

    # 3. Costruzione del Modello
    logger.info("Fase 2: Costruzione Modello EfficientNetV2...")
    try:
        model = build_model(num_classes)
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Errore nella creazione del modello: {e}")
        return

    # 4. Training
    logger.info("Fase 3: Avvio Training...")
    trainer = Trainer(model, train_gen, val_gen, logger)
    try:
        history = trainer.train()
        trainer.save_final_model()
    except Exception as e:
        logger.error(f"Errore durante il training: {e}")
        return

    # 5. Plotting History
    logger.info("Salvataggio grafici training...")
    plot_path = config.RESULTS_DIR / "training_history.png"
    plot_training_history(history, save_path=plot_path)

    # 6. Evaluation
    logger.info("Fase 4: Valutazione Finale...")
    # NOTE: train_gen.class_indices doesn't exist anymore, use 'class_indices' variable
    evaluator = Evaluator(model, val_gen, class_indices)
    y_true, y_pred, _ = evaluator.evaluate()
    
    # Report
    report = evaluator.generate_report(y_true, y_pred)
    logger.info("\n" + report)
    
    # Matrice di Confusione
    cm_path = config.RESULTS_DIR / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    logger.info(f"Matrice di confusione salvata in: {cm_path}")

    # 7. Convert to TFLite (Optimized for Offline App)
    logger.info("Fase 5: Conversione in TFLite (Optimized)...")
    try:
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Enable FP16 quantization for 2x size reduction and speedup on supported hardware
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()

        # Save the model
        tflite_path = config.MODELS_DIR / "model_optimized.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"Modello TFLite salvato in: {tflite_path}")
        logger.info(f"Dimensione TFLite: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Errore durante la conversione TFLite: {e}")

    logger.info("Pipeline completata con successo.")

if __name__ == "__main__":
    main()
