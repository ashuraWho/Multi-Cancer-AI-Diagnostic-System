import sys
import os
from pathlib import Path

# Aggiunge la root del progetto al path per permettere gli import assoluti
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

import tensorflow as tf
from multi_cancer_ai.config import config
from multi_cancer_ai.src.data_loader import create_generators
import numpy as np
import os

def log(msg):
    with open("recovery_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

def recover():
    log("[INFO] Avvio recupero post-training...")
    
    model_path = config.MODELS_DIR / "best_model.h5"
    if not model_path.exists():
        log(f"[ERROR] Modello non trovato in {model_path}")
        return

    log(f"[INFO] Caricamento modello da: {model_path}")
    try:
        model = tf.keras.models.load_model(str(model_path))
        log("[INFO] Modello caricato correttemente.")
    except Exception as e:
        log(f"[ERROR] Errore caricamento modello: {e}")
        return

    log("Fase 5: Conversione in TFLite (Optimized)...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        log("[DEBUG] Converter created.")
        
        # Enable SELECT_TF_OPS for EfficientNetV2 support
        converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
          tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
        ]
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        log("[DEBUG] Starting conversion...")
        tflite_model = converter.convert()
        log(f"[DEBUG] Conversion finished. Model size: {len(tflite_model)} bytes")

        tflite_path = config.MODELS_DIR / "model_optimized.tflite"
        log(f"[DEBUG] Saving to: {tflite_path}")
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        log("[DEBUG] Write operation done.")
        
        if tflite_path.exists():
             log(f"[SUCCESS] Modello TFLite salvato in: {tflite_path}")
        else:
             log(f"[ERROR] Il file non è stato creato nonostante la scrittura!")
        
    except Exception as e:
        log(f"[ERROR] Errore durante la conversione TFLite: {e}")
        import traceback
        with open("recovery_log.txt", "a") as f:
            traceback.print_exc(file=f)

    log("\n[INFO] Recupero completato! Ora puoi lanciare l'app.")

if __name__ == "__main__":
    if os.path.exists("recovery_log.txt"): os.remove("recovery_log.txt")
    sys.stdout.reconfigure(line_buffering=True)
    recover()
