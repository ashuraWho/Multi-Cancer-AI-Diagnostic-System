import os
import sys
from pathlib import Path

# 1. Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"Python: {sys.version}")
print("Importing TensorFlow...")
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.h5"

print(f"Checking model path: {MODEL_PATH}")

if not MODEL_PATH.exists():
    print("ERROR: Model file does not exist!")
    sys.exit(1)

print("Attempting to load model (no GUI)...")
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("SUCCESS: Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
