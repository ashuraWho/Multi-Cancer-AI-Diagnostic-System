import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil
import datetime
from multi_cancer_ai.config import config

class ActiveTrainer:
    def __init__(self, model_path=None):
        self.model_path = model_path or (config.MODELS_DIR / "best_model.h5")
        self.custom_model_path = config.MODELS_DIR / "best_model_custom.h5"
        self.model = None
        
    def load_model_for_training(self):
        """Loads the model (preferring custom if exists) for training."""
        path_to_load = self.custom_model_path if self.custom_model_path.exists() else self.model_path
        print(f"[TRAINER] Loading model from: {path_to_load}")
        
        try:
            self.model = tf.keras.models.load_model(path_to_load)
            # Compile with low learning rate for fine-tuning
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Failed to load model: {e}"

    def train_on_single_image(self, img_array, label_index):
        """
        Fine-tunes the model on a single image (Active Learning).
        WARNING: Training on 1 image can be unstable. We use very low LR.
        """
        if self.model is None:
            return False, "Model not loaded"

        try:
            # Prepare data
            # img_array shape: (224, 224, 3) -> (1, 224, 224, 3)
            X = np.expand_dims(img_array, axis=0)
            
            # Prepare label (One-hot encoding)
            num_classes = len(config.CLASS_MAPPING)
            y = np.zeros((1, num_classes))
            y[0, label_index] = 1.0

            print(f"[TRAINER] Training on class index: {label_index}")
            
            # Train for a few epochs on this sample to enforce learning
            history = self.model.fit(X, y, epochs=5, verbose=0)
            loss = history.history['loss'][-1]
            
            # Save the updated model
            print(f"[TRAINER] Saving updated model to {self.custom_model_path}")
            self.model.save(self.custom_model_path)
            
            return True, f"Training complete. Loss: {loss:.4f}"
            
        except Exception as e:
            print(f"[TRAINER] Error during training: {e}")
            return False, str(e)

    def save_labeled_image(self, original_path, label_name):
        """Saves a copy of the user-labeled image for future full retraining."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_corrected_{timestamp}_{Path(original_path).name}"
            
            # Create user_data directory structure
            save_dir = config.BASE_DIR / "user_data" / "labeled" / label_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = save_dir / filename
            shutil.copy2(original_path, dest_path)
            print(f"[TRAINER] Image saved to {dest_path}")
            return True
        except Exception as e:
            print(f"[TRAINER] Failed to save image: {e}")
            return False
