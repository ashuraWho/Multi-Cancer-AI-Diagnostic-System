import tensorflow as tf
import sys
import os
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from multi_cancer_ai.config import config

def finetune_model():
    """
    Performs incremental training on data found in data/kaggle_import.
    """
    print("[INFO] Starting Fine-Tuning Process...")
    
    # 0. Performance Setup
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[INFO] Mixed Precision enabled for training speed.")
    except:
        pass

    # 1. Paths
    import_dir = config.BASE_DIR / "data" / "kaggle_import"
    model_path = config.MODELS_DIR / "best_model.h5"
    save_path = config.MODELS_DIR / "best_model_tuned.h5"
    
    if not import_dir.exists():
        print(f"[ERROR] No import directory found at {import_dir}")
        print("Run 'python tools/import_kaggle.py <handle>' first.")
        return

    # 2. Load Data & Optimize
    print(f"[INFO] Loading data from {import_dir}...")
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            import_dir,
            image_size=config.IMG_SIZE,
            batch_size=16, 
            label_mode='categorical',
            shuffle=True
        )
        
        # Performance Pipeline
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return

    # 3. Dynamic Class Adaptation
    new_classes = train_ds.class_names
    num_new_classes = len(new_classes)
    print(f"[INFO] New Dataset Classes ({num_new_classes}): {new_classes}")

    # 4. Load Model
    print(f"[INFO] Loading model from {model_path}...")
    original_model = tf.keras.models.load_model(model_path)
    
    # Check if we need to modify the head
    # We compare the output shape of existing model vs new classes
    # output_shape is usually (None, NumClasses)
    original_output_shape = original_model.output_shape[-1]
    
    if original_output_shape != num_new_classes:
        print(f"[WARN] Class mismatch detected! Model has {original_output_shape} outputs, new data has {num_new_classes}.")
        print("[INFO] Re-initializing Model Head for new classes...")
        
        # Create new model with new head
        # We assume the last layer is the classification head.
        # EfficientNetV2 -> Pooling -> BatchN -> Dropout -> Dense -> BN -> Dropout -> Output
        # We want to keep everything up to the layer BEFORE predictions.
        
        # Let's find the second to last layer output (or traverse up)
        # Using functional API logic:
        x = original_model.layers[-2].output # Get output of layer before 'predictions'
        
        # Create new output layer
        # Output must be float32 for mixed_precision stability if enabled
        predictions = tf.keras.layers.Dense(num_new_classes, activation='softmax', dtype='float32', name="predictions_new")(x)
        
        model = tf.keras.models.Model(inputs=original_model.input, outputs=predictions)
        
        # Freeze the base layers to avoid destroying features with uninitialized head
        # Let's freeze everything except the new head
        for layer in model.layers[:-1]:
            layer.trainable = False
            
        print("[INFO] Base model frozen. Training only new head.")
        
    else:
        print("[INFO] Class count matches. Fine-tuning entire model (low LR).")
        model = original_model
    
    # 5. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 6. Train
    EPOCHS = 5
    print(f"[INFO] Training for {EPOCHS} epochs...")
    history = model.fit(train_ds, epochs=EPOCHS)
    
    # 7. Save Model & Classes
    print(f"[INFO] Saving tuned model to {save_path}...")
    model.save(save_path)
    
    # Save Class Mapping JSON
    import json
    classes_json_path = config.MODELS_DIR / "classes.json"
    
    # Create a simple mapping dict: {0: "ClassA", 1: "ClassB", ...} 
    # Or match the config format {"ClassA": "Class A Desc"}?
    # For now, we save the raw list and let the App generate a default mapping if missing from config.
    
    mapping_data = {
        "classes": new_classes,
        "mapping": {c: c for c in new_classes} # Simple identity mapping for now
    }
    
    with open(classes_json_path, 'w') as f:
        json.dump(mapping_data, f, indent=4)

    print(f"[SUCCESS] Saved new class mapping to {classes_json_path}")
    print("[SUCCESS] Fine-tuning complete!")

if __name__ == "__main__":
    finetune_model()
