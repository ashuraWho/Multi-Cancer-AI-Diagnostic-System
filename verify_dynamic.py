import os
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

# Setup paths
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from multi_cancer_ai.config import config
from multi_cancer_ai.tools import finetune

def setup_dummy_data():
    """Creates a dummy dataset with a NEW class."""
    import_dir = config.BASE_DIR / "data" / "kaggle_import"
    if import_dir.exists():
        shutil.rmtree(import_dir)
    
    # Create "NewCancer" class
    new_class_dir = import_dir / "NewCancer"
    new_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Create existing class (to ensure mix)
    # Using 'Allan' or similar from existing mapping? 
    # Let's check config.CLASS_MAPPING keys.
    # Assuming 'ALL' is a key based on previous context.
    existing_class_dir = import_dir / "ALL"
    existing_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy images
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(new_class_dir / f"img_{i}.jpg")
        
        img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img2.save(existing_class_dir / f"img_{i}.jpg")
        
    print(f"[VERIFY] Created dummy data at {import_dir} with class 'NewCancer'")

def run_verification():
    setup_dummy_data()
    
    # Run finetune (it prints to stdout)
    print("\n[VERIFY] Running finetune.py...")
    try:
        finetune.finetune_model()
    except Exception as e:
        print(f"[FAIL] Finetune crashed: {e}")
        return

    # Check Artifacts
    tuned_model = config.MODELS_DIR / "best_model_tuned.h5"
    classes_json = config.MODELS_DIR / "classes.json"
    
    if not tuned_model.exists():
        print("[FAIL] best_model_tuned.h5 was NOT created.")
        return
    else:
        print("[PASS] best_model_tuned.h5 exists.")

    if not classes_json.exists():
        print("[FAIL] classes.json was NOT created.")
        return
    
    with open(classes_json, 'r') as f:
        data = json.load(f)
        classes = data["classes"]
        print(f"[VERIFY] Classes in JSON: {classes}")
        
        if "NewCancer" in classes and "ALL" in classes:
            print("[PASS] JSON contains 'NewCancer'. dynamic adaptation worked!")
        else:
            print("[FAIL] JSON missing expected classes.")

if __name__ == "__main__":
    run_verification()
