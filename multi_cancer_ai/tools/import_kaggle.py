import kagglehub
import shutil
import os
import sys
from pathlib import Path

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from multi_cancer_ai.config import config

def import_dataset(kaggle_handle):
    """
    Downloads a dataset from Kaggle and moves it to the project's data directory.
    Args:
        kaggle_handle (str): The dataset handle (e.g., 'username/dataset-name')
    """
    print(f"[INFO] Downloading dataset: {kaggle_handle}...")
    try:
        # Download latest version
        path = kagglehub.dataset_download(kaggle_handle)
        print(f"[INFO] Downloaded to cache: {path}")
        
        # Define destination
        dest_dir = config.BASE_DIR / "data" / "kaggle_import"
        
        # Clean destination if exists to avoid mixing datasets
        if dest_dir.exists():
            print(f"[WARN] Cleaning existing import directory: {dest_dir}")
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files logic (optimized for storage)
        print(f"[INFO] Moving files (Zero-Copy) to: {dest_dir}")
        path_obj = Path(path)
        
        for item in path_obj.iterdir():
            if item.is_dir():
                # If dir exists in dest, merge or overwrite? Simple overwrite here to match strict cleanup
                dest_item = dest_dir / item.name
                if dest_item.exists():
                     shutil.rmtree(dest_item)
                shutil.move(str(item), str(dest_dir))
            else:
                shutil.move(str(item), str(dest_dir))
        
        # Cleanup Cache (Kagglehub leaves the folder structure)
        print(f"[INFO] Cleaning up download cache: {path}")
        try:
            shutil.rmtree(path)
            # Optional: aggressive cleaning of parent hash folder if empty? 
            # Usually path is like .../datasets/user/dataset/versions/1
            # We can leave the structure, it's small. The big files are moved.
        except Exception as e:
            print(f"[WARN] Could not full clean path {path}: {e}")

        print(f"[SUCCESS] Dataset imported & aggregated to {dest_dir}")
        print("[INFO] Storage optimized: Zero duplicates remain.")
        print("You can now run 'python tools/finetune.py' to train on this data.")
        
    except Exception as e:
        print(f"[ERROR] Failed to download/import dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/import_kaggle.py <kaggle_handle>")
        print("Example: python tools/import_kaggle.py jessicali9530/stanford-dogs-dataset")
    else:
        import_dataset(sys.argv[1])
