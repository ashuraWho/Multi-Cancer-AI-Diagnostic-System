import sys
from pathlib import Path
import os

# Setup paths
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from multi_cancer_ai.config import config

import io

def get_structure_string(glossary=None):
    """
    Crawls the dataset directory and returns the class hierarchy as a string.
    Args:
        glossary (dict, optional): Dictionary mapping folder names to descriptions.
    """
    output = io.StringIO()
    
    # 1. Check Main Dataset
    main_ds = config.DATASET_PATH
    output.write(f"📊 Dataset Inspection\n")
    output.write(f"Root: {main_ds}\n")
    
    if not main_ds.exists():
        output.write("[ERR] Main dataset not found!\n")
    else:
        output.write("\n[Main Dataset Structure]\n")
        _write_tree(main_ds, output, glossary=glossary)

    # 2. Check Kaggle Import
    kaggle_dir = config.BASE_DIR / "data" / "kaggle_import"
    if kaggle_dir.exists():
        output.write(f"\n[Kaggle Import Staging Area] ({kaggle_dir})\n")
        _write_tree(kaggle_dir, output, glossary=glossary)
    else:
        output.write("\n[Kaggle Import] (Empty/Not initialized)\n")
        
    return output.getvalue()

def _write_tree(directory, buffer, level=0, glossary=None):
    """
    Recursively writes folder structure to buffer.
    """
    indent = "  " * level
    PREFIX = "├── " if level > 0 else ""
    
    # Get subdirs
    try:
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        subdirs.sort()
        
        for d in subdirs:
            # Count images
            image_count = len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            count_str = f" ({image_count} images)" if image_count > 0 else ""
            
            # Lookup description
            desc = ""
            if glossary and d.name in glossary:
                desc = f"\n{indent}    ℹ️ {glossary[d.name]}"
            
            buffer.write(f"{indent}{PREFIX}{d.name}{count_str}{desc}\n")
            
            # Recurse if not too deep (we want to see subclasses)
            if level < 2: 
                _write_tree(d, buffer, level + 1, glossary)
                
    except PermissionError:
        buffer.write(f"{indent}[Access Denied]\n")

if __name__ == "__main__":
    print(get_structure_string())
