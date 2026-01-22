#!/bin/bash

# Attiva l'ambiente virtuale se esiste
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Verifica se customtkinter è installato, altrimenti installa i requisiti
python -c "import customtkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Required modules not found. Installing dependencies..."
    pip install -r multi_cancer_ai/requirements.txt
fi

# Aggiungi la directory corrente al PYTHONPATH per sicurezza
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Esegui l'app desktop
python multi_cancer_ai/desktop_app.py

