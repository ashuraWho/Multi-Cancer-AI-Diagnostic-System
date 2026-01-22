#!/bin/bash
# Script helper per avviare il training usando l'ambiente virtuale configurato
# Fix per crash su Mac (richiede Python 3.10 e numpy<2.0)

# Ottieni la directory dello script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Attiva venv ed esegui
echo "Avvio training con ambiente virtuale (Python 3.10)..."
export PYTHONPATH="$DIR"
"$DIR/venv/bin/python" "$DIR/multi_cancer_ai/train_main.py"
