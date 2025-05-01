#!/bin/bash

# Skrypt do uruchomienia inferencji evo2 na GPU A100 na klastrze Entropy
# Użycie: ./run_evo2_inference.sh

# Sprawdzenie czy jesteśmy w katalogu głównym czy w src/inference
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Jesteśmy w katalogu głównym projektu."
    PROJECT_ROOT="$(pwd)"
else
    echo "Jesteśmy w katalogu src/inference, przechodzę do katalogu głównego..."
    PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$0")")")"
    cd "$PROJECT_ROOT"
fi

echo "Katalog główny projektu: $PROJECT_ROOT"

# Sprawdzenie czy środowisko wirtualne już istnieje
if [ ! -d "evo2_env" ]; then
    echo "Tworzenie nowego środowiska wirtualnego..."
    python3 -m venv evo2_env
else
    echo "Środowisko wirtualne już istnieje."
fi

# Aktywacja środowiska wirtualnego
source evo2_env/bin/activate

# Instalacja wymaganych pakietów
pip install -r requirements.txt

# Instalacja pakietu evo2 w trybie deweloperskim
echo "Instalacja pakietu evo2 z katalogu: $(pwd)"
pip install -e .

# Parametry zadania dla Slurm
# --partition=common - partycja obliczeniowa
# --qos=student - quality of service dla studentów
# --gres=gpu:a100:1 - rezerwacja jednej karty A100
# --time=00:30:00 - limit czasu: 30 minut

# Uruchomienie zadania za pomocą srun
srun --partition=a100 \
     --qos=wsitek_a100 \
     --gres=gpu:a100:1 \
     --time=00:30:00 \
     evo2_env/bin/python src/inference/run_evo2_inference.py \
     --sequence "ACTGACTGACTGACTG" \
     --model_name "arcinstitute/evo2_40b" \
     --num_tokens 8 \
     --output_file "inference_output.json"

echo "Zadanie zostało uruchomione. Wyniki zostaną zapisane w pliku inference_output.json"

# Deaktywacja środowiska wirtualnego
deactivate 