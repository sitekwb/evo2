#!/bin/bash

# Skrypt do uruchomienia klasyfikatora enhancerów na GPU A100 na klastrze Entropy
# Użycie: ./run_enhancer_inference.sh <plik_wejściowy>

# Sprawdzenie, czy podano nazwę pliku wejściowego
if [ $# -lt 1 ]; then
    echo "Użycie: $0 <plik_wejściowy>"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE="${INPUT_FILE%.txt}_results.json"

# Parametry zadania dla Slurm
# --partition=common - partycja obliczeniowa
# --qos=student - quality of service dla studentów
# --gres=gpu:a100:1 - rezerwacja jednej karty A100
# --time=01:00:00 - limit czasu: 1 godzina

# Uruchomienie zadania za pomocą srun
srun --partition=common \
     --qos=student \
     --gres=gpu:a100:1 \
     --time=01:00:00 \
     python3 run_enhancer_inference.py \
     --input_file "$INPUT_FILE" \
     --model_name "arcinstitute/evo2_7b" \
     --output_file "$OUTPUT_FILE" \
     --batch_size 4

echo "Zadanie zostało uruchomione. Wyniki zostaną zapisane w pliku $OUTPUT_FILE" 