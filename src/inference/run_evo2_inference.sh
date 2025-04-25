#!/bin/bash

# Skrypt do uruchomienia inferencji evo2 na GPU A100 na klastrze Entropy
# Użycie: ./run_evo2_inference.sh

# Parametry zadania dla Slurm
# --partition=common - partycja obliczeniowa
# --qos=student - quality of service dla studentów
# --gres=gpu:a100:1 - rezerwacja jednej karty A100
# --time=00:30:00 - limit czasu: 30 minut

# Uruchomienie zadania za pomocą srun
srun --partition=common \
     --qos=student \
     --gres=gpu:a100:1 \
     --time=00:30:00 \
     python3 run_evo2_inference.py \
     --sequence "ACTGACTGACTGACTG" \
     --model_name "arcinstitute/evo2_40b" \
     --num_tokens 8 \
     --output_file "inference_output.json"

echo "Zadanie zostało uruchomione. Wyniki zostaną zapisane w pliku inference_output.json" 