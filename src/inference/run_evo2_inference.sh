#!/bin/bash

# Skrypt do uruchomienia inferencji evo2 na GPU A100 na klastrze Entropy
# Użycie: ./run_evo2_inference.sh

# Funkcja do obsługi błędów
handle_error() {
    echo "Wystąpił błąd w linii $1"
    echo "Komenda: $2"
    exit 1
}

# Ustawienie obsługi błędów
trap 'handle_error ${LINENO} "$BASH_COMMAND"' ERR

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

# Przygotowanie skryptu wykonawczego, który będzie uruchomiony na węźle obliczeniowym
cat > run_on_compute_node.sh << 'EOF'
#!/bin/bash

echo "Uruchamianie na węźle obliczeniowym"
echo "Katalog bieżący: $(pwd)"

# Aktywacja środowiska wirtualnego
source evo2_env/bin/activate

# Inicjalizacja systemu modułów dla Linux
echo "Inicjalizacja systemu modułów..."
if [ -f "/etc/profile.d/lmod.sh" ]; then
    source /etc/profile.d/lmod.sh
elif [ -f "/etc/profile.d/modules.sh" ]; then
    source /etc/profile.d/modules.sh
fi

# Ładowanie modułu CUDA
if command -v module &> /dev/null; then
    echo "Ładowanie modułu CUDA..."
    module load cuda
    echo "Załadowane moduły: $(module list 2>&1)"
fi

# Sprawdzenie czy NVCC jest dostępny
if ! command -v nvcc &> /dev/null; then
    echo "Ostrzeżenie: NVCC nie jest dostępny, niektóre pakiety mogą nie zostać poprawnie zainstalowane."
    # Kontynuujemy mimo błędu, ponieważ niektóre pakiety mogą nie wymagać NVCC
else
    echo "NVCC jest dostępny: $(nvcc --version)"
fi

# Upewnienie się, że pip jest zainstalowany i zaktualizowany
echo "Aktualizacja pip..."
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip

# Instalacja narzędzi do budowania
echo "Instalacja narzędzi do budowania..."
pip install wheel setuptools build

# Instalacja wymaganych narzędzi systemowych
echo "Instalacja narzędzi systemowych..."
pip install ninja cmake pybind11

# Instalacja wymaganych pakietów
echo "Instalacja wymaganych pakietów..."
pip install -r requirements.txt

# Instalacja vortex
echo "Instalacja pakietu vortex..."
pip install git+https://github.com/arcinstitute/vortex.git

# Instalacja pakietu evo2 w trybie deweloperskim
echo "Instalacja pakietu evo2 z katalogu: $(pwd)"
pip install -e .

# Uruchomienie skryptu inferencji
echo "Uruchamianie skryptu inferencji..."
python src/inference/run_evo2_inference.py \
    --sequence "ACTGACTGACTGACTG" \
    --model_name "arcinstitute/evo2_40b" \
    --num_tokens 8 \
    --output_file "inference_output.json"

# Deaktywacja środowiska wirtualnego
deactivate
EOF

chmod +x run_on_compute_node.sh

echo "Przygotowano skrypt do uruchomienia na węźle obliczeniowym: run_on_compute_node.sh"

# Uruchomienie zadania za pomocą sbatch
echo "Uruchamianie zadania na Slurm..."
sbatch --partition=a100 \
       --qos=wsitek_a100 \
       --gres=gpu:a100:1 \
       --time=00:30:00 \
       --output=slurm_output_%j.log \
       ./run_on_compute_node.sh

echo "Zadanie zostało wysłane do Slurm. Sprawdź status za pomocą komendy 'squeue'"
echo "Wyniki i logi będą dostępne w pliku slurm_output_XXXX.log, gdzie XXXX to ID zadania" 