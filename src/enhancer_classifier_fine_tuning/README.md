# Enhancer Classifier Fine-tuning

This script fine-tunes a pre-trained Evo DNA language model (specifically `arcinstitute/evo2_7b`) for enhancer classification.

It freezes the base Evo model and trains a simple MLP classifier head on top of the pooled embeddings from the base model's last layer.

## Setup

1.  **Clone the repository:** Ensure you have cloned the repository containing this script and the associated `evo2` module.

2.  **Create Environment:** It is recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    conda create -n evo_finetune python=3.10 # Or your preferred python version
    conda activate evo_finetune
    ```

3.  **Install Dependencies:**
    *   **Install PyTorch:** Install PyTorch matching your system (CPU or CUDA version). Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for instructions. Example for CUDA 12.1:
        ```bash
        # Make sure you are in the correct conda environment
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
        *Note: The Evo model and its dependencies (`transformer-engine`) require a CUDA-enabled environment to run correctly. Running on CPU is currently not supported due to these dependencies.*

    *   **Install other requirements:**
        ```bash
        pip install -r requirements.txt
        ```

4.  **(Optional but recommended) Login to Hugging Face Hub:** If you encounter issues downloading models or need access to private/gated models in the future:
    ```bash
    huggingface-cli login
    ```

## Running the Script

1.  **Prepare Your Data:**
    *   Modify the `if __name__ == "__main__":` block in `enhancer_classifier.py`.
    *   Replace the dummy data generation section (`# --- Data Preparation (Dummy Data Example) --- #`) with your actual data loading and preprocessing logic.
    *   Ensure your data provides sequences (strings) and corresponding binary labels (0 or 1).
    *   The script uses the model's tokenizer to prepare `input_ids` and `attention_mask` suitable for the model.

2.  **Configure Training Parameters:**
    *   Adjust parameters like `HIDDEN_MLP_DIM`, `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`, and `MAX_SEQ_LENGTH` in the `if __name__ == "__main__":` block according to your dataset and hardware capabilities.

3.  **Run Training:**
    ```bash
    python src/enhancer_classifier_fine_tuning/enhancer_classifier.py
    ```
    The script will automatically use the GPU if available and detected by PyTorch.

## Notes

*   The base `evo2` model parameters are frozen during training; only the MLP classifier head is trained.
*   The script uses mean pooling over the hidden states of the last layer of the base model.
*   The code currently assumes the underlying model structure allows accessing the hidden dimension via `model.config.d_model`. This might need adjustment depending on the exact model implementation loaded by the `evo2` wrapper. 