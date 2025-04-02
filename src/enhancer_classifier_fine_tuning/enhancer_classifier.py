# Standard library imports first
# (None used directly in this file after removing path logic)

# Third-party imports
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Local application/library specific imports
# Requires the package to be installed (e.g., `pip install -e .`)
from evo2.models import Evo2


# Removed unused imports:
# from transformers import AutoModel, AutoTokenizer, AutoConfig


class EnhancerClassifier(nn.Module):
    """Classifier using a pre-trained Evo model with a MLP head."""
    def __init__(self, model_name="arcinstitute/evo2_7b", hidden_mlp_dim=512):
        super().__init__()
        # Load model and tokenizer using the custom Evo2 wrapper
        print(f"Initializing Evo2 wrapper for {model_name}...")
        # This might download the model if not cached
        try:
            evo2_wrapper = Evo2(model_name=model_name)
        except Exception as e:
            print(f"Error initializing Evo2 wrapper: {e}")
            print("Ensure model name is correct and files accessible.")
            raise

        # Get the underlying StripedHyena model
        self.evo_model = evo2_wrapper.model
        self.tokenizer = evo2_wrapper.tokenizer
        print("Evo2 model and tokenizer loaded successfully.")

        # Freeze the base model parameters
        print("Freezing base model parameters...")
        for param in self.evo_model.parameters():
            param.requires_grad = False
        print("Base model parameters frozen.")

        # Get the hidden size from the base model config
        try:
            # Assuming hidden size is d_model based on StripedHyena structure
            hidden_size = self.evo_model.config.d_model
            print(f"Base model hidden size (d_model): {hidden_size}")
        except AttributeError as e:
            print("Could not get hidden size from evo_model.config.d_model.")
            print(f"Error: {e}")
            print("Check StripedHyena model config and update code.")
            raise ValueError("Failed to get hidden size for MLP.") from e

        # Define the MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Optional dropout for regularization
            nn.Linear(hidden_mlp_dim, 1),
            nn.Sigmoid()
        )
        print("MLP classifier head initialized.")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Uses a forward hook to capture the output of the last layer
        of the base model (self.evo_model), performs mean pooling,
        and passes the result through the classifier head.

        Args:
            input_ids (torch.Tensor): Tensor of token ids,
                                    shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Tensor indicating padding,
                                        shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Output probabilities, shape (batch_size, 1)
        """
        # --- Hook Setup --- # 
        last_layer_output = None

        # Define the hook function
        def hook_fn(module, input, output):
            nonlocal last_layer_output
            # Output might be a tuple, we typically want the first element (hidden states)
            if isinstance(output, tuple):
                last_layer_output = output[0].detach()
            else:
                last_layer_output = output.detach()

        # --- Register Hook --- #
        # IMPORTANT: Assumes last layer is self.evo_model.layers[-1].
        # Verify/adjust if needed.
        try:
            last_layer = self.evo_model.layers[-1]
            hook_handle = last_layer.register_forward_hook(hook_fn)
        except (AttributeError, IndexError) as e:
            print(f"Hook registration error on layers[-1]: {e}")  # noqa: E501
            print("Cannot capture embeddings. Check model structure.")
            raise

        # --- Forward Pass (Base Model) --- #
        # Run forward pass. Hook captures the output.
        # Base model logits are not needed here.
        # Classifier needs grads later.
        _ = self.evo_model(input_ids=input_ids)

        # --- Remove Hook --- #
        hook_handle.remove()

        # --- Check if Hook Captured Output --- #
        if last_layer_output is None:
            raise RuntimeError("Hook did not capture the last layer output.")

        # --- Pooling --- #
        # Perform mean pooling, ignoring padding tokens
        # last_layer_output shape: (batch_size, seq_length, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_layer_output.size()
        ).float()
        # Ensure sum_mask is never zero to avoid division by zero
        sum_embeddings = torch.sum(last_layer_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # --- Classifier Head --- #
        # Pass pooled output through the classifier
        # Ensure pooled_output requires grad for classifier training
        pooled_output.requires_grad_(True)
        logits = self.classifier(pooled_output)
        return logits


# --- Training Setup --- #

def train_model(
    model, dataloader, optimizer, criterion, device, epochs=3
):
    """Basic training loop."""
    model.train()  # Set model to training mode
    model.to(device)  # Move model to the specified device

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        batch_count = 0
        for batch in dataloader:
            # Move batch to device
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels.float().unsqueeze(1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Average Training Loss: {avg_loss:.4f}")
        else:
            print("No batches processed in this epoch.")

    print("Training finished.")


# --- Main Execution Block --- #

if __name__ == "__main__":
    # --- Configuration --- #
    MODEL_NAME = "arcinstitute/evo2_7b"
    # Reduce hidden dim for faster testing if needed
    HIDDEN_MLP_DIM = 128
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4      # Adjust based on GPU memory
    EPOCHS = 1          # Start with 1 epoch for quick testing
    MAX_SEQ_LENGTH = 1024  # Example max length, adjust as needed

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Initialization --- #
    print("Initializing model...")
    # Pass trust_remote_code=True if needed for the specific model revision
    model = EnhancerClassifier(
        model_name=MODEL_NAME, hidden_mlp_dim=HIDDEN_MLP_DIM
    )
    # Get tokenizer from the model instance
    tokenizer = model.tokenizer
    print("Model initialized.")

    # --- Data Preparation (Dummy Data Example) --- #
    # !! Replace this with your actual data loading and preprocessing !!
    print("Preparing dummy data...")
    # Example sequences (replace with your real data)
    sequences = [
        "ACGT" * (MAX_SEQ_LENGTH // 4),  # Simulating a longer sequence
        "TTAACCGG" * (MAX_SEQ_LENGTH // 8),
        "CAGT" * (MAX_SEQ_LENGTH // 4),
        "GGGAAATTTCCC" * (MAX_SEQ_LENGTH // 12)
    ]
    # Example labels (0 or 1)
    labels = [1, 0, 1, 0]

    # Tokenize sequences
    # Ensure padding='max_length' and truncation=True are used
    # Add special tokens if the model expects them (check model documentation)
    print(
        f"Tokenizing {len(sequences)} sequences "
        f"with max length {MAX_SEQ_LENGTH}..."
    )
    encodings = tokenizer(
        sequences,
        padding='max_length',   # Pad to max_length
        truncation=True,        # Truncate longer sequences
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",    # Return PyTorch tensors
        # Add verbose=False if tokenizer prints too much
        # verbose=False
    )

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels_tensor = torch.tensor(labels)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(
        f"Dummy data prepared. Dataset size: {len(dataset)}, DataLoader ready."
    )

    # --- Training Setup --- #
    # Define Loss and Optimizer
    # Optimize only the classifier parameters
    optimizer = AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    # Binary Cross Entropy for binary classification
    criterion = nn.BCELoss()

    # --- Start Training --- #
    print("Starting training...")
    train_model(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        epochs=EPOCHS
    )
