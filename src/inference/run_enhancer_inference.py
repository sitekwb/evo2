#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
import json

# Import the classifier
from src.enhancer_classifier_fine_tuning.enhancer_classifier import EnhancerClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Run enhancer classifier inference")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Input file with DNA sequences, one per line"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="arcinstitute/evo2_7b",
        help="Base model name to use"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to trained classifier checkpoint"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="enhancer_predictions.json",
        help="Output file to save results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for inference"
    )
    return parser.parse_args()

def load_sequences(file_path):
    with open(file_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    return sequences

def main():
    args = parse_args()
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"Initializing EnhancerClassifier with {args.model_name}...")
    model = EnhancerClassifier(model_name=args.model_name)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Load input sequences
    print(f"Loading sequences from {args.input_file}")
    sequences = load_sequences(args.input_file)
    
    # Tokenize sequences
    results = {}
    
    with torch.no_grad():
        for i in range(0, len(sequences), args.batch_size):
            batch_sequences = sequences[i:i + args.batch_size]
            
            # Tokenize
            inputs = model.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # Move inputs to device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Run inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Process results
            probs = outputs.cpu().numpy().flatten().tolist()
            
            # Add to results dictionary
            for seq, prob in zip(batch_sequences, probs):
                results[seq] = {"probability": prob, "is_enhancer": prob > 0.5}
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 