#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path

# Import from local evo2 package
from evo2.models import Evo2

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with evo2 model")
    parser.add_argument("--sequence", type=str, default="ACTGACTGACTGACTG", 
                        help="DNA sequence for inference")
    parser.add_argument("--model_name", type=str, default="arcinstitute/evo2_40b",
                        help="Model name to use for inference")
    parser.add_argument("--num_tokens", type=int, default=8,
                        help="Number of tokens to generate")
    parser.add_argument("--output_file", type=str, default="inference_output.json",
                        help="Output file to save results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    print(f"Initializing Evo2 with model {args.model_name}...")
    model = Evo2(model_name=args.model_name)
    
    print(f"Running inference on sequence: {args.sequence}")
    
    # Generate output using the model
    output = model.generate(
        args.sequence,
        num_tokens=args.num_tokens,
        top_k=1,
        enable_sampled_probs=True
    )
    
    # Save output to file
    output_path = Path(args.output_file)
    output_path.write_text(str(output))
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 