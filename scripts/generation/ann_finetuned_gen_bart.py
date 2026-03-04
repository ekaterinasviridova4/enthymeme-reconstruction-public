"""
Generate reconstructions using a fine-tuned BART model.
"""

import os
import json
import logging
import argparse
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_length", type=int, default=4096, help="Max generation length")

    args = parser.parse_args()

    # Load Model and Tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load Data
    logger.info(f"Loading data from {args.test_file}")
    data = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"Found {len(data)} examples")

    results = []
    
    batch_data = []
    batch_indices = []

    for i, item in enumerate(tqdm(data)):
        source_text = item.get("output", item.get("input", ""))
        input_text = "Reconstruct implicit premises and claims from the tagged dialogue: " + source_text
        
        batch_data.append(input_text)
        batch_indices.append(i)

        if len(batch_data) == args.batch_size or i == len(data) - 1:
            input_text_batch = batch_data
            # Tokenize
            inputs = tokenizer(
                input_text_batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=4096
            ).to(device)

            if "led" in args.model_path:
                 # Create global_attention_mask for LED
                 # Set global attention on the first token (<s>)
                 inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
                 inputs["global_attention_mask"][:, 0] = 1

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=args.max_length,
                    num_beams=4, 
                    early_stopping=True
                )
            
            # Decode
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Save results
            for idx, text in zip(batch_indices, decoded):
                original_item = data[idx]
                original_item["reconstructed_text"] = text
                results.append(original_item)
            
            batch_data = []
            batch_indices = []

    # Write output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
            
    logger.info(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
