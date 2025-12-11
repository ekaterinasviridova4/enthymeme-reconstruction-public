# -*- coding: utf-8 -*-
"""
Zero-shot text reconstruction using LiteLLM (Gemini/GPT)
Analyzes JSONL input texts and reconstructs implicit parts
Uses CoT prompting for better reasoning
"""

import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from litellm import completion
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="litellm_generation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ensure_api_keys():
    """Ensure necessary API keys are set"""
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not google_key and not openai_key:
        raise ValueError("No API keys found. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
    
    if google_key:
        logging.info("Google API key found.")
    if openai_key:
        logging.info("OpenAI API key found.")

ensure_api_keys()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def load_jsonl_dataset(path):
    """Load JSONL dataset with input texts"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            # Only load input field for reconstruction
            input_text = ex.get("input", "").strip()
            if input_text:
                texts.append({
                    "input": input_text,
                    "original_output": ex.get("output", ""),  # Keep for optional comparison
                    "dialogue_id": ex.get("dialogue_id"),  # Preserve dialogue metadata
                    "exchange_id": ex.get("exchange_id"),
                    "total_exchanges": ex.get("total_exchanges")
                })
    return texts

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Text reconstruction using LiteLLM (Gemini/GPT)")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cot_reconstructed_litellm",
        help="Directory to save reconstructed outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model to use (e.g., 'gemini/gemini-2.0-flash', 'openai/gpt-4o', 'openai/gpt-4-turbo')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation"
    )
    return parser.parse_args()

def build_prompt(text):
    """Build prompt for text reconstruction with Chain-of-Thought reasoning"""
    prompt = f"""Analyze the given dialogue exchange and reconstruct implicit parts of it. This is an argumentative exchange between two speakers (speaker1 and speaker2). Your goal is to identify unstated premises and conclusions that are necessary for the argument to work logically.

Text:
{text}

Instructions:

Step 1 - Reproduce the dialogue:
First, write out the entire original dialogue exactly as provided.

Step 2 - Chain-of-Thought Reasoning:
Think step-by-step about what is implicit:
a) What assumptions does speaker1 make but not state explicitly?
b) What logical conclusions are assumed but not stated in speaker1's argument?
c) What unstated beliefs underlie speaker2's response?
d) What premises does speaker2 assume when responding to speaker1?
e) Are there any implicit conclusions that follow from the stated premises but are left unstated?

Step 3 - Reconstruct the dialogue:
Insert the identified implicit premises and conclusions into the dialogue at appropriate points.
- Format implicit elements as: "speaker1: [Implicit premise: ...]" or "speaker2: [Implicit conclusion: ...]"
- Place implicit elements near the related explicit statements
- Identify at least 2-4 implicit elements per speaker (depending on the length of their turn)
- Each implicit element should be necessary for the logical coherence of the argument
- Maintain the chronological flow of the dialogue

Important: Focus on logical necessity - only include premises/conclusions that are truly required for the argument to work, not just tangentially related ideas.

Output:
"""
    return prompt

# def build_prompt(text):
#     """Build prompt for text reconstruction"""
#     prompt = f"""Your task is to analyze the given text and reconstruct implicit parts of the text. The text is argumentative, so implicit parts can be premises or conclusions that are not explicitly stated but are necessary for the argument to hold.

# As an output, provide a complete text including all original and reconstructed implicit sentences.

# Text:
# {text}

# Instructions:
# - Identify all explicit sentences that are already present in the text
# - Identify and reconstruct any implicit premises or conclusions
# - Maintain the logical flow of the argument

# Output:
# """
#     return prompt

def save_predictions(predictions, output_dir, model_name):
    """Save reconstructed texts to JSONL file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"cot_reconstructed_gpt_{timestamp}.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    logging.info(f"Reconstructed texts saved to {output_file}")
    return output_file

def generate_reconstruction(model, text, max_tokens=2048, temperature=0.0):
    """Generate reconstruction for a single text using LiteLLM"""
    prompt = build_prompt(text)
    
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        reconstruction = response.choices[0].message.content.strip()
        return reconstruction
        
    except Exception as e:
        logging.error(f"Error in generation: {str(e)}")
        raise

def main():
    """Main function to run text reconstruction"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    
    logging.info("="*50)
    logging.info("Starting text reconstruction process")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info("="*50)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load data
    logging.info("Loading dataset...")
    data = load_jsonl_dataset(args.input_file)
    logging.info(f"Loaded {len(data)} examples")
    
    # Limit if specified
    if args.limit:
        data = data[:args.limit]
        logging.info(f"Limited to first {args.limit} examples")
    
    # Process each text
    logging.info(f"Processing {len(data)} texts with model: {args.model}")
    predictions = []
    
    for idx, item in enumerate(tqdm(data, desc="Reconstructing texts")):
        try:
            reconstruction = generate_reconstruction(
                model=args.model,
                text=item["input"],
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            predictions.append({
                "index": idx,
                "dialogue_id": item.get("dialogue_id"),
                "exchange_id": item.get("exchange_id"),
                "total_exchanges": item.get("total_exchanges"),
                "original_text": item["input"],
                "reconstructed_text": reconstruction,
                "original_output": item.get("original_output", ""),
                "model": args.model
            })
            
            # Log progress every 10 examples
            if (idx + 1) % 10 == 0:
                logging.info(f"Processed {idx + 1}/{len(data)} examples")
                
        except Exception as e:
            logging.error(f"Error processing example {idx}: {str(e)}")
            predictions.append({
                "index": idx,
                "dialogue_id": item.get("dialogue_id"),
                "exchange_id": item.get("exchange_id"),
                "total_exchanges": item.get("total_exchanges"),
                "original_text": item["input"],
                "reconstructed_text": f"ERROR: {str(e)}",
                "original_output": item.get("original_output", ""),
                "model": args.model
            })
    
    # Save results
    output_file = save_predictions(predictions, args.output_dir, args.model)
    
    logging.info("="*50)
    logging.info(f"Reconstruction complete! Processed {len(predictions)} examples")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*50)

if __name__ == "__main__":
    main()