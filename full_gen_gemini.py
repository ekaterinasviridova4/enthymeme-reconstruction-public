# -*- coding: utf-8 -*-
"""
Zero-shot text reconstruction using LiteLLM (Gemini/GPT)
Analyzes JSONL input texts and reconstructs implicit parts
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
                    "original_output": ex.get("output", "")  # Keep for optional comparison
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
        default="./reconstructed_litellm",
        help="Directory to save reconstructed outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-2.0-flash",
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
        help="Temperature for generation (0.0 = deterministic)"
    )
    return parser.parse_args()

def build_prompt(text):
    """Build prompt for text reconstruction"""
    prompt = f"""Your task is to analyze the given dialogue and reconstruct implicit parts of it. There are two speakers in the dialogue, speaker1 and speaker2. The change of roles is determined by "speaker1:" and "speaker2:" marks. The dialogue is argumentative, so implicit parts can be premises or conclusions of each speaker that are not explicitly stated but are necessary for the argument to hold.

As an output, provide a complete dialogue including all original and reconstructed implicit sentences in the same format as the input.

Text:
{text}

Instructions:
- First, reproduce the entire original dialogue exactly as provided
- Then, at the end, add reconstructed implicit content attributed to the appropriate speaker
- Format: "speaker1: [Implicit premise: ...]" or "speaker1: [Implicit conclusion: ...]" or "speaker2: [Implicit premise: ...]" or "speaker2: [Implicit conclusion: ...]"
- You MUST identify and add at least 3-5 implicit premises or conclusions for each dialogue
- Each implicit premise/conclusion should be attributed to the speaker who holds that assumption or reaches that conclusion
- Even if the dialogue contains many questions or is philosophical in nature, identify the underlying assumptions and logical connections
- For long speaker turns, break down the argument into components and identify what is assumed but not stated
- Maintain the logical flow of the argument

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

def save_predictions(predictions, output_dir):
    """Save reconstructed texts to JSONL file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"reconstructed_gemini_texts_{timestamp}.jsonl")
    
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
                "original_text": item["input"],
                "reconstructed_text": f"ERROR: {str(e)}",
                "original_output": item.get("original_output", ""),
                "model": args.model
            })
    
    # Save results
    output_file = save_predictions(predictions, args.output_dir)
    
    logging.info("="*50)
    logging.info(f"Reconstruction complete! Processed {len(predictions)} examples")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*50)

if __name__ == "__main__":
    main()