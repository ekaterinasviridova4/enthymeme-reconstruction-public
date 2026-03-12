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
import litellm
from litellm import completion
import logging
from datetime import datetime

# Enable dropping unsupported parameters (e.g. temperature=0 for gpt-5)
litellm.drop_params = True

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
            # Load 'output' as annotated_text because it contains the <Implicit>/<Explicit> tags
            annotated_text = ex.get("output", "").strip()
            if annotated_text:
                texts.append({
                    "annotated_text": annotated_text,
                    "original_input": ex.get("input", ""),
                    "dialogue_id": ex.get("dialogue_id"),
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
        default="../../results_dev/ann_short_reconstructed_litellm",
        help="Directory to save reconstructed outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model to use (e.g., 'gemini/gemini-2.5-flash', 'gemini/gemini-3-flash-preview', 'openai/gpt-4o', 'openai/gpt-5')"
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
        default=16384,
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
    prompt = f"""You are an expert in argumentation mining. Your task is to reconstruct implicit premises and claims in an argumentative dialogue.

INPUT FORMAT:
The input text is a dialogue between "speaker1" and "speaker2".
Sentences are annotated with tags:
- <Explicit>...</Explicit>: Explicitly stated content.
- <Implicit>...</Implicit>: Implicit content that needs reconstruction.

TASK:
For each sentence marked as <Implicit>...</Implicit>:
1. Identify the missing logical link (premise or claim).
2. Formulate it as a clear sentence.
3. Classify as <IMPLICIT_PREMISE> or <IMPLICIT_CLAIM>.
4. Insert it immediately before or after the relevant <Implicit> sentence, whichever makes the most logical sense.

OUTPUT FORMAT:
- Output ONLY the final reconstructed dialogue.
- Do NOT include any explanations, reasoning, lists, or notes.
- Do NOT output the text "Reconstructed Dialogue" or any preamble.
- Start directly with "speaker1:" or "speaker2:".
- Keep all original <Explicit> and <Implicit> tags exactly as is.
- Insert your reconstructions in <IMPLICIT_PREMISE>...</IMPLICIT_PREMISE> or <IMPLICIT_CLAIM>...</IMPLICIT_CLAIM> tags.

EXAMPLE:
Input:
speaker1: <Explicit> It is raining. </Explicit> <Implicit> I will take an umbrella. </Implicit>

Output:
speaker1: <Explicit> It is raining. </Explicit> <IMPLICIT_PREMISE>Umbrellas protect from rain.</IMPLICIT_PREMISE> <Implicit> I will take an umbrella.</Implicit>

INPUT TEXT:
{text}
"""
    return prompt

def save_predictions(predictions, output_dir):
    """Save reconstructed texts to JSONL file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"reconstructed_dev_gpt4o_{timestamp}.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    logging.info(f"Reconstructed texts saved to {output_file}")
    return output_file

def generate_reconstruction(model, text, max_tokens=16384, temperature=0.0):
    """Generate reconstruction for a single text using LiteLLM"""
    prompt = build_prompt(text)
    
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Log generation details
        finish_reason = response.choices[0].finish_reason
        usage = getattr(response, 'usage', 'Unknown')
        logging.info(f"Generation finished. Reason: {finish_reason}, Usage: {usage}")
        
        if finish_reason == "length":
            logging.warning("WARNING: Generation was truncated due to token limit!")
            
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

    # For Gemini 3
    if "gemini-3" in args.model.lower() and args.temperature < 1.0:
        logging.warning(f"Warning: Temperature set to 1.0.")
        args.temperature = 1.0

    predictions = []
    
    for idx, item in enumerate(tqdm(data, desc="Reconstructing texts")):
        try:
            rec_text = generate_reconstruction(
                model=args.model,
                text=item["annotated_text"],
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            predictions.append({
                "index": idx,
                "original_text": item["annotated_text"],
                "reconstructed_text": rec_text,
                "original_input": item.get("original_input", ""),
                "model": args.model
            })
            
            # Log progress every 10 examples
            if (idx + 1) % 10 == 0:
                logging.info(f"Processed {idx + 1}/{len(data)} examples")
                
        except Exception as e:
            logging.error(f"Error processing example {idx}: {str(e)}")
            predictions.append({
                "index": idx,
                "original_text": item["annotated_text"],
                "reconstructed_text": f"ERROR: {str(e)}",
                "original_input": item.get("original_input", ""),
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