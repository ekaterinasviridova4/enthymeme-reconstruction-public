"""
Zero-shot text reconstruction using BART (Seq2Seq)
Analyzes JSONL input texts and reconstructs implicit parts
"""

import os
import json
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import logging
from datetime import datetime
import transformers
import bitsandbytes as bnb
import accelerate


def ensure_huggingface_token():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        logging.warning("Hugging Face token not found.")
    else:
        logging.info("Hugging Face token found. Logging in...")
        try:
            login(token=token)
        except Exception as e:
            logging.warning(f"Login failed: {e}")


def set_seed(seed):
    """Random seeds to make results reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_dataset(path):
    """Load JSONL dataset with input texts"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
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
    parser = argparse.ArgumentParser(description="Text reconstruction using BART")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../results/ann_full_reconstructed_bart",
        help="Directory to save reconstructed outputs"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/bart-large",
        help="Hugging Face model ID e.g. 'facebook/bart-base', 'facebook/bart-large'"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024, 
        help="Maximum number of tokens to output"
    )
    return parser.parse_args()

def setup_model(model_id):
    """Load model and tokenizer"""
    logging.info(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # AutoModelForSeq2SeqLM for BART
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
        
    logging.info("Model loaded successfully")
    return model, tokenizer

def build_prompt(text):
    """Build prompt for text reconstruction"""
    prompt = f"""You are an expert in argumentation mining. Your task is to reconstruct implicit premises and claims in an argumentative dialogue.

INPUT FORMAT:
The input text is a dialogue between "speaker1" and "speaker2".
Sentences are annotated with tags:
- <Explicit>...</Explicit>: Explicitly stated content.
- <Implicit>...</Implicit>: Implicit content that needs reconstruction.

TASK:
For each <Implicit> sentence, you must insert a missing logical link (Premise or Claim) that explains it.
1. Formulate the missing link.
2. Determine if it is a PREMISE or a CLAIM.
3. Insert it using strictly these tags:
   - <IMPLICIT_PREMISE>...content...</IMPLICIT_PREMISE>
   - <IMPLICIT_CLAIM>...content...</IMPLICIT_CLAIM>
4. Do NOT use <Implicit> tags for your inserted text. The <Implicit> tag is ONLY for the original text.

OUTPUT FORMAT:
- Output the full dialogue including original tags.
- Insert your <IMPLICIT_PREMISE> or <IMPLICIT_CLAIM> tags immediately next to the relevant <Implicit> sentence.
- Do NOT change the content inside <Explicit> or <Implicit> tags.
- Maintain "speaker1:" and "speaker2:" labels.

EXAMPLE:
Input:
speaker1: <Explicit> It is raining. </Explicit> <Implicit> I will take an umbrella. </Implicit>

Output:
speaker1: <Explicit> It is raining. </Explicit> <IMPLICIT_PREMISE>Umbrellas protect from rain.</IMPLICIT_PREMISE> <Implicit> I will take an umbrella. </Implicit>

INPUT TEXT:
{text}

OUTPUT:
"""
    return prompt

def save_predictions(predictions, output_dir, model_name):
    """Save reconstructed texts to JSONL file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = model_name.replace("/", "_")
    output_file = os.path.join(output_dir, f"reconstructed_test_{clean_name}_{timestamp}.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    logging.info(f"Reconstructed texts saved to {output_file}")
    return output_file

def generate_reconstruction(model, tokenizer, text, max_new_tokens=1024):
    """Reconstruct implicitness using Seq2Seq model"""
    prompt = build_prompt(text)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # num_beams=5, # Beam search is common for seq2seq to improve quality
            # early_stopping=True
        )
    
    # Decode output
    reconstruction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    return reconstruction

def main():
    """Main function"""

    args = parse_args()
    
    ensure_huggingface_token()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"reconstruction_bart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    
    logging.info("="*50)
    logging.info("Starting text reconstruction process with BART")
    logging.info(f"Transformers version: {transformers.__version__}")
    logging.info(f"BitsAndBytes version: {bnb.__version__}")
    logging.info(f"Accelerate version: {accelerate.__version__}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model_id}")
    logging.info(f"Seed: {args.seed}")
    logging.info("="*50)
    
    # Set seed 
    set_seed(args.seed)
    
    # Load data
    logging.info("Loading dataset...")
    data = load_jsonl_dataset(args.input_file)
    logging.info(f"Loaded {len(data)} examples")
    
    # Limit
    if args.limit:
        data = data[:args.limit]
        logging.info(f"Limited to first {args.limit} examples")
    
    # Load model and tokenizer
    logging.info("Loading model...")
    model, tokenizer = setup_model(args.model_id)
    model.eval()
    
    # Process each text
    logging.info(f"Processing {len(data)} texts...")
    predictions = []
    
    # Create a progress bar
    pbar = tqdm(data, desc="Reconstructing texts")
    
    for idx, item in enumerate(pbar):
        rec_text = generate_reconstruction(
            model, 
            tokenizer, 
            item["annotated_text"],
            max_new_tokens=args.max_new_tokens
        )
        
        predictions.append({
            "index": idx,
            "original_text": item["annotated_text"],
            "reconstructed_text": rec_text,
            "original_input": item.get("original_input", "")
        })
        
        # Log progress occasionally to file
        if (idx + 1) % 10 == 0:
            logging.info(f"Processed {idx + 1}/{len(data)} examples")
            
    # Save results
    output_file = save_predictions(predictions, args.output_dir, args.model_id)
    
    logging.info("="*50)
    logging.info(f"Reconstruction complete! Processed {len(predictions)} examples")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*50)

if __name__ == "__main__":
    main()
