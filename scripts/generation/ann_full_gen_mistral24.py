"""
Zero-shot text reconstruction using Mistral and Olmo
Analyzes JSONL input texts and reconstructs implicit parts
"""

import os
import re
import json
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import logging
from datetime import datetime
import transformers
import bitsandbytes as bnb
import accelerate

# Logging
logging.basicConfig(
    filename="olmo_zero_generation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ensure_huggingface_token():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Hugging Face token not found. Please ensure it is set in the environment.")
    else:
        logging.info("Hugging Face token found. Logging in...")
        login(token=token)

ensure_huggingface_token()

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
            # 'output' = annotated_text because it contains the <Implicit>/<Explicit> tags
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
    parser = argparse.ArgumentParser(description="Text reconstruction using Mistral or Olmo")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../results/ann_full_reconstructed_mistral_olmo",
        help="Directory to save reconstructed outputs"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="allenai/Olmo-3.1-32B-Instruct",
        help="Hugging Face model ID 'allenai/Olmo-3.1-32B-Instruct', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'"
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
        default=16384,
        help="Maximum number of tokens to output"
    )
    return parser.parse_args()

def setup_model(model_id):
    """4-bit quantization (Mistral and Olmo)"""
    logging.info(f"Loading model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    if "olmo" in model_id.lower():
        logging.info("Detected Olmo, using AutoModelForCausalLM")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb_config
        )
    else:
        logging.info("Using Mistral loading")
        tokenizer = MistralTokenizer.from_hf_hub(model_id)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb_config
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

def save_predictions(predictions, output_dir):
    """Save reconstructed texts to JSONL file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"reconstructed_test_olmo_{timestamp}.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    logging.info(f"Reconstructed texts saved to {output_file}")
    return output_file

def generate_reconstruction(model, tokenizer, text, max_new_tokens=16384):
    """Reconstruct implicitness"""
    prompt = build_prompt(text)
    
    messages = [{"role": "user", "content": prompt}]

    # For different tokenizer types
    if hasattr(tokenizer, "encode_chat_completion"):
        # MistralTokenizer from mistral-common
        chat_request = ChatCompletionRequest(messages=messages)
        tokenized = tokenizer.encode_chat_completion(chat_request)
        input_ids = torch.tensor([tokenized.tokens], device=model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        generated_tokens = outputs[0][len(tokenized.tokens):]
        reconstruction = tokenizer.decode(generated_tokens).strip()

    else:
        # Tokenizer for Olmo
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        
        if isinstance(inputs, dict) or hasattr(inputs, "keys"):
            inputs = inputs.to(model.device)
            input_length = inputs["input_ids"].shape[1]
            generate_kwargs = inputs
        else:
            inputs = inputs.to(model.device)
            input_length = inputs.shape[1]
            generate_kwargs = {"input_ids": inputs}
            
        with torch.no_grad():
            outputs = model.generate(
                **generate_kwargs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        generated_tokens = outputs[0][input_length:]
        reconstruction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Check stop reason
    stop_reason = "stop"
    if len(generated_tokens) >= max_new_tokens:
         stop_reason = "length"
         # Double check if the last token is actually EOS (rare edge case where it fits exactly)
         if hasattr(model.config, "eos_token_id") and model.config.eos_token_id is not None:
             eos_ids = model.config.eos_token_id
             if not isinstance(eos_ids, list):
                 eos_ids = [eos_ids]
             
             if generated_tokens[-1].item() in eos_ids:
                 stop_reason = "stop"
    
    # Log generation details similar to ann_full_gen_gemini.py
    logging.info(f"Generation finished. Reason: {stop_reason}")
    if stop_reason == "length":
        logging.warning("WARNING: Generation was truncated due to token limit!")

    return reconstruction

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logging.info("="*50)
    logging.info("Starting text reconstruction process")
    logging.info(f"Transformers version: {transformers.__version__}")
    logging.info(f"BitsAndBytes version: {bnb.__version__}")
    logging.info(f"Accelerate version: {accelerate.__version__}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logging.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
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
    
    # Limit if specified
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
    for idx, item in enumerate(tqdm(data, desc="Reconstructing texts")):
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
        
        # Log progress
        if (idx + 1) % 10 == 0:
            logging.info(f"Processed {idx + 1}/{len(data)} examples")
    
    # Save results
    output_file = save_predictions(predictions, args.output_dir)
    
    logging.info("="*50)
    logging.info(f"Reconstruction complete! Processed {len(predictions)} examples")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*50)

if __name__ == "__main__":
    main()