"""
Fine-tune BART (Seq2Seq) for implicit premise/claim reconstruction.
"""

import os
import json
import logging
import argparse
import random
import torch
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_dataset(path, limit=None):
    """
    Load JSONL and prepare source/target pairs.
    Source: 'output' field (<Explicit>/<Implicit> tags).
    Target: 'reconstructed_text' field.
    """
    data = []
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            if "output" in item and "reconstructed_text" in item:
                source_text = item["output"]
                target_text = item["reconstructed_text"]
                
                # Instruction prefix
                input_text = "Reconstruct implicit premises and claims from the tagged dialogue: " + source_text
                
                data.append({
                    "input_text": input_text,
                    "target_text": target_text
                })
                count += 1
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, default="./results/bart_finetuned", help="Output directory")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large", help="Model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (small for small data)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--limit_train_data", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Output directory
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info(f"Loading data from {args.train_file}")
    raw_data = load_jsonl_dataset(args.train_file, args.limit_train_data)
    if not raw_data:
        logger.error("No data found. Check file format (must contain 'output' and 'reconstructed_text').")
        return
    logger.info(f"Loaded {len(raw_data)} training examples")
    
    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(raw_data)

    # Tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function_dynamic(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        # Tokenize inputs 
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True, padding="max_length")

        # Tokenize labels
        labels = tokenizer(text_target=targets, max_length=args.max_length, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]

        # LED specific: create global_attention_mask
        if "led" in args.model_name:
            batch_size = len(model_inputs["input_ids"])
            # Initialize with 0
            global_attention_mask = [
                [0 for _ in range(len(model_inputs["input_ids"][i]))] 
                for i in range(batch_size)
            ]
            
            # Set global attention on the first token (usually <s>)
            for i in range(batch_size):
                global_attention_mask[i][0] = 1
                
            model_inputs["global_attention_mask"] = global_attention_mask

        return model_inputs

    tokenized_dataset = hf_dataset.map(preprocess_function_dynamic, batched=True, remove_columns=hf_dataset.column_names)
    
    train_dataset = tokenized_dataset

    # Model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Check vocab size mismatch and resize if needed (common source of index out of bounds)
    if len(tokenizer) > model.config.vocab_size:
        logger.warning(f"Resizing token embeddings: Tokenizer len ({len(tokenizer)}) > Model vocab size ({model.config.vocab_size})")
        model.resize_token_embeddings(len(tokenizer))

    # Gradient checkpoint for memory efficiency if using LED
    if "led" in args.model_name and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for LED")
    
    # Training Args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=1,
        fp16=torch.cuda.is_available(), 
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Verify generation on one sample
    logger.info("Verifying generation on a sample input...")
    input_text = raw_data[0]["input_text"]
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    if hasattr(model.config, "model_type") and model.config.model_type == "led":
        inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
        inputs["global_attention_mask"][:, 0] = 1

    outputs = model.generate(**inputs, max_length=args.max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Input: {input_text[:100]}...")
    logger.info(f"Generated: {generated_text[:200]}...")

if __name__ == "__main__":
    main()
