import argparse
import json
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer

def load_data(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            texts.append(data.get("reconstructed_text", ""))
    return texts

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ROUGE scores between full reconstructed texts.")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to the human annotated gold data (JSONL).")
    parser.add_argument("--gen_file", type=str, required=True, help="Path to the LLM generated data (JSONL).")
    parser.add_argument("--output_file", type=str, default="evaluation_results_fulltext_rouge.json", help="Path to save the detailed results.")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading gold data from: {args.gold_file}")
    gold_texts = load_data(args.gold_file)
    print(f"Loaded {len(gold_texts)} gold examples.")

    print(f"Loading generated data from: {args.gen_file}")
    gen_texts = load_data(args.gen_file)
    print(f"Loaded {len(gen_texts)} generated examples.")

    # Validate alignment
    if len(gold_texts) != len(gen_texts):
        print(f"Warning: Number of examples differ (Gold: {len(gold_texts)}, Gen: {len(gen_texts)}). Truncating to minimum.")
        min_len = min(len(gold_texts), len(gen_texts))
        gold_texts = gold_texts[:min_len]
        gen_texts = gen_texts[:min_len]
    
    # Initialize ROUGE scorer
    # evaluating ROUGE-1, ROUGE-2, and ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    
    # Accumulators for average calculation
    total_scores = {
        'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
        'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
        'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
    }
    
    print("Computing ROUGE scores...")
    for i in tqdm(range(len(gold_texts))):
        gold_text = gold_texts[i]
        gen_text = gen_texts[i]
        
        # Calculate scores for this pair
        scores = scorer.score(gold_text, gen_text)
        
        # Format individual results
        example_result = {
            "index": i,
            "gold_text": gold_text,
            "gen_text": gen_text,
            "scores": {
                "rouge1": scores['rouge1']._asdict(),
                "rouge2": scores['rouge2']._asdict(),
                "rougeL": scores['rougeL']._asdict()
            }
        }
        results.append(example_result)
        
        # Accumulate for average
        for metric in total_scores:
            total_scores[metric]['precision'] += scores[metric].precision
            total_scores[metric]['recall'] += scores[metric].recall
            total_scores[metric]['fmeasure'] += scores[metric].fmeasure

    # Calculate averages
    num_examples = len(gold_texts)
    avg_scores = {}
    if num_examples > 0:
        for metric in total_scores:
            avg_scores[metric] = {
                'precision': total_scores[metric]['precision'] / num_examples,
                'recall': total_scores[metric]['recall'] / num_examples,
                'fmeasure': total_scores[metric]['fmeasure'] / num_examples
            }

    # Print summary
    print(f"\nAverage ROUGE Scores (F-measure):")
    print(f"  ROUGE-1: {avg_scores['rouge1']['fmeasure']:.4f}")
    print(f"  ROUGE-2: {avg_scores['rouge2']['fmeasure']:.4f}")
    print(f"  ROUGE-L: {avg_scores['rougeL']['fmeasure']:.4f}")

    # Save details
    output_data = {
        "gold_file": args.gold_file,
        "gen_file": args.gen_file,
        "average_scores": avg_scores,
        "details": results
    }

    if args.output_file:
        import os
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
