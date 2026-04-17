
import json
import argparse
from rouge_score import rouge_scorer
import os
from tqdm import tqdm
import numpy as np

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_rouge_bag(gold_file, gen_file, output_file):
    print(f"Loading gold data from {gold_file}...")
    gold_data = load_data(gold_file)
    
    print(f"Loading generated data from {gen_file}...")
    gen_data = load_data(gen_file)
    
    if len(gold_data) != len(gen_data):
        print(f"Warning: Number of lines in gold ({len(gold_data)}) and generated ({len(gen_data)}) files do not match.")
    
    # Initialize ROUGE scorer
    # 'rouge1': unigram overlap
    # 'rouge2': bigram overlap
    # 'rougeL': longest common subsequence
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Store aggregated scores
    all_scores = {
        'rouge1': {'p': [], 'r': [], 'f': []},
        'rouge2': {'p': [], 'r': [], 'f': []},
        'rougeL': {'p': [], 'r': [], 'f': []}
    }
    
    details = []
    
    # Process each dialogue/line
    for i, (gold_entry, gen_entry) in enumerate(tqdm(zip(gold_data, gen_data), total=min(len(gold_data), len(gen_data)), desc="Evaluating")):
        
        gold_assoc = gold_entry.get('associations', [])
        gen_assoc = gen_entry.get('associations', [])
        
        gold_sentences = []
        for item in gold_assoc:
            if item.get('tag', '').startswith('IMPLICIT_'):
                text = item.get('text', '').strip()
                if text: gold_sentences.append(text)
                
        gen_sentences = []
        for item in gen_assoc:
            if item.get('tag', '').startswith('IMPLICIT_'):
                text = item.get('text', '').strip()
                if text: gen_sentences.append(text)
        
        # Handle empty cases
        if not gold_sentences and not gen_sentences:
            # Perfect match (empty vs empty) -> Score 1.0 across board
            for metric in all_scores:
                all_scores[metric]['p'].append(1.0)
                all_scores[metric]['r'].append(1.0)
                all_scores[metric]['f'].append(1.0)
            continue
            
        if not gold_sentences or not gen_sentences:
            # Mismatch (one empty) -> Score 0.0 across board
            for metric in all_scores:
                all_scores[metric]['p'].append(0.0)
                all_scores[metric]['r'].append(0.0)
                all_scores[metric]['f'].append(0.0)
            continue
        
        gold_text = "\n".join(gold_sentences)
        gen_text = "\n".join(gen_sentences)
        
        scores = scorer.score(gold_text, gen_text)
        
        for metric, score_obj in scores.items():
            all_scores[metric]['p'].append(score_obj.precision)
            all_scores[metric]['r'].append(score_obj.recall)
            all_scores[metric]['f'].append(score_obj.fmeasure)
            
        details.append({
            "dialogue_index": i,
            "rouge1_f": scores['rouge1'].fmeasure,
            "rougeL_f": scores['rougeL'].fmeasure,
            "gold_count": len(gold_sentences),
            "gen_count": len(gen_sentences)    
        })

    # Average results
    final_results = {}
    for metric, values in all_scores.items():
        final_results[metric] = {
            "precision": np.mean(values['p']),
            "recall": np.mean(values['r']),
            "fmeasure": np.mean(values['f'])
        }
    
    results_json = {
        "metrics": final_results,
        "details_subset": details[:20]
    }
    
    print("ROUGE Results:")
    for metric, res in final_results.items():
        print(f"  {metric}: F-measure: {res['fmeasure']:.4f} (P: {res['precision']:.4f}, R: {res['recall']:.4f})")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ROUGE score of reconstructed premises/claims (Bag of Sentences approach).")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to the gold standard JSONL file.")
    parser.add_argument("--gen_file", type=str, required=True, help="Path to the generated JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gold_file):
        print(f"Error: Gold file not found at {args.gold_file}")
        exit(1)
    if not os.path.exists(args.gen_file):
        print(f"Error: Generated file not found at {args.gen_file}")
        exit(1)
        
    calculate_rouge_bag(args.gold_file, args.gen_file, args.output_file)
