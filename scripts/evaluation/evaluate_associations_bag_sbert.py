
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_all_reconstructed_sentences(associations):
    """
    Extracts all reconstructed sentences (IMPLICIT_*) regardless of associated_id.
    Returns a list of strings.
    """
    sentences = []
    for item in associations:
        tag = item.get('tag', '')
        if tag.startswith('IMPLICIT_'):
            text = item.get('text', '').strip()
            if text:
                sentences.append(text)
    return sentences

def calculate_similarity_bag_of_words(gold_file, gen_file, output_file, model_name='all-MiniLM-L6-v2'):
    print(f"Loading SBERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Loading gold data from {gold_file}...")
    gold_data = load_data(gold_file)
    
    print(f"Loading generated data from {gen_file}...")
    gen_data = load_data(gen_file)
    
    if len(gold_data) != len(gen_data):
        print(f"Warning: Number of lines in gold ({len(gold_data)}) and generated ({len(gen_data)}) files do not match.")
    
    dialogue_scores = []
    details = []
    
    # Process each dialogue/line
    for i, (gold_entry, gen_entry) in enumerate(tqdm(zip(gold_data, gen_data), total=min(len(gold_data), len(gen_data)), desc="Evaluating")):
        
        gold_assoc = gold_entry.get('associations', [])
        gen_assoc = gen_entry.get('associations', [])
        
        gold_sentences = extract_all_reconstructed_sentences(gold_assoc)
        gen_sentences = extract_all_reconstructed_sentences(gen_assoc)
        
        if not gold_sentences and not gen_sentences:
            dialogue_scores.append(1.0)
            continue

        if not gold_sentences or not gen_sentences:
            dialogue_scores.append(0.0)
            details.append({
                "dialogue_index": i,
                "score": 0.0,
                "gold_count": len(gold_sentences),
                "gen_count": len(gen_sentences),
                "note": "One side empty"
            })
            continue
            
        # Encode all sentences
        gold_embs = model.encode(gold_sentences)
        gen_embs = model.encode(gen_sentences)
        
        # Calculate similarity matrix (num_gold x num_gen)
        sim_matrix = cosine_similarity(gold_embs, gen_embs)
        
        # Greedy Matching Strategy
        
        max_sim_gold = np.max(sim_matrix, axis=1)
        recall_score = np.mean(max_sim_gold)
        
        max_sim_gen = np.max(sim_matrix, axis=0)
        precision_score = np.mean(max_sim_gen)
        
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        
        score = (recall_score + precision_score) / 2
        
        dialogue_scores.append(score)
        
        details.append({
            "dialogue_index": i,
            "score": float(score),
            "recall_sim": float(recall_score),
            "precision_sim": float(precision_score),
            "gold_count": len(gold_sentences),
            "gen_count": len(gen_sentences)
        })

    # Aggregate results
    if dialogue_scores:
        avg_score = sum(dialogue_scores) / len(dialogue_scores)
    else:
        avg_score = 0.0
    
    results = {
        "average_similarity": avg_score,
        "num_dialogues_processed": len(dialogue_scores),
        "details_subset": details[:20] 
    }
    
    print(f"Average Similarity (Bag-of-Sentences): {avg_score:.4f}")
    
    # Ensure directory 
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate semantic similarity of reconstructed premises/claims using SBERT (Bag of Sentences approach).")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to the gold standard JSONL file.")
    parser.add_argument("--gen_file", type=str, required=True, help="Path to the generated JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="SBERT model name.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gold_file):
        print(f"Error: Gold file not found at {args.gold_file}")
        exit(1)
    if not os.path.exists(args.gen_file):
        print(f"Error: Generated file not found at {args.gen_file}")
        exit(1)
        
    calculate_similarity_bag_of_words(args.gold_file, args.gen_file, args.output_file, args.model_name)
