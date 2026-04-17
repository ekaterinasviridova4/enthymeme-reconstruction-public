
import json
import argparse
from sacrebleu import sacrebleu  # Using sacrebleu library
import os
from tqdm import tqdm

try:
    import sacrebleu
except ImportError:
    print("Sacrebleu not installed. Please install it using `pip install sacrebleu`")
    exit(1)

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_all_reconstructed_sentences(associations):
    """
    Extracts all reconstructed sentences (IMPLICIT_*) regardless of associated_id.
    Returns a single string of all sentences concatenated, or a list.
    """
    sentences = []
    for item in associations:
        tag = item.get('tag', '')
        if tag.startswith('IMPLICIT_'):
            text = item.get('text', '').strip()
            if text:
                sentences.append(text)
    
    # We sort the sentences to minimize the impact of ordering differences 
    sentences.sort()
    
    return sentences

def calculate_corups_bleu_bag(gold_file, gen_file, output_file):
    print(f"Loading gold data from {gold_file}...")
    gold_data = load_data(gold_file)
    
    print(f"Loading generated data from {gen_file}...")
    gen_data = load_data(gen_file)
    
    if len(gold_data) != len(gen_data):
        print(f"Warning: Number of lines in gold ({len(gold_data)}) and generated ({len(gen_data)}) files do not match.")
    
    all_best_bleu_scores = []
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
        
        if not gold_sentences and not gen_sentences:
            all_best_bleu_scores.append(100.0)
            continue
            
        if not gold_sentences or not gen_sentences:
            all_best_bleu_scores.append(0.0)
            continue
        
        dialogue_bleu_sum = 0
        
        for gen_sent in gen_sentences:
            
            best_score_for_sent = 0
            for gold_sent in gold_sentences:
                score = sacrebleu.sentence_bleu(gen_sent, [gold_sent], smooth_method='exp').score
                if score > best_score_for_sent:
                    best_score_for_sent = score
            
            dialogue_bleu_sum += best_score_for_sent
            
        # Average BLEU for this dialogue (Precision perspective)
        avg_dialogue_bleu = dialogue_bleu_sum / len(gen_sentences) if gen_sentences else 0
        all_best_bleu_scores.append(avg_dialogue_bleu)
        
        details.append({
            "dialogue_index": i,
            "avg_sentence_bleu": avg_dialogue_bleu,
            "gold_count": len(gold_sentences),
            "gen_count": len(gen_sentences)
        })

    # Average over corpus
    final_avg_bleu = sum(all_best_bleu_scores) / len(all_best_bleu_scores) if all_best_bleu_scores else 0
    
    results = {
        "average_best_match_bleu": final_avg_bleu,
        "details_subset": details[:20]
    }
    
    print(f"Average Best-Match Sentence BLEU: {final_avg_bleu:.2f}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BLEU score of reconstructed premises/claims (Bag of Sentences approach).")
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
        
    calculate_corups_bleu_bag(args.gold_file, args.gen_file, args.output_file)
