import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate semantic similarity between full reconstructed texts using SBERT.")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to the human annotated gold data (JSONL).")
    parser.add_argument("--gen_file", type=str, required=True, help="Path to the LLM generated data (JSONL).")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SBERT model name.")
    parser.add_argument("--output_file", type=str, default="results_fulltext_sbert.json", help="Path to save the detailed results.")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading SBERT model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print(f"Loading gold data from: {args.gold_file}")
    gold_data = load_data(args.gold_file)
    print(f"Loaded {len(gold_data)} gold examples.")

    print(f"Loading generated data from: {args.gen_file}")
    gen_data = load_data(args.gen_file)
    print(f"Loaded {len(gen_data)} generated examples.")

    # Validate alignment
    if len(gold_data) != len(gen_data):
        print(f"Warning: Number of examples differ (Gold: {len(gold_data)}, Gen: {len(gen_data)}). Truncating to minimum.")
        min_len = min(len(gold_data), len(gen_data))
        gold_data = gold_data[:min_len]
        gen_data = gen_data[:min_len]
    
    similarities = []
    results = []

    print("Computing similarities...")
    for i in tqdm(range(len(gold_data))):
        gold_text = gold_data[i].get("reconstructed_text", "")
        gen_text = gen_data[i].get("reconstructed_text", "")

        if not gold_text or not gen_text:
            print(f"Warning: Empty text at index {i}. Skipping.")
            continue
        
        # Compute embeddings
        # Numpy array or list of numpy arrays
        embedding_gold = model.encode(gold_text, convert_to_tensor=True)
        embedding_gen = model.encode(gen_text, convert_to_tensor=True)

        # Compute cosine similarity
        score = util.pytorch_cos_sim(embedding_gold, embedding_gen).item()
        similarities.append(score)

        results.append({
            "index": i,
            "gold_text": gold_text,
            "gen_text": gen_text,
            "similarity_score": score
        })

    avg_similarity = np.mean(similarities) if similarities else 0.0
    print(f"\nAverage Semantic Similarity: {avg_similarity:.4f}")

    # Save details
    output_data = {
        "model_name": args.model_name,
        "gold_file": args.gold_file,
        "gen_file": args.gen_file,
        "average_similarity": avg_similarity,
        "details": results
    }

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
