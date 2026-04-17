import argparse
import json
import sacrebleu
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Evaluate BLEU score between full reconstructed texts.")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to the human annotated gold data (JSONL).")
    parser.add_argument("--gen_file", type=str, required=True, help="Path to the LLM generated data (JSONL).")
    parser.add_argument("--output_file", type=str, default="evaluation_results_fulltext_bleu.json", help="Path to save the detailed results.")
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
    
    # Calculate corpus BLEU (overall metric)
    refs = [gold_texts] 
    hyps = gen_texts

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    print(f"\nCorpus BLEU Score: {bleu.score:.2f}")

    # Calculate sentence-level BLEU
    results = []
    print("Computing sentence-level BLEU scores...")
    for i in tqdm(range(len(gold_texts))):
        gold_text = gold_texts[i]
        gen_text = gen_texts[i]
        
        sentence_score = sacrebleu.sentence_bleu(gen_text, [gold_text])
        
        results.append({
            "index": i,
            "gold_text": gold_text,
            "gen_text": gen_text,
            "bleu_score": sentence_score.score
        })

    # Save details
    output_data = {
        "gold_file": args.gold_file,
        "gen_file": args.gen_file,
        "corpus_bleu": bleu.score,
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
