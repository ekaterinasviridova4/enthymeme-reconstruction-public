# Associations ROUGE Evaluation (Bag of Sentences)

This script calculates the ROUGE score (ROUGE-1, ROUGE-2, ROUGE-L) for reconstructed premises/claims using a concatenated "Bag of Sentences" approach.

## Logic
1. Reads both Gold and Generated JSONL files.
2. For each dialogue, it extracts all `IMPLICIT_` sentences.
3. **Concatenation**: It joins all extracted sentences into a single text block separated by newlines (`\n`).
   - This treats the collection of implicit premises as a "multi-sentence summary".
   - ROUGE-1 (unigram overlap) is largely order-independent, making it excellent for "bag of content" evaluation.
   - ROUGE-L measures the longest common subsequence, handling sentence boundaries appropriately when newlines are used.
4. Computes Precision, Recall, and F-measure for each dialogue and averages them.

## Usage

1. **Install Requirements**:
   ```bash
   pip install -r scripts/evaluation/requirements_fulltext_rouge.txt
   ```

2. **Run the Script**:
   ```bash
   python scripts/evaluation/evaluate_associations_bag_rouge.py \
     --gold_file data/dialogue/out_dial_jsonl/post_processed_test.jsonl \
     --gen_file results/post_processed_litellm/post_processed_test_gemini2.5.jsonl \
     --output_file results/evaluation_results/results_associations_bag_gemini2.5_rouge.json
   ```