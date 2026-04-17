# Associations BLEU Evaluation (Bag of Sentences)

This script calculates the BLEU score for reconstructed `IMPLICIT_` premises and claims using a "Bag of Sentences" approach.

## Logic
1. Reads both Gold and Generated JSONL files.
2. For each dialogue, it extracts all `IMPLICIT_` sentences from the `associations` field.
3. **Sorting**: It sorts the extracted sentences by the best match.
4. **Corpus BLEU**: It calculates the standard Corpus BLEU score (using `sacrebleu`) comparing the generated dialogue strings against the gold dialogue strings.

## Usage

1. **Install Requirements**:
   ```bash
   pip install -r scripts/evaluation/requirements_fulltext_bleu.txt
   ```

2. **Run the Script**:
   ```bash
   python scripts/evaluation/evaluate_associations_bag_bleu.py \
     --gold_file data/dialogue/out_dial_jsonl/post_processed_test.jsonl \
     --gen_file results/post_processed_litellm/post_processed_test_gemini2.5.jsonl \
     --output_file results/evaluation_results/associations/results_associations_bag_gemini2.5_bleu.json
   ```