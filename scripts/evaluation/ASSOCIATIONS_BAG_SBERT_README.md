# Associations SBERT Evaluation (Bag of Sentences)

This script evaluates the semantic similarity of reconstructed implicit premises and claims by treating them as a "bag of sentences" for each dialogue, disregarding the specific `associated_id` link. This is useful when the alignment of IDs between Gold and Generated data is inconsistent.

## Logic
1. Reads both Gold and Generated JSONL files.
2. For each dialogue, extracts **all** sentences from `associations` where the tag starts with `IMPLICIT_`.
3. Encodes all Gold sentences and all Generated sentences into embeddings.
4. Computes a similarity matrix between every Gold sentence and every Generated sentence.
5. Calculates a score using a greedy matching strategy (similar to BERTScore):
   - **Recall-oriented Similarity**: For every Gold sentence, finds the best matching Generated sentence. Takes the average.
   - **Precision-oriented Similarity**: For every Generated sentence, finds the best matching Gold sentence. Takes the average.
   - **Final Score**: The average of the Recall and Precision scores.

## Usage

1. **Install Requirements**:
   ```bash
   pip install -r scripts/evaluation/requirements_fulltext_sbert.txt
   ```

2. **Run the Script**:
   ```bash
   python scripts/evaluation/evaluate_associations_bag_sbert.py \
     --gold_file data/dialogue/out_dial_jsonl/post_processed_test.jsonl \
     --gen_file results/post_processed_litellm/post_processed_test_gemini2.5.jsonl \
     --output_file results/evaluation_results/associations/results_associations_bag_gemini2.5_sbert.json
   ```
   