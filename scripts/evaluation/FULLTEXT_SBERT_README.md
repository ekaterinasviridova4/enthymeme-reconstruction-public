# Full-Text SBERT Similarity Evaluation

This script evaluates the semantic similarity between reconstructed texts in a gold standard dataset and an LLM-generated dataset using Sentence-BERT (SBERT).

## Requirements

```bash
pip install -r scripts/evaluation/requirements_fulltext_sbert.txt
```

## Usage

Run the script from the project root directory:

```bash
python3 scripts/evaluation/evaluate_fulltext_sbert.py \
  --gold_file <path_to_gold_jsonl> \
  --gen_file <path_to_generated_jsonl> \
  --output_file <path_to_output_json> \
  [--model_name <sbert_model_name>]
```

### Arguments

- `--gold_file`: Path to the human-annotated gold data (JSONL format). Each line must contain a `reconstructed_text` field.
- `--gen_file`: Path to the LLM-generated data (JSONL format). Each line must contain a `reconstructed_text` field.
- `--output_file`: (Optional) Path where the detailed results JSON will be saved. Defaults to `results/evaluation_results/results_fulltext_sbert.json.json`.
- `--model_name`: (Optional) The SBERT model to use. Defaults to `sentence-transformers/all-MiniLM-L6-v2`.

### Example

To compare the Gemini 2.5 generated results against the test set:

```bash
python3 scripts/evaluation/evaluate_fulltext_sbert.py \
  --gold_file data/dialogue/out_dial_jsonl/post_processed_test.jsonl \
  --gen_file results/post_processed_litellm/post_processed_test_gemini2.5.jsonl \
  --output_file results/evaluation_results/fulltext/results_fulltext_gemini2.5_test_sbert.json
```

## Output

The script prints the **Average Semantic Similarity** score to the console.

It also produces a JSON file containing:
- Metadata matches (files used, model name).
- The calculated average similarity.
- A `details` list with the score and text pair for every example evaluated.
