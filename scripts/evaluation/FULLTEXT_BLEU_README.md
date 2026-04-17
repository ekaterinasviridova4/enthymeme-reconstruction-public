# Full-Text BLEU Score Evaluation

This script evaluates the BLEU score between reconstructed texts in a gold standard dataset and an LLM-generated dataset using `sacrebleu`.

## Requirements

```bash
pip install -r scripts/evaluation/requirements_fulltext_bleu.txt
```

## Usage

Run the script from the project root directory:

```bash
python3 scripts/evaluation/evaluate_fulltext_bleu.py \
  --gold_file <path_to_gold_jsonl> \
  --gen_file <path_to_generated_jsonl> \
  --output_file <path_to_output_json>
```

### Arguments

- `--gold_file`: Path to the human-annotated gold data (JSONL format). Each line must contain a `reconstructed_text` field.
- `--gen_file`: Path to the LLM-generated data (JSONL format). Each line must contain a `reconstructed_text` field.
- `--output_file`: (Optional) Path where the detailed results JSON will be saved. Defaults to `evaluation_results_fulltext_bleu.json`.

### Example

To compare the Gemini 2.5 generated results against the test set:

```bash
python scripts/evaluation/evaluate_fulltext_bleu.py \
  --gold_file data/dialogue/out_dial_jsonl/post_processed_test.jsonl \
  --gen_file results/post_processed_litellm/post_processed_test_gemini2.5.jsonl \
  --output_file results/evaluation_results/fulltext/results_fulltext_gemini2.5_test_bleu.json
```

## Output

The script prints the **Corpus BLEU Score** to the console.

It also produces a JSON file containing:
- Metadata about files used.
- The overall Corpus BLEU score.
- A `details` list with the per-sentence BLEU score and text pair for every example evaluated.
