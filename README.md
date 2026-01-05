# Project Overview

This project focuses on reconstructing implicit premises and claims in argumentative texts, using both standard generative models and Chain-of-Thought (CoT) prompting. The dataset consists of argumentative dialogues, processed in various formats to test different reconstruction techniques.

## 📂 Data Structure

The data is organized into two main categories in the `data/` directory: **Non-dialogue** (linear text) and **Dialogue-structured** (speaker separated).

### Non-Dialogue Data (`data/linear/`)
Original data formatted for LLMs without explicit dialogue separation in the output structure.
- **`out_jsonl/`**: Standard dataset with Implicit/Explicit annotations.
- **`out_fine_grained_jsonl/`**: Dataset with fine-grained annotations.
- **`out_premise_claim_jsonl/`**: Dataset with premise and claim annotations.

### Dialogue Data (`data/dialogue/`)
Data preserving the dialogue structure with `speaker1` and `speaker2` labels.
- **`out_dial_jsonl/`**: Standard dialogue dataset with Implicit/Explicit annotations.
- **`out_dial_fine_grained_jsonl/`**: Dialogue dataset with fine-grained annotations.
- **`out_dial_premise_claim_jsonl/`**: Dialogue dataset with premise and claim annotations.

### Processed Data (`data/processed/`)
Intermediate data generated for specific experiments.
- **`speaker_exchanges_dev.jsonl`**: Extracted speaker exchanges used for CoT experiments (dev set).

---

## 🤖 Generation & Reconstruction

### Standard Generation (Dialogue)
Scripts located in `scripts/generation/` used to reconstruct implicit information.
- **`full_gen_gemini.py`**: Generation script using Gemini/GPT models.
- **`full_gen_mistral24.py`**: Generation script using Mistral models.

**Results:**
- **`results/reconstructed_litellm/`**: Output files from Gemini/GPT generations.
- **`results/reconstructed_mistral/`**: Output files from Mistral generations.

### Chain of Thought (CoT) Experiments
Experiments using Chain-of-Thought prompting on shorter dialogue segments.
- **`scripts/processing/extract_speaker_exchanges.py`**: Script to extract specific speaker exchanges from the dialogues.
- **`scripts/generation/cot_full_gen_gemini.py`**: Script to perform CoT generation on the extracted exchanges.
- **`results/cot_reconstructed_litellm/`**: Output files from CoT experiments.

---

## 🛠️ Data Processing Tools

### Dialogue Structure Reconstruction
Tools located in `scripts/processing/` for mapping dialogue labels back onto annotated outputs.
- **`add_speaker_labels.py`**: A script that automatically adds `speaker1` and `speaker2` labels to the annotated `output` field of the data.
  - **Usage**: Applied to `out_dial_jsonl` to create `_labeled.jsonl` files (e.g., `dev_labeled.jsonl`), enabling LLMs to process the output as a structured dialogue with implicit/explicit tags.

---

## ⚙️ Configuration

Configuration files for job submission (OAR scheduler) are located in `configs/`.
- **`oar_config.sh`**: Config for Mistral generation.
- **`oar_config_litellm.sh`**: Config for LiteLLM (Gemini/GPT) generation.

