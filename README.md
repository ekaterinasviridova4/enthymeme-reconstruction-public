# Project Overview

This project focuses on reconstructing implicit premises and claims in argumentative texts, using both standard generative models and Chain-of-Thought (CoT) prompting. The dataset consists of argumentative dialogues, processed in various formats to test different reconstruction techniques.

## 📂 Data Structure

The data is organized into two main categories: **Non-dialogue** (linear text) and **Dialogue-structured** (speaker separated).

### Non-Dialogue Data
Original data formatted for LLMs without explicit dialogue separation in the output structure.
- **`out_jsonl/`**: Standard dataset with Implicit/Explicit annotations.
- **`out_fine_grained_jsonl/`**: Dataset with fine-grained annotations.
- **`out_premise_claim_jsonl/`**: Dataset with premise and claim annotations.

### Dialogue Data
Data preserving the dialogue structure with `speaker1` and `speaker2` labels.
- **`out_dial_jsonl/`**: Standard dialogue dataset with Implicit/Explicit annotations.
- **`out_dial_fine_grained_jsonl/`**: Dialogue dataset with fine-grained annotations.
- **`out_dial_premise_claim_jsonl/`**: Dialogue dataset with premise and claim annotations.

---

## 🤖 Generation & Reconstruction

### Standard Generation (Non-Dialogue)
Scripts used to reconstruct implicit information on the non-dialogue data.
- **`full_gen_gemini.py`**: Generation script using Gemini/GPT models.
- **`full_gen_mistral24.py`**: Generation script using Mistral models.

**Results:**
- **`reconstructed_litellm/`**: Output files from Gemini/GPT generations.
- **`reconstructed_mistral/`**: Output files from Mistral generations.

### Chain of Thought (CoT) Experiments
Experiments using Chain-of-Thought prompting on shorter dialogue segments.
- **`extract_speaker_exchanges.py`**: Script to extract specific speaker exchanges from the dialogues to create shorter contexts.
- **`speaker_exchanges_dev.jsonl`**: The extracted data used for CoT experiments (currently generated for the **dev set** only).
- **`cot_full_gen_gemini.py`**: Script to perform CoT generation on the extracted exchanges.

---

## 🛠️ Data Processing Tools

### Dialogue Structure Reconstruction
Tools for mapping dialogue labels back onto annotated outputs.
- **`add_speaker_labels.py`**: A script that automatically adds `speaker1` and `speaker2` labels to the annotated `output` field of the data.
  - **Usage**: Applied to `out_dial_jsonl` to create `_labeled.jsonl` files (e.g., `dev_labeled.jsonl`), enabling LLMs to process the output as a structured dialogue with implicit/explicit tags.

---

## ⚙️ Configuration

- **`oar_config.sh`** & **`oar_config_litellm.sh`**: Configuration files for job submission (OAR scheduler) to run the generation scripts.
