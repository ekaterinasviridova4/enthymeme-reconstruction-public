# Overview

This repository contains the implementation details of the paper "Better than the Gold Standard? Evaluating Human vs. LLM Enthymeme Reconstruction in Natural Language Argumentation".
This work focuses on reconstructing implicit premises and claims in argumentative texts, using different families of generative models and zero-shot prompting. 
While the full reconstruction pipeline is operational and can be tested on the provided CMV datasets, the human-reconstructed gold-standard data (used for evaluation) will be published upon paper acceptance.

## Data Structure

The data is organized into **Dialogue-structured** (speaker separated) texts including raw (non-annotated) and human-annotated CMV text: 

- **`out_dial_jsonl/`**: Dialogue dataset with Implicit/Explicit annotations.
- **`out_dial_fine_grained_jsonl/`**: Dialogue dataset with fine-grained annotations.
- **`out_dial_premise_claim_jsonl/`**: Dialogue dataset with premise and claim annotations.

---

## Reconstruction & Evaluation

Scripts located in `scripts/generation/` used to reconstruct implicit information.
- **`ann_full_gen_gemini.py`**: Generation script using Gemini/GPT models.
- **`ann_full_gen_mistral24.py`**: Generation script using Mistral/OLMo models.

Scripts located in `scripts/evaluation/` used to evaluate automatic reconstruction against human gold-standard reconstruction (available upon acceptance).
- Fulltext scripts: evaluation of complete reconstructions with BLEU ROUGE and SBERT.
- Associations scripts: evaluation of reconstruction pairs with BLEU ROUGE and SBERT.

---

## Data Processing Tools

### Files for breaking down and re-joining dialogues (speaker exchanges)
Tools located in `scripts/processing/` for splitting dialogues into speaker exchanges and joining back the reconstructions.

---

## Configuration

Configuration files for job submission (OAR scheduler) are located in `configs/`.
- **`oar_config.sh`**: Config for Mistral/OLMo generation.
- **`oar_config_litellm.sh`**: Config for LiteLLM (Gemini/GPT) generation.

---
## Evaluation Guidelines

Evaluation guidelines file contains complete guidelines for human evaluation study.

---

## Installation
```bash
# General dependencies
pip install -r requirements.txt

# For API-based reconstruction (LiteLLM)
pip install -r requirements_litellm.txt

