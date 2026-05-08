# ELARA

**Cognition Research Classifier**  
San Francisco, CA · ML / AI · 2026

A multi-label NLP classifier trained on declassified CIA and DARPA research 
documents on human consciousness. Maps each document across four cognitive 
layers — memory, belief, emotion, behavior — revealing the latent causal 
architecture of how identity states form, break, and recover under pressure.

Built with TF-IDF, transformer embeddings (SPECTER), and fine-tuned SciBERT.
Output is not a topic label. It is a position in identity space.

---

## What it classifies

Each document is mapped to one label per layer:

| Layer | What it captures |
|---|---|
| `memory` | How experience gets encoded and consolidated |
| `belief` | How the mind constructs and updates its model of reality |
| `emotion` | The affective states bridging belief into bodily readiness |
| `behavior` | Observable performance, action patterns, and their limits |

---

## Label taxonomy

**Memory**
- Encoding Under Stress
- Traumatic Consolidation
- Identity Anchoring

**Belief**
- Cognitive Bias Formation
- Self-Concept Updating
- Threat Appraisal
- Meaning Construction

**Emotion**
- Stress & Arousal Regulation
- Fear & Threat Response
- Flow & Absorption
- Emotional Dysregulation

**Behavior**
- Decision Under Duress
- Habit & Automaticity
- Peak Performance Ceiling
- Behavioral Breakdown

---

## Pipeline

01_problem/        → task definition, taxonomy, success metrics
02_data/           → raw declassified docs, annotation schema, labeled corpus
03_features/       → preprocessing, TF-IDF, SPECTER embeddings, UMAP clustering
04_training/       → 60/20/20 split, model experiments, SciBERT fine-tune
05_evaluation/     → per-class F1, confusion matrix, error analysis
06_deployment/     → FastAPI inference endpoint, drift monitoring, review queue

---

## Data sources

Declassified documents sourced from:
- CIA CREST (CIA Records Search Tool) — publicly available via FOIA
- DARPA technical reports — defense.gov open archive
- DTIC (Defense Technical Information Center)
- RAND Corporation open research corpus

---

## Stack

`sentence-transformers` · `scikit-learn` · `transformers` · `spaCy` · 
`FastAPI` · `UMAP` · `Weights & Biases` · `Google Colab` · `HuggingFace Hub`
