# 🩻 Medical VQA — Chest Baseline

A multimodal Visual Question Answering system for chest X-rays. Given an image and a clinical question, the model predicts a binary yes/no answer.

---

## Stack

`PyTorch` · `HuggingFace Datasets` · `DistilBERT` · `MLflow` / `Weights & Biases` · `Grad-CAM` · `scikit-learn` · `uv` · `Docker`

---

## What Was Built

- **EDA + risk discovery** on [`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) (2,248 QA pairs) — including hash-based duplicate detection that found 202 shared image hashes across train/test (potential leakage)
- **Binary baseline**: filtered to yes/no pairs, near-balanced splits (940 train / 251 test)
- **Multimodal model**: CNN image branch + frozen DistilBERT embeddings (768-dim), fused via concatenation → MLP head
- **Full eval suite**: accuracy, AUC-ROC, PR curves, confusion matrix, confidence distribution, error slicing, Grad-CAM
- **Config-driven pipelines** under `src/` (feature-eng, training, evaluation, inference) with CLI entrypoints
- **Swappable experiment tracking**: switch between self-hosted **MLflow** (private) and **Weights & Biases** with one config line
- **Fast offline tests**, **GitHub Actions CI** (lint + tests), and a **Dockerfile** for reproducible runs

---

## Results

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 85.3% |
| Test accuracy | 60.16% |
| AUC-ROC | 0.6465 |
| Avg Precision | 0.6042 |

The gap between train and test accuracy reflects the image-level leakage risk — a known issue flagged during EDA, not discovered after the fact.

---

## Known Risks

- **Split leakage**: 202 shared image hashes across train/test; image-disjoint splits are the fix
- **Frozen embeddings**: CLS-only DistilBERT is a fast approximation; token-level or end-to-end finetuning will likely close the gap
- **Architecture ceiling**: CNN + DistilBERT fusion is intentionally simple

---

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) for reproducible environments.

```bash
uv sync   # install the exact locked dependencies from uv.lock
```

The dataset streams from the Hugging Face Hub on first run (downloaded once, then cached) — there is no raw data stored in the repo.

## Usage

```bash
# Train (writes a checkpoint to data/05-models/<run_name>.pt)
uv run train --config config/local.yaml

# Batch inference over the test split → predictions CSV in data/04-predictions/
uv run predict --config config/local.yaml --checkpoint data/05-models/local-run.pt

# Single image + question
uv run predict --config config/local.yaml --checkpoint data/05-models/local-run.pt \
  --image path/to/xray.png --question "Is there cardiomegaly?"

# Fast, offline tests
uv run pytest -m "not slow"
```

## Experiment tracking (swappable)

Pick a backend in the `tracker` block of your config — no code changes:

```yaml
tracker:
  backend: "mlflow"     # mlflow | wandb | none
  project: "vqa-chest"
  tracking_uri: ""      # mlflow: "" → local private SQLite (mlflow.db)
```

- **MLflow (private / on-prem)** — with an empty `tracking_uri`, runs log to a local, offline SQLite db (`mlflow.db`) with zero network calls; view them via `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. For teams, point `tracking_uri` at your own DB or a self-hosted server (e.g. `postgresql://...` or `http://mlflow.internal:5000`). Recommended for regulated / clinical data.
- **Weights & Biases (cloud)** — set `backend: wandb` and run `wandb login` first. Convenient for public datasets.

> ⚠️ Regardless of backend, never log PHI into params, metric names, or artifact filenames.

## Docker & CI

```bash
docker build -t vqa-chest .
docker run --rm vqa-chest uv run pytest -m "not slow"
```

CI (`.github/workflows/ci.yml`) runs `ruff` lint/format checks and the fast test suite on every push / PR to `main`.

---

## Next Phase

- ✅ Notebooks migrated to modular `src/pipelines` with tests, CI, and Docker
- ✅ Config-driven runs with swappable, private-capable experiment tracking
- **Image-disjoint split** to eliminate the 202-hash train/test leakage
- Unfreeze / token-level DistilBERT, stronger multimodal fusion, and calibration experiments
