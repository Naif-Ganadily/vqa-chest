# 🩻 Medical VQA — Chest Baseline

A multimodal Visual Question Answering system for chest X-rays. Given an image and a clinical question, the model predicts a binary yes/no answer.

---

## Stack

`PyTorch` · `HuggingFace Datasets` · `DistilBERT` · `Weights & Biases` · `Grad-CAM` · `scikit-learn`

---

## What Was Built

- **EDA + risk discovery** on [`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) (2,248 QA pairs) — including hash-based duplicate detection that found 202 shared image hashes across train/test (potential leakage)
- **Binary baseline**: filtered to yes/no pairs, near-balanced splits (940 train / 251 test)
- **Multimodal model**: CNN image branch + frozen DistilBERT embeddings (768-dim), fused via concatenation → MLP head
- **Full eval suite**: accuracy, AUC-ROC, PR curves, confusion matrix, confidence distribution, error slicing, Grad-CAM
- **W&B tracking** integrated across training and eval runs

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

## Next Phase

- Migrate notebooks → modular `src/pipelines` with formal tests
- Config-driven, reproducible ablation pipelines
- Image-disjoint split strategy
- Stronger multimodal architectures + calibration experiments
