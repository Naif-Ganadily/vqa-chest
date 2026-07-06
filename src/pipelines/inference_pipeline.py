"""Inference pipeline.

Loads a trained checkpoint and produces predictions in one of two modes:

- **batch**  : run over a dataset split (default ``test``) and write a
               predictions CSV to ``config["predictions_dir"]``.
- **single** : answer one ``(image, question)`` pair.

Both modes reuse the exact feature-engineering used in training
(``build_text_encoder`` / ``build_img_transform`` / ``encode_question``) so
train-time and inference-time preprocessing can never drift apart.
"""

import os

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import VQARADBinaryDataset
from src.models.model import ImageTextBinaryModel
from src.pipelines.feature_eng_pipeline import (
    build_img_transform,
    build_text_encoder,
    encode_question,
)

LABEL_MAP = {"yes": 1, "no": 0}


def _resolve_device(config: dict) -> torch.device:
    return torch.device(config["device"] if torch.cuda.is_available() else "cpu")


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> ImageTextBinaryModel:
    """Rebuild the model architecture and load trained weights."""
    model = ImageTextBinaryModel(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_single(
    config: dict,
    checkpoint_path: str,
    image_path: str,
    question: str,
    device: torch.device | None = None,
) -> dict:
    """Predict a yes/no answer for a single image + question."""
    device = device or _resolve_device(config)
    tokenizer, text_model = build_text_encoder(config)
    img_transform = build_img_transform(config)
    model = load_model(config, checkpoint_path, device)

    image = Image.open(image_path).convert("RGB")
    img_tensor = img_transform(image).unsqueeze(0).to(device)
    text_emb = encode_question(question, tokenizer, text_model, config).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor, text_emb)).item()

    answer = "yes" if prob >= 0.5 else "no"
    print(f"Q: {question}\nAnswer: {answer} (p={prob:.4f})")
    return {"question": question, "answer": answer, "prob": prob}


def run_batch(
    config: dict,
    checkpoint_path: str,
    split: str = "test",
    device: torch.device | None = None,
) -> str:
    """Run predictions over a dataset split and write a CSV. Returns the CSV path."""
    device = device or _resolve_device(config)
    tokenizer, text_model = build_text_encoder(config)
    img_transform = build_img_transform(config)

    ds = load_dataset(config["dataset"])
    subset = (
        ds[split]
        .filter(lambda e: e["answer"].lower() in LABEL_MAP)
        .map(lambda e: {"label": LABEL_MAP[e["answer"].lower()]})
    )
    dataset = VQARADBinaryDataset(subset, tokenizer, text_model, img_transform, config)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    model = load_model(config, checkpoint_path, device)

    rows = []
    idx = 0
    with torch.no_grad():
        for images, text_embs, labels in tqdm(loader, desc="Predicting", leave=True):
            images = images.to(device)
            text_embs = text_embs.to(device)
            probs = torch.sigmoid(model(images, text_embs)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            for j in range(len(preds)):
                rows.append(
                    {
                        "question": subset[idx]["question"],
                        "true_label": int(labels[j]),
                        "pred_prob": float(probs[j]),
                        "pred_label": int(preds[j]),
                    }
                )
                idx += 1

    predictions_dir = config.get("predictions_dir", "data/04-predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    run_name = config.get("run_name", "model")
    out_path = os.path.join(predictions_dir, f"{run_name}_{split}_predictions.csv")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    accuracy = (df["true_label"] == df["pred_label"]).mean() if len(df) else float("nan")
    print(f"Wrote {len(df)} predictions to {out_path} (accuracy={accuracy:.4f})")
    return out_path


def run(
    config: dict,
    checkpoint_path: str,
    image: str | None = None,
    question: str | None = None,
    split: str = "test",
):
    """Dispatch to single-item or batch prediction based on provided args."""
    device = _resolve_device(config)
    if image and question:
        return predict_single(config, checkpoint_path, image, question, device)
    return run_batch(config, checkpoint_path, split=split, device=device)
