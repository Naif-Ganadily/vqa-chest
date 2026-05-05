import torch
import wandb
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

def run(model, test_loader, config: dict, device) -> dict:
    """
    Evaluate the model on test set and logs all metrics to W&B
    Returns dict of metrics   
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating", leave=True)
        for images, text_embs, labels in loop:
            images = images.to(device)
            text_embs = text_embs.to(device)

            probs = torch.sigmoid(model(images, text_embs)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            # Showing a live AUC in the progress bar as batches are processed
            if len(all_labels) > 1:
                loop.set_postfix(batches_done=len(all_labels))

    # Metrics
    auc_roc = roc_auc_score(all_labels, all_probs)
    avg_prec = average_precision_score(all_labels, all_probs)
    accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)

    print(classification_report(all_labels, all_preds, target_names=["No", "Yes"]))
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Log metrics to W&B
    wandb.log({
        "test_accuracy": accuracy,
        "test_auc_roc": auc_roc,
        "test_avg_precision": avg_prec,
    })

    # Lets also log the confusion matrics to W&B
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    # Log ROC curve to W&B
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(all_labels, all_probs, ax=ax)
    wandb.log({"roc_curve": wandb.Image(fig)})
    plt.close(fig)

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "avg_precision": avg_prec
    }