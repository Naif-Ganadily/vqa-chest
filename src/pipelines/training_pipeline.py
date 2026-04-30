import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb

from src.pipelines.feature_eng_pipeline import (
    build_text_encoder,
    build_img_transform,
    encode_question
)
from src.models.model import ImageTextBinaryModel
from src.data import VQARADBinaryDataset
from src.utils import set_seed

def run(config: dict):
    # Set the Seed for reproducibility
    set_seed(config["seed"])
    
    # Device setup
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Feature Engineerng setup
    tokenizer, text_model = build_text_encoder(config)
    img_transform = build_img_transform(config)

    # Data
    ds = load_dataset(config["dataset"])
    label_map = {"yes": 1, "no": 0}

    def is_yes_no(example):
        return example["answer"].lower() in label_map
    
    def add_label(example):
        return {"label": label_map[example["answer"].lower()]}
    
    train_yn = ds["train"].filter(is_yes_no).map(add_label)
    test_yn = ds["test"].filter(is_yes_no).map(add_label)

    train_dataset = VQARADBinaryDataset(train_yn, tokenizer, text_model, img_transform, config)
    test_dataset = VQARADBinaryDataset(test_yn, tokenizer, text_model, img_transform, config)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model, Loss, Optimizer
    model = ImageTextBinaryModel(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # WandB setup
    wandb.init(project=config["wandb_project"], name=config["run_name"], config=config)

    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = correct = total = 0

        for images, text_embs, labels in train_loader:
            images, text_embs, labels = (
                images.to(device),
                text_embs.to(device),
                labels.float().to(device)
            )
            optimizer.zero_grad()
            logits = model(images, text_embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.cpu() == labels.cpu().long()).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total

        wandb.log({"epoch": epoch, "train_loss": epoch_loss, "train_accuracy": epoch_acc})
        print(f"Epoch {epoch:02d}/{config['epochs']} loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

    wandb.finish()