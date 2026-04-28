import yaml
import torch
import src.pipelines.feature_eng_pipeline import binary_dataloaders

with open("config/local.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda")
train_loader, test_loader, tokenizer, text_model = binary_dataloaders(cfg, device)