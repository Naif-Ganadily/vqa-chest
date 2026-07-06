import torch
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms


def build_text_encoder(config: dict):
    """Load tokenizer and frozen text model from config."""
    tokenizer = AutoTokenizer.from_pretrained(config["text_model_id"])
    text_model = AutoModel.from_pretrained(config["text_model_id"])
    text_model.eval()
    for param in text_model.parameters():
        param.requires_grad = False
    return tokenizer, text_model


def encode_question(question: str, tokenizer, text_model, config: dict) -> torch.Tensor:
    """Returns DistilBERT CLS-token embedding. Shape: [text_emb_dim]"""
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=config["max_text_len"],
    )
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[0, 0]


def build_img_transform(config: dict) -> transforms.Compose:
    """Image transform pipeline from config."""
    return transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
