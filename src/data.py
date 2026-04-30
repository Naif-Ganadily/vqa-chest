import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.pipelines.feature_eng_pipeline import encode_question


class VQARADBinaryDataset(Dataset):

    def __init__(self, hf_split, tokenizer, text_model, img_transform, config: dict):
        self.data = hf_split
        self.img_transform = img_transform
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.config = config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return (
            self.img_transform(ex["image"].convert("RGB")), # [3, 224, 224]
            encode_question(ex["question"], self.tokenizer, self.text_model, self.config), # [768]
            ex["label"] # 0 or 1
        )