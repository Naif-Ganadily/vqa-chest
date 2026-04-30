from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def binary_dataloaders(cfg, tokenizer, text_model, device):
    "Streamlining the binary dataloaders between train test split"
    ds = load_dataset(cfg["dataset"])
    train_yn = ds["train"].filter(lambda x:x["answer"].lower() in ["yes", "no"])
    test_yn  = ds["test"].filter(lambda x: x["answer"].lower() in ["yes","no"])