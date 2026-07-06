import torch
import torch.nn as nn


class ImageTextBinaryModel(nn.Module):
    # 2x MaxPool2d on 224x224 input
    CNN_OUTPUT_DIM = 32 * 56 * 56

    def __init__(self, config: dict):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.CNN_OUTPUT_DIM + config["text_emb_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, images: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        img_features = self.cnn(images).flatten(1)
        fused = torch.cat([img_features, text_embs], dim=1)
        return self.classifier(fused).squeeze(1)
