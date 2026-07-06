"""Fast, offline tests for the model. No dataset or network needed."""

import torch

from src.models.model import ImageTextBinaryModel


def test_forward_output_shape():
    config = {"text_emb_dim": 768}
    model = ImageTextBinaryModel(config)
    batch = 4
    images = torch.randn(batch, 3, 224, 224)
    text_embs = torch.randn(batch, config["text_emb_dim"])

    out = model(images, text_embs)

    assert out.shape == (batch,)


def test_forward_is_deterministic_in_eval():
    config = {"text_emb_dim": 768}
    model = ImageTextBinaryModel(config).eval()
    images = torch.randn(2, 3, 224, 224)
    text_embs = torch.randn(2, config["text_emb_dim"])

    with torch.no_grad():
        first = model(images, text_embs)
        second = model(images, text_embs)

    assert torch.allclose(first, second)
