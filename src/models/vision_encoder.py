import torch
import torch.nn as nn
from transformers import ViTModel


class VisionEncoder(nn.Module):
    def __init__(self, out_dim=512, backbone='google/vit-base-patch16-224-in21k'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(backbone)
        self.adapt = nn.Linear(self.vit.config.hidden_size, out_dim)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_embeddings = outputs.last_hidden_state
        return self.adapt(cls_embeddings)
