import torch
import torch.nn as nn


class SimpleFusionHead(nn.Module):
    def __init__(self, img_dim=512, text_dim=768, hidden_dim=512, vocab_size=30522):
        super().__init__()
        self.fc1 = nn.Linear(img_dim + text_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_feats, text_feats):
        img_feats = img_feats.mean(dim=1)
        text_feats = text_feats[:, 0, :]
        fused = torch.cat([img_feats, text_feats], dim=-1)
        x = self.fc1(fused)
        x = self.relu(x)
        x = self.fc2(x)
        return x
