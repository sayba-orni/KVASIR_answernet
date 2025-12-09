import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='distilbert-base-uncased'):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
