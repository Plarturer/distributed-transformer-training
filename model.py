import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class ScalableTransformer(nn.Module):
    def __init__(self, config_name='bert-base-uncased'):
        super(ScalableTransformer, self).__init__()
        self.config = BertConfig.from_pretrained(config_name)
        self.transformer = BertModel(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.classifier(pooled_output)

print("Model architecture defined for distributed environments.")
