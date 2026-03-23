
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class DistributedTransformerTrainer:
    def __init__(self, rank, world_size, model_name="bert-base-uncased", num_labels=2):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.num_labels = num_labels
        self._setup_ddp()
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _setup_ddp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config).to(self.rank)
        return DDP(model, device_ids=[self.rank])

    def train(self, dataloader, optimizer, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.rank)
                attention_mask = batch["attention_mask"].to(self.rank)
                labels = batch["labels"].to(self.rank)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if self.rank == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}")

    def cleanup(self):
        destroy_process_group()

if __name__ == "__main__":
    # This is a simplified example. In a real scenario, you'd launch this with torch.distributed.launch
    # For demonstration, we'll just show the class definition.
    print("Distributed Transformer Trainer class defined. Ready for multi-GPU training.")
