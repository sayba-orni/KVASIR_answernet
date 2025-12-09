import torch
from transformers import DistilBertTokenizer, ViTImageProcessor
from datasets import load_dataset

from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion_head import SimpleFusionHead
from src.data.dataset import VQADataset
from src import hf_utils
from src.train import train


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("SimulaMet-HOST/kvasir-vqa", split="raw").shuffle(seed=42)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = train_test["train"]
    val_data = train_test["test"]

    train_dataset = VQADataset(train_data, tokenizer, image_processor)
    val_dataset = VQADataset(val_data, tokenizer, image_processor)

    vision_encoder = VisionEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    fusion_head = SimpleFusionHead(vocab_size=tokenizer.vocab_size).to(device)

    train(
        vision_encoder,
        text_encoder,
        fusion_head,
        train_dataset,
        val_dataset,
        tokenizer,
        device,
        num_epochs=10,
        batch_size=16,
        lr=3e-5,
        save_every=1,
        repo_name="vit-distilbert",
        hf_utils=hf_utils,
    )


if __name__ == "__main__":
    main()
