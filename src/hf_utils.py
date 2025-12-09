import os
import torch
from huggingface_hub import HfApi, upload_folder


def save_model_to_hub(vision_encoder, text_encoder, fusion_head, repo_name, epoch, device):
    model_dir = f"./{repo_name}_checkpoint_{epoch}"
    os.makedirs(model_dir, exist_ok=True)

    checkpoint = {
        'vision_encoder': vision_encoder.state_dict(),
        'text_encoder': text_encoder.state_dict(),
        'fusion_head': fusion_head.state_dict(),
    }
    checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
    torch.save(checkpoint, checkpoint_path)

    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(
            f"# VQA Model Checkpoint\n\n"
            f"Checkpoint from epoch {epoch}.\n\n"
            f"Model: ViT + DistilBERT Fusion for VQA task.\n"
        )

    upload_folder(
        folder_path=model_dir,
        path_in_repo=f"checkpoint-{epoch}",
        repo_id=f"SaybaKamal/{repo_name}",
        repo_type="model",
        commit_message=f"Checkpoint at epoch {epoch}",
    )


def load_local_checkpoint(vision_encoder, text_encoder, fusion_head, repo_name, epoch, device):
    model_dir = f"./{repo_name}_checkpoint_{epoch}"
    ckpt_path = os.path.join(model_dir, "pytorch_model.bin")
    checkpoint = torch.load(ckpt_path, map_location=device)
    vision_encoder.load_state_dict(checkpoint['vision_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    fusion_head.load_state_dict(checkpoint['fusion_head'])
