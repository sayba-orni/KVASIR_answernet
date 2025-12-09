import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
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
    hf_utils=None,
    metrics=None,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        list(vision_encoder.parameters())
        + list(text_encoder.parameters())
        + list(fusion_head.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train_losses = []

    for epoch in range(num_epochs):
        vision_encoder.train()
        text_encoder.train()
        fusion_head.train()

        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch['image'].to(device)
            question_ids = batch['question_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['target_id'].to(device)

            img_feats = vision_encoder(images)
            txt_feats = text_encoder(question_ids, attention_mask)

            logits = fusion_head(img_feats, txt_feats)
            loss = criterion(logits, target_ids)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Completed | Train Loss: {avg_loss:.4f}")

        scheduler.step()

        if hf_utils is not None and (epoch + 1) % save_every == 0:
            hf_utils.save_model_to_hub(
                vision_encoder, text_encoder, fusion_head, repo_name, epoch + 1, device
            )

    return train_losses
