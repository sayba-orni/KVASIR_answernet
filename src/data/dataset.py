import torch
from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, data, tokenizer, image_processor, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image'].convert('RGB')
        question = sample['question']
        answer = sample['answer']

        image = self.image_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        question_tokens = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
        )
        answer_tokens = self.tokenizer(
            answer,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
        )

        target_token_id = answer_tokens['input_ids'].squeeze(0)[1]

        return {
            'image': image,
            'question_ids': question_tokens['input_ids'].squeeze(0),
            'attention_mask': question_tokens['attention_mask'].squeeze(0),
            'target_id': target_token_id,
            'answer_text': answer,
        }
