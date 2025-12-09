# Kvasir AnswerNet

Vision–language VQA model on Kvasir-VQA (ViT + DistilBERT).

- src/models/vision_encoder.py     : ViT-based vision encoder
- src/models/text_encoder.py       : DistilBERT text encoder
- src/models/fusion_head.py        : Fusion head for VQA
- src/data/dataset.py              : VQADataset and data loaders
- src/train.py                     : Training loop and evaluation
- src/hf_utils.py                  : Save / load checkpoints on Hugging Face Hub
- scripts/run_train.py             : Entry script to train the model

After downloading this repo locally, open these files and paste your full code.
