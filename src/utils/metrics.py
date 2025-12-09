import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_accuracy(preds, targets):
    correct = sum(int(p == t) for p, t in zip(preds, targets))
    return correct / len(preds)


def bleu_and_em(pred_texts, ref_texts):
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    em_scores = []

    for pred, ref in zip(pred_texts, ref_texts):
        bleu = sentence_bleu(
            [ref.lower().split()],
            pred.lower().split(),
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie,
        )
        bleu_scores.append(bleu)
        em_scores.append(int(pred.strip().lower() == ref.strip().lower()))

    avg_bleu = float(np.mean(bleu_scores))
    avg_em = float(np.mean(em_scores))
    return avg_bleu, avg_em
