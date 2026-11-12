"""Transformer sentiment inference with PyTorch batching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class SentimentResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


class TransformerSentimentClassifier:
    """Production-shaped wrapper around a Hugging Face sequence classifier."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id_to_label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def predict(self, texts: list[str], batch_size: int = 32, max_length: int = 256) -> list[SentimentResult]:
        """Classify texts. Output order always matches input order.

        Texts are processed shortest-first so each batch is padded to a similar
        length; this is much faster on CPU than padding to the longest text.
        """
        count = len(texts)
        if count == 0:
            return []
        order = sorted(range(count), key=lambda i: len(str(texts[i])))
        ordered: list[SentimentResult] = []
        for start in range(0, count, batch_size):
            batch = [str(texts[i]) for i in order[start : start + batch_size]]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1).cpu()
            for row in probabilities:
                probs = {self.id_to_label[i]: float(value) for i, value in enumerate(row)}
                index = int(torch.argmax(row))
                ordered.append(SentimentResult(self.id_to_label[index], float(row[index]), probs))
        results: list[SentimentResult] = [None] * count  # type: ignore[list-item]
        for position, original_index in enumerate(order):
            results[original_index] = ordered[position]
        return results
