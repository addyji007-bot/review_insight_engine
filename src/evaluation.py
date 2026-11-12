"""Evaluation helpers for labelled sentiment datasets."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .sentiment import TransformerSentimentClassifier


def evaluate_classifier(frame: pd.DataFrame, text_column: str = "text", label_column: str = "label", model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> dict:
    """Evaluate predictions and return JSON-serializable model-card metrics."""
    if text_column not in frame or label_column not in frame:
        raise ValueError(f"Dataset must contain '{text_column}' and '{label_column}' columns.")
    classifier = TransformerSentimentClassifier(model_name)
    predictions = [result.label for result in classifier.predict(frame[text_column].astype(str).tolist())]
    labels = frame[label_column].astype(str).str.lower().tolist()
    ordered = sorted(set(labels) | set(predictions))
    return {
        "n_samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "labels": ordered,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=ordered).tolist(),
        "classification_report": classification_report(labels, predictions, labels=ordered, output_dict=True, zero_division=0),
    }

