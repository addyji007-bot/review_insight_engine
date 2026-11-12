"""Aspect-Based Sentiment Analysis (ABSA) on top of the existing sentiment model.

Overall sentiment says *whether* a customer is happy. ABSA says *what* they are
happy or unhappy about. For "The app crashes, but support resolved my issue":

    app              -> negative   (evidence: "The app crashes")
    customer support -> positive   (evidence: "support resolved my issue")

Method (lightweight, explainable, no extra model training):
  1. Split each review into clauses at sentence ends and contrast words
     (but, however, although ...). A clause holding several different aspects
     is split further at commas / "and" when every piece names an aspect.
  2. Find business aspects in each clause with the keyword lexicon in
     ``src.aspects``.
  3. Score each *clause* with the existing sentiment classifier, so an aspect
     is judged only by the words around it, not the whole review.
  4. The base model is binary (positive/negative). Predictions below
     ``neutral_threshold`` confidence are reported as "neutral" (heuristic).
"""

from __future__ import annotations

import re

import pandas as pd

from .aspects import aspect_names

COLUMNS = ["feedback_id", "review_text", "aspect", "clause", "aspect_sentiment", "confidence", "model_label"]

_CLAUSE_SPLIT = re.compile(
    r"\s*(?:[.;!?]+|,?\s+\b(?:but|however|although|though|whereas|yet)\b,?)\s*", flags=re.IGNORECASE
)
_SOFT_SPLIT = re.compile(r",|\band\b|\bwhile\b", flags=re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """Split a message into clauses at sentence ends and contrast words."""
    parts = [part.strip(" ,") for part in _CLAUSE_SPLIT.split(str(text))]
    return [part for part in parts if len(part.split()) >= 2]


def _split_by_aspect(clause: str) -> list[str]:
    """Split a clause that mixes aspects ("delivery was fast, packaging was bad")."""
    if len(aspect_names(clause)) < 2:
        return [clause]
    pieces = [p.strip(" ,") for p in _SOFT_SPLIT.split(clause) if len(p.split()) >= 2]
    kept = [p for p in pieces if aspect_names(p)]
    if len(kept) >= 2 and len({tuple(aspect_names(p)) for p in kept}) >= 2:
        return kept
    return [clause]


def analyze_aspects(classifier, frame: pd.DataFrame, neutral_threshold: float = 0.65) -> pd.DataFrame:
    """Return one row per (review, aspect) with the sentiment toward that aspect.

    ``frame`` needs ``feedback_id`` and ``text`` columns. ``classifier`` is any
    object whose ``predict(list[str])`` returns items with ``.label`` and
    ``.confidence`` (the existing TransformerSentimentClassifier).
    """
    pending: list[tuple] = []
    for feedback_id, text in zip(frame["feedback_id"], frame["text"]):
        text = str(text)
        seen: set[tuple[str, str]] = set()
        for clause in split_clauses(text) or [text]:
            for piece in _split_by_aspect(clause):
                for aspect in aspect_names(piece):
                    if (aspect, piece) not in seen:
                        seen.add((aspect, piece))
                        pending.append((feedback_id, text, aspect, piece))
    if not pending:
        return pd.DataFrame(columns=COLUMNS)

    clauses = list(dict.fromkeys(item[3] for item in pending))  # unique, order kept
    lookup = dict(zip(clauses, classifier.predict(clauses)))
    rows = []
    for feedback_id, text, aspect, clause in pending:
        prediction = lookup[clause]
        label = prediction.label if prediction.confidence >= neutral_threshold else "neutral"
        rows.append(
            {
                "feedback_id": feedback_id,
                "review_text": text,
                "aspect": aspect,
                "clause": clause,
                "aspect_sentiment": label,
                "confidence": round(float(prediction.confidence), 4),
                "model_label": prediction.label,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize_aspects(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """Per-aspect counts and shares, sorted by mentions."""
    if aspect_df is None or aspect_df.empty:
        return pd.DataFrame(columns=["aspect", "mentions", "positive", "negative", "neutral", "negative_share", "positive_share"])
    counts = aspect_df.groupby(["aspect", "aspect_sentiment"]).size().unstack(fill_value=0)
    for column in ("positive", "negative", "neutral"):
        if column not in counts.columns:
            counts[column] = 0
    counts["mentions"] = counts[["positive", "negative", "neutral"]].sum(axis=1)
    counts["negative_share"] = (counts["negative"] / counts["mentions"]).round(3)
    counts["positive_share"] = (counts["positive"] / counts["mentions"]).round(3)
    counts = counts.reset_index()[["aspect", "mentions", "positive", "negative", "neutral", "negative_share", "positive_share"]]
    return counts.sort_values("mentions", ascending=False).reset_index(drop=True)


def validation_template(aspect_df: pd.DataFrame, n: int = 40, seed: int = 42) -> pd.DataFrame:
    """A random sample of aspect rows with an empty ``true_sentiment`` column to label by hand."""
    if aspect_df is None or aspect_df.empty:
        return pd.DataFrame(columns=["feedback_id", "aspect", "clause", "aspect_sentiment", "true_sentiment"])
    sample = aspect_df.sample(n=min(n, len(aspect_df)), random_state=seed).sort_values("feedback_id")
    out = sample[["feedback_id", "aspect", "clause", "aspect_sentiment"]].copy()
    out["true_sentiment"] = ""
    return out.reset_index(drop=True)


def evaluate_labels(labeled: pd.DataFrame, pred_col: str = "aspect_sentiment", true_col: str = "true_sentiment") -> dict:
    """Accuracy, per-class precision/recall/F1 and a confusion matrix from hand-labelled rows."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    for column in (pred_col, true_col):
        if column not in labeled.columns:
            raise ValueError(f"The file needs a '{column}' column.")
    data = labeled[[pred_col, true_col]].dropna()
    data = data[data[true_col].astype(str).str.strip() != ""]
    if data.empty:
        raise ValueError(f"No rows have a value in '{true_col}' yet. Fill it with positive, negative or neutral.")
    y_true = data[true_col].astype(str).str.strip().str.lower()
    y_pred = data[pred_col].astype(str).str.strip().str.lower()
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    report = pd.DataFrame(
        {"sentiment": labels, "precision": precision.round(3), "recall": recall.round(3), "f1": f1.round(3), "support": support}
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true {label}" for label in labels],
        columns=[f"predicted {label}" for label in labels],
    )
    return {
        "n": int(len(data)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "report": report,
        "confusion": confusion,
    }
