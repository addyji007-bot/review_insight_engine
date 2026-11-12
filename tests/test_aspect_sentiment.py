"""Tests for aspect-based sentiment analysis (src/aspect_sentiment.py)."""

from dataclasses import dataclass

import pandas as pd
import pytest

from src.aspect_sentiment import analyze_aspects, evaluate_labels, split_clauses, summarize_aspects, validation_template


@dataclass
class Fake:
    label: str
    confidence: float


class FakeClassifier:
    """Word-list stand-in for DistilBERT so tests run instantly and offline."""

    NEGATIVE = {"crashes", "late", "rude", "broken"}

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self.calls: list[list[str]] = []

    def predict(self, texts):
        self.calls.append(list(texts))
        return [
            Fake("negative" if set(str(t).lower().split()) & self.NEGATIVE else "positive", self.confidence)
            for t in texts
        ]


def frame(*texts):
    return pd.DataFrame({"feedback_id": range(1, len(texts) + 1), "text": list(texts)})


def test_split_clauses_breaks_at_but_and_sentence_ends():
    parts = split_clauses("The app crashes, but support was helpful. Delivery was late.")
    assert parts == ["The app crashes", "support was helpful", "Delivery was late"]


def test_mixed_review_gets_a_different_sentiment_per_aspect():
    result = analyze_aspects(FakeClassifier(), frame("The app crashes when I pay, but support resolved my issue quickly."))
    by_aspect = dict(zip(result["aspect"], result["aspect_sentiment"]))
    assert by_aspect == {"app / website": "negative", "customer support": "positive"}


def test_aspect_is_scored_on_its_own_clause_not_the_whole_review():
    result = analyze_aspects(FakeClassifier(), frame("Delivery was late, but the price is fair"))
    row = result[result["aspect"] == "price"].iloc[0]
    assert row["clause"] == "the price is fair"
    assert row["aspect_sentiment"] == "positive"


def test_low_confidence_predictions_become_neutral():
    result = analyze_aspects(FakeClassifier(confidence=0.55), frame("Delivery was late"))
    assert result["aspect_sentiment"].tolist() == ["neutral"]
    assert result["model_label"].tolist() == ["negative"]


def test_reviews_without_aspects_produce_an_empty_result_with_columns():
    result = analyze_aspects(FakeClassifier(), frame("I am happy", "Nothing to say here"))
    assert result.empty and "aspect_sentiment" in result.columns


def test_identical_clauses_are_scored_once():
    classifier = FakeClassifier()
    analyze_aspects(classifier, frame("Delivery was late", "Delivery was late"))
    assert classifier.calls == [["Delivery was late"]]


def test_summary_counts_and_shares():
    df = pd.DataFrame(
        {"aspect": ["a"] * 4 + ["b"], "aspect_sentiment": ["positive", "negative", "negative", "negative", "positive"]}
    )
    summary = summarize_aspects(df).set_index("aspect")
    assert summary.loc["a", "mentions"] == 4
    assert summary.loc["a", "negative_share"] == 0.75
    assert summary.loc["b", "positive_share"] == 1.0


def test_evaluate_labels_matches_hand_calculation():
    df = pd.DataFrame(
        {
            "aspect_sentiment": ["positive", "positive", "negative", "negative", "positive", "negative"],
            "true_sentiment": ["positive", "negative", "negative", "negative", "positive", ""],
        }
    )
    metrics = evaluate_labels(df)
    assert metrics["n"] == 5  # the unlabelled row is ignored
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["confusion"].shape == (2, 2)


def test_evaluate_labels_explains_missing_labels():
    df = pd.DataFrame({"aspect_sentiment": ["positive"], "true_sentiment": [""]})
    with pytest.raises(ValueError, match="true_sentiment"):
        evaluate_labels(df)


def test_validation_template_is_blank_and_capped():
    df = pd.DataFrame(
        {"feedback_id": range(100), "aspect": "a", "clause": "x", "aspect_sentiment": "positive"}
    )
    template = validation_template(df, n=40)
    assert len(template) == 40 and (template["true_sentiment"] == "").all()
