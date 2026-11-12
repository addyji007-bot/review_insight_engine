"""End-to-end enrichment pipeline for batch and interactive use."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from .entities import EntityKeywordExtractor
from .sentiment import TransformerSentimentClassifier


class FeedbackInsightsPipeline:
    def __init__(self, sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment = TransformerSentimentClassifier(sentiment_model)
        self.extractor = EntityKeywordExtractor()

    def analyze_texts(
        self,
        texts: list[str],
        ids: list | None = None,
        progress_cb: Callable[[str, int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Analyze texts with batched inference and progress reporting.

        Args:
            texts: List of feedback texts to analyze.
            ids: Optional feedback IDs aligned with texts.
            progress_cb: Callback(stage, completed, total) for progress updates.

        Returns:
            DataFrame with sentiment, entities, keywords per text.
        """
        total = len(texts)
        if total == 0:
            return pd.DataFrame(columns=["text", "feedback_id", "sentiment", "sentiment_confidence",
                                         "sentiment_probabilities", "entities", "entity_text",
                                         "entity_labels", "keywords"])

        # Stage 1: Sentiment (batched PyTorch inference)
        if progress_cb:
            progress_cb("Loading models", 0, total)
        sentiment_results = self.sentiment.predict(texts)
        if progress_cb:
            progress_cb("Sentiment analysis", total, total)

        # Stage 2: Named Entity Recognition (spaCy pipe batching)
        if progress_cb:
            progress_cb("Named Entity Recognition", 0, total)
        entity_results = self.extractor.extract_entities_batch(texts)
        if progress_cb:
            progress_cb("Named Entity Recognition", total, total)

        # Stage 3: Keywords (spaCy pipe batching)
        if progress_cb:
            progress_cb("Keyword extraction", 0, total)
        keyword_results = self.extractor.extract_keywords_batch(texts)
        if progress_cb:
            progress_cb("Keyword extraction", total, total)

        # Stage 4: Assemble results
        if progress_cb:
            progress_cb("Preparing results", 0, total)
        rows = []
        for i, (text, sentiment_result, entities, keywords) in enumerate(
            zip(texts, sentiment_results, entity_results, keyword_results)
        ):
            row = {
                "text": text,
                "sentiment": sentiment_result.label,
                "sentiment_confidence": round(sentiment_result.confidence, 4),
                "sentiment_probabilities": sentiment_result.probabilities,
                "entities": [entity.to_dict() for entity in entities],
                "entity_text": ", ".join(entity.text for entity in entities),
                "entity_labels": ", ".join(entity.label for entity in entities),
                "keywords": keywords,
            }
            if ids is not None:
                row["feedback_id"] = ids[i]
            rows.append(row)
            if progress_cb and (i + 1) % 100 == 0:
                progress_cb("Preparing results", i + 1, total)
        if progress_cb:
            progress_cb("Preparing results", total, total)

        return pd.DataFrame(rows)

    def analyze_with_timing(
        self,
        texts: list[str],
        ids: list | None = None,
        progress_cb: Callable[[str, int, int, float], None] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze texts and return results with timing metrics.

        Args:
            texts: List of feedback texts to analyze.
            ids: Optional feedback IDs aligned with texts.
            progress_cb: Callback(stage, completed, total, elapsed_seconds).

        Returns:
            Tuple of (results DataFrame, timing metrics dict).
        """
        total = len(texts)
        timings: dict = {
            "total_records": total,
            "analyzed_records": 0,
            "sentiment_time": 0.0,
            "ner_time": 0.0,
            "keyword_time": 0.0,
            "overall_time": 0.0,
            "records_per_second": 0.0,
            "failures": 0,
        }

        if total == 0:
            return pd.DataFrame(), timings

        overall_start = time.time()

        # Stage 1: Sentiment
        t0 = time.time()
        sentiment_results = self.sentiment.predict(texts)
        timings["sentiment_time"] = time.time() - t0
        if progress_cb:
            progress_cb("Sentiment analysis", total, total, timings["sentiment_time"])

        # Stage 2: NER
        t0 = time.time()
        entity_results = self.extractor.extract_entities_batch(texts)
        timings["ner_time"] = time.time() - t0
        if progress_cb:
            progress_cb("Named Entity Recognition", total, total, timings["ner_time"])

        # Stage 3: Keywords
        t0 = time.time()
        keyword_results = self.extractor.extract_keywords_batch(texts)
        timings["keyword_time"] = time.time() - t0
        if progress_cb:
            progress_cb("Keyword extraction", total, total, timings["keyword_time"])

        # Stage 4: Assemble
        t0 = time.time()
        rows = []
        failures = 0
        for i, (text, sentiment_result, entities, keywords) in enumerate(
            zip(texts, sentiment_results, entity_results, keyword_results)
        ):
            try:
                row = {
                    "text": text,
                    "sentiment": sentiment_result.label,
                    "sentiment_confidence": round(sentiment_result.confidence, 4),
                    "sentiment_probabilities": sentiment_result.probabilities,
                    "entities": [entity.to_dict() for entity in entities],
                    "entity_text": ", ".join(entity.text for entity in entities),
                    "entity_labels": ", ".join(entity.label for entity in entities),
                    "keywords": keywords,
                }
                if ids is not None:
                    row["feedback_id"] = ids[i]
                rows.append(row)
            except Exception:
                failures += 1
            if progress_cb and (i + 1) % 100 == 0:
                progress_cb("Preparing results", i + 1, total, time.time() - overall_start)

        timings["overall_time"] = time.time() - overall_start
        timings["analyzed_records"] = len(rows)
        timings["failures"] = failures
        if timings["overall_time"] > 0:
            timings["records_per_second"] = round(len(rows) / timings["overall_time"], 1)

        return pd.DataFrame(rows), timings
