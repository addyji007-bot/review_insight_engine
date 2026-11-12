"""Topic discovery with BERTopic, hardened for small and repetitive datasets.

This module is self-contained on purpose: it does not import ``src.topics`` so an
old copy of that file can never interfere with topic discovery.

What it does differently from a default BERTopic call
-----------------------------------------------------
* Stop words are removed inside the tokenizer, so scikit-learn's strict
  ``stop_words`` parameter validation (which rejects sets) is never involved.
  Negation terms ("no", "not", "never") are kept so "not good" stays a phrase.
* Embeddings are computed here and passed to BERTopic. This makes the embedding
  step easy to swap or test.
* UMAP and HDBSCAN are configured from the dataset size. BERTopic's defaults
  (n_neighbors=15, spectral init) crash on small datasets.
* Fewer than ``MIN_DOCS`` unique reviews raises a clear, readable error.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

ENGINE_VERSION = "2026-11-11-r4"
MIN_DOCS = 10

_STOPWORDS = frozenset(set(ENGLISH_STOP_WORDS) - {"no", "not", "never"})
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_EMBEDDER_CACHE: dict[str, Callable] = {}


def _tokenize(text: str) -> list[str]:
    """Tokenise like scikit-learn's default and drop stop words (negations kept)."""
    return [token for token in _TOKEN_RE.findall(text) if token not in _STOPWORDS]


def _sentence_transformer_embedder(model_name: str) -> Callable:
    """Load a sentence-transformer once and reuse it."""
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _EMBEDDER_CACHE[model_name] = lambda docs: model.encode(list(docs), show_progress_bar=False)
    return _EMBEDDER_CACHE[model_name]


class TopicDiscoverer:
    """Cluster similar feedback into topics without manually assigned labels."""

    def __init__(
        self,
        embedder: Callable | None = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_topic_size: int | None = None,
        label_overrides: dict[int, str] | None = None,
        seed: int = 42,
    ):
        self._embedder = embedder
        self.embedding_model_name = embedding_model_name
        self.min_topic_size = min_topic_size
        self.label_overrides = label_overrides or {}
        self.seed = seed
        self._available = self._check_available()
        self._model = None
        self._topics: list[int] = []
        self._texts: list[str] = []
        self._ids: list = []

    def _check_available(self) -> bool:
        try:
            import bertopic  # noqa: F401
            import hdbscan  # noqa: F401
            import umap  # noqa: F401
        except ImportError:
            return False
        if self._embedder is None:
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                return False
        return True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit_transform(self, texts: list[str], ids: list) -> pd.DataFrame:
        """Fit BERTopic and return columns: feedback_id, text, topic, topic_name."""
        texts = [str(t) for t in texts]
        ids = list(ids)
        empty = pd.DataFrame({"feedback_id": ids, "text": texts, "topic": [None] * len(texts), "topic_name": [None] * len(texts)})
        if not texts or not self._available:
            return empty

        count = len(texts)
        if count < MIN_DOCS:
            raise ValueError(
                f"Topic discovery needs at least {MIN_DOCS} reviews, but only {count} were provided. "
                "If you removed duplicates, your file may contain very few unique reviews; "
                "use a larger, more varied dataset."
            )

        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP

        embedder = self._embedder or _sentence_transformer_embedder(self.embedding_model_name)
        embeddings = np.asarray(embedder(texts))

        min_topic_size = self.min_topic_size or max(3, min(10, count // 15))
        umap_model = UMAP(
            n_neighbors=max(2, min(15, count - 1)),
            n_components=max(2, min(5, count - 2)),
            min_dist=0.0,
            metric="cosine",
            random_state=self.seed,
            init="random" if count < 50 else "spectral",
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,
            stop_words=None,
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )

        self._model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=None,
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = self._model.fit_transform(texts, embeddings=embeddings)
        self._topics = [int(t) for t in topics]
        self._texts = texts
        self._ids = ids
        return pd.DataFrame(
            {
                "feedback_id": self._ids,
                "text": self._texts,
                "topic": self._topics,
                "topic_name": [self._topic_name(t) for t in self._topics],
            }
        )

    def summary(self) -> pd.DataFrame:
        """One row per topic: topic, topic_name, count, keywords, is_outlier."""
        columns = ["topic", "topic_name", "count", "keywords", "is_outlier"]
        if not self.is_fitted:
            return pd.DataFrame(columns=columns)
        info = self._model.get_topic_info()
        rows = []
        for _, row in info.iterrows():
            topic_id = int(row["Topic"])
            words = self._model.get_topic(topic_id) or []
            rows.append(
                {
                    "topic": topic_id,
                    "topic_name": self._topic_name(topic_id),
                    "count": int(row["Count"]),
                    "keywords": ", ".join(str(word) for word, _ in words[:6]),
                    "is_outlier": topic_id == -1,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def representative_docs(self, topic_id: int, n: int = 5) -> list[str]:
        """Up to n representative reviews for a topic; empty list if none."""
        if not self.is_fitted:
            return []
        try:
            reps = self._model.get_representative_docs(topic_id)
        except (ValueError, KeyError):
            return []
        if isinstance(reps, dict):
            reps = reps.get(topic_id, [])
        return list(reps or [])[:n]

    def _topic_name(self, topic_id: int) -> str:
        if topic_id in self.label_overrides:
            return self.label_overrides[topic_id]
        if topic_id == -1:
            return "Outliers / Unclustered"
        if self.is_fitted:
            words = self._model.get_topic(topic_id) or []
            if words:
                return " / ".join(str(word) for word, _ in words[:3]).title()
        return f"Topic {topic_id}"


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 over ordered (feedback_id, text) pairs, used as a cache key."""
    pairs = sorted((int(fid), str(t)) for fid, t in zip(frame["feedback_id"], frame["text"]))
    return hashlib.sha256(json.dumps({"pairs": pairs}, sort_keys=True).encode("utf-8")).hexdigest()[:16]
