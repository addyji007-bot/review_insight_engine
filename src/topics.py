"""Topic discovery over feedback texts using BERTopic.

Wraps BERTopic with graceful degradation when the optional heavy stack
(sentence-transformers, umap-learn, hdbscan) is not installed.

Architecture notes:
- Semantic clustering is driven by sentence-transformer embeddings + UMAP +
  HDBSCAN. This is separate from topic *representation*.
- Topic keywords/labels are produced by CountVectorizer + c-TF-IDF with a
  custom stopword list (negation terms preserved) and bigrams. This improves
  keyword quality without changing the underlying semantic clusters.
- Topic -1 is treated as the outlier / unclustered category and reported
  separately.
- No model caching lives here; dataset-fingerprint session caching is owned
  by the Streamlit layer (app.py).
"""

from __future__ import annotations

import hashlib
import json
import re

import pandas as pd


# Custom English stopwords that preserve negation terms (not, no, never, etc.)
# so sentiment-bearing phrases such as "not satisfied" keep their negation.
# CountVectorizer requires stop_words as a list, not a set.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_CUSTOM_STOPWORDS: list[str] = sorted(
    set(ENGLISH_STOP_WORDS) - {"no", "not", "never"}
)

assert type(_CUSTOM_STOPWORDS) is list
assert all(type(word) is str for word in _CUSTOM_STOPWORDS)

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_STOPWORD_SET = frozenset(_CUSTOM_STOPWORDS)


def _tokenize_without_stopwords(text: str) -> list[str]:
    """Tokenise like scikit-learn's default and drop the custom stop words.

    scikit-learn builds n-grams from the tokens this returns, so this is
    equivalent to passing ``stop_words=_CUSTOM_STOPWORDS`` (negation terms
    such as "not" are kept) without going through parameter validation.
    """
    return [token for token in _TOKEN_RE.findall(text) if token not in _STOPWORD_SET]


class TopicDiscoverer:
    """Cluster similar feedback into topics without manually assigned labels."""

    def __init__(
        self,
        min_topic_size: int = 2,
        nr_topics: str | int = "auto",
        label_overrides: dict[int, str] | None = None,
    ):
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.label_overrides = label_overrides or {}
        self._available = self._check_available()
        self._model = None
        self._topics: list[int] = []
        self._texts: list[str] = []
        self._ids: list = []

    def _check_available(self) -> bool:
        try:
            import bertopic  # noqa: F401
            import sentence_transformers  # noqa: F401
            import umap  # noqa: F401
            import hdbscan  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit_transform(self, texts: list[str], ids: list) -> pd.DataFrame:
        """Fit BERTopic on texts and return per-record topic assignments.

        Returns columns: feedback_id, text, topic, topic_name.
        Topic -1 is the outlier category.
        """
        result = pd.DataFrame(
            {
                "feedback_id": list(ids),
                "text": list(texts),
                "topic": [None] * len(texts),
                "topic_name": [None] * len(texts),
            }
        )
        if not texts:
            return result
        if not self._available:
            return result

        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        # Stop words are removed inside the tokenizer, so scikit-learn's strict
        # `stop_words` parameter validation is never involved (it rejects sets).
        vectorizer_model = CountVectorizer(
            tokenizer=_tokenize_without_stopwords,
            token_pattern=None,
            stop_words=None,
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )

        self._model = BERTopic(
            min_topic_size=self.min_topic_size,
            nr_topics=self.nr_topics,
            vectorizer_model=vectorizer_model,
            verbose=False,
        )

        topics, _ = self._model.fit_transform(texts)
        self._topics = [int(t) for t in topics]
        self._texts = list(texts)
        self._ids = list(ids)

        topic_names = self._make_labels()
        result = pd.DataFrame(
            {
                "feedback_id": self._ids,
                "text": self._texts,
                "topic": self._topics,
                "topic_name": topic_names,
            }
        )
        return result

    def transform(self, texts: list[str], ids: list) -> pd.DataFrame:
        """Assign topics to new texts using the fitted model.

        Raises RuntimeError if fit_transform has not been called first.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TopicDiscoverer has not been fitted. Call fit_transform() "
                "before transform()."
            )
        if not texts:
            return pd.DataFrame(
                {
                    "feedback_id": ids,
                    "text": texts,
                    "topic": [],
                    "topic_name": [],
                }
            )
        topics, _ = self._model.transform(texts)
        topics = [int(t) for t in topics]
        topic_names = [self._topic_name(int(t)) for t in topics]
        return pd.DataFrame(
            {
                "feedback_id": ids,
                "text": texts,
                "topic": topics,
                "topic_name": topic_names,
            }
        )

    def summary(self) -> pd.DataFrame:
        """Return topic -> name, count, and outlier flag.

        Topic -1 is marked is_outlier=True and reported separately.
        """
        if not self.is_fitted:
            return pd.DataFrame(
                columns=["topic", "topic_name", "count", "is_outlier"]
            )
        info = self._model.get_topic_info().copy()
        info = info.rename(columns={"Topic": "topic", "Name": "topic_name", "Count": "count"})
        info["topic"] = info["topic"].astype(int)
        info["is_outlier"] = info["topic"] == -1
        return info[["topic", "topic_name", "count", "is_outlier"]]

    def representative_docs(self, topic_id: int, n: int = 5) -> list[str]:
        """Return up to n representative reviews for a topic.

        Returns an empty list (with no crash) if the topic is missing or has
        no representative documents.
        """
        if not self.is_fitted:
            return []
        try:
            reps = self._model.get_representative_docs(topic_id)
        except (ValueError, KeyError):
            return []
        if isinstance(reps, dict):
            reps = reps.get(topic_id, [])
        if reps is None:
            return []
        return list(reps)[:n]

    def _topic_name(self, topic_id: int) -> str:
        if topic_id in self.label_overrides:
            return self.label_overrides[topic_id]
        if not self.is_fitted:
            return f"Topic {topic_id}"
        try:
            info = self._model.get_topic_info().set_index("Topic")
            if topic_id in info.index:
                return str(info.loc[topic_id, "Name"])
        except (KeyError, ValueError):
            pass
        return f"Topic {topic_id}"

    def _make_labels(self) -> list[str]:
        """Build readable topic labels from keywords + representative snippets.

        Manual overrides take precedence. Fallback is the cleaned top-term
        phrase. Never forces an incorrect category.
        """
        if not self.is_fitted:
            return [self._topic_name(int(t)) for t in self._topics]

        try:
            info = self._model.get_topic_info().set_index("Topic")
        except (KeyError, ValueError):
            return [self._topic_name(int(t)) for t in self._topics]

        names: list[str] = []
        for topic_id in self._topics:
            if topic_id in self.label_overrides:
                names.append(self.label_overrides[topic_id])
                continue
            try:
                row = info.loc[topic_id]
                raw_name = str(row["Name"])
            except (KeyError, ValueError):
                raw_name = f"Topic {topic_id}"
            if topic_id == -1:
                names.append("Outliers / Unclustered")
                continue
            cleaned = self._clean_label(raw_name)
            names.append(cleaned if cleaned else f"Topic {topic_id}")
        return names

    @staticmethod
    def _clean_label(raw_name: str) -> str:
        """Clean a BERTopic keyword-joined label into a readable phrase."""
        if not raw_name:
            return ""
        parts = [p.strip() for p in raw_name.split("_") if p.strip()]
        if not parts:
            return ""
        title = " ".join(parts).title()
        return title


def dataset_fingerprint(
    frame: pd.DataFrame, min_topic_size: int = 2, nr_topics: str | int = "auto"
) -> str:
    """SHA-256 over ordered (feedback_id, text) pairs plus model config.

    Pure helper (no caching) owned by the topic module so it can be imported
    without pulling in Streamlit. Session-state caching remains in app.py.
    """
    pairs = sorted((int(fid), str(t)) for fid, t in zip(frame["feedback_id"], frame["text"]))
    payload = json.dumps(
        {"pairs": pairs, "min_topic_size": min_topic_size, "nr_topics": nr_topics},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
