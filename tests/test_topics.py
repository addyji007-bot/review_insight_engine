"""Unit tests for the TopicDiscoverer module using a mocked BERTopic stack.

Tests are deterministic and do not download any models.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.topics import TopicDiscoverer


def _install_fake_bertopic():
    """Insert fake bertopic + dependencies into sys.modules.

    BERTopic(...) returns a MagicMock instance so fit_transform/transform
    can be exercised without downloading models.
    """
    fake_bertopic = types.ModuleType("bertopic")
    fake_bertopic.BERTopic = MagicMock()
    fake_st = types.ModuleType("sentence_transformers")
    fake_umap = types.ModuleType("umap")
    fake_hdbscan = types.ModuleType("hdbscan")
    return patch.dict(
        sys.modules,
        {
            "bertopic": fake_bertopic,
            "sentence_transformers": fake_st,
            "umap": fake_umap,
            "hdbscan": fake_hdbscan,
        },
        clear=False,
    )


def _topic_info_frame():
    return pd.DataFrame(
        {
            "Topic": [-1, 0, 1],
            "Name": ["-1", "Topic 0", "Topic 1"],
            "Count": [2, 1, 1],
        }
    )


def test_unavailable_stack_graceful():
    with patch.dict(
        sys.modules,
        {"bertopic": None, "sentence_transformers": None, "umap": None, "hdbscan": None},
        clear=False,
    ):
        td = TopicDiscoverer()
        assert td.available is False
        df = td.fit_transform(["a", "b"], [1, 2])
        assert list(df.columns) == ["feedback_id", "text", "topic", "topic_name"]
        # Stack unavailable: input records preserved, topics not computed (None).
        assert len(df) == 2
        assert list(df["feedback_id"]) == [1, 2]
        assert list(df["text"]) == ["a", "b"]
        assert df["topic"].isna().all()


def test_empty_input_returns_empty():
    td = TopicDiscoverer()
    df = td.fit_transform([], [])
    assert list(df.columns) == ["feedback_id", "text", "topic", "topic_name"]
    assert len(df) == 0


def test_transform_without_fit_raises():
    td = TopicDiscoverer()
    with pytest.raises(RuntimeError, match="not been fitted"):
        td.transform(["a"], [1])


def test_fit_transform_columns_with_mock():
    with _install_fake_bertopic() as p:
        td = TopicDiscoverer()
        assert td.available is True
        mock_model = p["bertopic"].BERTopic.return_value
        mock_model.fit_transform.return_value = ([0, 1, -1], None)
        mock_model.get_topic_info.return_value = _topic_info_frame()
        mock_model.get_representative_docs.return_value = ["rep doc"]
        df = td.fit_transform(["text a", "text b", "text c"], [1, 2, 3])
        assert list(df.columns) == ["feedback_id", "text", "topic", "topic_name"]
        assert len(df) == 3
        assert set(df["topic"]) == {0, 1, -1}
        assert td.is_fitted


def test_transform_after_fit():
    with _install_fake_bertopic() as p:
        td = TopicDiscoverer()
        mock_model = p["bertopic"].BERTopic.return_value
        mock_model.fit_transform.return_value = ([0, -1], None)
        mock_model.get_topic_info.return_value = _topic_info_frame()
        td.fit_transform(["a", "b"], [1, 2])

        mock_model.transform.return_value = ([0], None)
        df = td.transform(["c"], [3])
        assert list(df.columns) == ["feedback_id", "text", "topic", "topic_name"]
        assert len(df) == 1
        assert int(df.iloc[0]["topic"]) == 0


def test_summary_includes_outlier():
    with _install_fake_bertopic() as p:
        td = TopicDiscoverer()
        mock_model = p["bertopic"].BERTopic.return_value
        mock_model.fit_transform.return_value = ([-1, 0], None)
        mock_model.get_topic_info.return_value = _topic_info_frame()
        td.fit_transform(["a", "b"], [1, 2])
        s = td.summary()
        assert list(s.columns) == ["topic", "topic_name", "count", "is_outlier"]
        outlier = s[s["topic"] == -1]
        assert len(outlier) == 1
        assert bool(outlier.iloc[0]["is_outlier"]) is True


def test_label_overrides_applied():
    with _install_fake_bertopic() as p:
        td = TopicDiscoverer(label_overrides={0: "Payment Issues"})
        mock_model = p["bertopic"].BERTopic.return_value
        mock_model.fit_transform.return_value = ([0, 1], None)
        mock_model.get_topic_info.return_value = _topic_info_frame()
        df = td.fit_transform(["a", "b"], [1, 2])
        names = dict(zip(df["topic"], df["topic_name"]))
        assert names[0] == "Payment Issues"


def test_representative_docs_empty_when_not_fitted():
    td = TopicDiscoverer()
    assert td.representative_docs(0) == []
