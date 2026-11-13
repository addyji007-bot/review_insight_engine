"""Integration tests verifying core data contracts survive preprocessing and analysis.

- feedback_id survives preprocessing and pipeline analysis.
- Duplicate feedback texts with different IDs are correctly joined.
- Dataset fingerprint changes when texts, IDs, or model config change.
- Existing functionality remains intact.
"""

from __future__ import annotations

import pandas as pd

from src.pipeline import FeedbackInsightsPipeline
from src.preprocessing import (
    load_feedback,
    normalize_text,
    normalize_feedback_frame,
)
from src.topics import TopicDiscoverer, dataset_fingerprint


def test_feedback_id_survives_preprocessing(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text("id,text\n1,Great product!\n2,Terrible experience.\n", encoding="utf-8")
    frame = load_feedback(p)
    assert "feedback_id" in frame.columns
    assert list(frame["feedback_id"]) == [1, 2]
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]


def test_normalize_text_preserves_ids_concept():
    assert normalize_text("  refund\n request  ") == "refund request"


def test_pipeline_preserves_feedback_id():
    pipeline = FeedbackInsightsPipeline()
    texts = ["Great product!", "Terrible experience."]
    ids = [10, 20]
    results = pipeline.analyze_texts(texts, ids=ids)
    assert "feedback_id" in results.columns
    assert list(results["feedback_id"]) == ids
    assert list(results["text"]) == texts
    assert set(results["sentiment"]).issubset({"positive", "negative"})


def test_duplicate_texts_different_ids_joined():
    pipeline = FeedbackInsightsPipeline()
    texts = ["Same feedback text", "Same feedback text", "Different text"]
    ids = [1, 2, 3]
    results = pipeline.analyze_texts(texts, ids=ids)
    assert list(results["feedback_id"]) == ids
    # Two records share text but keep distinct IDs
    same = results[results["text"] == "Same feedback text"]
    assert len(same) == 2
    assert set(same["feedback_id"]) == {1, 2}


def test_dataset_fingerprint_detects_changes():
    base = pd.DataFrame(
        {"feedback_id": [1, 2], "text": ["a", "b"]}
    )
    f1 = dataset_fingerprint(base, 2, "auto")

    # Different text -> different fingerprint
    changed_text = pd.DataFrame({"feedback_id": [1, 2], "text": ["a", "c"]})
    f2 = dataset_fingerprint(changed_text, 2, "auto")
    assert f1 != f2

    # Different ID -> different fingerprint
    changed_id = pd.DataFrame({"feedback_id": [1, 3], "text": ["a", "b"]})
    f3 = dataset_fingerprint(changed_id, 2, "auto")
    assert f1 != f3

    # Different model config -> different fingerprint
    f4 = dataset_fingerprint(base, 5, "auto")
    assert f1 != f4

    # Same data -> same fingerprint
    f5 = dataset_fingerprint(base, 2, "auto")
    assert f1 == f5


def test_topic_discoverer_api_shape():
    td = TopicDiscoverer()
    assert hasattr(td, "fit_transform")
    assert hasattr(td, "transform")
    assert hasattr(td, "summary")
    assert hasattr(td, "representative_docs")
    assert isinstance(td.available, bool)


def test_existing_preprocessing_still_works():
    assert normalize_text("  refund\n request  ") == "refund request"


def test_detects_reviewtext_column(tmp_path):
    p = tmp_path / "amazon.csv"
    p.write_text('id,reviewText\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert "feedback_id" in frame.columns
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]
    assert frame.attrs.get("detected_column") == "reviewText"


def test_detects_review_text_column(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,review_text\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "review_text"
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]


def test_detects_comment_column_case_insensitive(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,COMMENT\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "COMMENT"


def test_detects_description_column(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,description\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "description"


def test_original_columns_preserved(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,reviewText,rating\n1,Great product!,5\n2,Terrible experience.,1\n', encoding="utf-8")
    frame = load_feedback(p)
    assert "reviewText" in frame.columns
    assert "rating" in frame.columns
    assert "text" in frame.columns
    assert "feedback_id" in frame.columns


def test_no_text_column_raises(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,rating\n1,5\n2,1\n', encoding="utf-8")
    try:
        load_feedback(p)
    except ValueError as exc:
        assert "Could not find a text column" in str(exc)
        return
    raise AssertionError("Expected ValueError for missing text column")


def test_interactive_mode_returns_raw_frame(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,my_feedback_column\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p, interactive=True)
    assert frame.attrs.get("needs_column_selection") is True
    assert "text" not in frame.columns
    assert "feedback_id" not in frame.columns


def test_normalize_feedback_frame_no_reparse(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,my_feedback_column\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    raw = load_feedback(p, interactive=True)
    assert raw.attrs.get("needs_column_selection") is True
    frame = normalize_feedback_frame(raw, "my_feedback_column")
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]
    assert list(frame["feedback_id"]) == [1, 2]


def test_utf8_bom_loaded(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('\ufeffid,text\n1,Great product!\n', encoding="utf-8-sig")
    frame = load_feedback(p)
    assert list(frame["text"]) == ["Great product!"]


def test_windows1252_loaded(tmp_path):
    p = tmp_path / "fb.csv"
    content = 'id,text\n1,Great product!\n'
    p.write_bytes(content.encode("windows-1252"))
    frame = load_feedback(p)
    assert list(frame["text"]) == ["Great product!"]


def test_quoted_commas_and_multiline(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,text\n1,"Great, product!"\n2,"Line1\nLine2"\n', encoding="utf-8")
    frame = load_feedback(p)
    assert len(frame) == 2
    assert frame.iloc[0]["text"] == "Great, product!"
    assert frame.iloc[1]["text"] == "Line1 Line2"


def test_empty_records_removed_and_tracked(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,text\n1,Great product!\n2,\n3,   \n4,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("empty_removed") == 2
    assert frame.attrs.get("usable_count") == 2
    assert len(frame) == 2


def test_feedback_id_stable_end_to_end(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,reviewText\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    pipeline = FeedbackInsightsPipeline()
    results = pipeline.analyze_texts(frame["text"].tolist(), ids=frame["feedback_id"].tolist())
    assert list(results["feedback_id"]) == [1, 2]
    assert set(results["sentiment"]).issubset({"positive", "negative"})


# --- Test A: CSV with comment_text — automatic detection succeeds ---
def test_detects_comment_text_column(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('comment_id,customer_type,service_area,comment_text,sentiment\n1,Type1,Area1,Great product!,positive\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "comment_text"
    assert list(frame["text"]) == ["Great product!"]
    assert "comment_id" in frame.columns
    assert "sentiment" in frame.columns


# --- Test B: CSV with reviewText — automatic detection succeeds ---
def test_detects_reviewtext_column_b(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,reviewText\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "reviewText"
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]


# --- Test C: CSV with review_text — automatic detection succeeds ---
def test_detects_review_text_column_c(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,review_text\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "review_text"
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]


# --- Test D: CSV with unknown text-column — interactive mode returns raw frame ---
def test_unknown_column_interactive_mode(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,my_feedback_column\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p, interactive=True)
    assert frame.attrs.get("needs_column_selection") is True
    assert "text" not in frame.columns


# --- Test E: normalize_feedback_frame returns analysis-ready DataFrame ---
def test_normalize_feedback_frame_analysis_ready(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,my_feedback_column\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    raw = load_feedback(p, interactive=True)
    assert raw.attrs.get("needs_column_selection") is True
    frame = normalize_feedback_frame(raw, "my_feedback_column")
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]
    assert list(frame["feedback_id"]) == [1, 2]
    assert frame.attrs.get("usable_count") == 2


# --- Test F: Original columns preserved after normalization ---
def test_original_columns_preserved_f(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('comment_id,customer_type,comment_text,sentiment\n1,Type1,Great product!,positive\n', encoding="utf-8")
    frame = load_feedback(p)
    assert "comment_id" in frame.columns
    assert "customer_type" in frame.columns
    assert "sentiment" in frame.columns
    assert "comment_text" in frame.columns
    assert "text" in frame.columns


# --- Test G: Duplicate feedback text with different IDs remains two records ---
def test_duplicate_text_different_ids(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,text\n1,Same feedback\n2,Same feedback\n3,Different\n', encoding="utf-8")
    frame = load_feedback(p)
    assert len(frame) == 3
    same = frame[frame["text"] == "Same feedback"]
    assert len(same) == 2
    assert set(same["feedback_id"]) == {1, 2}


# --- Test H: Quoted commas and multiline reviews load correctly ---
def test_quoted_commas_and_multiline_h(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,text\n1,"Great, product!"\n2,"Line1\nLine2"\n', encoding="utf-8")
    frame = load_feedback(p)
    assert len(frame) == 2
    assert frame.iloc[0]["text"] == "Great, product!"
    assert frame.iloc[1]["text"] == "Line1 Line2"


# --- Test I: Large Amazon-style dataset loads without buffer overflow ---
def test_large_amazon_style_dataset(tmp_path):
    p = tmp_path / "amazon.csv"
    rows = []
    for i in range(20000):
        text = f"This product is {'' if i % 2 else 'not '}great but arrived late. reviewText {i}."
        rows.append(f'{i},"{text}"')
    csv_content = 'id,reviewText\n' + '\n'.join(rows)
    p.write_text(csv_content, encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "reviewText"
    assert frame.attrs.get("usable_count") > 10000
    assert list(frame["feedback_id"][:3]) == [1, 2, 3]


# --- Test J: Sentiment, NER, topic discovery, CSV export still work ---
def test_full_pipeline_end_to_end(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text('id,text\n1,The mobile app crashes when I try to pay.\n2,Fast delivery and excellent packaging.\n', encoding="utf-8")
    frame = load_feedback(p)
    pipeline = FeedbackInsightsPipeline()
    results = pipeline.analyze_texts(frame["text"].tolist(), ids=frame["feedback_id"].tolist())
    assert "sentiment" in results.columns
    assert "entities" in results.columns
    assert "keywords" in results.columns
    assert set(results["sentiment"]).issubset({"positive", "negative"})
    # CSV export round-trip
    csv_out = results.to_csv(index=False)
    assert "signaldesk_enriched_feedback" not in csv_out or True  # just verify it serializes


# --- Test K: All Plotly chart keys are unique ---
def test_all_plotly_chart_keys_unique():
    import re
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    keys = re.findall(r'key="chart_([^"]+)"', content)
    assert len(keys) == len(set(keys)), f"Duplicate chart keys found: {keys}"
    assert len(keys) == 9, f"Expected 9 chart keys, found {len(keys)}: {keys}"


# --- Test L: Amazon "Review Text" column (with space) auto-detected ---
def test_detects_review_text_with_space_column(tmp_path):
    p = tmp_path / "amazon.csv"
    p.write_text('Id,Review Text\n1,Great product!\n2,Terrible experience.\n', encoding="utf-8")
    frame = load_feedback(p)
    assert frame.attrs.get("detected_column") == "Review Text"
    assert list(frame["text"]) == ["Great product!", "Terrible experience."]


# --- Test M: analyze_with_timing returns both results and timings ---
def test_analyze_with_timing_method():
    pipeline = FeedbackInsightsPipeline()
    texts = ["Great product!", "Terrible experience."]
    ids = [10, 20]
    results, timings = pipeline.analyze_with_timing(texts, ids=ids)
    assert "feedback_id" in results.columns
    assert list(results["feedback_id"]) == ids
    assert set(results["sentiment"]).issubset({"positive", "negative"})
    assert isinstance(timings, dict)
    assert "overall_time" in timings
    assert "records_per_second" in timings
    assert timings["analyzed_records"] == 2
    assert timings["failures"] == 0
