"""Tests for the one-slide business insight (src/insight_slide.py)."""

import pandas as pd

from src.aspect_sentiment import summarize_aspects
from src.insight_slide import build_insight, render_slide_html


def aspect_frame(spec):
    rows = []
    for aspect, sentiment, clause, count in spec:
        for i in range(count):
            rows.append(
                {"feedback_id": len(rows), "review_text": "r", "aspect": aspect, "clause": clause,
                 "aspect_sentiment": sentiment, "confidence": 0.99 - i * 0.001}
            )
    return pd.DataFrame(rows)


def test_headline_names_the_best_and_worst_aspect():
    df = aspect_frame([
        ("delivery", "negative", "The package arrived late and damaged", 8),
        ("delivery", "positive", "Delivery was fast", 2),
        ("customer support", "positive", "Support resolved my issue very quickly", 9),
        ("customer support", "negative", "Support never replied", 1),
    ])
    insight = build_insight(summarize_aspects(df), df, total_reviews=100)
    assert insight["headline"] == "Customers praise customer support, but delivery drives the most complaints"
    assert insight["worst"]["aspect"] == "delivery" and insight["worst"]["share"] == 0.8
    assert "courier" in insight["action"]


def test_thin_aspects_do_not_win_the_ranking():
    df = aspect_frame([
        ("price", "negative", "Far too expensive for me", 1),  # 100% negative but only 1 mention
        ("delivery", "negative", "Arrived late and damaged", 6),
        ("delivery", "positive", "Delivery was fast", 4),
    ])
    assert build_insight(summarize_aspects(df), df, 50)["worst"]["aspect"] == "delivery"


def test_no_negative_mentions_gives_a_calm_headline():
    df = aspect_frame([("price", "positive", "Great value for the price", 5)])
    insight = build_insight(summarize_aspects(df), df, 20)
    assert insight["has_problem"] is False
    assert "No aspect stands out" in insight["headline"]


def test_empty_input_returns_none():
    assert build_insight(summarize_aspects(pd.DataFrame()), pd.DataFrame(), 10) is None


def test_slide_html_escapes_customer_text():
    df = aspect_frame([("app / website", "negative", "The app <script>alert(1)</script> crashes on pay", 5)])
    html = render_slide_html(build_insight(summarize_aspects(df), df, 10))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_slide_is_a_complete_html_document():
    df = aspect_frame([("price", "negative", "Far too expensive for what you get", 4), ("price", "positive", "Good value", 3)])
    html = render_slide_html(build_insight(summarize_aspects(df), df, 30))
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "Based on 30 reviews" in html
