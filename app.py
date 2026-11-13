"""Executive-facing Streamlit application for customer feedback intelligence."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.aspects import aspect_names, find_aspect_mentions
from src.pipeline import FeedbackInsightsPipeline
from src.preprocessing import load_feedback, normalize_feedback_frame
from src.insight_slide import build_insight, render_slide_html
from src.aspect_sentiment import analyze_aspects, evaluate_labels, split_clauses, summarize_aspects, validation_template
from src.topic_engine import TopicDiscoverer, dataset_fingerprint


class _SafeStream:
    """Wrap stdout/stderr so a closed terminal can never crash the app.

    On Windows, writing to a console that no longer exists (for example after
    VS Code's "Reload Window" kills the terminal Streamlit was started from)
    raises ``OSError: [Errno 22] Invalid argument``. Model-loading progress
    bars (tqdm / Hugging Face) write to stderr, so they triggered it.
    """

    _is_safe_stream = True

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except (OSError, ValueError):
            return len(data) if data else 0

    def flush(self):
        try:
            self._stream.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


for _stream_name in ("stdout", "stderr"):
    _current = getattr(sys, _stream_name, None)
    if _current is not None and not getattr(_current, "_is_safe_stream", False):
        setattr(sys, _stream_name, _SafeStream(_current))


st.set_page_config(
    page_title="SignalDesk | Customer Feedback Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color:#0b1324; letter-spacing:-.025em; }
    .block-container { padding:2rem 3rem 3rem; max-width:1500px; }
    [data-testid="stSidebar"] { background:#0b1324; border-right:0; }
    [data-testid="stSidebar"] * { color:#dbeafe !important; }
    .brand-mark { font-family:'Space Grotesk'; color:#fff; font-size:1.25rem; font-weight:700; letter-spacing:-.04em; }
    .brand-sub { color:#94a3b8; font-size:.7rem; margin-top:.1rem; }
    .hero { background:linear-gradient(115deg,#0b1324 0%,#162b52 68%,#1d4ed8 100%); border-radius:18px; padding:2rem 2.25rem; color:white; position:relative; overflow:hidden; margin-bottom:1.6rem; }
    .hero:after { content:''; position:absolute; width:220px; height:220px; right:-70px; top:-100px; border:30px solid rgba(147,197,253,.12); border-radius:50%; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:700; color:#93c5fd; }
    .hero h1 { color:#fff !important; font-size:2rem; margin:.45rem 0 .4rem; }
    .hero p { color:#cbd5e1; margin:0; max-width:700px; }
    .hero-badge { display:inline-block; margin-top:1rem; padding:.35rem .7rem; border:1px solid rgba(191,219,254,.3); border-radius:999px; color:#dbeafe; font-size:.75rem; }
    .section-label { color:#64748b; text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:700; margin:1.4rem 0 .65rem; display:block; }
    .section-title { font-family:'Space Grotesk'; font-size:1.25rem; font-weight:700; color:#0f172a; margin:0 0 .5rem 0; }
    .kpi { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:1rem 1.1rem; min-height:108px; box-shadow:0 6px 18px rgba(15,23,42,.04); }
    .kpi-label { color:#64748b; font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
    .kpi-value { color:#0b1324; font-family:'Space Grotesk'; font-size:1.75rem; font-weight:700; margin:.25rem 0; }
    .kpi-note { color:#94a3b8; font-size:.7rem; }
    .insight { background:#f8fafc; border-left:4px solid #2563eb; border-radius:10px; padding:1rem 1.1rem; margin:.4rem 0; color:#334155; font-size:.85rem; }
    .insight strong { color:#0f172a; }
    .highlight-box { padding:1.25rem; border-radius:12px; background:#f8fafc; line-height:2.3; border:1px solid #e2e8f0; font-size:1rem; }
    mark.entity { padding:.2rem .42rem; border-radius:.4rem; background:#dbeafe; color:#1e3a8a; display:inline-block; margin:0 2px; }
    mark.entity small { margin-left:.35rem; font-size:.63rem; color:#1d4ed8; font-weight:700; text-transform:uppercase; }
    mark.aspect { padding:.2rem .42rem; border-radius:.4rem; background:#fef3c7; color:#92400e; display:inline-block; margin:0 2px; }
    mark.aspect small { margin-left:.35rem; font-size:.63rem; color:#b45309; font-weight:700; text-transform:uppercase; }
    .legend { color:#64748b; font-size:.75rem; }
    .empty-state { background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px; padding:2.5rem 2rem; text-align:center; color:#64748b; margin:1rem 0; }
    .empty-state h3 { color:#0f172a; margin:0 0 .5rem 0; }
    .success-banner { background:#ecfdf5; border:1px solid #bbf7d0; border-radius:10px; padding:1rem 1.25rem; color:#065f46; font-size:.875rem; margin:.5rem 0; }
    .warning-banner { background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:1rem 1.25rem; color:#713f12; font-size:.875rem; margin:.5rem 0; }
    .error-banner { background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:1rem 1.25rem; color:#991b1b; font-size:.875rem; margin:.5rem 0; }
    .validation-card { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:1.1rem 1.25rem; margin-bottom:1rem; box-shadow:0 4px 14px rgba(15,23,42,.03); }
    .validation-row { display:flex; justify-content:space-between; padding:.45rem 0; border-bottom:1px solid #f1f5f9; font-size:.875rem; }
    .validation-row:last-child { border-bottom:none; }
    .validation-key { color:#64748b; font-weight:600; }
    .validation-value { color:#0f172a; font-weight:700; font-family:'Space Grotesk'; }
    .topic-chip { display:inline-block; background:#e0e7ff; border:1px solid #c7d2fe; border-radius:999px; padding:.2rem .6rem; font-size:.75rem; font-weight:600; color:#1e3a8a; margin:.15rem .25rem .15rem 0; }
    .topic-chip.outlier { background:#f1f5f9; border-color:#cbd5e1; color:#475569; }
    .timing-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.25rem; margin-top:1rem; }
    .timing-row { display:flex; justify-content:space-between; padding:.35rem 0; border-bottom:1px solid #f1f5f9; font-size:.85rem; }
    .timing-row:last-child { border-bottom:none; }
    .timing-key { color:#64748b; }
    .timing-value { color:#0f172a; font-weight:700; font-family:'Space Grotesk'; }
    </style>
    """,
        unsafe_allow_html=True,
    )


inject_style()


# Bump this string whenever you replace source files, so Streamlit never reuses
# an object that was built from older code.
CACHE_VERSION = "2026-11-12-r6"


@st.cache_resource(show_spinner="Loading NLP models…")
def _build_pipeline(version: str) -> FeedbackInsightsPipeline:
    return FeedbackInsightsPipeline()


def get_pipeline() -> FeedbackInsightsPipeline:
    """Return the cached pipeline, rebuilding it if the cached object is stale."""
    pipeline = _build_pipeline(CACHE_VERSION)
    if not hasattr(pipeline, "analyze_with_timing"):
        _build_pipeline.clear()
        pipeline = _build_pipeline(CACHE_VERSION)
    return pipeline


def analyze_with_progress(
    texts: list[str],
    action: str,
    ids: list | None = None,
    chunk_size: int = 100,
) -> tuple[pd.DataFrame, dict]:
    """Run inference in chunks with a real progress bar, ETA and aggregated timings."""
    pipeline = get_pipeline()
    total = len(texts)
    totals = {"sentiment_time": 0.0, "ner_time": 0.0, "keyword_time": 0.0, "analyzed_records": 0, "failures": 0}
    frames: list[pd.DataFrame] = []

    with st.status(f"Starting {action}…", expanded=True) as status:
        bar = st.progress(0.0, text="Loading models… the first run can take a minute.")
        started = time.time()
        for start in range(0, total, chunk_size):
            chunk = texts[start : start + chunk_size]
            chunk_ids = ids[start : start + chunk_size] if ids is not None else None
            if hasattr(pipeline, "analyze_with_timing"):
                part, part_timing = pipeline.analyze_with_timing(chunk, ids=chunk_ids)
            else:  # stale pipeline class: fall back to analyze_texts
                t0 = time.time()
                part = pipeline.analyze_texts(chunk, ids=chunk_ids)
                part_timing = {"analyzed_records": len(part)}
            frames.append(part)
            for key in totals:
                totals[key] += part_timing.get(key, 0)
            done = min(start + chunk_size, total)
            elapsed = time.time() - started
            remaining = elapsed / done * (total - done)
            bar.progress(done / total, text=f"Analyzed {done:,} / {total:,} · about {remaining:.0f}s remaining")
        results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        overall = time.time() - started
        status.update(label=f"Analysis complete in {overall:.1f}s", state="complete", expanded=False)

    if len(results) and "aspects" not in results.columns:
        results["aspects"] = results["text"].map(lambda t: ", ".join(aspect_names(t)))
    timings = {
        "total_records": total,
        "analyzed_records": int(totals["analyzed_records"]) or len(results),
        "sentiment_time": totals["sentiment_time"],
        "ner_time": totals["ner_time"],
        "keyword_time": totals["keyword_time"],
        "overall_time": overall,
        "records_per_second": round(len(results) / overall, 1) if overall else 0.0,
        "failures": int(totals["failures"]),
    }
    return results, timings


def merge_mentions(entities: list[dict], aspects: list[dict]) -> list[dict]:
    """Combine spaCy entities with aspect mentions; entities win on overlap."""
    merged = [dict(e, kind="entity") for e in (entities or [])]
    for aspect in aspects:
        if not any(aspect["start"] < e["end"] and e["start"] < aspect["end"] for e in merged):
            merged.append(dict(aspect, kind="aspect"))
    return sorted(merged, key=lambda m: m["start"])


def highlight_mentions(text: str, mentions: list[dict]) -> str:
    """Tag entities (blue) and business aspects (amber) inside the text."""
    if not mentions:
        return html.escape(text)
    pieces: list[str] = []
    cursor = 0
    for mention in sorted(mentions, key=lambda item: item.get("start", 0)):
        start, end = int(mention.get("start", 0)), int(mention.get("end", 0))
        if start < cursor:
            continue
        css = "aspect" if mention.get("kind") == "aspect" else "entity"
        label = html.escape(str(mention.get("label", "ENTITY")))
        pieces.append(html.escape(text[cursor:start]))
        pieces.append(f'<mark class="{css}">{html.escape(text[start:end])}<small>{label}</small></mark>')
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _log_odds(probabilities: dict) -> float:
    """log(P(positive) / P(negative)); stays informative when the model is ~100% sure."""
    positive = max(float(probabilities.get("positive", 0.0)), 1e-9)
    negative = max(float(probabilities.get("negative", 0.0)), 1e-9)
    return math.log(positive) - math.log(negative)


def explain_sentiment(classifier, text: str, max_words: int = 60) -> dict:
    """Occlusion-based word influence using only classifier.predict().

    Each word is removed in turn and the change in log-odds is measured.
    A positive value means the word pushes the message toward positive, a
    negative value toward negative. Log-odds are used instead of raw
    probabilities because a 99.7% prediction barely moves in probability
    when one word is removed, which would make every word look unimportant.
    """
    words = str(text).split()[:max_words]
    if not words:
        return {"label": None, "contributions": []}
    base = classifier.predict([" ".join(words)])[0]
    base_score = _log_odds(base.probabilities)
    variants = [" ".join(words[:i] + words[i + 1 :]) or "." for i in range(len(words))]
    preds = classifier.predict(variants)
    contributions = [(word, base_score - _log_odds(pred.probabilities)) for word, pred in zip(words, preds)]
    return {"label": base.label, "contributions": contributions}


def highlight_reasoning(contributions: list) -> str:
    """Green = pushes toward positive, red = toward negative; darker = stronger."""
    if not contributions:
        return ""
    peak = max(abs(delta) for _, delta in contributions) or 1.0
    pieces: list[str] = []
    for word, delta in contributions:
        safe = html.escape(word)
        relative = abs(delta) / peak
        if abs(delta) < 0.25 or relative < 0.15:
            pieces.append(safe)
            continue
        rgb = "15,118,110" if delta > 0 else "220,38,38"
        alpha = 0.18 + 0.62 * relative
        pieces.append(
            f'<mark style="background:rgba({rgb},{alpha:.2f});padding:.15rem .3rem;border-radius:.35rem;">{safe}</mark>'
        )
    return " ".join(pieces)


def top_drivers(contributions: list, n: int = 3, min_effect: float = 0.25) -> tuple[list[str], list[str]]:
    """Return (words pushing positive, words pushing negative), strongest first."""
    clean = lambda word: word.strip(".,!?;:\"'()")
    positive = sorted([c for c in contributions if c[1] >= min_effect], key=lambda c: -c[1])[:n]
    negative = sorted([c for c in contributions if c[1] <= -min_effect], key=lambda c: c[1])[:n]
    return [clean(w) for w, _ in positive], [clean(w) for w, _ in negative]


def parse_upload(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded file into a raw DataFrame, storing metadata in attrs."""
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        path = handle.name
    return load_feedback(path, interactive=True)


def dataset_upload_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 over ordered (feedback_id, text) pairs for upload-level caching."""
    pairs = sorted((int(fid), str(t)) for fid, t in zip(frame["feedback_id"], frame["text"]))
    payload = json.dumps({"pairs": pairs}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark">◈ SignalDesk</div>'
            '<div class="brand-sub">Customer Feedback Intelligence Platform</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**Decision support**")
        st.caption("Convert unstructured feedback into prioritized signals for product, support, and operations teams.")
        st.markdown("**Model stack**")
        st.caption("DistilBERT sentiment · spaCy NER · aspect-based sentiment · BERTopic topics · word-level explanations")
        st.divider()
        st.markdown("**Product note**")
        st.caption("Use the scan to move from recurring customer friction to a prioritized conversation with the owning team.")
        with st.expander("Diagnostics", expanded=False):
            try:
                import sys
                import spacy.util
                from src import pipeline as _p, sentiment as _s, topics as _t
                st.caption(f"code version: {CACHE_VERSION}")
                st.caption(f"python: {sys.executable}")
                st.caption(f"pipeline.py: {getattr(_p, '__file__', 'n/a')}")
                st.caption(f"sentiment.py: {getattr(_s, '__file__', 'n/a')}")
                st.caption(f"topics.py: {getattr(_t, '__file__', 'n/a')}")
                st.caption(f"spaCy model en_core_web_sm installed: {spacy.util.is_package('en_core_web_sm')}")
            except Exception as exc:  # diagnostics must never break the app
                st.caption(f"diagnostics unavailable: {exc}")


def build_insights(results: pd.DataFrame) -> list[str]:
    if results is None or len(results) == 0:
        return ["<strong>No analyzable feedback records were found.</strong> Upload a file with non-empty text, or load the sample dataset."]
    negative = int((results.sentiment == "negative").sum())
    total = len(results)
    insights = [f"<strong>{negative / total:.0%} of analyzed feedback is negative.</strong> Treat this as the first triage signal, then use the enriched records below to identify the underlying customer friction."]
    high_conf = int((results["sentiment_confidence"] >= .9).sum())
    insights.append(f"<strong>{high_conf:,} records have high-confidence predictions.</strong> Confidence can support triage, but should be validated on labeled support data before automating escalation.")
    return insights


def normalize_uploaded_frame(raw_frame: pd.DataFrame, detected_column: str | None, source_label: str) -> pd.DataFrame | None:
    """Normalize an uploaded raw frame into an analysis-ready DataFrame.

    Returns None if normalization cannot proceed (e.g., no text column detected
    and user hasn't selected one yet).
    """
    meta = raw_frame.attrs or {}
    needs_selection = bool(meta.get("needs_column_selection", False))
    available_columns = meta.get("available_columns") or list(raw_frame.columns)

    if needs_selection or not detected_column:
        st.warning(
            "No feedback text column could be detected automatically. "
            "Please select the column containing the customer feedback text."
        )
        selected_column = st.selectbox(
            "Select the feedback text column",
            options=available_columns,
            index=0,
            key="text_column_selector",
        )
        if st.button("Use selected column", type="primary", key="use_selected_column"):
            try:
                frame = normalize_feedback_frame(raw_frame, selected_column)
                st.session_state["uploaded_frame"] = frame
                st.session_state["uploaded_selected_column"] = selected_column
                st.session_state["uploaded_source"] = source_label
                st.session_state.pop("batch_results", None)
                st.session_state.pop("batch_timings", None)
                st.session_state.pop("batch_fingerprint", None)
            except Exception as exc:
                st.error(f"Could not normalize the selected column: {exc}")
                return None
        if "uploaded_frame" not in st.session_state:
            return None
        return st.session_state["uploaded_frame"]

    # Already normalized or auto-detected column available.
    if ("uploaded_frame" in st.session_state
            and st.session_state.get("uploaded_source") == source_label
            and st.session_state.get("uploaded_selected_column") == detected_column):
        return st.session_state["uploaded_frame"]

    try:
        frame = normalize_feedback_frame(raw_frame, detected_column)
        st.session_state["uploaded_frame"] = frame
        st.session_state["uploaded_selected_column"] = detected_column
        st.session_state["uploaded_source"] = source_label
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_timings", None)
        st.session_state.pop("batch_fingerprint", None)
        return frame
    except Exception as exc:
        st.error(f"Could not normalize the detected column: {exc}")
        return None


def render_upload_section() -> tuple[pd.DataFrame, dict] | None:
    """Overview upload section — the main entry point for the application."""
    st.markdown('<div class="section-label">Upload Customer Feedback</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Get started</h2>', unsafe_allow_html=True)
    st.caption("Upload a customer feedback export (CSV, TSV, JSON, or TXT) to begin analysis.")

    uploaded = st.file_uploader(
        "Drop a feedback export",
        type=["csv", "tsv", "json", "txt", "text"],
        label_visibility="collapsed",
        key="overview_file_uploader",
    )
    sample = st.button("Load sample dataset", type="secondary", key="overview_sample_button")
    if uploaded is None and not sample:
        st.info("Start with the included sample dataset or upload your own feedback export.")
        return None

    try:
        if uploaded is not None:
            if ("uploaded_raw" not in st.session_state
                    or st.session_state.get("uploaded_name") != uploaded.name):
                raw_frame = parse_upload(uploaded)
                st.session_state["uploaded_raw"] = raw_frame
                st.session_state["uploaded_name"] = uploaded.name
                st.session_state.pop("uploaded_frame", None)
                st.session_state.pop("uploaded_selected_column", None)
                st.session_state.pop("batch_results", None)
                st.session_state.pop("batch_timings", None)
                st.session_state.pop("batch_fingerprint", None)
            raw_frame = st.session_state["uploaded_raw"]
            source_label = uploaded.name
        else:
            raw_frame = load_feedback(Path(__file__).parent / "data/sample/customer_feedback.csv")
            source_label = "customer_feedback.csv"
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        return None

    meta = raw_frame.attrs or {}
    original_count = int(meta.get("original_count", len(raw_frame)))
    parser_skipped = int(meta.get("parser_skipped", 0))
    detected_column = meta.get("detected_column")
    needs_selection = bool(meta.get("needs_column_selection", False))

    # File preview.
    with st.expander("File preview (first 5 rows)", expanded=False):
        st.dataframe(raw_frame.head(), use_container_width=True, hide_index=True)

    # Normalize the frame.
    frame = normalize_uploaded_frame(raw_frame, detected_column, source_label)
    if frame is None:
        return None

    meta = frame.attrs or {}
    empty_removed = int(meta.get("empty_removed", 0))
    usable_count = int(meta.get("usable_count", len(frame)))
    detected_column = meta.get("detected_column")

    st.caption(f"Source: **{source_label}**")

    # Dataset validation summary.
    st.markdown('<div class="validation-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Dataset validation</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(f'<div class="validation-row"><span class="validation-key">Original rows</span><span class="validation-value">{original_count:,}</span></div>', unsafe_allow_html=True)
    v2.markdown(f'<div class="validation-row"><span class="validation-key">Parsed rows</span><span class="validation-value">{original_count - parser_skipped:,}</span></div>', unsafe_allow_html=True)
    v3.markdown(f'<div class="validation-row"><span class="validation-key">Text column</span><span class="validation-value">{detected_column}</span></div>', unsafe_allow_html=True)
    v4.markdown(f'<div class="validation-row"><span class="validation-key">Empty removed</span><span class="validation-value">{empty_removed:,}</span></div>', unsafe_allow_html=True)
    v5, v6, v7, _ = st.columns(4)
    v5.markdown(f'<div class="validation-row"><span class="validation-key">Malformed excluded</span><span class="validation-value">{parser_skipped:,} <span style="font-size:.68rem;color:#94a3b8;">(approx.)</span></span></div>', unsafe_allow_html=True)
    v6.markdown(f'<div class="validation-row"><span class="validation-key">Usable records</span><span class="validation-value">{usable_count:,}</span></div>', unsafe_allow_html=True)
    status_text = "OK" if (parser_skipped == 0 and empty_removed == 0 and usable_count > 0) else ("Needs column selection" if needs_selection else "Review exclusions")
    v7.markdown(f'<div class="validation-row"><span class="validation-key">Validation status</span><span class="validation-value">{status_text}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if parser_skipped > 0 or empty_removed > 0:
        st.warning(
            "Not all records could be processed. "
            f"{parser_skipped:,} parser-detected malformed record(s) and "
            f"{empty_removed:,} empty record(s) were excluded. "
            "Customer feedback was never silently discarded."
        )
    if usable_count == 0:
        st.info("This file contained no analyzable feedback records.")
        return None

    # Speed controls: duplicates add no information and slow the run down.
    dup_count = int(frame["text"].duplicated().sum())
    opt_a, opt_b = st.columns(2)
    dedupe = opt_a.checkbox(f"Remove duplicate reviews ({dup_count:,} found)", value=True, key="opt_dedupe")
    work = frame.drop_duplicates(subset="text", keep="first") if dedupe else frame
    unique_count = max(len(work), 1)
    if int(st.session_state.get("opt_max_rows", 0)) > unique_count:
        st.session_state["opt_max_rows"] = unique_count
    cap_kwargs = {} if "opt_max_rows" in st.session_state else {"value": min(500, unique_count)}
    max_rows = opt_b.number_input(
        "Maximum reviews to analyze",
        min_value=1,
        max_value=unique_count,
        step=50,
        key="opt_max_rows",
        help="Recommended: 500-1,000 for a live demo; 2,000 at most on a laptop CPU.",
        **cap_kwargs,
    )
    if len(work) > int(max_rows):
        work = work.sample(n=int(max_rows), random_state=42).sort_values("feedback_id")
    work = work.reset_index(drop=True)
    removed = usable_count - len(frame.drop_duplicates(subset="text")) if dedupe else 0
    st.caption(
        f"Will analyze **{len(work):,}** of {usable_count:,} usable reviews "
        f"({removed:,} duplicates removed). Keep this at 500-1,000 for a smooth demo; 2,000 is the practical maximum on CPU."
    )
    if len(work) > 2000:
        st.warning("This is a large run and may take many minutes on a CPU. Consider lowering the maximum.")

    run_scan = st.button("Analyze Feedback", type="primary", use_container_width=True, key="overview_analyze_button")
    if sample or run_scan:
        fp = dataset_upload_fingerprint(work)
        if st.session_state.get("batch_fingerprint") != fp or st.session_state.get("batch_results") is None:
            results, timings = analyze_with_progress(
                work["text"].tolist(),
                "feedback analysis",
                ids=work["feedback_id"].tolist(),
            )
            st.session_state["batch_results"] = results
            st.session_state["batch_timings"] = timings
            st.session_state["batch_fingerprint"] = fp
            st.session_state["batch_source"] = source_label
        # Rerun so every tab (Overview dashboard, Feedback Analysis, Topic Intelligence)
        # sees the stored results immediately instead of on the next interaction.
        st.rerun()

    return None


def render_overview() -> None:
    """Overview — main entry point. Upload form when no data; executive dashboard after analysis."""
    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)

    results = st.session_state.get("batch_results")
    timings = st.session_state.get("batch_timings")

    if results is None or len(results) == 0:
        # Pre-analysis state: show upload section
        ret = render_upload_section()
        if ret is not None:
            results, timings = ret
            st.session_state["batch_results"] = results
            st.session_state["batch_timings"] = timings
    else:
        # Post-analysis state: executive dashboard
        st.markdown('<h2 class="section-title">Executive Dashboard</h2>', unsafe_allow_html=True)
        if st.button("Analyze a different dataset", key="reset_analysis_button"):
            for k in ("batch_results", "batch_timings", "batch_fingerprint"):
                st.session_state.pop(k, None)
            st.rerun()
        negative = int((results.sentiment == "negative").sum())
        positive = int((results.sentiment == "positive").sum())
        high_conf = int((results["sentiment_confidence"] >= .9).sum())

        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, note in [
            (k1, "Feedback analyzed", f"{len(results):,}", "usable records"),
            (k2, "Negative signal", f"{negative / len(results):.0%}", f"{negative:,} records"),
            (k3, "Positive signal", f"{positive / len(results):.0%}", f"{positive:,} records"),
            (k4, "High-confidence", f"{high_conf / len(results):.0%}", "confidence ≥ 90%"),
        ]:
            col.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label">What the model is telling us</div>', unsafe_allow_html=True)
        for message in build_insights(results):
            st.markdown(f'<div class="insight">{message}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Sentiment mix</div>', unsafe_allow_html=True)
        counts = results["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        fig = px.pie(
            counts, names="sentiment", values="count", hole=.68,
            color="sentiment", color_discrete_map={"positive":"#0f766e","negative":"#dc2626"},
        )
        fig.update_layout(
            title="Sentiment mix", margin=dict(t=55,b=10,l=10,r=10),
            legend=dict(orientation="h",y=-.05), paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textinfo="percent", textfont_size=14)
        st.plotly_chart(fig, use_container_width=True, key="chart_overview_sentiment_pie")

        # Timing report
        if timings:
            st.markdown('<div class="section-label">Processing performance</div>', unsafe_allow_html=True)
            st.markdown('<div class="timing-card">', unsafe_allow_html=True)
            t1, t2, t3, t4 = st.columns(4)
            t1.markdown(f'<div class="timing-row"><span class="timing-key">Total records</span><span class="timing-value">{timings.get("total_records", 0):,}</span></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="timing-row"><span class="timing-key">Analyzed</span><span class="timing-value">{timings.get("analyzed_records", 0):,}</span></div>', unsafe_allow_html=True)
            t3.markdown(f'<div class="timing-row"><span class="timing-key">Sentiment time</span><span class="timing-value">{timings.get("sentiment_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t4.markdown(f'<div class="timing-row"><span class="timing-key">NER time</span><span class="timing-value">{timings.get("ner_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t5, t6, t7, _ = st.columns(4)
            t5.markdown(f'<div class="timing-row"><span class="timing-key">Keyword time</span><span class="timing-value">{timings.get("keyword_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t6.markdown(f'<div class="timing-row"><span class="timing-key">Overall time</span><span class="timing-value">{timings.get("overall_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t7.markdown(f'<div class="timing-row"><span class="timing-key">Records/sec</span><span class="timing-value">{timings.get("records_per_second", 0):,.1f}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def render_feedback_analysis() -> None:
    """Feedback Analysis — detailed records table."""
    st.markdown('<div class="section-label">Feedback Analysis</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Detailed Feedback Records</h2>', unsafe_allow_html=True)
    st.caption("Review individual customer reviews with sentiment, confidence, and extracted entities.")

    results = st.session_state.get("batch_results")
    if results is None or len(results) == 0:
        st.markdown(
            '<div class="empty-state"><h3>No feedback analyzed yet</h3>'
            '<p>Upload and analyze a dataset from the Overview page to see detailed records.</p></div>',
            unsafe_allow_html=True,
        )
        return

    # Search and filters
    search = st.text_input("Search feedback text", key="feedback_search_input", placeholder="Search reviews…")
    sentiment_filter = st.selectbox(
        "Filter by sentiment",
        options=["All", "positive", "negative"],
        index=0,
        key="feedback_sentiment_filter",
    )

    filtered = results
    if search:
        filtered = filtered[filtered["text"].str.contains(search, case=False, na=False)]
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    st.caption(f"Showing {len(filtered):,} of {len(results):,} records")

    shown = filtered.copy()
    if "aspects" not in shown.columns:
        shown["aspects"] = shown["text"].map(lambda t: ", ".join(aspect_names(t)))
    display = shown[["text", "sentiment", "sentiment_confidence", "entity_text", "aspects", "keywords"]].copy()
    display["sentiment_confidence"] = display["sentiment_confidence"] * 100
    display["keywords"] = display["keywords"].apply(
        lambda k: ", ".join(map(str, k)) if isinstance(k, (list, tuple)) else str(k)
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "sentiment_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f%%"),
            "entity_text": "Named entities (spaCy)",
            "aspects": "Business aspects",
        },
    )
    st.caption(
        "Named entities are people, organizations, places, dates and similar items, so this column is empty for reviews "
        "that name none. Business aspects (delivery, price, support, app...) show what each review is about."
    )

    st.download_button(
        "Download enriched CSV",
        results.to_csv(index=False),
        "signaldesk_enriched_feedback.csv",
        "text/csv",
        key="feedback_download_button",
    )


def render_playground() -> None:
    """Quick Feedback Analysis — single message analysis."""
    st.markdown('<div class="section-label">Quick Feedback Analysis</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Inspect one customer message at a time</h2>', unsafe_allow_html=True)
    st.caption("Use this surface to demonstrate how a raw support message becomes a structured, explainable prediction.")
    text = st.text_area("Customer feedback", "The mobile app crashes when I try to pay, but support resolved my issue quickly.", height=145, label_visibility="collapsed", key="playground_text_area")
    if st.button("Analyze feedback", type="primary", key="playground_analyze_button") and text.strip():
        try:
            results, _ = analyze_with_progress([text], "live feedback analysis")
            result = results.iloc[0]
            classifier = get_pipeline().sentiment
            explanation = explain_sentiment(classifier, text)
            mentions = merge_mentions(result["entities"], find_aspect_mentions(text))
            positive_words, negative_words = top_drivers(explanation["contributions"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted sentiment", str(result["sentiment"]).title())
            c2.metric("Model confidence", f"{result['sentiment_confidence']:.1%}")
            c3.metric("Entities & aspects", len(mentions))
            left, right = st.columns([1.25, .75])
            with left:
                st.markdown('<div class="section-label">Why this prediction? (word influence)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="highlight-box">{highlight_reasoning(explanation["contributions"])}</div>', unsafe_allow_html=True)
                st.markdown('<div class="legend">Green words push toward positive, red toward negative. Darker means stronger influence.</div>', unsafe_allow_html=True)
                if positive_words or negative_words:
                    drivers = []
                    if negative_words:
                        drivers.append(f"<strong>Pushes negative:</strong> {html.escape(', '.join(negative_words))}")
                    if positive_words:
                        drivers.append(f"<strong>Pushes positive:</strong> {html.escape(', '.join(positive_words))}")
                    st.markdown(f'<div class="insight">{"<br>".join(drivers)}</div>', unsafe_allow_html=True)

                clauses = split_clauses(text)
                if len(clauses) > 1:
                    st.markdown('<div class="section-label">Clause breakdown (mixed feedback)</div>', unsafe_allow_html=True)
                    clause_preds = classifier.predict(clauses)
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Clause": clauses,
                                "Sentiment": [p.label.title() for p in clause_preds],
                                "Confidence": [f"{p.confidence:.0%}" for p in clause_preds],
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown('<div class="section-label">Entities and business aspects</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="highlight-box">{highlight_mentions(text, mentions)}</div>', unsafe_allow_html=True)
                if mentions:
                    st.markdown('<div class="legend">Blue tags are named entities (spaCy). Amber tags are business aspects such as delivery, price, support or app.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="legend">No named entities or business aspects were found in this message.</div>', unsafe_allow_html=True)
            with right:
                probs = result["sentiment_probabilities"]
                fig = go.Figure(go.Bar(x=list(probs.values()), y=[key.title() for key in probs], orientation="h", marker_color=["#dc2626" if key == "negative" else "#0f766e" for key in probs]))
                fig.update_layout(title="Confidence profile", xaxis_title="Probability", xaxis_range=[0,1], margin=dict(t=55,b=15,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, key="chart_playground_confidence_bar")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


def render_topic_discovery() -> None:
    st.markdown('<div class="section-label">Topic Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Discover recurring themes in feedback</h2>', unsafe_allow_html=True)
    st.caption("BERTopic clusters similar feedback into topics without manually assigned labels.")
    results = st.session_state.get("batch_results")
    if results is None or len(results) == 0:
        st.info("Run a feedback analysis first to discover topics.")
        return
    if len(results) < 20:
        st.markdown(
            '<div class="warning-banner"><strong>Small dataset:</strong> '
            f'{len(results)} records detected. Topic results are exploratory and may not be statistically stable.</div>',
            unsafe_allow_html=True,
        )
    fingerprint = dataset_fingerprint(results)
    cache_key = f"topic_model_{CACHE_VERSION}_{fingerprint}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {"discoverer": TopicDiscoverer(), "fitted": False, "topic_df": None}
    topic_state = st.session_state[cache_key]
    discoverer = topic_state["discoverer"]
    run = st.button("Discover Topic Clusters", type="primary", use_container_width=True, key="topic_discover_button")
    topic_df = topic_state["topic_df"]
    if run:
        if not discoverer.available:
            st.error(
                "BERTopic is not installed. Add the optional dependencies from "
                "requirements-topics.txt (bertopic, sentence-transformers, umap-learn, hdbscan)."
            )
            return
        try:
            with st.status("Fitting topic model…", expanded=True) as status:
                st.write("Downloading the sentence-transformer embedding model on first run.")
                topic_df = discoverer.fit_transform(results["text"].tolist(), results["feedback_id"].tolist())
                st.write("Topics extracted.")
                status.update(label="Topic discovery complete", state="complete", expanded=False)
            topic_state["fitted"] = True
            topic_state["topic_df"] = topic_df
        except Exception as exc:
            st.error(f"Topic discovery failed: {exc}")
            return
    if topic_df is None or len(topic_df) == 0:
        return
    # Merge topic assignments onto results by feedback_id (not raw text)
    merged = results.merge(topic_df[["feedback_id", "topic", "topic_name"]], on="feedback_id", how="left")
    st.markdown('<div class="section-label">Discovered topics</div>', unsafe_allow_html=True)
    summary = discoverer.summary()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-label">Topic keywords</div>', unsafe_allow_html=True)
    if discoverer.is_fitted and len(summary) > 1:
        try:
            fig_bar = discoverer._model.visualize_barchart(top_n_topics=min(10, len(summary)))
            st.plotly_chart(fig_bar, use_container_width=True, key="chart_topic_keywords_barchart")
        except Exception:
            st.info("Topic keyword chart could not be rendered.")
    st.markdown('<div class="section-label">Topic map</div>', unsafe_allow_html=True)
    if discoverer.is_fitted and len(summary) > 1:
        try:
            fig_map = discoverer._model.visualize_topics()
            st.plotly_chart(fig_map, use_container_width=True, key="chart_topic_map")
        except Exception:
            st.info("Topic map could not be rendered.")
    elif discoverer.is_fitted:
        st.info("Topic map requires more than one discovered topic.")
    st.markdown('<div class="section-label">Representative customer reviews</div>', unsafe_allow_html=True)
    if discoverer.is_fitted:
        for _, row in summary.iterrows():
            tid = int(row["topic"])
            name = row["topic_name"]
            is_outlier = bool(row["is_outlier"])
            chip_cls = "topic-chip outlier" if is_outlier else "topic-chip"
            st.markdown(f'<span class="{chip_cls}">{html.escape(str(name))}</span>', unsafe_allow_html=True)
            reps = discoverer.representative_docs(tid, n=3)
            if reps:
                for rep in reps:
                    st.markdown(f"> {html.escape(str(rep))}")
            else:
                st.caption("No representative documents available for this topic.")
            st.divider()
    st.markdown('<div class="section-label">Topic-wise sentiment distribution</div>', unsafe_allow_html=True)
    if discoverer.is_fitted and "sentiment" in merged.columns:
        grp = merged.groupby(["topic", "topic_name", "sentiment"]).size().unstack(fill_value=0).reset_index()
        for col in ["positive", "negative"]:
            if col not in grp.columns:
                grp[col] = 0
        grp["total"] = grp["positive"] + grp["negative"]
        grp = grp[grp["total"] > 0]
        if len(grp) > 0:
            grp["positive_pct"] = (grp["positive"] / grp["total"] * 100).round(1)
            grp["negative_pct"] = (grp["negative"] / grp["total"] * 100).round(1)
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Bar(x=grp["topic_name"], y=grp["positive_pct"], name="Positive", marker_color="#0f766e"))
            fig_ts.add_trace(go.Bar(x=grp["topic_name"], y=grp["negative_pct"], name="Negative", marker_color="#dc2626"))
            fig_ts.update_layout(
                barmode="stack", title="Sentiment by topic cluster",
                xaxis_title="Topic", yaxis_title="Share of records (%)",
                legend=dict(orientation="h", y=-.15), paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ts, use_container_width=True, key="chart_topic_sentiment_bar")
        else:
            st.info("No sentiment data available for topic-wise distribution.")
    else:
        st.info("Run topic discovery to view sentiment by topic.")


def show_error(label: str, exc: Exception) -> None:
    """Show a friendly error plus the full traceback so the real cause is never hidden."""
    st.error(f"{label} error: {exc}")
    with st.expander("Technical details (send this if the error persists)"):
        st.code(traceback.format_exc())


def render_aspect_insights() -> None:
    """Aspect-based sentiment: what customers are happy or unhappy about."""
    st.markdown('<div class="section-label">Aspect Insights</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">What customers are happy or unhappy about</h2>', unsafe_allow_html=True)
    st.caption(
        "Each review is split into clauses, business aspects are detected, and every aspect is scored "
        "by the clause that mentions it, so one mixed review can be positive about one thing and negative about another."
    )
    results = st.session_state.get("batch_results")
    if results is None or len(results) == 0:
        st.info("Run a feedback analysis from the Overview tab first, then come back here.")
        return

    cache_key = f"aspect_df_{CACHE_VERSION}_{st.session_state.get('batch_fingerprint')}"
    if st.button("Run aspect analysis", type="primary", use_container_width=True, key="aspect_run_button"):
        with st.spinner("Scoring aspects clause by clause…"):
            st.session_state[cache_key] = analyze_aspects(get_pipeline().sentiment, results)
    aspect_df = st.session_state.get(cache_key)
    if aspect_df is None:
        st.markdown(
            '<div class="empty-state"><h3>Aspects not analyzed yet</h3>'
            '<p>Click <strong>Run aspect analysis</strong> to score delivery, price, support, app and other aspects.</p></div>',
            unsafe_allow_html=True,
        )
        return
    if aspect_df.empty:
        st.info("No business aspects were found in these reviews. The keyword lexicon is in src/aspects.py and can be extended.")
        return

    summary = summarize_aspects(aspect_df)
    with_aspect = int(aspect_df["feedback_id"].nunique())
    top = summary.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, note in [
        (k1, "Reviews analyzed", f"{len(results):,}", "from the Overview run"),
        (k2, "Mention an aspect", f"{with_aspect / len(results):.0%}", f"{with_aspect:,} reviews"),
        (k3, "Aspect mentions", f"{len(aspect_df):,}", f"{len(summary)} different aspects"),
        (k4, "Most discussed", html.escape(str(top["aspect"]).title()), f"{int(top['mentions']):,} mentions"),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="font-size:1.35rem">{value}</div><div class="kpi-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

    rated = summary[summary["mentions"] >= 3]
    if rated.empty:
        rated = summary
    worst = rated.sort_values(["negative_share", "mentions"], ascending=False).iloc[0]
    best = rated.sort_values(["positive_share", "mentions"], ascending=False).iloc[0]
    st.markdown('<div class="section-label">What the aspects are telling us</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight"><strong>Biggest pain point: {html.escape(str(worst["aspect"]))}.</strong> '
        f'{worst["negative_share"]:.0%} of its {int(worst["mentions"])} mentions are negative.</div>'
        f'<div class="insight"><strong>Strongest asset: {html.escape(str(best["aspect"]))}.</strong> '
        f'{best["positive_share"]:.0%} of its {int(best["mentions"])} mentions are positive.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">One-slide business insight</div>', unsafe_allow_html=True)
    insight = build_insight(summary, aspect_df, len(results))
    if insight:
        slide_html = render_slide_html(insight)
        try:
            import streamlit.components.v1 as components

            components.html(slide_html, height=640, scrolling=False)
        except Exception:
            st.caption("Preview unavailable here; download the slide below.")
        st.download_button(
            "Download the insight slide (HTML)",
            slide_html,
            "business_insight_slide.html",
            "text/html",
            key="insight_slide_download",
        )
        st.caption("Open the file in a browser, then screenshot it or use Print > Save as PDF to place it in your presentation.")

    st.markdown('<div class="section-label">Sentiment by aspect</div>', unsafe_allow_html=True)
    long = summary.melt(id_vars=["aspect"], value_vars=["positive", "negative", "neutral"], var_name="sentiment", value_name="count")
    fig = px.bar(
        long, x="aspect", y="count", color="sentiment", barmode="group",
        category_orders={"aspect": summary["aspect"].tolist()},
        color_discrete_map={"positive": "#0f766e", "negative": "#dc2626", "neutral": "#94a3b8"},
    )
    fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=-.25), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key="chart_aspect_grouped")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-label">Top negative aspects (share of mentions)</div>', unsafe_allow_html=True)
        neg = rated.sort_values("negative_share", ascending=False).head(6).sort_values("negative_share")
        fig_neg = go.Figure(go.Bar(x=neg["negative_share"] * 100, y=neg["aspect"], orientation="h", marker_color="#dc2626"))
        fig_neg.update_layout(xaxis_title="% negative", margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_neg, use_container_width=True, key="chart_aspect_negative")
    with right:
        st.markdown('<div class="section-label">Top positive aspects (share of mentions)</div>', unsafe_allow_html=True)
        pos = rated.sort_values("positive_share", ascending=False).head(6).sort_values("positive_share")
        fig_pos = go.Figure(go.Bar(x=pos["positive_share"] * 100, y=pos["aspect"], orientation="h", marker_color="#0f766e"))
        fig_pos.update_layout(xaxis_title="% positive", margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pos, use_container_width=True, key="chart_aspect_positive")
    st.caption("Aspects with fewer than 3 mentions are left out of these rankings when enough others exist.")

    st.markdown('<div class="section-label">Evidence: the clause behind each prediction</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    chosen_aspect = f1.selectbox("Aspect", ["All"] + summary["aspect"].tolist(), key="aspect_filter_aspect")
    chosen_sentiment = f2.selectbox("Sentiment", ["All", "negative", "positive", "neutral"], key="aspect_filter_sentiment")
    view = aspect_df
    if chosen_aspect != "All":
        view = view[view["aspect"] == chosen_aspect]
    if chosen_sentiment != "All":
        view = view[view["aspect_sentiment"] == chosen_sentiment]
    st.caption(f"Showing {min(len(view), 200):,} of {len(view):,} aspect mentions")
    st.dataframe(
        view[["aspect", "aspect_sentiment", "confidence", "clause", "review_text"]].head(200),
        use_container_width=True,
        hide_index=True,
        column_config={"confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f")},
    )
    st.download_button(
        "Download aspect results (CSV)",
        aspect_df.to_csv(index=False),
        "aspect_results.csv",
        "text/csv",
        key="aspect_download_button",
    )

    with st.expander("Validate against your own labels (for the evaluation section of your report)"):
        st.caption(
            "1) Download the sample below. 2) Fill the true_sentiment column by hand with positive, negative or neutral. "
            "3) Upload the file to get accuracy, precision, recall, F1 and a confusion matrix."
        )
        st.download_button(
            "Download labelling template (40 rows)",
            validation_template(aspect_df).to_csv(index=False),
            "aspect_validation_template.csv",
            "text/csv",
            key="aspect_template_button",
        )
        labeled_file = st.file_uploader("Upload the labelled CSV", type=["csv"], key="aspect_labeled_upload")
        if labeled_file is not None:
            try:
                metrics = evaluate_labels(pd.read_csv(labeled_file))
            except Exception as exc:
                st.error(f"Could not evaluate this file: {exc}")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Labelled rows", metrics["n"])
                m2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                m3.metric("Macro F1", f"{metrics['macro_f1']:.2f}")
                st.dataframe(metrics["report"], hide_index=True, use_container_width=True)
                fig_cm = px.imshow(metrics["confusion"], text_auto=True, color_continuous_scale="Blues", aspect="auto")
                fig_cm.update_layout(title="Confusion matrix", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm, use_container_width=True, key="chart_aspect_confusion")


def render_model_methodology() -> None:
    st.markdown('<div class="section-label">Model & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">How SignalDesk works</h2>', unsafe_allow_html=True)

    def box(title: str, body: str) -> None:
        st.markdown(f'<div class="insight"><strong>{title}</strong> {body}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)
    box(
        "Flow:",
        "Upload → clean and de-duplicate → overall sentiment → entities and keywords → aspect detection → "
        "clause-level aspect sentiment → topic clustering → dashboard and CSV export.",
    )
    st.markdown('<div class="section-label">Models and methods</div>', unsafe_allow_html=True)
    box(
        "Overall sentiment:",
        "DistilBERT fine-tuned on SST-2 (distilbert-base-uncased-finetuned-sst-2-english), run in length-sorted batches on CPU or GPU. "
        "It is a two-class model (positive/negative) trained on movie-review sentences, so scores on product or support text are "
        "indicative, not guaranteed. A confidence of 99% means the model is sure, not that it is right.",
    )
    box(
        "Word influence (\"why this prediction?\"):",
        "Occlusion. Each word is removed in turn and the change in the model's log-odds is measured. Words that push toward the "
        "predicted class are shaded; green pushes positive, red pushes negative, darker is stronger. Log-odds are used because "
        "probabilities saturate near 100% and would hide every word.",
    )
    box(
        "Named entities:",
        "spaCy NER (en_core_web_sm) finds people, organizations, places, dates and similar named items, with a regex fallback if the "
        "model is missing. Most reviews contain none, so an empty entity column is normal.",
    )
    box(
        "Business aspects and aspect-based sentiment (ABSA):",
        "A keyword lexicon (src/aspects.py) detects 12 aspects such as delivery, price, support, app and refunds. Each review is split "
        "into clauses at sentence ends and contrast words (but, however, although), and a clause that mixes aspects is split further. "
        "Each aspect is scored by the clause that mentions it using the same sentiment model, so no extra training is needed. "
        "Because the model is two-class, predictions under 65% confidence are reported as neutral (a heuristic, not a trained class).",
    )
    box(
        "Topic discovery:",
        "Sentence-transformer embeddings (all-MiniLM-L6-v2) are reduced with UMAP and clustered with HDBSCAN through BERTopic. Topic labels come "
        "from c-TF-IDF over unigrams and bigrams with a stop-word list that keeps negations such as not and never. UMAP and HDBSCAN settings adapt "
        "to dataset size, topic -1 means unclustered outliers, and at least 10 unique reviews are required.",
    )
    st.markdown('<div class="section-label">Data handling</div>', unsafe_allow_html=True)
    box(
        "Input rules:",
        "CSV, TSV, JSON and TXT are accepted. Empty rows are removed and counted, exact duplicates can be removed, and the number of analyzed "
        "reviews can be capped. For a laptop CPU, 500-1,000 reviews is comfortable and about 2,000 is the practical maximum.",
    )
    st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
    box(
        "How accuracy is measured:",
        "Overall sentiment and aspect results are not scored against ground truth automatically. In Aspect Insights you can download a "
        "40-row sample, label it by hand, and upload it to get accuracy, precision, recall, F1 and a confusion matrix. "
        "Any accuracy figure should be quoted with that sample size.",
    )
    st.markdown('<div class="section-label">Known limitations</div>', unsafe_allow_html=True)
    box(
        "Limits to state openly:",
        "Sentiment is two-class and trained on a different domain. Aspect detection is keyword-based, so unlisted wording is missed and a keyword "
        "in an unrelated sense can be matched. Sarcasm and negation spanning clauses can be misread. Two aspects joined without a comma, "
        "\"and\" or contrast word share one score. Topic results on small or repetitive data are exploratory.",
    )


render_sidebar()
st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">Unstructured feedback → prioritized decisions</div>'
    '<h1>Customer Feedback Intelligence Platform</h1>'
    '<p>SignalDesk converts reviews and support tickets into sentiment, entities, recurring themes, and an evidence trail for the next product decision.</p>'
    '<span class="hero-badge">● Live NLP workspace · Hugging Face + spaCy + Plotly</span>'
    '</div>',
    unsafe_allow_html=True,
)

# Navigation — Overview is first and handles both upload and results
overview_tab, batch_tab, aspect_tab, playground_tab, topic_tab, method_tab = st.tabs(
    ["◈ Overview", "📥 Feedback Analysis", "◎ Aspect Insights", "⌁ Quick Feedback Analysis", "⌂ Topic Intelligence", "📘 Model & Methodology"]
)

# Render each tab independently — errors in one tab do not affect others
with overview_tab:
    try:
        render_overview()
    except Exception as exc:
        show_error("Overview", exc)

with batch_tab:
    try:
        render_feedback_analysis()
    except Exception as exc:
        show_error("Feedback Analysis", exc)

with aspect_tab:
    try:
        render_aspect_insights()
    except Exception as exc:
        show_error("Aspect Insights", exc)

with playground_tab:
    try:
        render_playground()
    except Exception as exc:
        show_error("Quick Feedback Analysis", exc)

with topic_tab:
    try:
        render_topic_discovery()
    except Exception as exc:
        show_error("Topic Intelligence", exc)

with method_tab:
    try:
        render_model_methodology()
    except Exception as exc:
        show_error("Model & Methodology", exc)
