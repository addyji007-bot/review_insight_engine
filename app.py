"""AspectLens: Streamlit app that turns customer reviews into aspect-level insight."""

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
    page_title="AspectLens | See what customers really mean",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
    :root {
        --ink:#1c1033; --muted:#6b6485; --soft:#9a94b0;
        --violet:#6d28d9; --violet-d:#4c1d95; --pink:#c026d3; --amber:#f59e0b;
        --line:#e7e1f5; --tint:#f7f4ff; --pos:#059669; --neg:#e11d48; --neu:#a8a29e;
    }
    html, body, [class*="css"] { font-family:'Manrope', 'Segoe UI', sans-serif; }
    .stApp { background:linear-gradient(180deg,#faf8ff 0%,#ffffff 420px); }
    h1, h2, h3 { font-family:'Fraunces', Georgia, serif !important; color:var(--ink); letter-spacing:-.01em; }
    .block-container { padding:3.6rem 2.6rem 3rem; max-width:1400px; }

    /* ---------- sidebar: light lavender ---------- */
    [data-testid="stSidebar"] { background:#f3eeff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:#3b2d63 !important; }
    .brand { display:flex; align-items:center; gap:.65rem; }
    .brand svg { flex:none; }
    .brand-name { font-family:'Fraunces',Georgia,serif; font-size:1.35rem; font-weight:700; color:var(--violet-d) !important; line-height:1; }
    .brand-tag { font-size:.68rem; color:var(--muted) !important; margin-top:.2rem; letter-spacing:.02em; }
    .side-h { font-size:.66rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--violet) !important; margin:.9rem 0 .25rem; display:block; }
    .side-step { display:flex; gap:.55rem; align-items:flex-start; margin:.35rem 0; font-size:.8rem; line-height:1.35; }
    .side-step b { flex:none; width:1.35rem; height:1.35rem; border-radius:50%; background:var(--violet); color:#fff !important; font-size:.68rem; display:flex; align-items:center; justify-content:center; }

    /* ---------- hero ---------- */
    .hero { background:linear-gradient(120deg,#2e1065 0%,#5b21b6 58%,#c026d3 125%); border-radius:26px; padding:2.2rem 2.4rem 2rem; color:#fff; position:relative; overflow:hidden; margin-bottom:1.4rem; box-shadow:0 18px 40px -18px rgba(76,29,149,.55); }
    .hero .lens { position:absolute; right:2.2rem; top:50%; transform:translateY(-50%); width:150px; height:150px; border-radius:50%; border:14px solid rgba(255,255,255,.16); box-shadow:0 0 0 22px rgba(255,255,255,.05), inset 0 0 40px rgba(255,255,255,.12); }
    .hero .lens:after { content:''; position:absolute; width:16px; height:78px; background:rgba(255,255,255,.22); border-radius:10px; right:-26px; bottom:-52px; transform:rotate(-40deg); }
    .eyebrow { letter-spacing:.2em; font-size:.68rem; font-weight:800; color:#f5d0fe; }
    .hero h1 { color:#fff !important; font-size:2.3rem; line-height:1.12; margin:.55rem 0 .6rem; max-width:640px; }
    .hero h1 em { font-style:italic; color:#fde68a; }
    .hero p { color:#e9ddff; margin:0; max-width:620px; font-size:.95rem; line-height:1.55; }
    .chips { margin-top:1.1rem; display:flex; flex-wrap:wrap; gap:.5rem; }
    .chip { padding:.32rem .8rem; border-radius:999px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28); color:#fff; font-size:.74rem; font-weight:600; }

    /* ---------- pill tabs ---------- */
    [role="tablist"], [data-baseweb="tab-list"] { gap:.35rem !important; background:var(--tint) !important; padding:.35rem !important; border-radius:999px !important; border:1px solid var(--line) !important; width:fit-content; max-width:100%; overflow-x:auto; box-shadow:none !important; }
    [role="tablist"] { border-bottom:1px solid var(--line) !important; }
    [role="tab"], [data-baseweb="tab"] { height:auto !important; padding:.5rem 1.1rem !important; border-radius:999px !important; background:transparent !important; color:var(--muted) !important; font-weight:700 !important; font-size:.84rem !important; white-space:nowrap; }
    [role="tab"] p, [data-baseweb="tab"] p { color:inherit !important; font-weight:700 !important; margin:0 !important; }
    [role="tab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] { background:linear-gradient(120deg,var(--violet-d),var(--violet)) !important; color:#fff !important; box-shadow:0 6px 14px -6px rgba(76,29,149,.6); }
    .react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none !important; }
    [role="tabpanel"], [data-baseweb="tab-panel"] { padding-top:1.1rem; }

    /* ---------- buttons ---------- */
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:linear-gradient(120deg,var(--violet-d),var(--violet) 60%,var(--pink)); border:0; border-radius:12px; font-weight:700; color:#fff; padding:.55rem 1.2rem; box-shadow:0 10px 20px -12px rgba(109,40,217,.8); }
    .stButton > button[kind="primary"]:hover { filter:brightness(1.08); }
    .stButton > button[kind="secondary"], .stDownloadButton > button { border-radius:12px; border:1px solid #d8ccf3; color:var(--violet-d); background:#fff; font-weight:600; }

    /* ---------- headings & cards ---------- */
    .section-label { color:var(--violet); text-transform:uppercase; letter-spacing:.16em; font-size:.66rem; font-weight:800; margin:1.5rem 0 .55rem; display:block; }
    .section-title { font-family:'Fraunces',Georgia,serif; font-size:1.5rem; font-weight:700; color:var(--ink); margin:0 0 .4rem 0; }
    .kpi { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1rem 1.15rem 1rem; min-height:126px; box-shadow:0 12px 26px -20px rgba(76,29,149,.5); position:relative; overflow:hidden; }
    .kpi:before { content:''; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--accent,#6d28d9); }
    .kpi-label { color:var(--muted); font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
    .kpi-value { color:var(--ink); font-family:'Fraunces',Georgia,serif; font-size:2rem; font-weight:700; margin:.2rem 0 .1rem; }
    .kpi-note { color:var(--soft); font-size:.72rem; }
    .insight { background:var(--tint); border:1px solid var(--line); border-left:5px solid var(--pink); border-radius:14px; padding:.95rem 1.15rem; margin:.45rem 0; color:#463a6b; font-size:.86rem; line-height:1.55; }
    .insight strong { color:var(--ink); }
    .highlight-box { padding:1.25rem; border-radius:16px; background:#fff; line-height:2.3; border:1px solid var(--line); font-size:1rem; color:var(--ink); }
    mark.entity { padding:.2rem .42rem; border-radius:.5rem; background:#e0f2fe; color:#075985; display:inline-block; margin:0 2px; }
    mark.entity small { margin-left:.35rem; font-size:.62rem; color:#0369a1; font-weight:800; text-transform:uppercase; }
    mark.aspect { padding:.2rem .42rem; border-radius:.5rem; background:#fef3c7; color:#92400e; display:inline-block; margin:0 2px; }
    mark.aspect small { margin-left:.35rem; font-size:.62rem; color:#b45309; font-weight:800; text-transform:uppercase; }
    .legend { color:var(--muted); font-size:.76rem; margin-top:.35rem; }
    .empty-state { background:var(--tint); border:2px dashed #d3c5f0; border-radius:20px; padding:2.6rem 2rem; text-align:center; color:var(--muted); margin:1rem 0; }
    .empty-state h3 { color:var(--ink); margin:0 0 .5rem 0; }
    .success-banner { background:#ecfdf5; border:1px solid #a7f3d0; border-radius:14px; padding:1rem 1.25rem; color:#065f46; font-size:.875rem; margin:.5rem 0; }
    .warning-banner { background:#fffbeb; border:1px solid #fde68a; border-radius:14px; padding:1rem 1.25rem; color:#713f12; font-size:.875rem; margin:.5rem 0; }
    .error-banner { background:#fff1f2; border:1px solid #fecdd3; border-radius:14px; padding:1rem 1.25rem; color:#9f1239; font-size:.875rem; margin:.5rem 0; }
    .validation-row, .timing-row { display:flex; justify-content:space-between; gap:.6rem; padding:.65rem .9rem; margin-bottom:.5rem; background:#fff; border:1px solid var(--line); border-radius:14px; font-size:.85rem; box-shadow:0 10px 22px -20px rgba(76,29,149,.5); }
    .validation-key, .timing-key { color:var(--muted); font-weight:600; }
    .validation-value, .timing-value { color:var(--ink); font-weight:800; }
    .topic-chip { display:inline-block; background:#f3e8ff; border:1px solid #e9d5ff; border-radius:999px; padding:.22rem .7rem; font-size:.76rem; font-weight:700; color:#6b21a8; margin:.15rem .25rem .15rem 0; }
    .topic-chip.outlier { background:#f5f5f4; border-color:#e7e5e4; color:#57534e; }

    /* ---------- 3-step strip ---------- */
    .steps { display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin:.4rem 0 1.1rem; }
    .step { background:#fff; border:1px solid var(--line); border-radius:16px; padding:.85rem 1rem; display:flex; gap:.75rem; align-items:flex-start; }
    .step .n { flex:none; width:1.8rem; height:1.8rem; border-radius:10px; background:var(--tint); color:var(--violet); font-weight:800; display:flex; align-items:center; justify-content:center; }
    .step b { color:var(--ink); font-size:.86rem; display:block; }
    .step span { color:var(--muted); font-size:.76rem; }
    @media (max-width:800px) { .steps { grid-template-columns:1fr; } .hero .lens { display:none; } .block-container { padding:1rem; } }
    </style>
    """,
        unsafe_allow_html=True,
    )


inject_style()


def kpi_card(label: str, value: str, note: str, accent: str = "#6d28d9", small: bool = False) -> str:
    size = ' style="font-size:1.7rem"' if small else ""
    return (
        f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value"{size}>{value}</div><div class="kpi-note">{note}</div></div>'
    )


# Bump this string whenever you replace source files, so Streamlit never reuses
# an object that was built from older code.
CACHE_VERSION = "2026-11-13-lens1"


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
        rgb = "5,150,105" if delta > 0 else "225,29,72"
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
            '<div class="brand">'
            '<svg width="38" height="38" viewBox="0 0 38 38" fill="none">'
            '<rect width="38" height="38" rx="11" fill="#5b21b6"/>'
            '<circle cx="17" cy="17" r="8" stroke="#fff" stroke-width="3"/>'
            '<path d="M23 23l7 7" stroke="#fde68a" stroke-width="3.5" stroke-linecap="round"/>'
            '<circle cx="17" cy="17" r="2.6" fill="#f0abfc"/></svg>'
            '<div><div class="brand-name">AspectLens</div>'
            '<div class="brand-tag">Review Insight Engine</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="side-h">What it does</span>', unsafe_allow_html=True)
        st.caption("Reads thousands of customer reviews and tells you what people like, what they hate, and why.")
        st.markdown('<span class="side-h">How to use it</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-step"><b>1</b><div>Load reviews or use the sample</div></div>'
            '<div class="side-step"><b>2</b><div>Press <em>Run analysis</em></div></div>'
            '<div class="side-step"><b>3</b><div>Explore the other tabs</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="side-h">Under the hood</span>', unsafe_allow_html=True)
        st.caption("DistilBERT sentiment · spaCy entities · aspect-level scoring · BERTopic themes · word-level explanations")
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
        return ["<strong>Nothing to analyze yet.</strong> Upload a file that has review text, or use the sample reviews."]
    negative = int((results.sentiment == "negative").sum())
    total = len(results)
    insights = [f"<strong>{negative / total:.0%} of the reviews sound unhappy.</strong> That is your starting point: open the Aspect Lens tab to see which parts of the experience are behind it."]
    high_conf = int((results["sentiment_confidence"] >= .9).sum())
    insights.append(f"<strong>The model is very sure about {high_conf:,} reviews (90%+).</strong> Being sure is not the same as being right, so spot-check a few in the Review Explorer before acting.")
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
            "We could not tell which column holds the reviews. "
            "Please pick it below."
        )
        selected_column = st.selectbox(
            "Which column holds the review text?",
            options=available_columns,
            index=0,
            key="text_column_selector",
        )
        if st.button("Use this column", type="primary", key="use_selected_column"):
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
    st.markdown('<div class="section-label">Step 1 · Bring your reviews</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Start with your own reviews, or try ours</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="steps">'
        '<div class="step"><div class="n">1</div><div><b>Load</b><span>CSV, TSV, JSON or TXT file</span></div></div>'
        '<div class="step"><div class="n">2</div><div><b>Run</b><span>Sentiment, aspects, entities</span></div></div>'
        '<div class="step"><div class="n">3</div><div><b>Explore</b><span>Dashboard, themes, slide</span></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop your reviews file here",
        type=["csv", "tsv", "json", "txt", "text"],
        label_visibility="collapsed",
        key="overview_file_uploader",
    )
    sample = st.button("Use the sample reviews", type="secondary", key="overview_sample_button")
    if uploaded is None and not sample:
        st.info("No file yet. Drop one above, or press \"Use the sample reviews\" to see AspectLens in action.")
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
    with st.expander("Peek at the file (first 5 rows)", expanded=False):
        st.dataframe(raw_frame.head(), use_container_width=True, hide_index=True)

    # Normalize the frame.
    frame = normalize_uploaded_frame(raw_frame, detected_column, source_label)
    if frame is None:
        return None

    meta = frame.attrs or {}
    empty_removed = int(meta.get("empty_removed", 0))
    usable_count = int(meta.get("usable_count", len(frame)))
    detected_column = meta.get("detected_column")

    st.caption(f"Loaded: **{source_label}**")

    # Dataset validation summary.
    st.markdown('<div class="section-label">Data check</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(f'<div class="validation-row"><span class="validation-key">Rows in file</span><span class="validation-value">{original_count:,}</span></div>', unsafe_allow_html=True)
    v2.markdown(f'<div class="validation-row"><span class="validation-key">Rows read</span><span class="validation-value">{original_count - parser_skipped:,}</span></div>', unsafe_allow_html=True)
    v3.markdown(f'<div class="validation-row"><span class="validation-key">Review column</span><span class="validation-value">{detected_column}</span></div>', unsafe_allow_html=True)
    v4.markdown(f'<div class="validation-row"><span class="validation-key">Blank rows dropped</span><span class="validation-value">{empty_removed:,}</span></div>', unsafe_allow_html=True)
    v5, v6, v7, _ = st.columns(4)
    v5.markdown(f'<div class="validation-row"><span class="validation-key">Broken rows skipped</span><span class="validation-value">{parser_skipped:,} <span style="font-size:.68rem;color:#9a94b0;">(approx.)</span></span></div>', unsafe_allow_html=True)
    v6.markdown(f'<div class="validation-row"><span class="validation-key">Reviews ready</span><span class="validation-value">{usable_count:,}</span></div>', unsafe_allow_html=True)
    status_text = "OK" if (parser_skipped == 0 and empty_removed == 0 and usable_count > 0) else ("Pick a column" if needs_selection else "Check skipped rows")
    v7.markdown(f'<div class="validation-row"><span class="validation-key">Status</span><span class="validation-value">{status_text}</span></div>', unsafe_allow_html=True)

    if parser_skipped > 0 or empty_removed > 0:
        st.warning(
            "Some rows were left out. "
            f"{parser_skipped:,} parser-detected malformed record(s) and "
            f"{empty_removed:,} empty record(s) were excluded. "
            "Nothing is dropped silently."
        )
    if usable_count == 0:
        st.info("This file has no reviews we can read.")
        return None

    # Speed controls: duplicates add no information and slow the run down.
    dup_count = int(frame["text"].duplicated().sum())
    opt_a, opt_b = st.columns(2)
    dedupe = opt_a.checkbox(f"Skip repeated reviews ({dup_count:,} found)", value=True, key="opt_dedupe")
    work = frame.drop_duplicates(subset="text", keep="first") if dedupe else frame
    unique_count = max(len(work), 1)
    if int(st.session_state.get("opt_max_rows", 0)) > unique_count:
        st.session_state["opt_max_rows"] = unique_count
    cap_kwargs = {} if "opt_max_rows" in st.session_state else {"value": min(500, unique_count)}
    max_rows = opt_b.number_input(
        "Analyze at most this many reviews",
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
        f"Plan: analyze **{len(work):,}** of {usable_count:,} reviews "
        f"({removed:,} repeats skipped). 500-1,000 keeps a live demo smooth; around 2,000 is the most a laptop CPU handles comfortably."
    )
    if len(work) > 2000:
        st.warning("That is a big batch and may take several minutes on a CPU. Consider lowering the limit.")

    run_scan = st.button("Run analysis", type="primary", use_container_width=True, key="overview_analyze_button")
    if sample or run_scan:
        fp = dataset_upload_fingerprint(work)
        if st.session_state.get("batch_fingerprint") != fp or st.session_state.get("batch_results") is None:
            results, timings = analyze_with_progress(
                work["text"].tolist(),
                "review analysis",
                ids=work["feedback_id"].tolist(),
            )
            st.session_state["batch_results"] = results
            st.session_state["batch_timings"] = timings
            st.session_state["batch_fingerprint"] = fp
            st.session_state["batch_source"] = source_label
        # Rerun so every tab (Dashboard, Review Explorer, Theme Finder)
        # sees the stored results immediately instead of on the next interaction.
        st.rerun()

    return None


def render_overview() -> None:
    """Overview — main entry point. Upload form when no data; executive dashboard after analysis."""
    st.markdown('<div class="section-label">Dashboard</div>', unsafe_allow_html=True)

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
        st.markdown('<h2 class="section-title">The big picture</h2>', unsafe_allow_html=True)
        if st.button("Start over with new data", key="reset_analysis_button"):
            for k in ("batch_results", "batch_timings", "batch_fingerprint"):
                st.session_state.pop(k, None)
            st.rerun()
        negative = int((results.sentiment == "negative").sum())
        positive = int((results.sentiment == "positive").sum())
        high_conf = int((results["sentiment_confidence"] >= .9).sum())

        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, note, accent in [
            (k1, "Reviews analyzed", f"{len(results):,}", "ready to explore", "#6d28d9"),
            (k2, "Unhappy", f"{negative / len(results):.0%}", f"{negative:,} reviews", "#e11d48"),
            (k3, "Happy", f"{positive / len(results):.0%}", f"{positive:,} reviews", "#059669"),
            (k4, "Model is sure", f"{high_conf / len(results):.0%}", "confidence 90% or more", "#f59e0b"),
        ]:
            col.markdown(kpi_card(label, value, note, accent), unsafe_allow_html=True)

        st.markdown('<div class="section-label">Quick read</div>', unsafe_allow_html=True)
        for message in build_insights(results):
            st.markdown(f'<div class="insight">{message}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Mood of the reviews</div>', unsafe_allow_html=True)
        counts = results["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        fig = px.pie(
            counts, names="sentiment", values="count", hole=.68,
            color="sentiment", color_discrete_map={"positive":"#059669","negative":"#e11d48"},
        )
        fig.update_layout(
            title="Happy vs. unhappy", margin=dict(t=55,b=10,l=10,r=10),
            legend=dict(orientation="h",y=-.05), paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textinfo="percent", textfont_size=14)
        st.plotly_chart(fig, use_container_width=True, key="chart_overview_sentiment_pie")

        # Timing report
        if timings:
            st.markdown('<div class="section-label">How fast it ran</div>', unsafe_allow_html=True)
            t1, t2, t3, t4 = st.columns(4)
            t1.markdown(f'<div class="timing-row"><span class="timing-key">Reviews in</span><span class="timing-value">{timings.get("total_records", 0):,}</span></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="timing-row"><span class="timing-key">Reviews done</span><span class="timing-value">{timings.get("analyzed_records", 0):,}</span></div>', unsafe_allow_html=True)
            t3.markdown(f'<div class="timing-row"><span class="timing-key">Sentiment</span><span class="timing-value">{timings.get("sentiment_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t4.markdown(f'<div class="timing-row"><span class="timing-key">Entities</span><span class="timing-value">{timings.get("ner_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t5, t6, t7, _ = st.columns(4)
            t5.markdown(f'<div class="timing-row"><span class="timing-key">Keywords</span><span class="timing-value">{timings.get("keyword_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t6.markdown(f'<div class="timing-row"><span class="timing-key">Total</span><span class="timing-value">{timings.get("overall_time", 0):.2f}s</span></div>', unsafe_allow_html=True)
            t7.markdown(f'<div class="timing-row"><span class="timing-key">Reviews / sec</span><span class="timing-value">{timings.get("records_per_second", 0):,.1f}</span></div>', unsafe_allow_html=True)


def render_feedback_analysis() -> None:
    """Feedback Analysis — detailed records table."""
    st.markdown('<div class="section-label">Review Explorer</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Every review, one row each</h2>', unsafe_allow_html=True)
    st.caption("Search, filter and read individual reviews with their mood, confidence, named entities and topics.")

    results = st.session_state.get("batch_results")
    if results is None or len(results) == 0:
        st.markdown(
            '<div class="empty-state"><h3>Nothing to explore yet</h3>'
            '<p>Load reviews and press Run analysis on the Dashboard tab first.</p></div>',
            unsafe_allow_html=True,
        )
        return

    # Search and filters
    search = st.text_input("Search inside reviews", key="feedback_search_input", placeholder="Try: refund, battery, rude…")
    sentiment_filter = st.selectbox(
        "Show only",
        options=["All", "positive", "negative"],
        index=0,
        key="feedback_sentiment_filter",
    )

    filtered = results
    if search:
        filtered = filtered[filtered["text"].str.contains(search, case=False, na=False)]
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    st.caption(f"Showing {len(filtered):,} of {len(results):,} reviews")

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
            "entity_text": "Named entities",
            "aspects": "Topics mentioned",
        },
    )
    st.caption(
        "Named entities are people, organizations, places, dates and similar items, so this column is empty for reviews "
        "that name none. Business aspects (delivery, price, support, app...) show what each review is about."
    )

    st.download_button(
        "Download results (CSV)",
        results.to_csv(index=False),
        "aspectlens_enriched_reviews.csv",
        "text/csv",
        key="feedback_download_button",
    )


def render_playground() -> None:
    """Quick Feedback Analysis — single message analysis."""
    st.markdown('<div class="section-label">Try a Sentence</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Type a review, see how the model reads it</h2>', unsafe_allow_html=True)
    st.caption("Paste any customer comment. You will get the mood, the words that drove it, and the topics it mentions.")
    text = st.text_area("Customer feedback", "The mobile app crashes when I try to pay, but support resolved my issue quickly.", height=145, label_visibility="collapsed", key="playground_text_area")
    if st.button("Explain this sentence", type="primary", key="playground_analyze_button") and text.strip():
        try:
            results, _ = analyze_with_progress([text], "sentence check")
            result = results.iloc[0]
            classifier = get_pipeline().sentiment
            explanation = explain_sentiment(classifier, text)
            mentions = merge_mentions(result["entities"], find_aspect_mentions(text))
            positive_words, negative_words = top_drivers(explanation["contributions"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Mood", str(result["sentiment"]).title())
            c2.metric("Certainty", f"{result['sentiment_confidence']:.1%}")
            c3.metric("Things spotted", len(mentions))
            left, right = st.columns([1.25, .75])
            with left:
                st.markdown('<div class="section-label">Which words decided it?</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="highlight-box">{highlight_reasoning(explanation["contributions"])}</div>', unsafe_allow_html=True)
                st.markdown('<div class="legend">Green words pull the mood up, red words pull it down. The darker the shade, the stronger the pull.</div>', unsafe_allow_html=True)
                if positive_words or negative_words:
                    drivers = []
                    if negative_words:
                        drivers.append(f"<strong>Pulls it down:</strong> {html.escape(', '.join(negative_words))}")
                    if positive_words:
                        drivers.append(f"<strong>Pulls it up:</strong> {html.escape(', '.join(positive_words))}")
                    st.markdown(f'<div class="insight">{"<br>".join(drivers)}</div>', unsafe_allow_html=True)

                clauses = split_clauses(text)
                if len(clauses) > 1:
                    st.markdown('<div class="section-label">Mixed feelings? Part by part</div>', unsafe_allow_html=True)
                    clause_preds = classifier.predict(clauses)
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Part": clauses,
                                "Sentiment": [p.label.title() for p in clause_preds],
                                "Confidence": [f"{p.confidence:.0%}" for p in clause_preds],
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown('<div class="section-label">Names and topics spotted</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="highlight-box">{highlight_mentions(text, mentions)}</div>', unsafe_allow_html=True)
                if mentions:
                    st.markdown('<div class="legend">Blue tags are named entities (people, places, brands). Amber tags are topics like delivery, price, support or app.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="legend">No names or topics were spotted in this sentence.</div>', unsafe_allow_html=True)
            with right:
                probs = result["sentiment_probabilities"]
                fig = go.Figure(go.Bar(x=list(probs.values()), y=[key.title() for key in probs], orientation="h", marker_color=["#e11d48" if key == "negative" else "#059669" for key in probs]))
                fig.update_layout(title="How sure is the model?", xaxis_title="Probability", xaxis_range=[0,1], margin=dict(t=55,b=15,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, key="chart_playground_confidence_bar")
        except Exception as exc:
            st.error(f"That did not work: {exc}")


def render_topic_discovery() -> None:
    st.markdown('<div class="section-label">Theme Finder</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">What do people keep talking about?</h2>', unsafe_allow_html=True)
    st.caption("Similar reviews are grouped automatically into themes. Nobody has to define the groups first.")
    results = st.session_state.get("batch_results")
    if results is None or len(results) == 0:
        st.info("Run the analysis on the Dashboard tab first, then come back to find themes.")
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
    run = st.button("Find themes", type="primary", use_container_width=True, key="topic_discover_button")
    topic_df = topic_state["topic_df"]
    if run:
        if not discoverer.available:
            st.error(
                "BERTopic is not installed. Add the optional dependencies from "
                "requirements-topics.txt (bertopic, sentence-transformers, umap-learn, hdbscan)."
            )
            return
        try:
            with st.status("Looking for themes…", expanded=True) as status:
                st.write("Downloading the sentence-transformer embedding model on first run.")
                topic_df = discoverer.fit_transform(results["text"].tolist(), results["feedback_id"].tolist())
                st.write("Topics extracted.")
                status.update(label="Themes ready", state="complete", expanded=False)
            topic_state["fitted"] = True
            topic_state["topic_df"] = topic_df
        except Exception as exc:
            st.error(f"Topic discovery failed: {exc}")
            return
    if topic_df is None or len(topic_df) == 0:
        return
    # Merge topic assignments onto results by feedback_id (not raw text)
    merged = results.merge(topic_df[["feedback_id", "topic", "topic_name"]], on="feedback_id", how="left")
    st.markdown('<div class="section-label">Themes found</div>', unsafe_allow_html=True)
    summary = discoverer.summary()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-label">Words behind each theme</div>', unsafe_allow_html=True)
    if discoverer.is_fitted and len(summary) > 1:
        try:
            fig_bar = discoverer._model.visualize_barchart(top_n_topics=min(10, len(summary)))
            st.plotly_chart(fig_bar, use_container_width=True, key="chart_topic_keywords_barchart")
        except Exception:
            st.info("Topic keyword chart could not be rendered.")
    st.markdown('<div class="section-label">Theme map</div>', unsafe_allow_html=True)
    if discoverer.is_fitted and len(summary) > 1:
        try:
            fig_map = discoverer._model.visualize_topics()
            st.plotly_chart(fig_map, use_container_width=True, key="chart_topic_map")
        except Exception:
            st.info("Topic map could not be rendered.")
    elif discoverer.is_fitted:
        st.info("Topic map requires more than one discovered topic.")
    st.markdown('<div class="section-label">Example reviews per theme</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-label">Mood inside each theme</div>', unsafe_allow_html=True)
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
            fig_ts.add_trace(go.Bar(x=grp["topic_name"], y=grp["positive_pct"], name="Positive", marker_color="#059669"))
            fig_ts.add_trace(go.Bar(x=grp["topic_name"], y=grp["negative_pct"], name="Negative", marker_color="#e11d48"))
            fig_ts.update_layout(
                barmode="stack", title="Happy vs. unhappy by theme",
                xaxis_title="Theme", yaxis_title="Share of reviews (%)",
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
    st.markdown('<div class="section-label">Aspect Lens</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">What exactly do customers love or hate?</h2>', unsafe_allow_html=True)
    st.caption(
        "Each review is split into clauses, business aspects are detected, and every aspect is scored "
        "by the clause that mentions it, so one mixed review can be positive about one thing and negative about another."
    )
    results = st.session_state.get("batch_results")
    if results is None or len(results) == 0:
        st.info("Run the analysis on the Dashboard tab first, then come back here.")
        return

    cache_key = f"aspect_df_{CACHE_VERSION}_{st.session_state.get('batch_fingerprint')}"
    if st.button("Find aspects", type="primary", use_container_width=True, key="aspect_run_button"):
        with st.spinner("Scoring aspects clause by clause…"):
            st.session_state[cache_key] = analyze_aspects(get_pipeline().sentiment, results)
    aspect_df = st.session_state.get(cache_key)
    if aspect_df is None:
        st.markdown(
            '<div class="empty-state"><h3>No aspects scored yet</h3>'
            '<p>Press <strong>Find aspects</strong> to score delivery, price, support, app and other aspects.</p></div>',
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
    for col, label, value, note, accent in [
        (k1, "Reviews analyzed", f"{len(results):,}", "from the Dashboard run", "#6d28d9"),
        (k2, "Name a topic", f"{with_aspect / len(results):.0%}", f"{with_aspect:,} reviews", "#c026d3"),
        (k3, "Topic mentions", f"{len(aspect_df):,}", f"{len(summary)} different topics", "#f59e0b"),
        (k4, "Talked about most", html.escape(str(top["aspect"]).title()), f"{int(top['mentions']):,} mentions", "#059669"),
    ]:
        col.markdown(kpi_card(label, value, note, accent, small=(label == "Talked about most")), unsafe_allow_html=True)

    rated = summary[summary["mentions"] >= 3]
    if rated.empty:
        rated = summary
    worst = rated.sort_values(["negative_share", "mentions"], ascending=False).iloc[0]
    best = rated.sort_values(["positive_share", "mentions"], ascending=False).iloc[0]
    st.markdown('<div class="section-label">Headlines</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight"><strong>Biggest headache: {html.escape(str(worst["aspect"]))}.</strong> '
        f'{worst["negative_share"]:.0%} of its {int(worst["mentions"])} mentions are negative.</div>'
        f'<div class="insight"><strong>Biggest win: {html.escape(str(best["aspect"]))}.</strong> '
        f'{best["positive_share"]:.0%} of its {int(best["mentions"])} mentions are positive.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">The one-slide takeaway</div>', unsafe_allow_html=True)
    insight = build_insight(summary, aspect_df, len(results))
    if insight:
        slide_html = render_slide_html(insight)
        try:
            import streamlit.components.v1 as components

            components.html(slide_html, height=640, scrolling=False)
        except Exception:
            st.caption("Preview unavailable here; download the slide below.")
        st.download_button(
            "Download the slide (HTML)",
            slide_html,
            "business_insight_slide.html",
            "text/html",
            key="insight_slide_download",
        )
        st.caption("Open the file in a browser, then screenshot it or use Print > Save as PDF to drop it into your presentation.")

    st.markdown('<div class="section-label">Mood per aspect</div>', unsafe_allow_html=True)
    long = summary.melt(id_vars=["aspect"], value_vars=["positive", "negative", "neutral"], var_name="sentiment", value_name="count")
    fig = px.bar(
        long, x="aspect", y="count", color="sentiment", barmode="group",
        category_orders={"aspect": summary["aspect"].tolist()},
        color_discrete_map={"positive": "#059669", "negative": "#e11d48", "neutral": "#a8a29e"},
    )
    fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=-.25), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key="chart_aspect_grouped")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-label">Most criticized (share of mentions)</div>', unsafe_allow_html=True)
        neg = rated.sort_values("negative_share", ascending=False).head(6).sort_values("negative_share")
        fig_neg = go.Figure(go.Bar(x=neg["negative_share"] * 100, y=neg["aspect"], orientation="h", marker_color="#e11d48"))
        fig_neg.update_layout(xaxis_title="% negative", margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_neg, use_container_width=True, key="chart_aspect_negative")
    with right:
        st.markdown('<div class="section-label">Most praised (share of mentions)</div>', unsafe_allow_html=True)
        pos = rated.sort_values("positive_share", ascending=False).head(6).sort_values("positive_share")
        fig_pos = go.Figure(go.Bar(x=pos["positive_share"] * 100, y=pos["aspect"], orientation="h", marker_color="#059669"))
        fig_pos.update_layout(xaxis_title="% positive", margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pos, use_container_width=True, key="chart_aspect_positive")
    st.caption("Aspects with fewer than 3 mentions are left out of these rankings when enough others exist.")

    st.markdown('<div class="section-label">Proof: the exact words behind each score</div>', unsafe_allow_html=True)
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
        "Download aspect scores (CSV)",
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
                fig_cm = px.imshow(metrics["confusion"], text_auto=True, color_continuous_scale="Purples", aspect="auto")
                fig_cm.update_layout(title="Confusion matrix", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm, use_container_width=True, key="chart_aspect_confusion")


def render_model_methodology() -> None:
    st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">How AspectLens works</h2>', unsafe_allow_html=True)

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
        "Overall sentiment and aspect results are not scored against ground truth automatically. In the Aspect Lens tab you can download a "
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
    '<div class="hero"><div class="lens"></div>'
    '<div class="eyebrow">REVIEWS → REASONS → ACTIONS</div>'
    '<h1>See what customers really mean, <em>aspect by aspect.</em></h1>'
    '<p>Drop in a pile of reviews. AspectLens tells you who is happy, who is not, which part of the '
    'experience is responsible, and which single fix would matter most.</p>'
    '<div class="chips"><span class="chip">Overall mood</span><span class="chip">Per-aspect sentiment</span>'
    '<span class="chip">Word-level reasons</span><span class="chip">Theme finder</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

# Navigation: the Dashboard tab handles both upload and results
overview_tab, batch_tab, aspect_tab, playground_tab, topic_tab, method_tab = st.tabs(
    ["▦ Dashboard", "☰ Review Explorer", "◎ Aspect Lens", "✎ Try a Sentence", "❖ Theme Finder", "ⓘ How It Works"]
)

# Render each tab independently — errors in one tab do not affect others
with overview_tab:
    try:
        render_overview()
    except Exception as exc:
        show_error("Dashboard", exc)

with batch_tab:
    try:
        render_feedback_analysis()
    except Exception as exc:
        show_error("Review Explorer", exc)

with aspect_tab:
    try:
        render_aspect_insights()
    except Exception as exc:
        show_error("Aspect Lens", exc)

with playground_tab:
    try:
        render_playground()
    except Exception as exc:
        show_error("Try a Sentence", exc)

with topic_tab:
    try:
        render_topic_discovery()
    except Exception as exc:
        show_error("Theme Finder", exc)

with method_tab:
    try:
        render_model_methodology()
    except Exception as exc:
        show_error("How It Works", exc)
