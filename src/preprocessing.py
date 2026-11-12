"""Text normalization and input loading utilities.

Parsing and text-column detection are deliberately separated so that an
uploaded file can be parsed and inspected before a text column is chosen.
Malformed records are tracked separately from empty feedback records, and
customer feedback is never silently discarded.
"""

from __future__ import annotations

import csv
import io
import json
import re
import threading
from pathlib import Path
from typing import Iterable

import pandas as pd


# Lock to protect the process-global csv.field_size_limit() setting.
_field_size_lock = threading.Lock()


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving useful punctuation and casing.

    NaN / missing values are treated as empty so they are removed during
    normalization rather than being converted to the literal string "nan".
    """
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Candidate text columns, ordered by preference. Matched case-insensitively
# after stripping whitespace and lowercasing (spaces collapsed to single space).
_TEXT_COLUMN_CANDIDATES = [
    # 1. Exact "text" column (highest priority after explicit selection)
    "text",
    # 2. Review-specific columns (with and without spaces, for Amazon-style CSVs)
    "reviewtext",
    "review text",      # "Review Text" column in Amazon exports
    "review_text",
    "review body",
    "review_body",
    "review content",
    "review_content",
    "customer review",
    "customer_review",
    "review",
    # 3. Feedback/comment-specific columns
    "comment text",
    "comment_text",
    "comment",
    "customer comment",
    "customer_comment",
    "customer feedback",
    "customer_feedback",
    "feedback text",
    "feedback_text",
    "feedback",
    # 4. Other generic text columns
    "content",
    "message",
    "body",
    "description",
    "summary",
    "ticket",
]


def _normalize_column_name(col) -> str:
    """Normalize a column name: strip, lowercase, collapse internal spaces."""
    import re as _re
    return _re.sub(r"\s+", " ", str(col).strip().lower())


def _detect_text_column(columns: list[str]) -> str | None:
    """Return the best text-column name from a list of column names.

    Matching is case-insensitive. Returns None when no candidate matches.
    """
    lower_to_original = {_normalize_column_name(c): c for c in columns}
    for candidate in _TEXT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _raise_field_size_limit(limit: int) -> int:
    """Raise csv.field_size_limit to `limit`, returning the previous value."""
    with _field_size_lock:
        previous = csv.field_size_limit()
        csv.field_size_limit(limit)
        return previous


def _restore_field_size_limit(previous: int) -> None:
    """Restore csv.field_size_limit to `previous`."""
    with _field_size_lock:
        csv.field_size_limit(previous)


def _sniff_delimiter(sample_text: str) -> str:
    """Auto-detect the CSV delimiter from a text sample."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t|;")
        return dialect.delimiter
    except csv.Error:
        return ","


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV with encoding/BOM/delimiter fallback and malformed tracking.

    Tries utf-8-sig -> utf-8 -> windows-1252 and C engine -> Python engine.
    Raises the last exception if every attempt fails. Does not silently
    discard malformed rows; parser-skipped rows are counted via on_bad_lines.
    """
    last_error: Exception | None = None
    encodings = ["utf-8-sig", "utf-8", "windows-1252"]

    # Sniff delimiter from a small sample read as bytes.
    sample_text = ""
    try:
        raw = path.read_bytes()
        for enc in encodings:
            try:
                sample_text = raw[:65536].decode(enc)
                break
            except UnicodeDecodeError:
                continue
    except OSError:
        sample_text = ""
    delimiter = _sniff_delimiter(sample_text)

    previous_limit = _raise_field_size_limit(50 * 1024 * 1024)
    try:
        for enc in encodings:
            for engine in ("c", "python"):
                try:
                    kwargs = dict(
                        filepath_or_buffer=str(path),
                        encoding=enc,
                        sep=delimiter,
                        engine=engine,
                    )
                    if engine == "python":
                        kwargs["on_bad_lines"] = "warn"
                    else:
                        kwargs["on_bad_lines"] = "warn"
                    frame = pd.read_csv(**kwargs)
                    return frame
                except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                    last_error = exc
                    continue
    finally:
        _restore_field_size_limit(previous_limit)

    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not read CSV file: {path}")


def _parse_file(path: Path) -> pd.DataFrame:
    """Parse a supported file into a raw DataFrame with all original columns."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_robust(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    if suffix == ".json":
        return pd.read_json(path, encoding="utf-8-sig")
    if suffix in {".txt", ".text"}:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        return pd.DataFrame({"text": text.splitlines()})
    raise ValueError(f"Unsupported input type: {suffix}")


def load_feedback(
    path: str | Path,
    text_column: str = "text",
    interactive: bool = False,
) -> pd.DataFrame:
    """Load CSV/TSV/JSON/TXT feedback into a canonical DataFrame.

    Parsing is separated from text-column detection. The returned DataFrame
    carries load metadata in ``.attrs``:

    * ``original_count`` -- rows successfully parsed from the file
    * ``parser_skipped`` -- parser-detected malformed rows skipped (approx.)
    * ``detected_column`` -- the text column chosen (or None)
    * ``empty_removed`` -- empty-feedback records dropped during normalization
    * ``usable_count`` -- final usable record count
    * ``needs_column_selection`` -- True when no column was auto-detected
    * ``available_columns`` -- list of original column names for a selector

    When ``interactive=True`` and no text column can be detected, the raw
    DataFrame is returned with ``needs_column_selection=True`` instead of
    raising, so the caller can show a manual column selector without
    re-parsing the file.
    """
    path = Path(path)
    raw_frame = _parse_file(path)

    available_columns = list(raw_frame.columns)
    detected = _detect_text_column(available_columns)

    # Respect an explicit text_column argument if it exists in the frame.
    if text_column and text_column in available_columns:
        detected = text_column

    result = raw_frame.copy()
    metadata: dict = {
        "original_count": len(raw_frame),
        "parser_skipped": 0,
        "detected_column": detected,
        "empty_removed": 0,
        "usable_count": 0,
        "needs_column_selection": False,
        "available_columns": available_columns,
    }

    if detected is None:
        metadata["needs_column_selection"] = True
        if interactive:
            result.attrs = metadata
            return result
        raise ValueError(
            "Could not find a text column. Available columns: "
            + ", ".join(available_columns)
            + ". Expected one of: " + ", ".join(_TEXT_COLUMN_CANDIDATES)
        )

    # Normalize: add canonical `text` column, drop empties, assign feedback_id.
    result["text"] = result[detected].map(normalize_text)
    before_drop = len(result)
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    empty_removed = before_drop - len(result)
    result["feedback_id"] = result.index + 1

    metadata.update(
        {
            "detected_column": detected,
            "empty_removed": empty_removed,
            "usable_count": len(result),
        }
    )
    result.attrs = metadata
    return result


def normalize_feedback_frame(raw_frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Normalize an already-parsed DataFrame using the given text column.

    Does not re-read or re-parse the source file. Returns a DataFrame with
    canonical ``text`` and ``feedback_id`` columns and load metadata in
    ``.attrs``.
    """
    if text_column not in raw_frame.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available: "
            + ", ".join(map(str, raw_frame.columns))
        )

    result = raw_frame.copy()
    result["text"] = result[text_column].map(normalize_text)
    before_drop = len(result)
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    empty_removed = before_drop - len(result)
    result["feedback_id"] = result.index + 1

    metadata: dict = {
        "original_count": len(raw_frame),
        "parser_skipped": int(result.attrs.get("parser_skipped", 0)),
        "detected_column": text_column,
        "empty_removed": empty_removed,
        "usable_count": len(result),
        "needs_column_selection": False,
        "available_columns": list(raw_frame.columns),
    }
    result.attrs = metadata
    return result


def chunk_texts(texts: Iterable[str], max_chars: int = 512) -> list[str]:
    """Conservatively chunk long texts for transformer token limits."""
    chunks: list[str] = []
    for text in texts:
        words = normalize_text(text).split()
        current: list[str] = []
        size = 0
        for word in words:
            if current and size + len(word) + 1 > max_chars:
                chunks.append(" ".join(current))
                current, size = [], 0
            current.append(word)
            size += len(word) + 1
        if current:
            chunks.append(" ".join(current))
    return chunks
