"""Build a clean, de-duplicated review sample for the demo.

Usage (PowerShell, from the project folder):
    python scripts/prepare_sample.py path\to\reviews.csv --n 1000 --out data/sample/reviews_1000.csv
    python scripts/prepare_sample.py path\to\reviews.csv --column "reviews.text"

It picks the review-text column, drops empty and duplicate reviews, drops very
short reviews, takes a random sample and saves a CSV that has a `text` column,
so the app detects it automatically.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CANDIDATES = [
    "reviews.text", "review text", "review_text", "review body", "review_body", "review content",
    "review_content", "customer review", "customer_review", "review", "text", "content",
    "comment", "feedback", "body", "message",
]


def read_any(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "windows-1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def pick_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise SystemExit(f"Column {requested!r} not found. Available: {', '.join(map(str, frame.columns))}")
        return requested
    lookup = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in frame.columns}
    for candidate in CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    raise SystemExit("Could not guess the review column. Re-run with --column. Available: " + ", ".join(map(str, frame.columns)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/sample/reviews_sample.csv"))
    parser.add_argument("--n", type=int, default=1000, help="reviews to keep (default 1000)")
    parser.add_argument("--column", default=None, help="review text column (auto-detected if omitted)")
    parser.add_argument("--min-words", type=int, default=6, help="drop reviews shorter than this")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = read_any(args.input)
    column = pick_column(frame, args.column)
    total = len(frame)

    frame = frame.copy()
    frame["text"] = frame[column].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame = frame[frame[column].notna() & (frame["text"].str.lower() != "nan") & (frame["text"] != "")]
    after_empty = len(frame)
    frame = frame[~frame["text"].str.lower().duplicated()]
    after_dupes = len(frame)
    frame = frame[frame["text"].str.split().str.len() >= args.min_words]
    after_short = len(frame)
    if len(frame) > args.n:
        frame = frame.sample(n=args.n, random_state=args.seed)
    frame = frame.reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"column used      : {column}")
    print(f"rows in file     : {total}")
    print(f"after empties    : {after_empty}")
    print(f"after duplicates : {after_dupes}")
    print(f"after min words  : {after_short}")
    print(f"saved            : {len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    main()
