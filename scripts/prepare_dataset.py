"""Create a normalized dataset from a supported feedback file."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing import load_feedback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV, TSV, JSON, or TXT input")
    parser.add_argument("--output", default="data/processed/feedback.csv")
    parser.add_argument("--text-column", default="text")
    args = parser.parse_args()
    frame = load_feedback(args.input, args.text_column)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} normalized records to {output}")


if __name__ == "__main__":
    main()

