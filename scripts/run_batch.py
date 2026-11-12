"""Run the NLP enrichment pipeline on a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import FeedbackInsightsPipeline
from src.preprocessing import load_feedback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="data/processed/enriched_feedback.csv")
    args = parser.parse_args()
    frame = load_feedback(args.input)
    enriched = FeedbackInsightsPipeline().analyze_texts(frame["text"].tolist())
    result = frame.drop(columns=["text"], errors="ignore").join(enriched)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result):,} enriched records to {output}")


if __name__ == "__main__":
    main()

