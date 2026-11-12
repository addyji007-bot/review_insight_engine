"""Evaluate a labelled CSV containing text and label columns."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.evaluation import evaluate_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="evaluation.json")
    args = parser.parse_args()
    metrics = evaluate_classifier(pd.read_csv(args.input), args.text_column, args.label_column)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Accuracy: {metrics['accuracy']:.3f} | Macro-F1: {metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    main()

