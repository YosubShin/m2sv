#!/usr/bin/env python3
"""Re-evaluate an existing predictions CSV using normalize_letter."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any, List

from m2sv_eval_utils import normalize_letter


def parse_options(raw: str) -> List[str]:
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]
    except Exception:
        pass
    # fallback: split by comma
    return [tok.strip() for tok in raw.strip("[]()") .split(",") if tok.strip()]


def reevaluate(csv_path: Path, out_csv: Path | None = None) -> None:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results: List[dict[str, Any]] = []
    total = len(rows)
    correct = 0
    empty_preds = 0
    random_expectation = 0.0

    for row in rows:
        options = parse_options(row.get("options", ""))
        num_options = len(options) if options else 0
        random_expectation += (1.0 / num_options) if num_options else 0.0

        raw_response = row.get("raw_response", "") or row.get("prediction", "")
        normalized = normalize_letter(raw_response, num_options)
        if not normalized:
            empty_preds += 1

        ground_truth = (row.get("ground_truth") or "").strip().upper()
        is_correct = normalized == ground_truth and bool(ground_truth)
        if is_correct:
            correct += 1

        updated = dict(row)
        updated["normalized_prediction"] = normalized
        updated["normalized_correct"] = str(is_correct)
        results.append(updated)

    accuracy = correct / total if total else 0.0
    random_baseline = random_expectation / total if total else 0.0

    print(f"Total rows: {total}")
    print(f"Accuracy (normalized): {accuracy:.4%} ({correct}/{total})")
    print(f"Random baseline: {random_baseline:.4%}")
    print(f"Empty predictions: {empty_preds}")

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(results[0].keys()) if results else []
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote updated CSV to {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate predictions CSV.")
    parser.add_argument("csv", type=Path, help="Path to predictions.csv")
    parser.add_argument("--out", type=Path, help="Optional path to write augmented CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reevaluate(args.csv, args.out)


if __name__ == "__main__":
    main()

