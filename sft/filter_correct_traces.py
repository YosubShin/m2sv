#!/usr/bin/env python3
"""
Filter a model results JSON to extract only correct answers and their raw traces.

Input format (expected):
{
  "accuracy": float,
  "correct": int,
  "total": int,
  "results": [
    {
      "id": str,
      "pred": str,
      "gold": str,
      "raw": str,    # reasoning/trace text
      "correct": bool
    }, ...
  ]
}

Output format (JSONL): one object per line with keys: id, pred, gold, raw

Usage examples:
  python filter_correct_traces.py \
    --input gemini-2-5-pro.json \
    --output gemini-2-5-pro-correct.jsonl

  python filter_correct_traces.py \
    --input past_results/2025-10-15-v1/gemini-2-5-pro.json \
    --output past_results/2025-10-15-v1/gemini-2-5-pro-correct.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_correct(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return minimal records for entries marked correct with available raw traces."""
    filtered: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("correct") is True and isinstance(item.get("raw"), str):
            filtered.append(
                {
                    "id": item.get("id"),
                    "pred": item.get("pred"),
                    "gold": item.get("gold"),
                    "raw": item.get("raw"),
                }
            )
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter correct answers and raw traces from a results JSON into JSONL",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("gemini-2-5-pro.json"),
        help="Path to input results JSON (default: gemini-2-5-pro.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gemini-2-5-pro-correct.jsonl"),
        help="Path to output JSONL (default: gemini-2-5-pro-correct.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_json(args.input)
    if not isinstance(data, dict) or "results" not in data:
        raise SystemExit(f"Input JSON does not have expected 'results' array: {args.input}")

    results = data.get("results")
    if not isinstance(results, list):
        raise SystemExit("'results' must be a list")

    filtered = extract_correct(results)
    write_jsonl(filtered, args.output)

    total = len(results)
    kept = len(filtered)
    print(f"Wrote {kept} records to {args.output} (from {total} total results)")


if __name__ == "__main__":
    main()


