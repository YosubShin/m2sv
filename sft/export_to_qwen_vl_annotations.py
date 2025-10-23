#!/usr/bin/env python3
"""
Convert an HF JSONL dataset (map-to-street-view SFT) into a LLaVA-style
annotation JSON with multi-image conversations.

Input JSONL rows (per merge_traces_into_dataset.py):
  - id (str)
  - image_map (str)
  - image_sv (str)
  - question (str)
  - options (list[str])
  - answer (str)
  - trace (str, optional; chain-of-thought explanation)

Output JSON (list[dict]):
[
  {
    "image": ["images/map_....jpg", "images/sv_....jpg"],
    "conversations": [
      {"from": "human", "value": "<image>\n<image>\n...question..."},
      {"from": "gpt",   "value": "...trace or answer..."}
    ]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            yield json.loads(line_stripped)


def convert_row_to_conversation(row: Dict[str, Any]) -> Dict[str, Any]:
    image_map_rel = row.get("image_map")
    image_sv_rel = row.get("image_sv")
    question_text = row.get("question", "").rstrip("\n")
    trace_text = row.get("trace")
    answer_text = row.get("answer")

    if not isinstance(image_map_rel, str) or not isinstance(image_sv_rel, str):
        raise ValueError("Row is missing image_map or image_sv string fields")

    human_value = f"<image>\n<image>\n{question_text}" if question_text else "<image>\n<image>"
    assistant_value = (
        trace_text if isinstance(trace_text, str) and trace_text.strip() else (answer_text or "")
    )

    return {
        "image": [image_map_rel, image_sv_rel],
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": assistant_value},
        ],
    }


def convert_dataset(input_jsonl: Path) -> List[Dict[str, Any]]:
    conversations: List[Dict[str, Any]] = []
    for row in read_jsonl(input_jsonl):
        try:
            conversations.append(convert_row_to_conversation(row))
        except Exception:
            # Skip malformed rows to keep conversion robust for large datasets
            continue
    return conversations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HF JSONL to LLaVA-style conversation annotations")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/hf/m2sv-sft-11k-7k-in-progress"),
        help="Dataset directory containing train.jsonl and images/",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional explicit path to the input JSONL (defaults to dataset-dir/train.jsonl)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write the output annotations JSON (defaults to dataset-dir/annotations.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = args.input_jsonl or (args.dataset_dir / "train.jsonl")
    output_json = args.output_json or (args.dataset_dir / "annotations.json")

    if not input_jsonl.exists():
        raise SystemExit(f"Input JSONL not found: {input_jsonl}")

    conversations = convert_dataset(input_jsonl)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(conversations)} conversations to {output_json}")


if __name__ == "__main__":
    main()


