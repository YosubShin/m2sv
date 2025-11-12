#!/usr/bin/env python3
"""
Create a clean validation split by removing rows that already belong to train.

Example:
  python create_validation_split.py \
    --train blueprints/20k/train.jsonl \
    --train-val blueprints/20k/train-val-20k.jsonl \
    --output blueprints/20k/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter validation rows not present in train")
    parser.add_argument("--train", type=Path, required=True, help="Path to the train JSONL file")
    parser.add_argument("--train-val", type=Path, required=True, help="Combined train+validation JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the filtered validation JSONL")
    parser.add_argument(
        "--uid-field",
        default="uid",
        help="JSON field used to detect overlap between train and validation (default: uid)",
    )
    return parser.parse_args()


def load_uid_set(path: Path, uid_field: str) -> set[str]:
    """Load every uid from a JSONL file into a set."""
    uids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse {path}:{line_num}: {exc}") from exc
            if uid_field not in record:
                raise SystemExit(f"Missing '{uid_field}' in {path}:{line_num}")
            uid = str(record[uid_field])
            if uid in uids:
                raise SystemExit(f"Duplicate '{uid_field}' value '{uid}' found in {path}:{line_num}")
            uids.add(uid)
    return uids


def filter_validation(train_val_path: Path, output_path: Path, excluded_uids: set[str], uid_field: str) -> tuple[int, int]:
    """Return (kept, dropped) counts."""
    kept = 0
    dropped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with train_val_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_num, raw in enumerate(src, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse {train_val_path}:{line_num}: {exc}") from exc
            if uid_field not in record:
                raise SystemExit(f"Missing '{uid_field}' in {train_val_path}:{line_num}")
            uid = str(record[uid_field])
            if uid in excluded_uids:
                dropped += 1
                continue
            dst.write(raw if raw.endswith("\n") else raw + "\n")
            kept += 1
    return kept, dropped


def main() -> None:
    args = parse_args()
    train_uids = load_uid_set(args.train, args.uid_field)
    print(f"Loaded {len(train_uids):,} unique {args.uid_field} values from {args.train}")

    kept, dropped = filter_validation(args.train_val, args.output, train_uids, args.uid_field)
    total = kept + dropped
    print(f"Wrote {kept:,} validation rows to {args.output}")
    print(f"Skipped {dropped:,} overlapping rows out of {total:,} total in {args.train_val}")


if __name__ == "__main__":
    main()

