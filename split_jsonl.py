#!/usr/bin/env python3
"""
Shuffle and split a JSONL file into multiple parts by exact row counts.

Examples:
  # 11k -> 10k train, 1k validation
  python split_jsonl.py \
    --input blueprints/11k/train-val-11k.jsonl \
    --output-dir blueprints \
    --split train=10000 --split validation=1000

  # Deterministic shuffle with seed
  python split_jsonl.py --input data.jsonl --output-dir out --seed 123 --split a=800 --split b=200

Behavior:
  - Shuffles all lines deterministically with --seed
  - Writes <output-dir>/<name>.jsonl for each split
  - If requested counts sum < total, the remaining rows go to a file named 'remainder.jsonl' unless --drop-remainder is set
  - If requested counts sum > total, exits with an error
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line.rstrip("\n"))
    return lines


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def parse_split_arg(values: list[str]) -> list[tuple[str, int]]:
    splits: list[tuple[str, int]] = []
    for v in values:
        if "=" not in v:
            raise SystemExit(f"Invalid --split '{v}'. Expected name=count, e.g. train=10000")
        name, cnt = v.split("=", 1)
        name = name.strip()
        try:
            count = int(cnt.replace(",", "").strip())
        except ValueError:
            raise SystemExit(f"Invalid count in --split '{v}' (must be integer)")
        if count < 0:
            raise SystemExit(f"Split count must be non-negative: {v}")
        if not name:
            raise SystemExit(f"Split name must be non-empty: {v}")
        splits.append((name, count))
    # ensure unique names
    names = [n for n, _ in splits]
    if len(set(names)) != len(names):
        raise SystemExit("Duplicate split names are not allowed")
    return splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shuffle and split JSONL by exact counts")
    p.add_argument("--input", type=Path, required=True, help="Path to input JSONL")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write split files")
    p.add_argument("--split", action="append", default=[], help="Split spec 'name=count' (repeatable)")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    p.add_argument("--drop-remainder", action="store_true", help="If set, drop leftover rows instead of writing remainder.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = parse_split_arg(args.split)
    if not splits:
        raise SystemExit("Provide at least one --split name=count")

    lines = read_lines(args.input)
    total = len(lines)
    rng = random.Random(args.seed)
    rng.shuffle(lines)

    requested = sum(c for _, c in splits)
    if requested > total:
        raise SystemExit(f"Requested {requested} rows across splits but input has only {total}")

    start = 0
    for name, count in splits:
        chunk = lines[start:start + count]
        start += count
        out_path = args.output_dir / f"{name}.jsonl"
        write_lines(out_path, chunk)
        print(f"Wrote {len(chunk)} -> {out_path}")

    leftover = total - start
    if leftover > 0 and not args.drop_remainder:
        out_path = args.output_dir / "remainder.jsonl"
        write_lines(out_path, lines[start:])
        print(f"Wrote {leftover} -> {out_path}")
    elif leftover > 0 and args.drop_remainder:
        print(f"Dropped {leftover} leftover rows (use without --drop-remainder to save as remainder.jsonl)")


if __name__ == "__main__":
    main()


