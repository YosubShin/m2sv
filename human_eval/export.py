#!/usr/bin/env python3
"""Export collected human-eval responses.

Produces, under an output directory:
  per_participant/<email>.json  one file per person in the evaluate_vlm format
                                (id, pred, gold, raw, correct, elapsed_seconds)
                                so the existing analysis/ scripts work unchanged.
  responses.csv                 flat dump of every response with all timing fields.

Usage:
  python human_eval/export.py --out human_eval/export
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

try:
    from . import db
except ImportError:  # allow `python human_eval/export.py`
    import db

HERE = Path(__file__).resolve().parent
PROBLEMS = {p["id"]: p for p in json.loads((HERE / "problems.json").read_text())["problems"]}


def safe_name(email: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", email.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "export")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    per_dir = args.out / "per_participant"
    per_dir.mkdir(exist_ok=True)

    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM responses ORDER BY email, position"
        ).fetchall()]

    # Flat CSV with every field.
    csv_path = args.out / "responses.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Per-participant evaluate_vlm-format JSON.
    by_email: dict[str, list[dict]] = {}
    for r in rows:
        by_email.setdefault(r["email"], []).append(r)

    for email, recs in by_email.items():
        results = []
        for r in recs:
            gold = PROBLEMS.get(r["problem_id"], {}).get("answer", "")
            results.append({
                "id": r["problem_id"],
                "pred": r["selected"],
                "gold": gold,
                "raw": r["selected"],
                "correct": bool(r["is_correct"]),
                "elapsed_seconds": (r["client_elapsed_ms"] or 0) / 1000.0,
                "hidden_seconds": (r["hidden_ms"] or 0) / 1000.0,
                "position": r["position"],
                "flagged": bool(r["flagged"]),
                "revision": (r["revision"] if "revision" in r.keys() else 1),
            })
        correct = sum(1 for x in results if x["correct"])
        total = len(results)
        out = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "results": results,
        }
        (per_dir / f"{safe_name(email)}.json").write_text(json.dumps(out, indent=2))

    print(f"Exported {len(rows)} responses from {len(by_email)} participants -> {args.out}")


if __name__ == "__main__":
    main()
