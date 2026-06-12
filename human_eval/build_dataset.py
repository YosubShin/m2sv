#!/usr/bin/env python3
"""Freeze the 200-problem human-eval set and copy its images.

Joins the IDs in ``results/manual.json`` (the 200 problems you already
hand-evaluated) with their rows in ``data/hf/m2sv-11k-validation`` and writes:

  human_eval/problems.json            frozen set (id, question, options, answer, meta)
  human_eval/static/img/<id>_map.jpg  overhead map image
  human_eval/static/img/<id>_sv.jpg   street view image

The frozen JSON keeps the gold ``answer`` for server-side grading; the API
never sends it to the client. Run once (re-run is idempotent).

Usage:
  python human_eval/build_dataset.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def load_manual_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    ids = [str(r["id"]) for r in data.get("results", [])]
    if not ids:
        raise SystemExit(f"No result ids found in {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate ids in {path}")
    return ids


def load_dataset_rows(jsonl: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manual", type=Path, default=REPO_ROOT / "results/manual.json")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data/hf/m2sv-11k-validation",
        help="Directory holding train.jsonl and an images/ subdir",
    )
    ap.add_argument("--out", type=Path, default=HERE / "problems.json")
    ap.add_argument("--img-out", type=Path, default=HERE / "static/img")
    args = ap.parse_args()

    jsonl = args.dataset / "train.jsonl"
    ids = load_manual_ids(args.manual)
    rows = load_dataset_rows(jsonl)

    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} manual ids missing from {jsonl}: {missing[:5]}...")

    args.img_out.mkdir(parents=True, exist_ok=True)

    problems = []
    for pid in ids:
        r = rows[pid]
        # Copy both images to stable, client-facing names.
        for kind, field in (("map", "image_map"), ("sv", "image_sv")):
            src = args.dataset / r[field]
            if not src.exists():
                raise SystemExit(f"Missing image {src}")
            shutil.copyfile(src, args.img_out / f"{pid}_{kind}{src.suffix}")

        suffix = (args.dataset / r["image_map"]).suffix
        problems.append(
            {
                "id": pid,
                "question": r.get("question", ""),
                "options": r.get("options", []),
                "answer": str(r.get("answer", "")).strip().upper(),  # server-side only
                "image_map": f"img/{pid}_map{suffix}",
                "image_sv": f"img/{pid}_sv{suffix}",
                "meta": r.get("meta", {}),
            }
        )

    args.out.write_text(json.dumps({"problems": problems}, indent=2))
    print(f"Wrote {len(problems)} problems -> {args.out}")
    print(f"Copied {len(problems) * 2} images -> {args.img_out}")


if __name__ == "__main__":
    main()
