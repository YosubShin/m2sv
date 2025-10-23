#!/usr/bin/env python3
"""
Merge filtered traces into an existing dataset JSONL by id, adding a 'trace' field,
and emit a new dataset directory mirroring the original schema. Optionally, split
into train/validation (e.g. 9:1) and upload to Hugging Face as a split dataset.

Input A (dataset JSONL): rows produced by render_from_blueprint.py, containing keys:
  - id (str)
  - image_map (str)
  - image_sv (str)
  - question (str)
  - options (list[str])
  - answer (str)

Input B (traces JSONL): rows produced by filter_correct_traces.py with keys:
  - id (str)
  - pred (str)
  - gold (str)
  - raw (str)

Output:
  - New dataset directory under output_root/dataset_name_with_traces/
    - images/ copied from original dataset for matched ids
    - train.jsonl: same fields as original plus 'trace' (raw)
  - Optional: push to HF typed dataset with an added 'trace' feature
"""

from __future__ import annotations

import argparse
import json
import shutil
import random
from pathlib import Path
from typing import Any, Dict


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge traces into dataset by id and add 'trace' field")
    parser.add_argument("--base-dataset", required=True, type=Path, help="Path to original dataset dir (containing train.jsonl and images/")
    parser.add_argument("--traces", required=True, type=Path, help="Path to traces JSONL from filter_correct_traces.py")
    parser.add_argument("--out-dataset-name", required=True, help="Name of new dataset directory to create under output root")
    parser.add_argument("--output-root", default="data/hf", help="Output root directory")
    parser.add_argument("--split", default=None, help="Train:validation ratio, e.g. '9:1'. If omitted, no validation split is created.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling prior to split")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo id (e.g. username/dataset-name)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token (or set HF_TOKEN env)")
    parser.add_argument("--hf-private", action="store_true", help="Create/update the HF repo as private")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.base_dataset
    base_jsonl = base_dir / "train.jsonl"
    base_images = base_dir / "images"
    if not base_jsonl.exists():
        raise SystemExit(f"Base dataset not found: {base_jsonl}")
    if not base_images.exists():
        raise SystemExit(f"Base images dir not found: {base_images}")

    traces_by_id: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(args.traces):
        _id = r.get("id")
        raw = r.get("raw")
        if isinstance(_id, str) and isinstance(raw, str):
            traces_by_id[_id] = r

    out_root = Path(args.output_root)
    out_dir = out_root / args.out_dataset_name
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    kept = 0
    out_rows = []
    for row in read_jsonl(base_jsonl):
        _id = row.get("id")
        if _id in traces_by_id:
            trace = traces_by_id[_id]["raw"]
            # copy images referenced by row
            map_rel = row.get("image_map")
            sv_rel = row.get("image_sv")
            if not isinstance(map_rel, str) or not isinstance(sv_rel, str):
                continue
            # copy files
            map_src = base_dir / map_rel
            sv_src = base_dir / sv_rel
            map_dst = out_images / Path(map_rel).name
            sv_dst = out_images / Path(sv_rel).name
            map_dst.parent.mkdir(parents=True, exist_ok=True)
            sv_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(map_src, map_dst)
            shutil.copyfile(sv_src, sv_dst)

            new_row = dict(row)
            new_row["image_map"] = str(map_dst.relative_to(out_dir))
            new_row["image_sv"] = str(sv_dst.relative_to(out_dir))
            new_row["trace"] = trace
            out_rows.append(new_row)
            kept += 1

    # Determine split
    train_rows = list(out_rows)
    val_rows = []
    if args.split:
        try:
            parts = [p.strip() for p in str(args.split).split(":")]
            if len(parts) != 2:
                raise ValueError
            tr_part = float(parts[0])
            va_part = float(parts[1])
            if tr_part < 0 or va_part < 0 or (tr_part + va_part) <= 0:
                raise ValueError
            val_ratio = va_part / (tr_part + va_part)
        except Exception:
            raise SystemExit("--split must be in the form 'train:validation', e.g. '9:1'")

        if val_ratio > 0.0 and len(train_rows) > 0:
            rng = random.Random(args.seed)
            rng.shuffle(train_rows)
            if len(train_rows) >= 2:
                n_val = int(len(train_rows) * val_ratio)
                n_val = max(1, min(n_val, len(train_rows) - 1))
            else:
                n_val = 0
            val_rows = train_rows[:n_val]
            train_rows = train_rows[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "train.jsonl"
    write_jsonl(train_jsonl, train_rows)
    if val_rows:
        val_jsonl = out_dir / "validation.jsonl"
        write_jsonl(val_jsonl, val_rows)
    print(
        f"Merged traces for {kept} rows -> train={len(train_rows)}"
        + (f", validation={len(val_rows)}" if val_rows else "")
        + f" in {out_dir}"
    )

    # Optional HF push
    if args.hf_repo and kept > 0:
        try:
            from datasets import Dataset as HFDataset, DatasetDict, Features, Value, Sequence, Image as HFImage
        except Exception as e:
            raise SystemExit(f"Install 'datasets' to push to HF (pip install datasets): {e}")

        features = Features({
            "id": Value("string"),
            "image_map": HFImage(),
            "image_sv": HFImage(),
            "question": Value("string"),
            "options": Sequence(Value("string")),
            "answer": Value("string"),
            "trace": Value("string"),
        })
        def rows_to_cols(rows):
            cols = {"id": [], "image_map": [], "image_sv": [], "question": [], "options": [], "answer": [], "trace": []}
            for r in rows:
                cols["id"].append(r["id"])
                cols["image_map"].append(str((out_dir / r["image_map"]).resolve()))
                cols["image_sv"].append(str((out_dir / r["image_sv"]).resolve()))
                cols["question"].append(r["question"])
                cols["options"].append(r["options"])
                cols["answer"].append(r["answer"])
                cols["trace"].append(r["trace"])
            return cols

        print(f"Pushing typed dataset to {args.hf_repo} (private={args.hf_private})")
        token = args.hf_token
        ds_train = HFDataset.from_dict(rows_to_cols(train_rows), features=features)
        if val_rows:
            ds_val = HFDataset.from_dict(rows_to_cols(val_rows), features=features)
            dsd = DatasetDict({"train": ds_train, "validation": ds_val})
            dsd.push_to_hub(args.hf_repo, token=token if token else None, private=args.hf_private)
        else:
            ds_train.push_to_hub(args.hf_repo, token=token if token else None, private=args.hf_private)
        print("HF Datasets push completed. Dataset preview should render images.")


if __name__ == "__main__":
    main()


