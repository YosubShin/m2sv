#!/usr/bin/env python3
"""
Create a new dataset from m2sv-sft that concatenates image_map and image_sv
into a single image, and writes a JSONL with one 'image' field.

Default input dataset: data/hf/m2sv-sft
Output: data/hf/m2sv-sft-singleimage (customizable)

Schema preserved: id, image (new), question, options, answer, trace
Optionally pushes a typed HF dataset with Image feature for 'image'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from PIL import Image


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def concat_images_horiz(paths: Tuple[Path, Path]) -> Image.Image:
    left = Image.open(paths[0]).convert("RGB")
    right = Image.open(paths[1]).convert("RGB")

    # Match heights by simple resize of right to left's height while preserving aspect
    if left.height != right.height:
        new_w = int(right.width * (left.height / right.height))
        right = right.resize((new_w, left.height), Image.BILINEAR)

    out = Image.new("RGB", (left.width + right.width, left.height), (0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate map and sv images into a single image column dataset")
    parser.add_argument("--input-dataset", type=Path, default=Path("data/hf/m2sv-sft"), help="Path to base dataset directory")
    parser.add_argument("--output-name", default="m2sv-sft-singleimage", help="Name of the output dataset directory under output root")
    parser.add_argument("--output-root", default="data/hf", help="Output root directory")
    parser.add_argument("--orientation", choices=["h", "v"], default="h", help="Concatenation orientation: h (horizontal) or v (vertical)")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo id (e.g. username/dataset-name)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token (or set HF_TOKEN env)")
    parser.add_argument("--hf-private", action="store_true", help="Create/update the HF repo as private")
    return parser.parse_args()


def concat_pair(map_path: Path, sv_path: Path, orientation: str) -> Image.Image:
    left = Image.open(map_path).convert("RGB")
    right = Image.open(sv_path).convert("RGB")
    if orientation == "h":
        # Match heights
        if left.height != right.height:
            new_w = int(right.width * (left.height / right.height))
            right = right.resize((new_w, left.height), Image.BILINEAR)
        out = Image.new("RGB", (left.width + right.width, left.height), (0, 0, 0))
        out.paste(left, (0, 0))
        out.paste(right, (left.width, 0))
        return out
    else:
        # vertical: match widths
        if left.width != right.width:
            new_h = int(right.height * (left.width / right.width))
            right = right.resize((left.width, new_h), Image.BILINEAR)
        out = Image.new("RGB", (left.width, left.height + right.height), (0, 0, 0))
        out.paste(left, (0, 0))
        out.paste(right, (0, left.height))
        return out


def main() -> None:
    args = parse_args()
    base_dir = args.input_dataset
    src_jsonl = base_dir / "train.jsonl"
    if not src_jsonl.exists():
        raise SystemExit(f"Input dataset JSONL not found: {src_jsonl}")

    rows = read_jsonl(src_jsonl)

    out_root = Path(args.output_root)
    out_dir = out_root / args.output_name
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        map_rel = r.get("image_map")
        sv_rel = r.get("image_sv")
        _id = r.get("id")
        if not (isinstance(map_rel, str) and isinstance(sv_rel, str) and isinstance(_id, (str, int))):
            continue
        map_path = base_dir / map_rel
        sv_path = base_dir / sv_rel
        out_img = concat_pair(map_path, sv_path, args.orientation)
        out_path = out_images / f"img_{_id}.jpg"
        out_img.save(out_path, format="JPEG", quality=92)

        new_row = {
            "id": _id,
            "image": str(out_path.relative_to(out_dir)),
            "question": r.get("question"),
            "options": r.get("options"),
            "answer": r.get("answer"),
        }
        # Preserve trace if present
        if isinstance(r.get("trace"), str):
            new_row["trace"] = r["trace"]
        out_rows.append(new_row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "train.jsonl"
    write_jsonl(out_jsonl, out_rows)
    print(f"Wrote {len(out_rows)} rows -> {out_jsonl}")

    # Optional push to HF typed dataset
    if args.hf_repo and len(out_rows) > 0:
        try:
            from datasets import Dataset as HFDataset, Features, Value, Sequence, Image as HFImage
        except Exception as e:
            raise SystemExit(f"Install 'datasets' to push to HF (pip install datasets): {e}")

        features = Features({
            "id": Value("string"),
            "image": HFImage(),
            "question": Value("string"),
            "options": Sequence(Value("string")),
            "answer": Value("string"),
            "trace": Value("string"),
        })
        cols = {"id": [], "image": [], "question": [], "options": [], "answer": [], "trace": []}
        for r in out_rows:
            cols["id"].append(str(r["id"]))
            cols["image"].append(str((out_dir / r["image"]).resolve()))
            cols["question"].append(r.get("question"))
            cols["options"].append(r.get("options"))
            cols["answer"].append(r.get("answer"))
            cols["trace"].append(r.get("trace", ""))

        ds = HFDataset.from_dict(cols, features=features)
        token = args.hf_token
        print(f"Pushing typed dataset to {args.hf_repo} (private={args.hf_private})")
        ds.push_to_hub(args.hf_repo, token=token if token else None, private=args.hf_private)
        print("HF Datasets push completed. Dataset preview should render images.")


if __name__ == "__main__":
    main()


