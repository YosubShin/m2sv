#!/usr/bin/env python3
# /// script
# dependencies = [
#   "datasets>=2.19.0",
# ]
# ///
"""
Download allenai/Molmo2-SynMultiImageQA and export LLaVA-style annotations.

Input HF rows:
  - images (list of images)
  - qa_pairs (dict with "question", "explanation", "answer" arrays)
  - category/subset field (configurable) used for per-category sampling

Output:
  - train.jsonl (sampled rows)
  - annotations.json (LLaVA-style list with multi-image conversations)
  - images/ (downloaded images)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def resolve_image_paths(
    dataset_dir: Path,
    image_dir: Path,
    image_ext: str,
    row_index: int,
    image_count: int,
    prefix: str,
) -> Tuple[List[str], List[Path]]:
    if image_dir.is_absolute():
        base_dir = image_dir
    else:
        base_dir = dataset_dir / image_dir

    try:
        rel_dir = base_dir.relative_to(dataset_dir)
        rel_dir_str = rel_dir.as_posix()
    except ValueError:
        rel_dir_str = base_dir.as_posix()

    ext = image_ext or ""
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    if not ext:
        ext = ".jpg"

    rel_paths = [f"{rel_dir_str}/{prefix}{row_index:08d}_{idx:02d}{ext}" for idx in range(image_count)]
    fs_paths = [base_dir / f"{prefix}{row_index:08d}_{idx:02d}{ext}" for idx in range(image_count)]
    return rel_paths, fs_paths


def save_image_entry(image_entry: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    if isinstance(image_entry, dict):
        source_path = image_entry.get("path")
        payload = image_entry.get("bytes")
        if isinstance(source_path, str) and Path(source_path).exists():
            shutil.copyfile(source_path, output_path)
            return
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, (bytes, bytearray)):
            with output_path.open("wb") as f:
                f.write(payload)
            return

    save_fn = getattr(image_entry, "save", None)
    if callable(save_fn):
        image_entry.save(output_path)
        return

    raise ValueError("Unsupported image entry type; expected dict/path/bytes or PIL image")


def sample_indices_by_category(
    dataset: Any,
    category_field: str,
    sample_fraction: float,
    seed: int,
    min_per_category: int,
) -> List[int]:
    if category_field in getattr(dataset, "column_names", []):
        categories = dataset[category_field]
    else:
        categories = ["unknown"] * len(dataset)

    category_to_indices: Dict[str, List[int]] = {}
    for idx, category in enumerate(categories):
        category_to_indices.setdefault(str(category), []).append(idx)

    rng = random.Random(seed)
    sampled: List[int] = []
    for indices in category_to_indices.values():
        if not indices:
            continue
        rng.shuffle(indices)
        k = int(len(indices) * sample_fraction)
        if min_per_category > 0:
            k = max(k, min_per_category)
        k = min(k, len(indices))
        sampled.extend(indices[:k])

    return sorted(sampled)


def get_row_images(dataset: Any, images_field: str, row_idx: int) -> List[Any]:
    try:
        column = dataset.data.column(images_field)
    except Exception as exc:
        raise ValueError(f"Missing images field: {images_field}") from exc

    entry = column[row_idx].as_py()
    if entry is None:
        return []
    if isinstance(entry, list):
        return entry
    return [entry]


def get_row_dict(dataset: Any, row_idx: int) -> Dict[str, Any]:
    row = dataset.data.slice(row_idx, 1).to_pylist()
    if not row:
        return {}
    if isinstance(row[0], dict):
        return row[0]
    return {}


def serialize_row_for_jsonl(
    row: Dict[str, Any],
    images_field: str,
    image_paths: List[str],
    config_name: str,
) -> Dict[str, Any]:
    sanitized = dict(row)
    sanitized[images_field] = image_paths
    sanitized["config"] = config_name
    return sanitized


def format_conversations(
    image_paths: List[str],
    qa_pairs: Dict[str, Any],
    multi_turn: bool,
    use_explanations: bool,
    explanation_with_final: bool,
) -> List[Dict[str, Any]]:
    questions = qa_pairs.get("question")
    answers = qa_pairs.get("answer")
    explanations = qa_pairs.get("explanation")

    if not isinstance(questions, list) or not isinstance(answers, list):
        raise ValueError("Row qa_pairs is missing question/answer lists")
    if len(questions) != len(answers):
        raise ValueError("Row qa_pairs question/answer lengths do not match")

    images_block = "\n".join("<image>" for _ in image_paths)
    conversations: List[Dict[str, Any]] = []
    if multi_turn:
        convo: List[Dict[str, str]] = []
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError("Row qa_pairs question/answer entries must be strings")
            question_text = question.rstrip("\n")
            human_value = f"{images_block}\n{question_text}" if question_text else images_block
            assistant_value = answer
            if use_explanations and isinstance(explanations, list) and idx < len(explanations):
                exp = explanations[idx]
                if isinstance(exp, str) and exp.strip():
                    if explanation_with_final:
                        assistant_value = f"{exp}\nFinal answer: {answer}"
                    else:
                        assistant_value = exp
            convo.append({"from": "human", "value": human_value})
            convo.append({"from": "gpt", "value": assistant_value})
        conversations.append({"image": image_paths, "conversations": convo})
    else:
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError("Row qa_pairs question/answer entries must be strings")
            question_text = question.rstrip("\n")
            human_value = f"{images_block}\n{question_text}" if question_text else images_block
            assistant_value = answer
            if use_explanations and isinstance(explanations, list) and idx < len(explanations):
                exp = explanations[idx]
                if isinstance(exp, str) and exp.strip():
                    if explanation_with_final:
                        assistant_value = f"{exp}\nFinal answer: {answer}"
                    else:
                        assistant_value = exp
            conversations.append(
                {
                    "image": image_paths,
                    "conversations": [
                        {"from": "human", "value": human_value},
                        {"from": "gpt", "value": assistant_value},
                    ],
                }
            )
    return conversations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Molmo2-SynMultiImageQA and export LLaVA-style annotations"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/hf/molmo2-synmultiimageqa"),
        help="Dataset directory to write train.jsonl, annotations.json, and images/",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="allenai/Molmo2-SynMultiImageQA",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hugging Face dataset config name (e.g., chart, chemical, circuit, diagram, doc, graphic, music, table)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Process all known configs and sample within each",
    )
    parser.add_argument(
        "--config-names",
        type=str,
        default=None,
        help="Comma-separated list of config names to process (overrides --config-name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download",
    )
    parser.add_argument(
        "--category-field",
        type=str,
        default="category",
        help="Field name for sub-category sampling",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Fraction to sample per category (default: 0.1)",
    )
    parser.add_argument(
        "--min-per-category",
        type=int,
        default=0,
        help="Minimum rows to keep per category (default: 0)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=17,
        help="Random seed for category sampling",
    )
    parser.add_argument(
        "--images-field",
        type=str,
        default="images",
        help="Field name containing image list",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("images"),
        help="Directory (relative to dataset-dir) to store images",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg",
        help="Image filename extension without dot (default: jpg)",
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Emit a single multi-turn conversation per row instead of one entry per QA pair",
    )
    parser.add_argument(
        "--use-explanations",
        action="store_true",
        help="Use qa_pairs.explanation for assistant responses when present",
    )
    parser.add_argument(
        "--explanation-with-final",
        action="store_true",
        help="Append 'Final answer: <answer>' after explanations when --use-explanations is set",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Include rows even if some images are missing on disk",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N processed rows (0 disables)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume writing train.jsonl/annotations.json if train.jsonl exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install 'datasets' to download from HF (pip install datasets)") from exc

    known_configs = ["chart", "chemical", "circuit", "diagram", "doc", "graphic", "music", "table"]
    if args.config_names:
        config_names = [name.strip() for name in args.config_names.split(",") if name.strip()]
    elif args.all_configs:
        config_names = known_configs
    else:
        config_names = [args.config_name]

    if not any(config_names):
        raise SystemExit("Config name is required. Use --config-name or --all-configs.")

    dataset_dir = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.image_dir if args.image_dir.is_absolute() else (dataset_dir / args.image_dir)
    train_jsonl = dataset_dir / "train.jsonl"
    annotations_json = dataset_dir / "annotations.json"

    start_index = 0
    if args.resume and train_jsonl.exists():
        start_index = count_jsonl_rows(train_jsonl)

    mode = "a" if start_index > 0 else "w"
    processed_rows = 0
    written_rows = 0
    annotation_entries = 0
    start_time = time.time()

    with train_jsonl.open(mode, encoding="utf-8") as train_f, annotations_json.open("w", encoding="utf-8") as ann_f:
        ann_f.write("[\n")
        first_ann = True
        total_rows = 0
        for config_name in config_names:
            dataset = load_dataset(args.dataset_name, config_name, split=args.split)
            sample_indices = sample_indices_by_category(
                dataset=dataset,
                category_field=args.category_field,
                sample_fraction=args.sample_fraction,
                seed=args.sample_seed,
                min_per_category=args.min_per_category,
            )
            total_rows += len(sample_indices)

            for pos, row_idx in enumerate(sample_indices):
                skip_train_write = processed_rows < start_index
                row = get_row_dict(dataset, row_idx)
                processed_rows += 1
                images = get_row_images(dataset, args.images_field, row_idx)
                qa_pairs = row.get("qa_pairs", {})
                if not isinstance(images, list) or not images:
                    if args.allow_missing_images:
                        continue
                    raise ValueError("Row is missing images list")

                rel_paths, fs_paths = resolve_image_paths(
                    dataset_dir=dataset_dir,
                    image_dir=args.image_dir,
                    image_ext=args.image_ext,
                    row_index=row_idx,
                    image_count=len(images),
                    prefix=f"{config_name}_",
                )

                missing_images = False
                for image_entry, output_path in zip(images, fs_paths):
                    if skip_train_write and output_path.exists():
                        continue
                    try:
                        save_image_entry(image_entry, output_path)
                    except Exception:
                        missing_images = True
                        break
                if missing_images and not args.allow_missing_images:
                    continue

                if not skip_train_write:
                    serial_row = serialize_row_for_jsonl(
                        row=row,
                        images_field=args.images_field,
                        image_paths=rel_paths,
                        config_name=config_name,
                    )
                    json.dump(serial_row, train_f, ensure_ascii=False)
                    train_f.write("\n")
                    written_rows += 1

                for convo in format_conversations(
                    image_paths=rel_paths,
                    qa_pairs=qa_pairs,
                    multi_turn=args.multi_turn,
                    use_explanations=args.use_explanations,
                    explanation_with_final=args.explanation_with_final,
                ):
                    if not first_ann:
                        ann_f.write(",\n")
                    json.dump(convo, ann_f, ensure_ascii=False)
                    first_ann = False
                    annotation_entries += 1

                if args.progress_every > 0 and processed_rows % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    rate = processed_rows / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Rows {processed_rows}/{total_rows} written={written_rows} ann={annotation_entries} "
                        f"({rate:.2f} rows/s)"
                    )
        ann_f.write("\n]\n")

    print(f"Wrote {written_rows} rows to {train_jsonl}")
    print(f"Wrote {annotation_entries} conversations to {annotations_json}")


if __name__ == "__main__":
    main()
