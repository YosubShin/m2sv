#!/usr/bin/env python3
# /// script
# dependencies = [
#   "datasets>=2.19.0",
# ]
# ///
"""
Convert the allenai/Molmo2-MultiImageQA HF JSONL dataset into LLaVA-style
annotation JSON with multi-image conversations.

Input JSONL rows:
  - image_urls (list[str])
  - image_sha256s (list[str])
  - qa_pairs (dict with "question" and "answer" arrays of equal length)

Output JSON (list[dict]):
[
  {
    "image": ["images/<sha>.jpg", "images/<sha>.jpg", ...],
    "conversations": [
      {"from": "human", "value": "<image>\n<image>\n...question..."},
      {"from": "gpt",   "value": "...answer..."}
    ]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            yield json.loads(line_stripped)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    return count


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    return count


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def download_images_for_row(
    row: Dict[str, Any],
    image_base_dir: Path,
    image_ext: str,
    verify_sha: bool,
) -> None:
    image_urls = row.get("image_urls")
    image_sha256s = row.get("image_sha256s")
    if not isinstance(image_urls, list) or not isinstance(image_sha256s, list):
        raise ValueError("Row is missing image_urls or image_sha256s lists")
    if len(image_urls) != len(image_sha256s):
        raise ValueError("Row image_urls/image_sha256s lengths do not match")

    ext = image_ext or ""
    if ext and not ext.startswith("."):
        ext = f".{ext}"

    image_base_dir.mkdir(parents=True, exist_ok=True)
    for url, sha in zip(image_urls, image_sha256s):
        if not isinstance(url, str) or not isinstance(sha, str):
            raise ValueError("Row image_urls/image_sha256s entries must be strings")
        output_path = image_base_dir / f"{sha}{ext}"
        if output_path.exists():
            continue
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        if verify_sha:
            observed = hashlib.sha256(payload).hexdigest()
            if observed != sha:
                print(f"Warning: SHA mismatch for {url} expected {sha} got {observed}")
        with output_path.open("wb") as f:
            f.write(payload)

    return None


def _download_row_images_safe(
    row: Dict[str, Any],
    image_base_dir: Path,
    image_ext: str,
    verify_sha: bool,
) -> bool:
    try:
        download_images_for_row(
            row,
            image_base_dir=image_base_dir,
            image_ext=image_ext,
            verify_sha=verify_sha,
        )
        return True
    except Exception as exc:
        print(f"Warning: failed to download images for row: {exc}")
        return False


def _print_progress(prefix: str, done: int, total: int, ok: int, failed: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    print(f"{prefix} {done}/{total} ok={ok} failed={failed} ({rate:.2f} rows/s)")


def download_hf_dataset(
    dataset_name: str,
    split: str,
    output_jsonl: Path,
    image_base_dir: Path,
    image_ext: str,
    download_images: bool,
    verify_sha: bool,
    download_workers: int,
    progress_every: int,
    resume: bool,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install 'datasets' to download from HF (pip install datasets)") from exc

    dataset = load_dataset(dataset_name, split=split)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def iter_rows() -> Iterable[Dict[str, Any]]:
        for row in dataset:
            yield row

    existing_rows = 0
    if resume and output_jsonl.exists():
        existing_rows = count_jsonl_rows(output_jsonl)

    if download_images:
        written = 0
        ok = 0
        failed = 0
        start_time = time.time()
        futures: List[concurrent.futures.Future[bool]] = []
        mode = "a" if existing_rows > 0 else "w"
        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
            with output_jsonl.open(mode, encoding="utf-8") as f:
                for idx, row in enumerate(dataset):
                    if idx < existing_rows:
                        continue
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")
                    written += 1
                    futures.append(
                        executor.submit(
                            _download_row_images_safe,
                            row,
                            image_base_dir,
                            image_ext,
                            verify_sha,
                        )
                    )
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    ok += 1
                else:
                    failed += 1
                done = ok + failed
                if progress_every > 0 and (done % progress_every == 0 or done == written):
                    _print_progress("Images", done, written, ok, failed, start_time)
        if progress_every > 0 and (ok + failed) != 0 and (ok + failed) != written:
            _print_progress("Images", ok + failed, written, ok, failed, start_time)
        return existing_rows + written

    if existing_rows > 0:
        new_rows = append_jsonl(output_jsonl, (row for idx, row in enumerate(dataset) if idx >= existing_rows))
        return existing_rows + new_rows

    return write_jsonl(output_jsonl, iter_rows())


def resolve_image_paths(
    image_sha256s: List[str], dataset_dir: Path, image_dir: Path, image_ext: str
) -> tuple[List[str], List[Path]]:
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

    rel_paths = [f"{rel_dir_str}/{sha}{ext}" for sha in image_sha256s]
    fs_paths = [base_dir / f"{sha}{ext}" for sha in image_sha256s]
    return rel_paths, fs_paths


def convert_row_to_conversations(
    row: Dict[str, Any],
    dataset_dir: Path,
    image_dir: Path,
    image_ext: str,
    multi_turn: bool,
    require_images: bool,
) -> List[Dict[str, Any]]:
    image_sha256s = row.get("image_sha256s")
    qa_pairs = row.get("qa_pairs", {})

    if not isinstance(image_sha256s, list) or not all(isinstance(s, str) for s in image_sha256s):
        raise ValueError("Row is missing image_sha256s list[str]")
    if not image_sha256s:
        raise ValueError("Row has empty image_sha256s")

    questions = qa_pairs.get("question")
    answers = qa_pairs.get("answer")
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise ValueError("Row qa_pairs is missing question/answer lists")
    if len(questions) != len(answers):
        raise ValueError("Row qa_pairs question/answer lengths do not match")

    image_paths, fs_paths = resolve_image_paths(image_sha256s, dataset_dir, image_dir, image_ext)
    if require_images:
        missing = [path for path in fs_paths if not path.exists()]
        if missing:
            raise ValueError("Row has missing image files")
    images_block = "\n".join("<image>" for _ in image_paths)

    conversations: List[Dict[str, Any]] = []
    if multi_turn:
        convo: List[Dict[str, str]] = []
        for question, answer in zip(questions, answers):
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError("Row qa_pairs question/answer entries must be strings")
            question_text = question.rstrip("\n")
            human_value = f"{images_block}\n{question_text}" if question_text else images_block
            convo.append({"from": "human", "value": human_value})
            convo.append({"from": "gpt", "value": answer})
        conversations.append({"image": image_paths, "conversations": convo})
    else:
        for question, answer in zip(questions, answers):
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError("Row qa_pairs question/answer entries must be strings")
            question_text = question.rstrip("\n")
            human_value = f"{images_block}\n{question_text}" if question_text else images_block
            conversations.append(
                {
                    "image": image_paths,
                    "conversations": [
                        {"from": "human", "value": human_value},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )

    return conversations


def convert_dataset(
    input_jsonl: Path,
    dataset_dir: Path,
    image_dir: Path,
    image_ext: str,
    multi_turn: bool,
    require_images: bool,
) -> List[Dict[str, Any]]:
    conversations: List[Dict[str, Any]] = []
    for row in read_jsonl(input_jsonl):
        try:
            conversations.extend(
                convert_row_to_conversations(
                    row,
                    dataset_dir=dataset_dir,
                    image_dir=image_dir,
                    image_ext=image_ext,
                    multi_turn=multi_turn,
                    require_images=require_images,
                )
            )
        except Exception:
            # Skip malformed rows to keep conversion robust for large datasets
            continue
    return conversations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Molmo2-MultiImageQA JSONL to LLaVA-style conversation annotations"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/hf/molmo2-multiimageqa"),
        help="Dataset directory containing train.jsonl and images/",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional explicit path to the input JSONL (defaults to dataset-dir/train.jsonl)",
    )
    parser.add_argument(
        "--download-hf",
        action="store_true",
        help="Download the dataset from Hugging Face into dataset-dir when input JSONL is missing",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="allenai/Molmo2-MultiImageQA",
        help="Hugging Face dataset name for download",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download images from image_urls into image-dir",
    )
    parser.add_argument(
        "--verify-sha",
        action="store_true",
        help="Verify downloaded image bytes against image_sha256s",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Number of parallel download workers when --download-images is set",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N completed rows (0 disables)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume JSONL/image download if output JSONL already exists",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write the output annotations JSON (defaults to dataset-dir/annotations.json)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("images"),
        help="Directory (relative to dataset-dir) containing images",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg",
        help="Image filename extension without dot (default: jpg). Use empty string for no extension.",
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Emit a single multi-turn conversation per row instead of one entry per QA pair",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Include rows even if some images are missing on disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_jsonl is not None:
        input_jsonl = args.input_jsonl
    elif args.download_hf:
        input_jsonl = args.dataset_dir / f"{args.split}.jsonl"
    else:
        input_jsonl = args.dataset_dir / "train.jsonl"
    output_json = args.output_json or (args.dataset_dir / "annotations.json")
    image_dir = args.image_dir if args.image_dir.is_absolute() else (args.dataset_dir / args.image_dir)

    if not input_jsonl.exists() or (args.download_hf and args.resume):
        if not args.download_hf:
            raise SystemExit(f"Input JSONL not found: {input_jsonl}")
        rows = download_hf_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            output_jsonl=input_jsonl,
            image_base_dir=image_dir,
            image_ext=args.image_ext,
            download_images=args.download_images,
            verify_sha=args.verify_sha,
            download_workers=args.download_workers,
            progress_every=args.progress_every,
            resume=args.resume,
        )
        print(f"Wrote {rows} rows to {input_jsonl}")

    conversations = convert_dataset(
        input_jsonl=input_jsonl,
        dataset_dir=args.dataset_dir,
        image_dir=args.image_dir,
        image_ext=args.image_ext,
        multi_turn=args.multi_turn,
        require_images=not args.allow_missing_images,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(conversations)} conversations to {output_json}")


if __name__ == "__main__":
    main()
