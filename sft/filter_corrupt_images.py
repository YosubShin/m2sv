#!/usr/bin/env python3
# /// script
# dependencies = [
#   "Pillow",
#   "tqdm",
# ]
# ///
"""
Filter out entries whose image files are missing or unreadable.

Supports JSON and JSONL annotations. For packed entries (list of items), the
entire pack is removed if any item has a bad image.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def _load_annotations(path: Path) -> List[Any]:
    if path.suffix == ".jsonl":
        items: List[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    return json.loads(path.read_text(encoding="utf-8"))


def _write_annotations(path: Path, items: List[Any]) -> None:
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=True))
                f.write("\n")
        return
    path.write_text(json.dumps(items, indent=2, ensure_ascii=True), encoding="utf-8")


def _normalize_images(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _resolve_path(image: str, base_path: Path) -> Path:
    image_path = Path(image)
    if image_path.is_absolute():
        return image_path
    return base_path / image


def _iter_item_images(item: Dict[str, Any]) -> Iterable[str]:
    for key in ("image", "images"):
        for image in _normalize_images(item.get(key)):
            yield image


def _is_image_readable(path: Path) -> Tuple[bool, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(path) as img:
                img.verify()
        return True, ""
    except FileNotFoundError:
        return False, "missing"
    except UnidentifiedImageError:
        return False, "corrupt"
    except SyntaxError:
        return False, "corrupt"
    except Warning:
        return False, "corrupt_warning"
    except OSError as exc:
        return False, f"oserror:{exc.__class__.__name__}"


def _check_entry(
    entry: Any, data_root: Path | None
) -> Tuple[bool, List[Tuple[str, str]]]:
    failures: List[Tuple[str, str]] = []

    def check_item(item: Dict[str, Any]) -> None:
        base_path = Path(item.get("data_path", "")) if item.get("data_path") else Path("")
        if data_root:
            base_path = data_root
        for image in _iter_item_images(item):
            image_path = _resolve_path(image, base_path)
            ok, reason = _is_image_readable(image_path)
            if not ok:
                failures.append((str(image_path), reason))

    if isinstance(entry, list):
        for sub in entry:
            if isinstance(sub, dict):
                check_item(sub)
    elif isinstance(entry, dict):
        check_item(entry)

    return (len(failures) == 0), failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove entries with missing or unreadable images."
    )
    parser.add_argument("annotations", type=Path, help="Path to JSON/JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>_filtered.json).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Base directory for relative image paths.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report of removed entries and bad files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write output.")
    args = parser.parse_args()

    annotations_path: Path = args.annotations
    items = _load_annotations(annotations_path)

    kept: List[Any] = []
    removed: List[Dict[str, Any]] = []

    for idx, entry in tqdm(
        enumerate(items), total=len(items), desc="Checking images"
    ):
        ok, failures = _check_entry(entry, args.data_root)
        if ok:
            kept.append(entry)
        else:
            removed.append({"index": idx, "failures": failures})

    output_path = args.output
    if output_path is None:
        suffix = annotations_path.suffix or ".json"
        output_path = annotations_path.with_suffix("")
        output_path = output_path.with_name(output_path.name + "_filtered").with_suffix(
            suffix
        )

    print(f"Total entries: {len(items)}")
    print(f"Kept entries: {len(kept)}")
    print(f"Removed entries: {len(removed)}")

    if removed:
        print("Sample failures:")
        for entry in removed[:5]:
            failure_paths = ", ".join(path for path, _ in entry["failures"])
            print(f"  - index {entry['index']}: {failure_paths}")

    if args.dry_run:
        print("Dry run enabled; no files written.")
        return

    _write_annotations(output_path, kept)
    print(f"Wrote filtered annotations to: {output_path}")

    if args.report:
        report = {
            "input": str(annotations_path),
            "output": str(output_path),
            "total": len(items),
            "kept": len(kept),
            "removed": len(removed),
            "removed_entries": removed,
        }
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(f"Wrote report to: {args.report}")


if __name__ == "__main__":
    main()
