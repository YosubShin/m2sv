import argparse
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset, DatasetDict, Image as HFImage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("merge_jsonl_splits")


def _resolve_image_paths(jsonl_path: Path, field_map_image: Optional[str], field_sv_image: Optional[str]):
    base_dir = jsonl_path.parent.resolve()

    map_key = field_map_image or "image_map"
    sv_key = field_sv_image or "image_sv"

    def _mapper(example):
        map_rel = example.get(map_key)
        sv_rel = example.get(sv_key)
        return {
            "image_map": str((base_dir / map_rel).resolve()) if map_rel else None,
            "image_sv": str((base_dir / sv_rel).resolve()) if sv_rel else None,
        }

    return _mapper


def load_split(jsonl_path: Path, field_map_image: Optional[str] = None, field_sv_image: Optional[str] = None):
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    raw = load_dataset("json", data_files={"_": str(jsonl_path)})["_"]
    raw = raw.map(_resolve_image_paths(jsonl_path, field_map_image, field_sv_image))
    # Cast to Image features so previews/rendering work on the Hub
    raw = raw.cast_column("image_map", HFImage())
    raw = raw.cast_column("image_sv", HFImage())
    return raw


def main():
    parser = argparse.ArgumentParser(description="Merge train/dev JSONL files into a single HF Dataset with train/validation splits")
    parser.add_argument("train_jsonl", type=str, help="Path to train split JSONL")
    parser.add_argument("dev_jsonl", type=str, help="Path to dev split JSONL (will be renamed to 'validation')")
    parser.add_argument("--out-dir", type=str, default="data/hf/merged", help="Directory to save the merged dataset (save_to_disk)")
    parser.add_argument("--repo", type=str, default=None, help="Optional Hugging Face repo id to push (e.g., username/dataset)")
    parser.add_argument("--private", action="store_true", help="Mark HF repo as private when pushing")
    parser.add_argument("--field-map-image", type=str, default=None, help="Field name for map image (default: image_map)")
    parser.add_argument("--field-sv-image", type=str, default=None, help="Field name for street view image (default: image_sv)")

    args = parser.parse_args()

    train_path = Path(args.train_jsonl)
    dev_path = Path(args.dev_jsonl)

    logger.info(f"Loading train split from {train_path}")
    ds_train = load_split(train_path, args.field_map_image, args.field_sv_image)
    logger.info(f"Train split: {len(ds_train)} rows")

    logger.info(f"Loading dev split (as validation) from {dev_path}")
    ds_val = load_split(dev_path, args.field_map_image, args.field_sv_image)
    logger.info(f"Validation split: {len(ds_val)} rows")

    ds = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged DatasetDict to {out_dir}")
    ds.save_to_disk(str(out_dir))

    if args.repo:
        logger.info(f"Pushing to Hugging Face Hub: {args.repo} (private={args.private})")
        ds.push_to_hub(args.repo, private=args.private)
        logger.info("Push completed")
    else:
        logger.info("Skip pushing to Hub (no --repo provided)")


if __name__ == "__main__":
    main()


