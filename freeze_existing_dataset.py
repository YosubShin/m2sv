import json
import logging
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("freeze_existing_dataset")


def load_hf_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_filtered_json(path: Path) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    # pandas to_json orient=records already yields list; but be defensive
    return data.get("data", [])


def to_blueprint_from_hf(row: dict, place: str, defaults: dict) -> dict:
    meta = row.get("meta", {})
    labels = meta.get("labels") or row.get("options")
    azimuths = meta.get("azimuths")
    answer = row.get("answer")
    if labels is None or azimuths is None:
        raise ValueError("HF JSONL row missing labels/azimuths in meta")
    try:
        idx = labels.index(answer)
    except ValueError:
        raise ValueError("Answer label not found in labels list")
    answer_heading = azimuths[idx]

    return {
        "uid": f"{place}|{row['id']}",
        "intersection_id": str(row["id"]),
        "place": place,
        "lat": meta.get("lat"),
        "lng": meta.get("lng"),
        "azimuths": azimuths,
        "labels": labels,
        "answer": answer,
        "answer_heading": answer_heading,
        "sv_pano_id": meta.get("pano_id"),
        "sv_distance_m": meta.get("distance_m"),
        "params": {
            "map_zoom": defaults["map_zoom"],
            "map_type": defaults["map_type"],
            "map_size": defaults["map_size"],
            "sv_fov": defaults["sv_fov"],
            "sv_pitch": defaults["sv_pitch"],
        },
    }


def to_blueprint_from_filtered(row: dict, place: str, defaults: dict) -> dict:
    roads = row.get("roads", [])
    labels = [r.get("label") for r in roads]
    azimuths = [r.get("azimuth") for r in roads]
    answer = row.get("vlm_answer")
    try:
        idx = labels.index(answer)
    except ValueError:
        raise ValueError("Answer label not found in roads labels")
    answer_heading = azimuths[idx]

    return {
        "uid": f"{place}|{row['intersection_id']}",
        "intersection_id": str(row["intersection_id"]),
        "place": place,
        "lat": row.get("lat"),
        "lng": row.get("lng"),
        "azimuths": azimuths,
        "labels": labels,
        "answer": answer,
        "answer_heading": answer_heading,
        "sv_pano_id": row.get("sv_pano_id"),
        "sv_distance_m": row.get("sv_distance_m"),
        "params": {
            "map_zoom": defaults["map_zoom"],
            "map_type": defaults["map_type"],
            "map_size": defaults["map_size"],
            "sv_fov": defaults["sv_fov"],
            "sv_pitch": defaults["sv_pitch"],
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Freeze metadata for an already-created dev dataset into a blueprint JSONL")
    parser.add_argument("--source-jsonl", type=str, default=None, help="Path to HF JSONL (e.g., data/hf/<name>/train.jsonl)")
    parser.add_argument("--source-json", type=str, default=None, help="Path to filtered intersections JSON (e.g., data/intersection_dataset_filtered.json)")
    parser.add_argument("--out", type=str, default="data/blueprints/dev.jsonl", help="Output blueprint JSONL path")
    parser.add_argument("--place", type=str, default="Honolulu, Hawaii, USA", help="Place name to record in the blueprint")
    parser.add_argument("--map-zoom", type=int, default=19)
    parser.add_argument("--map-type", choices=["satellite", "roadmap", "terrain", "hybrid"], default="satellite")
    parser.add_argument("--map-size", default="640x640")
    parser.add_argument("--sv-fov", type=float, default=120.0)
    parser.add_argument("--sv-pitch", type=float, default=0.0)

    args = parser.parse_args()

    defaults = {
        "map_zoom": args.map_zoom,
        "map_type": args.map_type,
        "map_size": args.map_size,
        "sv_fov": args.sv_fov,
        "sv_pitch": args.sv_pitch,
    }

    rows: List[dict] = []
    if args.source_jsonl:
        logger.info(f"Loading HF JSONL from {args.source_jsonl}")
        for r in load_hf_jsonl(Path(args.source_jsonl)):
            rows.append(to_blueprint_from_hf(r, args.place, defaults))
    elif args.source_json:
        logger.info(f"Loading filtered JSON from {args.source_json}")
        for r in load_filtered_json(Path(args.source_json)):
            rows.append(to_blueprint_from_filtered(r, args.place, defaults))
    else:
        raise SystemExit("Provide either --source-jsonl or --source-json")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(rows)} blueprint rows to {out_path}")


if __name__ == "__main__":
    main()


