import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict

import requests
from PIL import Image
from dotenv import load_dotenv
from datasets import Dataset as HFDataset, Features, Value, Sequence, Image as HFImage
from tqdm import tqdm


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("render_from_blueprint")

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


def static_map_url(lat, lon, zoom=19, size="640x640", maptype="satellite"):
    return (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}&maptype={maptype}&key={API_KEY}"
    )


def streetview_url_from_pano(pano_id, azimuth, size, fov, pitch):
    return (
        "https://maps.googleapis.com/maps/api/streetview?"
        f"size={size}&pano={pano_id}&heading={azimuth}&fov={fov}&pitch={pitch}&key={API_KEY}"
    )


def _format_float_for_name(value):
    return f"{value:.6f}".replace("-", "m").replace(".", "p")


def cache_paths(root: Path, lat: float, lon: float, heading: float, size: str, fov: float, pitch: float):
    lat_s = _format_float_for_name(lat)
    lon_s = _format_float_for_name(lon)
    head_s = _format_float_for_name(heading)
    size_s = size.replace("x", "_")
    fov_s = _format_float_for_name(fov)
    pitch_s = _format_float_for_name(pitch)
    maps = root / "cache" / "maps"
    sv = root / "cache" / "sv"
    maps.mkdir(parents=True, exist_ok=True)
    sv.mkdir(parents=True, exist_ok=True)
    return (
        maps / f"map_{lat_s}_{lon_s}.png",
        sv / f"sv_{lat_s}_{lon_s}_h{head_s}_{size_s}_fov{fov_s}_p{pitch_s}.png",
    )


def overlay_arrows_on_map(base_map_path: Path, azimuths, labels, out_path: Path):
    # Reuse logic by importing from create_dataset would risk side effects; replicate minimal logic
    from PIL import ImageDraw, ImageFont
    import math

    image = Image.open(base_map_path).convert("RGBA")
    w, h = image.size
    cx, cy = w / 2, h / 2
    draw = ImageDraw.Draw(image)

    colors = [
        (255, 69, 58, 255),
        (52, 199, 89, 255),
        (0, 122, 255, 255),
        (255, 159, 10, 255),
        (175, 82, 222, 255),
        (255, 214, 10, 255),
    ]
    length = int(min(w, h) * 0.28)

    try:
        font = ImageFont.truetype("arial.ttf", size=int(min(w, h) * 0.05))
    except Exception:
        font = ImageFont.load_default()

    def _draw_arrow(draw, cx, cy, angle_deg, length, color):
        theta = math.radians(angle_deg)
        dx = math.sin(theta)
        dy = -math.cos(theta)
        x2 = cx + dx * length
        y2 = cy + dy * length
        draw.line((cx, cy, x2, y2), fill=color, width=6)
        head_len = max(12, int(length * 0.12))
        head_angle = math.radians(25)
        left_theta = theta + math.pi - head_angle
        right_theta = theta + math.pi + head_angle
        x3 = x2 + math.sin(left_theta) * head_len
        y3 = y2 - math.cos(left_theta) * head_len
        x4 = x2 + math.sin(right_theta) * head_len
        y4 = y2 - math.cos(right_theta) * head_len
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=color)
        return x2, y2

    for i, (az, lbl) in enumerate(zip(azimuths, labels)):
        color = colors[i % len(colors)]
        tip_x, tip_y = _draw_arrow(draw, cx, cy, az, length, color)
        label_radius = int(min(w, h) * 0.035)
        offset = int(min(w, h) * 0.015)
        theta = math.radians(az)
        dx = math.sin(theta)
        dy = -math.cos(theta)
        lx = tip_x + dx * offset
        ly = tip_y + dy * offset
        draw.ellipse((lx - label_radius, ly - label_radius, lx + label_radius, ly + label_radius), fill=(0, 0, 0, 180))
        try:
            bbox = draw.textbbox((0, 0), lbl, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textlength(lbl, font=font), font.size
        draw.text((lx - tw / 2, ly - th / 2), lbl, fill=(255, 255, 255, 255), font=font)

    image = image.convert("RGB")
    image.save(out_path)
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render dataset from a frozen blueprint JSONL")
    parser.add_argument("blueprint", help="Path to blueprint JSONL file")
    parser.add_argument("dataset_name", help="Dataset name (under data/hf)")
    parser.add_argument("--output-root", default="data/hf", help="Output root directory")
    parser.add_argument("--override-map-zoom", type=int, default=None)
    parser.add_argument("--override-map-type", choices=["satellite", "roadmap", "terrain", "hybrid"], default=None)
    parser.add_argument("--override-map-size", default=None)
    parser.add_argument("--override-sv-fov", type=float, default=None)
    parser.add_argument("--override-sv-pitch", type=float, default=None)
    parser.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo id (e.g. username/dataset-name)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token (or set HF_TOKEN env)")
    parser.add_argument("--hf-private", action="store_true", help="Create/update the HF repo as private")

    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("GOOGLE_MAPS_API_KEY is not set.")

    dataset_dir = Path(args.output_root) / args.dataset_name
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.blueprint, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    vlm_rows = []
    for row in tqdm(rows, total=len(rows), desc="Rendering"):
        lat = row["lat"]
        lon = row["lng"]
        azimuths = row["azimuths"]
        labels = row["labels"]
        pano_id = row["sv_pano_id"]

        params = row.get("params", {})
        map_zoom = args.override_map_zoom if args.override_map_zoom is not None else params.get("map_zoom", 19)
        map_type = args.override_map_type if args.override_map_type is not None else params.get("map_type", "satellite")
        map_size = args.override_map_size if args.override_map_size is not None else params.get("map_size", "640x640")
        sv_fov = args.override_sv_fov if args.override_sv_fov is not None else params.get("sv_fov", 120.0)
        sv_pitch = args.override_sv_pitch if args.override_sv_pitch is not None else params.get("sv_pitch", 0.0)

        # Fetch static map (cached)
        map_cache_path, sv_cache_path = cache_paths(Path.cwd(), lat, lon, row["answer_heading"], map_size, sv_fov, sv_pitch)
        if not map_cache_path.exists():
            url = static_map_url(lat, lon, zoom=map_zoom, size=map_size, maptype=map_type)
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            map_cache_path.write_bytes(r.content)

        # Overlay labeled arrows
        overlay_path = images_dir / f"map_{row['intersection_id']}.jpg"
        overlay_arrows_on_map(map_cache_path, azimuths, labels, overlay_path)

        # Fetch SV for the gold label heading (cached)
        if not sv_cache_path.exists():
            url = streetview_url_from_pano(pano_id, row["answer_heading"], map_size, sv_fov, sv_pitch)
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            sv_cache_path.write_bytes(r.content)
        sv_dst = images_dir / f"sv_{row['intersection_id']}.jpg"
        try:
            Image.open(sv_cache_path).convert("RGB").save(sv_dst, format="JPEG", quality=92)
        except Exception:
            shutil.copyfile(sv_cache_path, sv_dst)

        question = "Which labeled direction on the map corresponds to the direction in which the street view photo was taken?"
        opts = "\n".join(labels)

        vlm_rows.append({
            "id": row["intersection_id"],
            "image_map": str(overlay_path.relative_to(dataset_dir)),
            "image_sv": str(sv_dst.relative_to(dataset_dir)),
            "question": f"{question}\n\nOptions:\n{opts}",
            "options": labels,
            "answer": row["answer"],
            "meta": {
                "lat": lat,
                "lng": lon,
                "pano_id": pano_id,
                "distance_m": row.get("sv_distance_m"),
                "azimuths": azimuths,
                "labels": labels,
            },
        })

    dataset_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_dir / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in vlm_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Rendered {len(vlm_rows)} rows to {jsonl_path}")

    # Optionally push to Hugging Face (typed dataset with Image features)
    if args.hf_repo and len(vlm_rows) > 0:
        token = args.hf_token or os.getenv("HF_TOKEN")

        features = Features({
            "id": Value("string"),
            "image_map": HFImage(),
            "image_sv": HFImage(),
            "question": Value("string"),
            "options": Sequence(Value("string")),
            "answer": Value("string"),
        })
        rows = {
            "id": [],
            "image_map": [],
            "image_sv": [],
            "question": [],
            "options": [],
            "answer": [],
        }
        for row in vlm_rows:
            rows["id"].append(row["id"])
            # Convert to absolute paths for upload
            rows["image_map"].append(str((dataset_dir / row["image_map"]).resolve()))
            rows["image_sv"].append(str((dataset_dir / row["image_sv"]).resolve()))
            rows["question"].append(row["question"])
            rows["options"].append(row["options"])
            rows["answer"].append(row["answer"])

        ds = HFDataset.from_dict(rows, features=features)
        logger.info(f"Pushing typed dataset to {args.hf_repo} (private={args.hf_private})")
        ds.push_to_hub(args.hf_repo, token=token if token else None, private=args.hf_private)
        logger.info("HF Datasets push completed. Dataset preview should render images.")


if __name__ == "__main__":
    main()
