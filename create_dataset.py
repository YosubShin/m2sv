import os
import time
import random
import logging
from pathlib import Path
import numpy as np
import requests
import osmnx as ox
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import math
import json
import shutil
import argparse
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
from datasets import Dataset as HFDataset, Features, Value, Sequence, Image as HFImage

load_dotenv()

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("create_dataset")

# ---------- CONFIG ----------
CITY_NAME = "Honolulu, Hawaii, USA"
N_SAMPLES = 5  # number of intersections to test
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
IMAGE_SIZE = "640x640"
FOV = 120.0
SLEEP_SEC = 0.1  # be polite to the API
OUTPUT_JSON = "data/intersection_dataset_filtered.json"
# Maximum allowed distance between the requested intersection and the Street View panorama
MAX_SV_DISTANCE_M = 5.0
# Radius used in the metadata lookup to constrain the nearest pano search
METADATA_RADIUS_M = 5
# Image caches
CACHE_DIR = Path("cache")
CACHE_MAPS_DIR = CACHE_DIR / "maps"
CACHE_SV_DIR = CACHE_DIR / "sv"
# ----------------------------

if not API_KEY:
    logger.error(
        "GOOGLE_MAPS_API_KEY is not set. Please set it in your environment or .env file."
    )
    raise SystemExit(1)


# Step 1. Load city road graph
logger.info(f"Loading city graph for {CITY_NAME}...")
G = ox.graph_from_place(CITY_NAME, network_type="drive")
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
logger.info(
    f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges. Sampling {N_SAMPLES} intersections."
)
try:
    edges_index_names = list(edges.index.names) if hasattr(edges.index, "names") else []
    logger.debug(
        f"Edges columns: {list(edges.columns)} | index names: {edges_index_names} | index levels: {getattr(edges.index, 'nlevels', 'NA')}"
    )
except Exception:
    # best-effort logging only
    pass

# Step 2. Sample random intersections
sample_nodes = nodes.sample(N_SAMPLES, random_state=42)


# ---------- Helper functions ----------


def compute_azimuth(lat1, lon1, lat2, lon2):
    """Compute bearing (azimuth angle in degrees) from point 1 to 2."""
    dy = lat2 - lat1
    dx = np.cos(np.radians(lat1)) * (lon2 - lon1)
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def compute_edge_heading_at_node(node_id, row_u, row_v, edge_row, lat_node, lon_node):
    """Compute heading along the edge geometry at the given node.

    Uses the first segment of the LineString at the node side (u or v). Falls back
    to the node-to-node heading if geometry is missing or too short.
    """
    try:
        geom = edge_row.get("geometry", None)
        if geom is not None and hasattr(geom, "coords"):
            coords = list(geom.coords)
            if len(coords) >= 2:
                # Shapely coords are (x, y) == (lon, lat)
                if node_id == row_u:
                    (lon1, lat1), (lon2, lat2) = coords[0], coords[1]
                else:
                    (lon1, lat1), (lon2, lat2) = coords[-1], coords[-2]
                return compute_azimuth(lat1, lon1, lat2, lon2)
    except Exception:
        # best-effort only; fall back below
        pass

    # Fallback: use the straight line between nodes
    try:
        other_node = row_v if row_u == node_id else row_u
        lat2, lon2 = edge_row["_lat_other"], edge_row["_lon_other"]
        return compute_azimuth(lat_node, lon_node, lat2, lon2)
    except Exception:
        return None


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Return great-circle distance in meters between two lat/lon points."""
    r_earth_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_earth_m * c


def get_streetview_metadata(lat, lon, radius_m=None):
    """Query Street View metadata for the nearest panorama to a point.

    Returns metadata dict on success (status==OK), otherwise None.
    Optionally constrains search to radius_m meters.
    """
    base = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    radius_part = f"&radius={int(radius_m)}" if radius_m is not None else ""
    url = f"{base}location={lat},{lon}{radius_part}&key={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            logger.warning(
                f"Street View metadata request failed (HTTP {r.status_code}) for {lat},{lon}"
            )
            return None
        data = r.json()
        status = data.get("status")
        if status != "OK":
            logger.info(
                f"No Street View at {lat},{lon} within radius={radius_m} (status={status}, error_message={data.get('error_message')})"
            )
            return None
        return data
    except requests.RequestException as e:
        logger.error(f"Street View metadata request error for {lat},{lon}: {e}")
        return None


def streetview_url_from_pano(pano_id, azimuth, size, fov, pitch):
    """Generate a Street View image URL for a given heading using a specific pano."""
    return (
        "https://maps.googleapis.com/maps/api/streetview?"
        f"size={size}&pano={pano_id}&heading={azimuth}&fov={fov}&pitch={pitch}&key={API_KEY}"
    )


def static_map_url(lat, lon, zoom=19, size="640x640", maptype="satellite"):
    """Generate a 2D bird-eye map image URL."""
    return (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}&maptype={maptype}&key={API_KEY}"
    )


def _format_float_for_name(value):
    """Format float for stable file names."""
    return f"{value:.6f}".replace("-", "m").replace(".", "p")


def _ensure_dirs(extra_dirs=None):
    CACHE_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_SV_DIR.mkdir(parents=True, exist_ok=True)
    if extra_dirs:
        for d in extra_dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


def cache_map_path(lat, lon, zoom=19, size="640x640", maptype="satellite"):
    lat_s = _format_float_for_name(lat)
    lon_s = _format_float_for_name(lon)
    return CACHE_MAPS_DIR / f"map_{lat_s}_{lon_s}_z{zoom}_{size}_{maptype}.png"


def fetch_map_image(lat, lon, zoom=19, size="640x640", maptype="satellite"):
    """Download or load cached static map image and return its path."""
    _ensure_dirs()
    out_path = cache_map_path(lat, lon, zoom=zoom, size=size, maptype=maptype)
    if out_path.exists():
        return out_path
    url = static_map_url(lat, lon, zoom=zoom, size=size, maptype=maptype)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    except requests.RequestException as e:
        logger.error(f"Failed to download static map for {lat},{lon}: {e}")
        return None


def cache_sv_path(lat, lon, heading, size, fov, pitch):
    lat_s = _format_float_for_name(lat)
    lon_s = _format_float_for_name(lon)
    head_s = _format_float_for_name(heading)
    size_s = size.replace("x", "_")
    fov_s = _format_float_for_name(fov)
    pitch_s = _format_float_for_name(pitch)
    return CACHE_SV_DIR / f"sv_{lat_s}_{lon_s}_h{head_s}_{size_s}_fov{fov_s}_p{pitch_s}.png"


def fetch_sv_image_by_pano(lat, lon, pano_id, heading, size=IMAGE_SIZE, fov=90, pitch=0):
    """Download or load cached Street View image (cached by lat,lon,heading,size,fov,pitch)."""
    _ensure_dirs()
    out_path = cache_sv_path(lat, lon, heading, size, fov, pitch)
    if out_path.exists():
        return out_path
    url = streetview_url_from_pano(pano_id, heading, size, fov, pitch)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    except requests.RequestException as e:
        logger.error(f"Failed to download Street View (heading={heading}) for {lat},{lon}: {e}")
        return None


def _draw_arrow(draw, cx, cy, angle_deg, length, color):
    theta = math.radians(angle_deg)
    dx = math.sin(theta)
    dy = -math.cos(theta)  # image y-axis is downward
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


def overlay_arrows_on_map(base_map_path, azimuths, labels, out_path):
    """Overlay directional arrows and labels on a static map image."""
    image = Image.open(base_map_path).convert("RGBA")
    w, h = image.size
    cx, cy = w / 2, h / 2
    draw = ImageDraw.Draw(image)

    colors = [
        (255, 69, 58, 255),   # red
        (52, 199, 89, 255),   # green
        (0, 122, 255, 255),   # blue
        (255, 159, 10, 255),  # orange
        (175, 82, 222, 255),  # purple
        (255, 214, 10, 255),  # yellow
    ]
    length = int(min(w, h) * 0.28)

    try:
        font = ImageFont.truetype("arial.ttf", size=int(min(w, h) * 0.05))
    except Exception:
        font = ImageFont.load_default()

    for i, (az, lbl) in enumerate(zip(azimuths, labels)):
        color = colors[i % len(colors)]
        tip_x, tip_y = _draw_arrow(draw, cx, cy, az, length, color)
        # label background circle
        label_radius = int(min(w, h) * 0.035)
        offset = int(min(w, h) * 0.015)
        theta = math.radians(az)
        dx = math.sin(theta)
        dy = -math.cos(theta)
        lx = tip_x + dx * offset
        ly = tip_y + dy * offset
        draw.ellipse(
            (lx - label_radius, ly - label_radius, lx + label_radius, ly + label_radius),
            fill=(0, 0, 0, 180),
        )
        # text centered in circle
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


# --------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Create a HuggingFace-style VLM dataset from intersections and Street View")
    parser.add_argument("dataset_name", help="Name of the dataset directory under data/hf/")
    parser.add_argument("--city", default=CITY_NAME, help="City/place name to query with OSMnx")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Number of intersections to sample")
    parser.add_argument("--image-size", default=IMAGE_SIZE, help="Street View image size, e.g. 640x640")
    parser.add_argument("--map-size", default="640x640", help="Static map image size, e.g. 640x640")
    parser.add_argument("--map-type", default="satellite", choices=["satellite", "roadmap", "terrain", "hybrid"], help="Static map type")
    parser.add_argument("--zoom", type=int, default=19, help="Static map zoom level")
    parser.add_argument("--max-sv-distance-m", type=float, default=MAX_SV_DISTANCE_M, help="Max allowed pano distance from intersection in meters")
    parser.add_argument("--metadata-radius-m", type=int, default=METADATA_RADIUS_M, help="Radius used for Street View metadata search in meters")
    parser.add_argument("--sv-fov", type=float, default=FOV, help="Street View horizontal field of view in degrees")
    parser.add_argument("--sv-pitch", type=float, default=0.0, help="Street View pitch in degrees")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo id (e.g. username/dataset-name)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token (or set HF_TOKEN env)")
    parser.add_argument("--hf-private", action="store_true", help="Create/update the HF repo as private")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and choices")
    parser.add_argument("--output-root", default="data/hf", help="Root directory where dataset folder will be created")

    args = parser.parse_args()

    random.seed(args.seed)

    # Output directories for this dataset
    dataset_dir = Path(args.output_root) / args.dataset_name
    images_dir = dataset_dir / "images"
    _ensure_dirs([images_dir])

    # Step 1. Load city road graph
    logger.info(f"Loading city graph for {args.city}...")
    G = ox.graph_from_place(args.city, network_type="drive")
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    logger.info(
        f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges. Sampling {args.n_samples} intersections."
    )
    try:
        edges_index_names = list(edges.index.names) if hasattr(edges.index, "names") else []
        logger.debug(
            f"Edges columns: {list(edges.columns)} | index names: {edges_index_names} | index levels: {getattr(edges.index, 'nlevels', 'NA')}"
        )
    except Exception:
        pass

    # Step 2. Iterate shuffled intersections until we collect the requested number
    candidate_nodes = nodes.sample(frac=1.0, random_state=args.seed)

    dataset = []
    hf_rows = []
    created = 0

    logger.info("Checking Street View coverage and building dataset...")
    for node_id, node_data in tqdm(candidate_nodes.iterrows(), total=len(candidate_nodes)):
        lat, lon = node_data["y"], node_data["x"]
        logger.debug(f"Processing node {node_id} at lat={lat}, lon={lon}")

        # Step 3. Query Street View metadata near the intersection and filter by distance
        metadata = get_streetview_metadata(lat, lon, radius_m=args.metadata_radius_m)
        if metadata is None:
            logger.debug(f"Skipping node {node_id}: no Street View metadata within radius {args.metadata_radius_m}m")
            continue
        pano_loc = metadata.get("location") or {}
        pano_lat = pano_loc.get("lat")
        pano_lng = pano_loc.get("lng")
        pano_id = metadata.get("pano_id")
        if pano_lat is None or pano_lng is None or pano_id is None:
            logger.debug(f"Skipping node {node_id}: incomplete metadata {metadata}")
            continue
        dist_m = haversine_distance_m(lat, lon, pano_lat, pano_lng)
        if dist_m > args.max_sv_distance_m:
            logger.debug(
                f"Skipping node {node_id}: nearest pano {pano_id} is {dist_m:.2f}m away (> {args.max_sv_distance_m}m)"
            )
            continue
        logger.debug(
            f"Node {node_id}: using pano_id={pano_id} at distance {dist_m:.2f}m (radius={args.metadata_radius_m}m)"
        )
        time.sleep(SLEEP_SEC)

        # Step 4. Get connected roads and compute directions
        try:
            if "u" in edges.columns and "v" in edges.columns:
                connected_edges = edges[(edges["u"] == node_id) | (edges["v"] == node_id)]
                uv_in_columns = True
            else:
                index_names = list(edges.index.names) if hasattr(edges.index, "names") else []
                try:
                    u_vals = (
                        edges.index.get_level_values("u")
                        if "u" in index_names
                        else edges.index.get_level_values(0)
                    )
                    v_vals = (
                        edges.index.get_level_values("v")
                        if "v" in index_names
                        else edges.index.get_level_values(1)
                    )
                except Exception as e:
                    logger.error(f"Unable to access edge index levels for node {node_id}: {e}")
                    continue

                mask = (u_vals == node_id) | (v_vals == node_id)
                connected_edges = edges[mask]
                uv_in_columns = False
        except Exception as e:
            logger.error(f"Failed to filter connected edges for node {node_id}: {e}")
            continue
        logger.debug(f"Node {node_id} has {len(connected_edges)} connected edges")
        azimuths = []
        for _, e in connected_edges.iterrows():
            if uv_in_columns:
                row_u = e["u"]
                row_v = e["v"]
            else:
                try:
                    row_u, row_v = e.name[0], e.name[1]
                except Exception:
                    logger.debug(
                        f"Skipping an edge for node {node_id}: unable to determine u/v from index"
                    )
                    continue

            other_node = row_v if row_u == node_id else row_u
            if other_node in nodes.index:
                lat2, lon2 = nodes.loc[other_node, ["y", "x"]]
                # Attach for fallback use in heading computation
                try:
                    e_local = e.copy()
                    e_local["_lat_other"], e_local["_lon_other"] = lat2, lon2
                except Exception:
                    e_local = e
                az = compute_edge_heading_at_node(node_id, row_u, row_v, e_local, lat, lon)
                if az is None:
                    az = compute_azimuth(lat, lon, lat2, lon2)
                azimuths.append(az)
                logger.debug(f"Computed heading at node {node_id} via geometry: {az}")

        before_dedupe = len(azimuths)
        azimuths = sorted(list(set(int(round(a / 30) * 30) for a in azimuths)))
        logger.debug(
            f"Node {node_id}: {before_dedupe} azimuths -> {len(azimuths)} after dedupe: {azimuths}"
        )
        if not azimuths:
            logger.debug(f"Skipping node {node_id}: no valid azimuths after dedupe")
            continue
        if len(azimuths) < 2:
            logger.debug(f"Skipping node {node_id}: only one distinct road direction after dedupe")
            continue

        # Step 5. Create street view + map entries (use specific pano)
        labels = [chr(ord('A') + i) for i in range(len(azimuths))]
        road_views = [
            {"label": lbl, "azimuth": az}
            for lbl, az in zip(labels, azimuths)
        ]

        # Fetch and overlay static map
        map_path = fetch_map_image(lat, lon, zoom=args.zoom, size=args.map_size, maptype=args.map_type)
        if map_path is None:
            logger.debug(f"Skipping node {node_id}: failed to fetch static map")
            continue
        overlay_path = images_dir / f"map_{node_id}.jpg"
        overlay_arrows_on_map(map_path, azimuths, labels, overlay_path)

        # Pick a random road and fetch one SV image for VLM sample
        chosen_idx = random.randrange(len(road_views))
        chosen = road_views[chosen_idx]
        sv_image_path = fetch_sv_image_by_pano(
            lat, lon, pano_id, heading=chosen["azimuth"], size=args.image_size, fov=args.sv_fov, pitch=args.sv_pitch
        )
        if sv_image_path is None:
            logger.debug(f"Skipping node {node_id}: failed to fetch SV image for chosen road")
            continue
        sv_dst = images_dir / f"sv_{node_id}.jpg"
        try:
            Image.open(sv_image_path).convert("RGB").save(sv_dst, format="JPEG", quality=92)
        except Exception:
            shutil.copyfile(sv_image_path, sv_dst)

        entry = {
            "intersection_id": str(node_id),
            "lat": lat,
            "lng": lon,
            "roads": road_views,
            # map_url removed to avoid exposing API key via query string
            "sv_pano_id": pano_id,
            "sv_distance_m": round(dist_m, 3),
            "vlm_map_image": str(overlay_path.relative_to(dataset_dir)),
            "vlm_sv_image": str(sv_dst.relative_to(dataset_dir)),
            "vlm_answer": chosen["label"],
        }
        dataset.append(entry)
        created += 1

        question = "Which labeled direction on the map corresponds to the direction in which the street view photo was taken?"
        options = labels
        hf_rows.append({
            "id": str(node_id),
            "images": [str(overlay_path.relative_to(dataset_dir)), str(sv_dst.relative_to(dataset_dir))],
            "question": question,
            "options": options,
            "answer": chosen["label"],
            "meta": {
                "lat": lat,
                "lng": lon,
                "pano_id": pano_id,
                "distance_m": round(dist_m, 3),
                "azimuths": azimuths,
                "labels": labels,
            },
        })

        logger.debug(
            f"Added VLM sample for node {node_id} with {len(road_views)} options; answer={chosen['label']}"
        )
        time.sleep(SLEEP_SEC)

        if created >= args.n_samples:
            logger.info(f"Collected target of {args.n_samples} samples. Stopping early.")
            break

    # Save filtered dataset and HF-style dataset
    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dataset).to_json(out_path, orient="records", indent=2)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_dir / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in hf_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if len(dataset) == 0:
        logger.warning(
            f"Saved 0 intersections. Check API key, city selection, or Street View availability. Intersections JSON: {out_path}"
        )
    else:
        logger.info(
            f"Saved {len(dataset)} dataset entries to {out_path} and {len(hf_rows)} HF rows to {jsonl_path}"
        )

    if created < args.n_samples:
        logger.warning(
            f"Requested {args.n_samples} samples but only created {created}. Not enough eligible intersections matched the constraints."
        )

    # Optionally push to Hugging Face (typed dataset with Image features)
    if args.hf_repo:
        token = args.hf_token or os.getenv("HF_TOKEN")
        if not token:
            logger.info("No HF token provided via flag or env; using cached CLI login if available.")

        # Build typed dataset with explicit Image columns
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
        for row in hf_rows:
            rows["id"].append(row["id"])
            # Convert to absolute paths for upload; HF Datasets will package files
            rows["image_map"].append(str((dataset_dir / row["images"][0]).resolve()))
            rows["image_sv"].append(str((dataset_dir / row["images"][1]).resolve()))
            rows["question"].append(row["question"])
            rows["options"].append(row["options"])
            rows["answer"].append(row["answer"])

        ds = HFDataset.from_dict(rows, features=features)
        logger.info(f"Pushing typed dataset to {args.hf_repo} (private={args.hf_private})")
        ds.push_to_hub(args.hf_repo, token=token if token else None, private=args.hf_private)
        logger.info("HF Datasets push completed. Dataset preview should render images.")


if __name__ == "__main__":
    main()
