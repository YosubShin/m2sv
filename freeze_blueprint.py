import os
import json
import math
import random
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import osmnx as ox
from dotenv import load_dotenv
from tqdm import tqdm
import requests


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("freeze_blueprint")


API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
DEFAULT_TOTAL = 1000
DEFAULT_MAX_SV_DISTANCE_M = 5.0
DEFAULT_METADATA_RADIUS_M = 5
DEFAULT_MIN_AZIMUTHS = 2


def compute_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dy = lat2 - lat1
    dx = np.cos(np.radians(lat1)) * (lon2 - lon1)
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def compute_edge_heading_at_node(
    node_id,
    row_u,
    row_v,
    edge_row,
    lat_node: float,
    lon_node: float,
) -> Optional[float]:
    try:
        geom = edge_row.get("geometry", None)
        if geom is not None and hasattr(geom, "coords"):
            coords = list(geom.coords)
            if len(coords) >= 2:
                if node_id == row_u:
                    (lon1, lat1), (lon2, lat2) = coords[0], coords[1]
                else:
                    (lon1, lat1), (lon2, lat2) = coords[-1], coords[-2]
                return compute_azimuth(lat1, lon1, lat2, lon2)
    except Exception:
        pass

    try:
        lat2, lon2 = edge_row.get("_lat_other"), edge_row.get("_lon_other")
        if lat2 is None or lon2 is None:
            return None
        return compute_azimuth(lat_node, lon_node, lat2, lon2)
    except Exception:
        return None


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_earth_m * c


def get_streetview_metadata(lat: float, lon: float, radius_m: Optional[int] = None) -> Optional[dict]:
    base = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    radius_part = f"&radius={int(radius_m)}" if radius_m is not None else ""
    url = f"{base}location={lat},{lon}{radius_part}&key={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            logger.debug(f"Street View metadata HTTP {r.status_code} for {lat},{lon}")
            return None
        data = r.json()
        if data.get("status") != "OK":
            return None
        return data
    except requests.RequestException:
        return None


def get_streetview_metadata_by_pano(pano_id: str) -> Optional[dict]:
    base = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    url = f"{base}pano={pano_id}&key={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "OK":
            return None
        return data
    except requests.RequestException:
        return None


def load_places_from_file(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


DEFAULT_PLACES = [
    # US
    "New York City, New York, USA",
    "Los Angeles, California, USA",
    "Chicago, Illinois, USA",
    "Houston, Texas, USA",
    "Phoenix, Arizona, USA",
    "Seattle, Washington, USA",
    "Miami, Florida, USA",
    "Boston, Massachusetts, USA",
    "Denver, Colorado, USA",
    "Minneapolis, Minnesota, USA",
    # Europe
    "London, UK",
    "Paris, France",
    "Berlin, Germany",
    "Madrid, Spain",
    "Rome, Italy",
    "Amsterdam, Netherlands",
    # Asia
    "Tokyo, Japan",
    "Seoul, South Korea",
    "Singapore",
    "Bangkok, Thailand",
    "Delhi, India",
    # South America
    "Sao Paulo, Brazil",
    "Buenos Aires, Argentina",
    "Bogota, Colombia",
    # Africa
    "Lagos, Nigeria",
    "Nairobi, Kenya",
    # Oceania
    "Sydney, Australia",
    "Melbourne, Australia",
    "Auckland, New Zealand",
    # Middle East
    "Dubai, United Arab Emirates",
    "Riyadh, Saudi Arabia",
    "Istanbul, Turkey",
]


def choose_per_place_counts(total: int, places: List[str], cap: Optional[int]) -> List[int]:
    n = len(places)
    base = total // n
    counts = [base] * n
    remainder = total - base * n
    for i in range(remainder):
        counts[i % n] += 1
    if cap is not None:
        counts = [min(c, cap) for c in counts]
    # If capped caused deficit, fill round-robin where possible
    deficit = total - sum(counts)
    i = 0
    while deficit > 0 and any((cap is None or counts[j] < cap) for j in range(n)):
        if cap is None or counts[i % n] < cap:
            counts[i % n] += 1
            deficit -= 1
        i += 1
    return counts


def dedupe_too_close(samples: List[dict], lat: float, lon: float, radius_m: float) -> bool:
    for s in samples:
        if haversine_distance_m(lat, lon, s["lat"], s["lng"]) < radius_m:
            return True
    return False


def freeze_place(
    place: str,
    target_count: int,
    seed: int,
    max_sv_distance_m: float,
    metadata_radius_m: int,
    global_samples: List[dict],
    dedupe_radius_m: float,
    sv_fov: float,
    sv_pitch: float,
    map_zoom: int,
    map_type: str,
    map_size: str,
    min_azimuths: int,
    candidate_multiplier: int = 10,
) -> List[dict]:
    if target_count <= 0:
        return []

    logger.info(f"Loading graph for {place} ...")
    try:
        G = ox.graph_from_place(place, network_type="drive")
    except Exception as e:
        logger.warning(f"Skipping place '{place}' due to OSMnx error: {e}")
        return []
    logger.info(f"Graph loaded for {place} with {len(G.nodes)} nodes and {len(G.edges)} edges")
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    candidate_nodes = nodes.sample(frac=1.0, random_state=seed)

    # Cap the number of candidate nodes we will inspect to avoid hopelessly long runs
    max_checks = min(len(candidate_nodes), target_count * max(candidate_multiplier, 1))
    if max_checks < len(candidate_nodes):
        logger.info(
            f"Capping candidates for {place} to {max_checks} (= {candidate_multiplier}x target {target_count}); total nodes available: {len(candidate_nodes)}"
        )
        candidate_nodes = candidate_nodes.head(max_checks)
    else:
        logger.info(f"Sampled {len(candidate_nodes)} nodes for {place}")

    out: List[dict] = []
    stats: Dict[str, object] = {
        "checked": 0,
        "accepted": 0,
        "reasons": {
            "no_metadata": 0,
            "metadata_incomplete": 0,
            "too_far": 0,
            "pano_invalid": 0,
            "edges_access_error": 0,
            "no_azimuths": 0,
            "not_enough_azimuths": 0,
            "dedupe_blocked": 0,
        },
        "elapsed_sec": 0.0,
    }

    t0 = time.time()

    for node_id, node_data in tqdm(candidate_nodes.iterrows(), total=len(candidate_nodes), disable=True):
        if len(out) >= target_count:
            break
        stats["checked"] = int(stats["checked"]) + 1
        lat, lon = node_data["y"], node_data["x"]

        # Check Street View availability now (metadata only) for reproducibility
        meta = get_streetview_metadata(lat, lon, radius_m=metadata_radius_m) if API_KEY else None
        if meta is None:
            stats["reasons"]["no_metadata"] = int(stats["reasons"]["no_metadata"]) + 1
            continue
        pano_loc = meta.get("location") or {}
        pano_lat = pano_loc.get("lat")
        pano_lng = pano_loc.get("lng")
        pano_id = meta.get("pano_id")
        if pano_lat is None or pano_lng is None or pano_id is None:
            stats["reasons"]["metadata_incomplete"] = int(stats["reasons"]["metadata_incomplete"]) + 1
            continue
        dist_m = haversine_distance_m(lat, lon, pano_lat, pano_lng)
        if dist_m > max_sv_distance_m:
            stats["reasons"]["too_far"] = int(stats["reasons"]["too_far"]) + 1
            continue

        # Validate the pano_id directly to ensure it resolves later
        if get_streetview_metadata_by_pano(pano_id) is None:
            stats["reasons"]["pano_invalid"] = int(stats["reasons"]["pano_invalid"]) + 1
            continue

        # Build connected headings
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
                except Exception:
                    stats["reasons"]["edges_access_error"] = int(stats["reasons"]["edges_access_error"]) + 1
                    continue
                mask = (u_vals == node_id) | (v_vals == node_id)
                connected_edges = edges[mask]
                uv_in_columns = False
        except Exception:
            stats["reasons"]["edges_access_error"] = int(stats["reasons"]["edges_access_error"]) + 1
            continue

        azimuths: List[float] = []
        for _, e in connected_edges.iterrows():
            if uv_in_columns:
                row_u = e["u"]
                row_v = e["v"]
            else:
                try:
                    row_u, row_v = e.name[0], e.name[1]
                except Exception:
                    # treat as edges access error for stats purposes
                    stats["reasons"]["edges_access_error"] = int(stats["reasons"]["edges_access_error"]) + 1
                    continue
            other_node = row_v if row_u == node_id else row_u
            if other_node in nodes.index:
                lat2, lon2 = nodes.loc[other_node, ["y", "x"]]
                try:
                    e_local = e.copy()
                    e_local["_lat_other"], e_local["_lon_other"] = lat2, lon2
                except Exception:
                    e_local = e
                az = compute_edge_heading_at_node(node_id, row_u, row_v, e_local, lat, lon)
                if az is None:
                    az = compute_azimuth(lat, lon, lat2, lon2)
                azimuths.append(az)

        # Normalize and dedupe azimuths
        azimuths = sorted(list(set(int(round(a / 30) * 30) for a in azimuths)))
        if len(azimuths) < min_azimuths:
            if len(azimuths) == 0:
                stats["reasons"]["no_azimuths"] = int(stats["reasons"]["no_azimuths"]) + 1
            else:
                stats["reasons"]["not_enough_azimuths"] = int(stats["reasons"]["not_enough_azimuths"]) + 1
            continue

        # Dedupe near duplicates globally to avoid clustering
        if dedupe_radius_m > 0 and (dedupe_too_close(global_samples, lat, lon, dedupe_radius_m) or dedupe_too_close(out, lat, lon, dedupe_radius_m)):
            stats["reasons"]["dedupe_blocked"] = int(stats["reasons"]["dedupe_blocked"]) + 1
            continue

        labels = [chr(ord('A') + i) for i in range(len(azimuths))]
        chosen_idx = random.randrange(len(azimuths))
        chosen_label = labels[chosen_idx]
        chosen_heading = azimuths[chosen_idx]

        uid = f"{place}|{node_id}"
        out.append({
            "uid": uid,
            "intersection_id": str(node_id),
            "place": place,
            "lat": lat,
            "lng": lon,
            "azimuths": azimuths,
            "labels": labels,
            "answer": chosen_label,
            "answer_heading": chosen_heading,
            "sv_pano_id": pano_id,
            "sv_distance_m": round(dist_m, 3),
            # rendering defaults (can be overridden at render time)
            "params": {
                "map_zoom": map_zoom,
                "map_type": map_type,
                "map_size": map_size,
                "sv_fov": sv_fov,
                "sv_pitch": sv_pitch,
            },
        })
        stats["accepted"] = int(stats["accepted"]) + 1

    stats["elapsed_sec"] = round(time.time() - t0, 3)
    checked = int(stats["checked"]) if stats["checked"] else 1
    accepted = int(stats["accepted"]) if stats["accepted"] else 0
    acc_rate = accepted / checked
    nodes_per_sec = checked / max(stats["elapsed_sec"], 1e-6)
    logger.info(
        f"{place}: checked={checked}, accepted={accepted} (rate={acc_rate:.3f}), elapsed={stats['elapsed_sec']}s, nodes/s={nodes_per_sec:.2f}"
    )
    # Log detailed filter reasons with percentages
    reasons: Dict[str, int] = stats["reasons"]  # type: ignore[assignment]
    if checked:
        breakdown = ", ".join(
            f"{k}={v} ({(v/checked):.3f})" for k, v in reasons.items() if v
        )
        if breakdown:
            logger.info(f"{place}: filter breakdown -> {breakdown}")

    if len(out) < target_count and checked >= max_checks:
        logger.info(
            f"{place}: reached candidate cap ({max_checks}) with {len(out)} accepted (< target {target_count})."
        )

    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Freeze a reproducible blueprint (JSONL) of intersections with headings and labels")
    parser.add_argument("--out", default="data/blueprints/train-1k.jsonl", help="Output JSONL path for the blueprint")
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL, help="Total number of samples to freeze")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--places-file", type=str, default=None, help="Path to a file (txt/json) containing place names")
    parser.add_argument("--place", action="append", default=None, help="Add a place name (can repeat)")
    parser.add_argument("--per-place-cap", type=int, default=60, help="Max samples per place to avoid over-representation")
    parser.add_argument("--max-sv-distance-m", type=float, default=DEFAULT_MAX_SV_DISTANCE_M, help="Max allowed Street View pano distance in meters")
    parser.add_argument("--metadata-radius-m", type=int, default=DEFAULT_METADATA_RADIUS_M, help="Radius for Street View metadata search in meters")
    parser.add_argument("--dedupe-radius-m", type=float, default=20.0, help="Minimum distance between frozen intersections (meters)")
    parser.add_argument("--min-azimuths", type=int, default=DEFAULT_MIN_AZIMUTHS, help="Minimum distinct road directions required after dedupe")
    parser.add_argument("--sv-fov", type=float, default=120.0, help="Default SV horizontal field of view")
    parser.add_argument("--sv-pitch", type=float, default=0.0, help="Default SV pitch")
    parser.add_argument("--map-zoom", type=int, default=19, help="Default static map zoom")
    parser.add_argument("--map-type", choices=["satellite", "roadmap", "terrain", "hybrid"], default="satellite", help="Default static map type")
    parser.add_argument("--map-size", default="640x640", help="Default static map size")
    parser.add_argument("--candidate-multiplier", type=int, default=10, help="Cap per-place node checks to this multiple of the target count")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file; skip completed places and append new ones per city")

    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("GOOGLE_MAPS_API_KEY is not set; freezing requires metadata and pano validation.")

    random.seed(args.seed)

    places: List[str] = []
    if args.places_file:
        places.extend(load_places_from_file(Path(args.places_file)))
    if args.place:
        places.extend(args.place)
    if not places:
        places = DEFAULT_PLACES

    counts = choose_per_place_counts(args.total_samples, places, args.per_place_cap)
    logger.info(f"Freezing blueprint across {len(places)} places; per-place targets: {counts}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Seed from existing file when resuming; otherwise, truncate the file
    frozen: List[dict] = []
    place_to_existing: Dict[str, int] = {}
    if args.resume and out_path.exists():
        logger.info(f"Resuming from existing file: {out_path}")
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frozen.append(row)
                pl = row.get("place")
                if isinstance(pl, str):
                    place_to_existing[pl] = place_to_existing.get(pl, 0) + 1
        logger.info(f"Loaded {len(frozen)} existing rows across {len(place_to_existing)} places")
    else:
        # fresh run: truncate/create file
        with out_path.open("w", encoding="utf-8") as f:
            pass

    # Initial per-place pass
    for place, target in zip(places, counts):
        existing = place_to_existing.get(place, 0)
        remaining = max(0, target - existing)
        if existing >= target:
            logger.info(f"Skipping {place}: already have {existing}/{target} rows in output (resume)")
            continue
        place_rows = freeze_place(
            place=place,
            target_count=remaining,
            seed=args.seed,
            max_sv_distance_m=args.max_sv_distance_m,
            metadata_radius_m=args.metadata_radius_m,
            global_samples=frozen,
            dedupe_radius_m=args.dedupe_radius_m,
            sv_fov=args.sv_fov,
            sv_pitch=args.sv_pitch,
            map_zoom=args.map_zoom,
            map_type=args.map_type,
            map_size=args.map_size,
            min_azimuths=args.min_azimuths,
            candidate_multiplier=args.candidate_multiplier,
        )
        frozen.extend(place_rows)
        # Incremental save: append per city
        if place_rows:
            with out_path.open("a", encoding="utf-8") as f:
                for row in place_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"{place}: added {len(place_rows)} samples (now total={len(frozen)})")
        if len(frozen) >= args.total_samples:
            break

    # If still short, continue round-robin until filled (best-effort)
    if len(frozen) < args.total_samples:
        logger.info("Top-up round to reach target total...")
        i = 0
        while len(frozen) < args.total_samples and i < len(places) * 3:
            place = places[i % len(places)]
            more = freeze_place(
                place=place,
                target_count=min(10, args.total_samples - len(frozen)),
                seed=args.seed + i,
                max_sv_distance_m=args.max_sv_distance_m,
                metadata_radius_m=args.metadata_radius_m,
                global_samples=frozen,
                dedupe_radius_m=args.dedupe_radius_m,
                sv_fov=args.sv_fov,
                sv_pitch=args.sv_pitch,
                map_zoom=args.map_zoom,
                map_type=args.map_type,
                map_size=args.map_size,
                min_azimuths=args.min_azimuths,
                candidate_multiplier=args.candidate_multiplier,
            )
            frozen.extend(more)
            if more:
                with out_path.open("a", encoding="utf-8") as f:
                    for row in more:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            i += 1

    logger.info(f"Saved {min(len(frozen), args.total_samples)} frozen samples to {out_path} (incremental writes)")


if __name__ == "__main__":
    main()


