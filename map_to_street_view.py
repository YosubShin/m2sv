import requests
import random
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("map_to_street_view")

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    logger.error(
        "GOOGLE_MAPS_API_KEY is not set. Please set it in your environment or .env file."
    )
    raise SystemExit(1)

OUT_DIR = Path("data/maps_streetview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

coords = [
    (21.284428, -157.802295),
]  # sample lat/lons

for i, (lat, lon) in enumerate(coords):
    # Check if Street View panorama exists using metadata API
    metadata_url = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata?"
        f"location={lat},{lon}&key={API_KEY}"
    )
    try:
        metadata_resp = requests.get(metadata_url, timeout=10)
        if metadata_resp.status_code != 200:
            logger.warning(
                f"Location {i} ({lat},{lon}) metadata HTTP {metadata_resp.status_code}"
            )
            continue
        metadata_json = metadata_resp.json()
    except requests.RequestException as e:
        logger.error(f"Metadata request error for {lat},{lon}: {e}")
        continue

    status = metadata_json.get("status")
    if status != "OK":
        logger.info(
            f"Skipping location {i} ({lat}, {lon}): No Street View (status={status}, msg={metadata_json.get('error_message')})"
        )
        continue

    logger.info(
        f"Processing location {i} ({lat}, {lon}): pano_id={metadata_json.get('pano_id')}"
    )

    # Save metadata JSON
    with open(OUT_DIR / f"{i}_metadata.json", "w") as f:
        json.dump(metadata_json, f, indent=2)
    
    # Bird's-eye map
    map_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom=17&size=640x640&maptype=roadmap&key={API_KEY}"
    )
    try:
        map_resp = requests.get(map_url, timeout=20)
        map_resp.raise_for_status()
        with open(OUT_DIR / f"{i}_map.png", "wb") as f:
            f.write(map_resp.content)
        logger.debug(f"Saved map image for location {i} to {(OUT_DIR / f'{i}_map.png')}" )
    except requests.RequestException as e:
        logger.error(f"Failed to download map image for {lat},{lon}: {e}")

    # Street View at four directions
    for j, heading in enumerate([0, 90, 180, 270]):
        sv_url = (
            f"https://maps.googleapis.com/maps/api/streetview?"
            f"size=640x640&location={lat},{lon}&heading={heading}&fov=90&pitch=0&key={API_KEY}"
        )
        try:
            sv_resp = requests.get(sv_url, timeout=20)
            sv_resp.raise_for_status()
            with open(OUT_DIR / f"{i}_sv_{j}.png", "wb") as f:
                f.write(sv_resp.content)
            logger.debug(
                f"Saved Street View image heading={heading} for location {i} to {(OUT_DIR / f'{i}_sv_{j}.png')}"
            )
        except requests.RequestException as e:
            logger.error(
                f"Failed to download Street View (heading={heading}) for {lat},{lon}: {e}"
            )
