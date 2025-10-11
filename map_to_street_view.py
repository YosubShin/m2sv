import requests
import random
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OUT_DIR = Path("data/maps_streetview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

coords = [(37.7749, -122.4194), (34.0522, -118.2437)]  # sample lat/lons

for i, (lat, lon) in enumerate(coords):
    # Bird’s-eye map
    map_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom=17&size=640x640&maptype=roadmap&key={API_KEY}"
    )
    with open(OUT_DIR / f"{i}_map.png", "wb") as f:
        f.write(requests.get(map_url).content)

    # Street View at four directions
    for j, heading in enumerate([0, 90, 180, 270]):
        sv_url = (
            f"https://maps.googleapis.com/maps/api/streetview?"
            f"size=640x640&location={lat},{lon}&heading={heading}&fov=90&pitch=0&key={API_KEY}"
        )
        with open(OUT_DIR / f"{i}_sv_{j}.png", "wb") as f:
            f.write(requests.get(sv_url).content)
