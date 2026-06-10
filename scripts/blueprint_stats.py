#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"Invalid JSON on line {line_num}: {path}")


def describe_numeric(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    vals = sorted(values)
    n = len(vals)
    def pct(p):
        if n == 1:
            return vals[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        return vals[f] + (k - f) * (vals[c] - vals[f])

    return {
        "count": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": mean(vals),
        "median": median(vals),
        "p05": pct(0.05),
        "p95": pct(0.95),
    }


def format_stat_block(title, stats, unit=""):
    unit = unit or ""
    def fmt(x):
        if x is None:
            return "n/a"
        if isinstance(x, int):
            return f"{x}{unit}"
        return f"{x:.3f}{unit}"

    lines = [f"{title}:"]
    lines.append(f"  count: {stats['count']}")
    lines.append(f"  min: {fmt(stats['min'])}")
    lines.append(f"  max: {fmt(stats['max'])}")
    lines.append(f"  mean: {fmt(stats['mean'])}")
    lines.append(f"  median: {fmt(stats['median'])}")
    lines.append(f"  p05: {fmt(stats['p05'])}")
    lines.append(f"  p95: {fmt(stats['p95'])}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize blueprint JSONL: #options distribution, SV distance stats, and city caps."
    )
    parser.add_argument(
        "--input",
        default="blueprints/20k/train-val-20k.jsonl",
        help="Path to blueprint JSONL.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/blueprint_stats",
        help="Directory for plots (created if missing).",
    )
    parser.add_argument(
        "--max-cities-plot",
        type=int,
        default=50,
        help="Max cities to show in city caps plot (top-N by count).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    options_counts = []
    sv_distances = []
    city_counts = Counter()

    total = 0
    for row in load_rows(input_path):
        total += 1
        azimuths = row.get("azimuths")
        labels = row.get("labels")
        if isinstance(labels, list):
            options_counts.append(len(labels))
        elif isinstance(azimuths, list):
            options_counts.append(len(azimuths))
        else:
            options_counts.append(0)

        dist = row.get("sv_distance_m")
        if isinstance(dist, (int, float)):
            sv_distances.append(float(dist))

        place = row.get("place")
        if isinstance(place, str):
            city_counts[place] += 1

    print(f"Loaded {total} rows from {input_path}")
    print()

    options_stats = describe_numeric(options_counts)
    print(format_stat_block("#options stats", options_stats))
    print()

    dist_stats = describe_numeric(sv_distances)
    print(format_stat_block("Street View distance stats (m)", dist_stats, unit="m"))
    print()

    if city_counts:
        city_values = sorted(city_counts.items(), key=lambda x: (-x[1], x[0]))
        counts_only = [c for _, c in city_values]
        city_stats = describe_numeric(counts_only)
        print(format_stat_block("City caps (counts per place)", city_stats))
        print()
        top_n = min(args.max_cities_plot, len(city_values))
        print(f"Top {top_n} cities by count:")
        for name, cnt in city_values[:top_n]:
            print(f"  {name}: {cnt}")
    else:
        print("No place field found; skipping city caps.")

    if args.no_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\nPlotting skipped (matplotlib import failed): {exc}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histogram: #options
    if options_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = range(0, max(options_counts) + 2)
        ax.hist(options_counts, bins=bins, edgecolor="black", color="#6baed6")
        ax.set_title("#Options Distribution")
        ax.set_xlabel("Number of options")
        ax.set_ylabel("Count")
        ax.set_xticks(list(bins))
        fig.tight_layout()
        fig.savefig(out_dir / "options_hist.png", dpi=150)
        plt.close(fig)

    # Histogram: SV distance
    if sv_distances:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sv_distances, bins=40, edgecolor="black", color="#fd8d3c")
        ax.set_title("Street View Distance (m)")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "sv_distance_hist.png", dpi=150)
        plt.close(fig)

    # City caps bar chart (top-N)
    if city_counts:
        city_values = sorted(city_counts.items(), key=lambda x: (-x[1], x[0]))
        top_n = min(args.max_cities_plot, len(city_values))
        labels = [name for name, _ in city_values[:top_n]]
        values = [cnt for _, cnt in city_values[:top_n]]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.25)))
        ax.barh(labels[::-1], values[::-1], color="#74c476")
        ax.set_title(f"City Caps (Top {top_n})")
        ax.set_xlabel("Count")
        ax.set_ylabel("Place")
        fig.tight_layout()
        fig.savefig(out_dir / "city_caps_bar.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
