import csv
import json
import math
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUEPRINT_PATH = Path("blueprints/20k/validation.jsonl")
MODEL_PATHS = [
    ("Gemini 3 Pro", "json", Path("past_results/2026-01-23/gemini-3-pro.json")),
    ("GPT-5", "json", Path("past_results/2025-10-23/gpt-5.json")),
    ("Qwen3-VL-8B (SFT+RL)", "csv", Path("past_results/2026-01-23/qwen3-vl-8b-instruct-sft-rl.csv")),
]

OUT_DIR = Path("analysis/plots")
OUT_PNG = OUT_DIR / "symmetry_accuracy.png"
OUT_SCATTER_PNG = OUT_DIR / "symmetry_vs_time.png"
HUMAN_PATH = Path("past_results/2026-01-23/manual.json")


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot coerce to bool: {value!r}")


def _load_azimuths(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = str(obj.get("intersection_id"))
            az = obj.get("azimuths") or []
            if rid and az:
                out[rid] = list(map(float, az))
    return out


def _gap_ratio(azimuths):
    az = sorted([a % 360.0 for a in azimuths])
    if len(az) < 2:
        return None
    gaps = []
    for i in range(len(az) - 1):
        gaps.append(az[i + 1] - az[i])
    gaps.append((360.0 + az[0]) - az[-1])
    min_gap = min(gaps)
    max_gap = max(gaps)
    if min_gap <= 0:
        return None
    return max_gap / min_gap


def _load_model_csv(path: Path):
    out = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            if rid is None:
                continue
            out[str(rid)] = _to_bool(row.get("correct"))
    return out


def _load_model_json(path: Path):
    obj = json.loads(path.read_text())
    rows = obj.get("results", obj if isinstance(obj, list) else [])
    out = {}
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        out[str(rid)] = _to_bool(row.get("correct", False))
    return out


def _load_human_elapsed(path: Path):
    obj = json.loads(path.read_text())
    rows = obj.get("results", obj if isinstance(obj, list) else [])
    out = {}
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        elapsed = row.get("elapsed_seconds")
        if elapsed is None:
            continue
        out[str(rid)] = float(elapsed)
    return out


def _assign_quantile_bins(ratio_map, n_bins=5):
    items = sorted(ratio_map.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 0:
        return {}
    bins = {}
    for idx, (rid, _) in enumerate(items):
        b = min(n_bins - 1, int(idx * n_bins / n))
        bins[rid] = b
    return bins


def _wilson_half_width(k, n, z=1.96):
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + (z * z / n)
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return half


def main():
    azimuths = _load_azimuths(BLUEPRINT_PATH)
    ratios = {}
    for rid, az in azimuths.items():
        r = _gap_ratio(az)
        if r is not None:
            ratios[rid] = r

    bin_map = _assign_quantile_bins(ratios, n_bins=5)
    labels = [f"Q{i+1}" for i in range(5)]
    centers = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    for label, kind, path in MODEL_PATHS:
        if kind == "csv":
            correct = _load_model_csv(path)
        elif kind == "json":
            correct = _load_model_json(path)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        # Build quantile bins within this model's IDs to respect differing N.
        model_ratios = {rid: ratios[rid] for rid in correct if rid in ratios}
        model_bin_map = _assign_quantile_bins(model_ratios, n_bins=5)
        bucket_vals = {i: [] for i in range(5)}
        for rid, b in model_bin_map.items():
            bucket_vals[b].append(correct[rid])
        acc = []
        err = []
        counts = []
        for i in range(5):
            vals = bucket_vals[i]
            counts.append(len(vals))
            k = sum(1 for v in vals if v)
            acc.append(k / len(vals) if vals else 0.0)
            err.append(_wilson_half_width(k, len(vals)))
        ax.errorbar(centers, acc, yerr=err, marker="o", label=label, capsize=2, ecolor="#bdbdbd", elinewidth=0.8)
        print(label, "counts:", counts)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Symmetry quantile (Q5=most asymmetric)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy vs. road-azimuth symmetry")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print("Saved:", OUT_PNG)

    # Scatter: symmetry quantile vs human elapsed time (200 human examples)
    human_elapsed = _load_human_elapsed(HUMAN_PATH)
    human_ratios = {rid: ratios[rid] for rid in human_elapsed if rid in ratios}
    human_bin_map = _assign_quantile_bins(human_ratios, n_bins=5)
    xs = []
    ys = []
    for rid, elapsed in human_elapsed.items():
        b = human_bin_map.get(rid)
        if b is None:
            continue
        xs.append(b + 1)
        ys.append(elapsed)

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax2.scatter(xs, ys, s=18, alpha=0.6)

    # Mean time by the same Q1-Q5 bins used for symmetry quantiles.
    mean_x = []
    mean_y = []
    for b in range(5):
        bin_times = [elapsed for rid, elapsed in human_elapsed.items() if human_bin_map.get(rid) == b]
        if not bin_times:
            continue
        mean_x.append(b + 1)
        mean_y.append(sum(bin_times) / len(bin_times))
    if mean_x:
        ax2.plot(mean_x, mean_y, color="#ff7f0e", linewidth=2, label="Mean (Q1–Q5)")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax2.set_xlabel("Symmetry quantile (Q5=most asymmetric)")
    ax2.set_ylabel("Human elapsed time (s)")
    ax2.set_title("Human time vs. road-azimuth symmetry")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_SCATTER_PNG)
    print("Saved:", OUT_SCATTER_PNG)


if __name__ == "__main__":
    main()
