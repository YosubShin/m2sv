# /// script
# dependencies = ["matplotlib"]
# ///
"""Model accuracy vs. road-azimuth symmetry, stratified to 3-option intersections.

The earlier across-all-#options quintile binning confounded symmetry with
candidate count (the most symmetric examples were overwhelmingly 4-way junctions)
and sliced a heavily-tied S distribution arbitrarily. We instead hold #options
fixed (the 3-option stratum, the largest group) and bin by the symmetry scalar
  S = max_i g_i / min_i g_i   (consecutive azimuth gaps; see paper Eq.)
into three well-separated, untied groups: Symmetric (S<2), Intermediate (S=2),
Asymmetric (S>2).

Writes figures/symmetry_accuracy.png in the paper directory.
"""
import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUEPRINT = ROOT / "blueprints/20k/validation.jsonl"
OUT = ROOT / "publications/icml2026/2026-01-28/figures/symmetry_accuracy.png"
MODELS = [
    ("Gemini-3-Pro", "json", ROOT / "past_results/2026-01-23/gemini-3-pro.json"),
    ("GPT-5", "json", ROOT / "past_results/2025-10-23/gpt-5.json"),
    ("Qwen3-VL-8B (SFT+RL)", "csv", ROOT / "past_results/2026-01-23/qwen3-vl-8b-instruct-sft-rl.csv"),
]
BINS = ["Symmetric\n(S<2)", "Intermediate\n(S=2)", "Asymmetric\n(S>2)"]

def _to_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    return str(v).strip().lower() in {"true", "t", "1", "yes", "y"}

def _gap_ratio(az):
    az = sorted(a % 360.0 for a in az)
    if len(az) < 2: return None
    gaps = [az[i + 1] - az[i] for i in range(len(az) - 1)] + [360.0 + az[0] - az[-1]]
    return max(gaps) / min(gaps) if min(gaps) > 0 else None

def _bin(s):
    return 0 if s < 2 else 1 if s < 2.01 else 2

def _load_correct(kind, path):
    if kind == "csv":
        with path.open(newline="") as f:
            return {str(r["id"]): _to_bool(r.get("correct")) for r in csv.DictReader(f) if r.get("id")}
    obj = json.loads(path.read_text())
    rows = obj.get("results", obj if isinstance(obj, list) else [])
    return {str(r["id"]): _to_bool(r.get("correct", False)) for r in rows if r.get("id") is not None}

def _wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def main():
    # 3-option intersections only -> symmetry bin
    sbin = {}
    with BLUEPRINT.open() as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            az = o.get("azimuths") or []
            if len(az) != 3: continue
            s = _gap_ratio(az)
            if s is not None:
                sbin[str(o.get("intersection_id"))] = _bin(s)

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=200)
    x = list(range(3))
    for label, kind, path in MODELS:
        correct = _load_correct(kind, path)
        acc, err = [], []
        for b in range(3):
            vals = [correct[i] for i in correct if sbin.get(i) == b]
            k = sum(1 for v in vals if v)
            acc.append(k / len(vals) if vals else 0.0)
            err.append(_wilson(k, len(vals)))
        ax.errorbar(x, acc, yerr=err, marker="o", capsize=3, label=label,
                    ecolor="#bdbdbd", elinewidth=0.9)
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_xlabel("Road-azimuth symmetry (3-option intersections)")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("Model accuracy vs. symmetry, holding candidate count fixed")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
