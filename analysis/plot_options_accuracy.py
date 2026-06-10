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
GEMINI_3_PRO_PATH = Path("past_results/2026-01-23/gemini-3-pro.json")
GPT_5_PATH = Path("past_results/2025-10-23/gpt-5.json")
QWEN_SFT_RL_PATH = Path("past_results/2026-01-23/qwen3-vl-8b-instruct-sft-rl.csv")

OUT_DIR = Path("analysis/plots")
OUT_PNG = OUT_DIR / "options_accuracy.png"
OUT_PNG_NORM = OUT_DIR / "options_accuracy_normalized.png"


def _load_json_results(path: Path):
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "results" in obj:
        return obj["results"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON structure in {path}")


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


def _load_model_json(path: Path):
    rows = _load_json_results(path)
    out = {}
    for r in rows:
        rid = str(r.get("id"))
        if rid is None:
            continue
        out[rid] = _to_bool(r.get("correct", False))
    return out


def _load_model_csv(path: Path):
    out = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            if rid is None:
                continue
            out[rid] = _to_bool(row.get("correct"))
    return out


def _load_num_options(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = str(obj.get("intersection_id"))
            labels = obj.get("labels") or []
            out[rid] = len(labels)
    return out


def _accuracy_by_options(model_correct, num_options):
    buckets = {2: [], 3: [], 4: [], 5: []}
    for rid, correct in model_correct.items():
        n = num_options.get(rid)
        if n not in buckets:
            continue
        buckets[n].append(correct)
    acc = {}
    counts = {}
    for k, vals in buckets.items():
        counts[k] = len(vals)
        acc[k] = sum(1 for v in vals if v) / len(vals) if vals else 0.0
    return acc, counts


def _wilson_half_width(k, n, z=1.96):
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + (z * z / n)
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return half


def _chance_by_options(num_options):
    buckets = {2: [], 3: [], 4: []}
    for rid, n in num_options.items():
        if n not in buckets:
            continue
        buckets[n].append(1.0 / n)
    acc = {}
    counts = {}
    for k, vals in buckets.items():
        counts[k] = len(vals)
        acc[k] = sum(vals) / len(vals) if vals else 0.0
    return acc, counts


def main():
    num_options = _load_num_options(BLUEPRINT_PATH)

    models = [
        ("Gemini 3 Pro", "json", GEMINI_3_PRO_PATH),
        ("GPT-5", "json", GPT_5_PATH),
        ("SFT+RL (Qwen3-VL-8B)", "csv", QWEN_SFT_RL_PATH),
    ]

    model_results = []
    for label, kind, path in models:
        if not path.exists():
            raise FileNotFoundError(path)
        if kind == "json":
            correct = _load_model_json(path)
        elif kind == "csv":
            correct = _load_model_csv(path)
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        acc, counts = _accuracy_by_options(correct, num_options)
        overall = sum(1 for v in correct.values() if v) / len(correct) if correct else 0.0
        model_results.append((label, acc, counts, overall))

    chance_acc, chance_counts = _chance_by_options(num_options)
    model_results.append(("Chance", chance_acc, chance_counts, 0.0))

    labels = [2, 3, 4]
    x = [i * 1.2 for i in range(len(labels))]
    width = 0.2
    offsets = [(i - (len(model_results) - 1) / 2) * width for i in range(len(model_results))]
    counts_by_label = {label: counts for label, _, counts, _ in model_results}
    gemini_counts = counts_by_label.get("Gemini 3 Pro", {})
    gpt_counts = counts_by_label.get("GPT-5", {})
    qwen_counts = counts_by_label.get("SFT+RL (Qwen3-VL-8B)", {})
    xlabels = []
    for l in labels:
        gem = gemini_counts.get(l, 0)
        gpt = gpt_counts.get(l, 0)
        qwen = qwen_counts.get(l, 0)
        xlabels.append(f"{l}\n(Gemini, GPT: {gem})\n(Qwen3-VL: {qwen})")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)

    for (label, acc, counts, _), offset in zip(model_results, offsets):
        color = "#9e9e9e" if label == "Chance" else None
        err = None
        if label != "Chance":
            err = [
                _wilson_half_width(round(acc[l] * counts[l]), counts[l]) if counts[l] else 0.0
                for l in labels
            ]
        ax.bar(
            [i + offset for i in x],
            [acc[l] for l in labels],
            width,
            label=label,
            color=color,
            yerr=err,
            capsize=2 if err else 0,
            ecolor="#bdbdbd",
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("# Options")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. structural difficulty")
    ax.legend(frameon=False, ncols=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)

    for label, _, counts, _ in model_results:
        print(f"{label} counts:", counts)
    print("Saved:", OUT_PNG)

    # Chance-normalized gain plot: (a - c) / (1 - c)
    chance_by_k = {2: 1 / 2, 3: 1 / 3, 4: 1 / 4}
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for (label, acc, counts, _), offset in zip(model_results, offsets):
        if label == "Chance":
            continue
        gains = []
        err = []
        for k in labels:
            a = acc[k]
            c = chance_by_k[k]
            gain = (a - c) / (1 - c) if (1 - c) > 0 else 0.0
            gains.append(gain)
            # Approximate CI by transforming bounds from accuracy CI.
            if counts[k]:
                half = _wilson_half_width(round(a * counts[k]), counts[k])
                upper = min(1.0, a + half)
                lower = max(0.0, a - half)
                upper_g = (upper - c) / (1 - c) if (1 - c) > 0 else 0.0
                lower_g = (lower - c) / (1 - c) if (1 - c) > 0 else 0.0
                err.append((upper_g - lower_g) / 2)
            else:
                err.append(0.0)
        ax2.bar(
            [i + offset for i in x],
            gains,
            width,
            label=label,
            yerr=err,
            capsize=2 if err else 0,
            ecolor="#bdbdbd",
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(xlabels)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("# Options")
    ax2.set_ylabel("Chance-normalized gain")
    ax2.set_title("Chance-normalized accuracy gain vs. #Options")
    ax2.legend(frameon=False, ncols=2, fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_PNG_NORM)
    print("Saved:", OUT_PNG_NORM)


if __name__ == "__main__":
    main()
