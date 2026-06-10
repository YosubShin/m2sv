import csv
import json
import math
import os
from pathlib import Path
from statistics import quantiles

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUMAN_PATH = Path("past_results/2026-01-23/manual.json")
QWEN_SFT_RL_PATH = Path("past_results/2026-01-23/qwen3-vl-8b-instruct-sft-rl.csv")
GEMINI_3_PRO_PATH = Path("past_results/2026-01-23/gemini-3-pro.json")

GEMINI_2_5_PRO_PATH = Path("past_results/2025-10-23/gemini-2-5-pro.json")
GPT_5_PATH = Path("past_results/2025-10-23/gpt-5.json")
QWEN_8B_BASE_PATH = Path("past_results/2025-10-23/qwen3-vl-8b-instruct.json")
QWEN_235B_PATH = Path("past_results/2025-10-23/qwen3-vl-235b-a22b-thinking.json")
BLUEPRINT_PATH = Path("blueprints/11k/validation.jsonl")
OUT_DIR = Path("analysis/plots")
OUT_PNG = OUT_DIR / "time_difficulty_accuracy.png"


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


def _load_human(path: Path):
    rows = _load_json_results(path)
    out = {}
    for r in rows:
        rid = str(r.get("id"))
        elapsed = r.get("elapsed_seconds")
        if rid is None or elapsed is None:
            continue
        out[rid] = {
            "elapsed_seconds": float(elapsed),
            "correct": _to_bool(r.get("correct", False)),
        }
    return out


def _load_model_json(path: Path):
    rows = _load_json_results(path)
    out = {}
    all_correct = []
    for r in rows:
        rid = str(r.get("id"))
        if rid is None:
            continue
        val = _to_bool(r.get("correct", False))
        out[rid] = val
        all_correct.append(val)
    return out, all_correct


def _load_model_csv(path: Path):
    out = {}
    all_correct = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            if rid is None:
                continue
            val = _to_bool(row.get("correct"))
            out[rid] = val
            all_correct.append(val)
    return out, all_correct


def _load_num_options(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = str(obj.get("intersection_id"))
            labels = obj.get("labels") or []
            if rid is None:
                continue
            out[rid] = len(labels)
    return out


def _bucketize(times):
    # Use raw elapsed time to define tertiles; log used only for internal sanity checks.
    _ = [math.log(max(t, 1e-9)) for t in times]
    q1, q2 = quantiles(times, n=3, method="inclusive")
    def bucket(t):
        if t <= q1:
            return "Easy"
        if t <= q2:
            return "Medium"
        return "Hard"
    return bucket, (q1, q2)


def _accuracy_by_bucket(human, model_correct):
    buckets = {"Easy": [], "Medium": [], "Hard": []}
    times = [v["elapsed_seconds"] for v in human.values()]
    bucket_fn, cuts = _bucketize(times)

    for rid, h in human.items():
        b = bucket_fn(h["elapsed_seconds"])
        if model_correct is None:
            buckets[b].append(h["correct"])
        else:
            if rid in model_correct:
                buckets[b].append(model_correct[rid])

    acc = {}
    counts = {}
    for b, vals in buckets.items():
        counts[b] = len(vals)
        k = sum(1 for v in vals if v)
        acc[b] = k / len(vals) if vals else 0.0
    return acc, counts, cuts


def _wilson_half_width(k, n, z=1.96):
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + (z * z / n)
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return half


def _random_baseline_by_bucket(human, num_options):
    buckets = {"Easy": [], "Medium": [], "Hard": []}
    times = [v["elapsed_seconds"] for v in human.values()]
    bucket_fn, _ = _bucketize(times)
    for rid, h in human.items():
        n = num_options.get(rid)
        if not n:
            continue
        b = bucket_fn(h["elapsed_seconds"])
        buckets[b].append(1.0 / n)
    acc = {}
    counts = {}
    for b, vals in buckets.items():
        counts[b] = len(vals)
        acc[b] = sum(vals) / len(vals) if vals else 0.0
    return acc, counts


def main():
    human = _load_human(HUMAN_PATH)
    num_options = _load_num_options(BLUEPRINT_PATH)
    model_sources = [
        ("Human", None, None),
        ("SFT+RL (Qwen3-VL-8B)", "csv", QWEN_SFT_RL_PATH),
        ("Gemini 3 Pro", "json", GEMINI_3_PRO_PATH),
        ("GPT-5", "json", GPT_5_PATH),
    ]

    model_results = []
    cuts = None
    for label, kind, path in model_sources:
        if kind is None:
            acc, counts, cuts = _accuracy_by_bucket(human, None)
            overall = sum(1 for v in human.values() if v["correct"]) / len(human) if human else 0.0
            model_results.append((label, acc, counts, overall, overall, len(human)))
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        if kind == "json":
            correct, all_correct = _load_model_json(path)
        elif kind == "csv":
            correct, all_correct = _load_model_csv(path)
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        acc, counts, cuts = _accuracy_by_bucket(human, correct)
        # Overall accuracy on the human-matched IDs only.
        matched = [v for rid, v in correct.items() if rid in human]
        overall = sum(1 for v in matched if v) / len(matched) if matched else 0.0
        overall_full = sum(1 for v in all_correct if v) / len(all_correct) if all_correct else 0.0
        model_results.append((label, acc, counts, overall, overall_full, len(all_correct)))

    random_acc, random_counts = _random_baseline_by_bucket(human, num_options)
    random_vals = [1.0 / num_options[rid] for rid in human if rid in num_options]
    random_overall = sum(random_vals) / len(random_vals) if random_vals else 0.0
    model_results.append(("Chance", random_acc, random_counts, random_overall, random_overall, len(random_vals)))

    model_results.sort(key=lambda item: item[3], reverse=True)

    labels = ["Easy", "Medium", "Hard"]
    x = [i * 1.2 for i in range(len(labels))]
    width = 0.16
    offsets = [(i - (len(model_results) - 1) / 2) * width for i in range(len(model_results))]
    human_counts = model_results[0][2]
    xlabels = [f"{lbl} (n={human_counts.get(lbl, 0)})" for lbl in labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for (label, acc, counts, _, _, _), offset in zip(model_results, offsets):
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
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. human-perceived difficulty")
    ax.legend(frameon=False, ncols=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)

    print("Bucket cutoffs (seconds):", cuts)
    for label, _, counts, overall, overall_full, n_full in model_results:
        print(f"{label} overall accuracy (matched 200): {overall:.3f}")
        if n_full:
            print(f"{label} overall accuracy (all {n_full}): {overall_full:.3f}")
        print(f"{label} counts:", counts)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
