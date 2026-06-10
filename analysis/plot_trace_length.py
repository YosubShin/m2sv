import csv
import json
import math
import os
import re
from pathlib import Path
from statistics import mean, quantiles

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUMAN_PATH = Path("past_results/2026-01-23/manual.json")
MODEL_SOURCES = [
    ("Gemini 3 Pro", "json", Path("past_results/2026-01-23/gemini-3-pro.json"), "raw"),
    ("Qwen3-VL-235B (Thinking)", "json", Path("past_results/2025-10-23/qwen3-vl-235b-a22b-thinking.json"), "raw"),
    ("Qwen3-VL-8B (SFT+RL)", "csv", Path("past_results/2026-01-23/qwen3-vl-8b-instruct-sft-rl.csv"), "raw_response"),
    ("Qwen3-VL-8B (SFT)", "csv", Path("past_results/2026-01-23/qwen3-vl-8b-instruct-sft.csv"), "raw_response"),
    ("Qwen3-VL-8B (Instruct)", "json", Path("past_results/2025-10-23/qwen3-vl-8b-instruct.json"), "raw"),
]

OUT_DIR = Path("analysis/plots")
OUT_TRACE_BY_DIFFICULTY = OUT_DIR / "trace_length_by_difficulty.png"
OUT_EARLY_COMMIT = OUT_DIR / "early_commitment_hard.png"


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


def _load_human_elapsed(path: Path):
    rows = _load_json_results(path)
    out = {}
    for r in rows:
        rid = r.get("id")
        elapsed = r.get("elapsed_seconds")
        if rid is None or elapsed is None:
            continue
        out[str(rid)] = float(elapsed)
    return out


def _load_model_json(path: Path, raw_key: str):
    rows = _load_json_results(path)
    out = {}
    for r in rows:
        rid = r.get("id")
        if rid is None:
            continue
        out[str(rid)] = {
            "correct": _to_bool(r.get("correct", False)),
            "raw": r.get(raw_key) or "",
        }
    return out


def _load_model_csv(path: Path, raw_key: str):
    out = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            if rid is None:
                continue
            out[str(rid)] = {
                "correct": _to_bool(row.get("correct")),
                "raw": row.get(raw_key) or "",
            }
    return out


def _bucketize(elapsed_seconds):
    q1, q2 = quantiles(elapsed_seconds, n=3, method="inclusive")

    def bucket(t):
        if t <= q1:
            return "Easy"
        if t <= q2:
            return "Medium"
        return "Hard"

    return bucket, (q1, q2)


def _tokenize(text: str):
    # Approximate tokenization: split into word-like chunks + punctuation.
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _count_tokens(text: str) -> int:
    return len(_tokenize(text))


def _find_commit_char_index(text: str):
    patterns = [
        r"final answer[^A-C]*[A-C]",
        r"answer[^A-C]*[A-C]",
        r"\\boxed\{[A-C]\}",
        r"option\s*[A-C]\s*(is|seems)?\s*(not|incorrect|wrong)",
        r"\b[A-C]\s*(is|seems)?\s*(not|incorrect|wrong)\b",
        r"\b[A-C]\s*does\s*not\b",
        r"eliminate\s*[A-C]",
    ]
    earliest = None
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            idx = match.start()
            if earliest is None or idx < earliest:
                earliest = idx
    return earliest


def _commit_fraction(text: str):
    total_tokens = _count_tokens(text)
    if total_tokens == 0:
        return None
    idx = _find_commit_char_index(text)
    if idx is None:
        return None
    prefix_tokens = _count_tokens(text[:idx])
    return prefix_tokens / total_tokens


def _mean_and_sem(values):
    if not values:
        return 0.0, 0.0
    m = mean(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    sem = math.sqrt(var / len(values))
    return m, sem


def main():
    human_elapsed = _load_human_elapsed(HUMAN_PATH)
    if not human_elapsed:
        raise ValueError("No human elapsed times found.")

    bucket_fn, cuts = _bucketize(list(human_elapsed.values()))
    print("Difficulty cuts (elapsed seconds):", cuts)

    records = []
    for label, kind, path, raw_key in MODEL_SOURCES:
        if kind == "json":
            model = _load_model_json(path, raw_key)
        elif kind == "csv":
            model = _load_model_csv(path, raw_key)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        for rid, elapsed in human_elapsed.items():
            if rid not in model:
                continue
            raw = model[rid]["raw"]
            records.append({
                "model": label,
                "id": rid,
                "elapsed": elapsed,
                "bucket": bucket_fn(elapsed),
                "tokens": _count_tokens(raw),
                "correct": model[rid]["correct"],
                "commit_frac": _commit_fraction(raw),
            })

    if not records:
        raise ValueError("No matched records between human annotations and model outputs.")

    # Plot 1: Trace length vs difficulty bucket.
    buckets = ["Easy", "Medium", "Hard"]
    bucket_counts = {
        b: sum(1 for elapsed in human_elapsed.values() if bucket_fn(elapsed) == b)
        for b in buckets
    }
    models = [m[0] for m in MODEL_SOURCES]
    colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"]

    fig1, ax1 = plt.subplots(figsize=(7.4, 4.4), dpi=150)
    x = list(range(len(buckets)))

    for i, model in enumerate(models):
        means = []
        sems = []
        for b in buckets:
            vals = [r["tokens"] for r in records if r["model"] == model and r["bucket"] == b]
            m, s = _mean_and_sem(vals)
            means.append(m)
            sems.append(s)
        ax1.plot(x, means, marker="o", linewidth=2, label=model, color=colors[i % len(colors)])
        ax1.errorbar(x, means, yerr=sems, fmt="none", ecolor=colors[i % len(colors)], capsize=2, alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n(n={bucket_counts[b]})" for b in buckets])
    ax1.set_xlabel("Difficulty bucket (human elapsed time)")
    ax1.set_ylabel("Approx. trace tokens")
    ax1.set_title("Trace length vs. difficulty")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(OUT_TRACE_BY_DIFFICULTY)
    print("Saved:", OUT_TRACE_BY_DIFFICULTY)

    # Plot 3: Early commitment proxy on hard examples.
    fig3, ax3 = plt.subplots(figsize=(7.4, 4.2), dpi=150)
    for i, model in enumerate(models):
        hard = [r for r in records if r["model"] == model and r["bucket"] == "Hard"]
        hard = [r for r in hard if r["commit_frac"] is not None]
        if len(hard) < 4:
            continue
        commit_fracs = [r["commit_frac"] for r in hard]
        q1, q2, q3 = quantiles(commit_fracs, n=4, method="inclusive")

        def quartile(v):
            if v <= q1:
                return 1
            if v <= q2:
                return 2
            if v <= q3:
                return 3
            return 4

        bucketed = {1: [], 2: [], 3: [], 4: []}
        for r in hard:
            bucketed[quartile(r["commit_frac"])].append(r["correct"])

        xs = []
        ys = []
        for q in range(1, 5):
            vals = bucketed[q]
            if not vals:
                xs.append(q)
                ys.append(0.0)
                continue
            ys.append(sum(1 for v in vals if v) / len(vals))
            xs.append(q)

        ax3.plot(xs, ys, marker="o", label=model, color=colors[i % len(colors)])

    ax3.set_xticks([1, 2, 3, 4])
    ax3.set_xticklabels(["Q1 (earliest)", "Q2", "Q3", "Q4 (latest)"])
    ax3.set_xlabel("Commitment timing quartile (hard only)")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Early commitment proxy vs accuracy (hard bucket)")
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, fontsize=9)

    fig3.tight_layout()
    fig3.savefig(OUT_EARLY_COMMIT)
    print("Saved:", OUT_EARLY_COMMIT)


if __name__ == "__main__":
    main()
