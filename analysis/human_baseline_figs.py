# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate the human-baseline figures + table numbers for m2sv.

Reads ONLY analysis/human_baseline_data.json (committed, anonymized, no PII) and
writes the figures into the paper's figures/ directory. Fully self-contained, so
readers reproduce every figure with:
  uv run analysis/human_baseline_figs.py
(Authors regenerate the bundle from the private DB via analysis/anonymize_human_data.py.)
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "publications/icml2026/2026-01-28/figures"

DATA = json.loads((ROOT / "analysis/human_baseline_data.json").read_text())
LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
P = {pid: {"answer": d["answer"], "options": LETTERS[:d["n_options"]]}
     for pid, d in DATA["problems"].items()}
EXP = DATA["expert"]                        # expert annotator predictions (A1)
GEM = DATA["models"]["Gemini-3-Pro"]
QWEN = DATA["models"]["Qwen3-VL"]

NAIVE = sorted(DATA["annotators"])           # already anonymized: A2..A8
ans, rt, rtp, completed = {}, {}, {}, {}
for aid, a in DATA["annotators"].items():
    ans[aid] = {r["pid"]: r["selected"] for r in a["responses"]}
    rtp[aid] = {r["pid"]: (r["elapsed_ms"] or 0) / 1000 for r in a["responses"]}
    rt[aid] = float(np.median(list(rtp[aid].values()))) if rtp[aid] else 0.0
    completed[aid] = a["completed_all"]
LABEL = {aid: aid for aid in NAIVE}          # ids are already the anon labels

def accuracy(d):
    cm = [pid for pid in d if pid in P]
    return sum(1 for pid in cm if d[pid] == P[pid]["answer"]) / len(cm), len(cm)

def kappa(d1, d2):
    cm = [pid for pid in d1 if pid in d2 and pid in P]
    if len(cm) < 5:
        return np.nan
    po = sum(1 for pid in cm if d1[pid] == d2[pid]) / len(cm)
    pe = sum(1 / len(P[pid]["options"]) for pid in cm) / len(cm)
    return (po - pe) / (1 - pe)

# Engaged = excludes near-chance outliers, flagged by agreement (kappa vs expert
# >= 0.3), not by accuracy.
ENGAGED = [aid for aid in NAIVE if kappa(ans[aid], EXP) >= 0.3]

# =====================================================================
# Print the numbers that go in the LaTeX table
# =====================================================================
print("Annotator              n   acc   medRT")
naive_acc = []
for e in sorted(NAIVE, key=lambda e: -accuracy(ans[e])[0]):
    a, n = accuracy(ans[e]); naive_acc.append(a)
    print(f"{LABEL[e]:<14} {n:>5} {a:>5.0%} {rt[e]:>5.0f}s   kappa_vs_expert={kappa(ans[e],EXP):.2f}")
ea, _ = accuracy(EXP)
print(f"{'EXPERT':<14} {200:>5} {ea:>5.0%}     -")
print(f"\nAll naive (n={len(naive_acc)}): mean {np.mean(naive_acc):.1%}  SD {np.std(naive_acc):.1%}")
eng = [accuracy(ans[e])[0] for e in ENGAGED]
print(f"Engaged   (n={len(eng)}): mean {np.mean(eng):.1%}  SD {np.std(eng):.1%}")
print(f"Models: Gemini-3-Pro {accuracy(GEM)[0]:.0%}  Qwen-plus {accuracy(QWEN)[0]:.0%}")

FIGDIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
GEM_ACC, GPT5_ACC, RAND = accuracy(GEM)[0], 0.572, 0.317  # GPT-5 from paper

def save(fig, name):
    # PNG only: matches the paper's existing figures and avoids the
    # publications/**/*.pdf gitignore rule swallowing figure sources.
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---- Fig 1: per-annotator accuracy distribution ---------------------------
order = sorted(NAIVE, key=lambda e: accuracy(ans[e])[0])
fig, ax = plt.subplots(figsize=(5.0, 3.0))
for i, e in enumerate(order):
    a, n = accuracy(ans[e])
    out = kappa(ans[e], EXP) < 0.3
    ax.scatter(i, a, s=30 + n, color="#c0392b" if out else "#2e6fdb",
               zorder=3, edgecolor="white", linewidth=.5)
    ax.annotate(f"{a:.0%}\n(n={n})", (i, a), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7)
ax.scatter(len(order), ea, marker="*", s=180, color="#1e8449", zorder=3,
           edgecolor="white", linewidth=.5)
ax.annotate(f"{ea:.0%}", (len(order), ea), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7)
for y, lab, c in [(GEM_ACC, f"Gemini-3-Pro {GEM_ACC:.0%}", "#7f8c8d"),
                  (GPT5_ACC, f"GPT-5 {GPT5_ACC:.0%}", "#95a5a6"),
                  (RAND, f"random {RAND:.0%}", "#bdc3c7")]:
    ax.axhline(y, ls="--", lw=.9, color=c)
    ax.text(len(order) + .15, y, lab, va="center", fontsize=7, color=c)
ax.set_xticks(list(range(len(order))) + [len(order)])
ax.set_xticklabels([LABEL[e] for e in order] + ["A1\n(expert)"], rotation=0, ha="center", fontsize=7)
ax.set_ylabel("Accuracy"); ax.set_ylim(0.25, 1.0)
ax.set_yticks(np.arange(0.3, 1.01, 0.1)); ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0.3, 1.01, 0.1)])
ax.set_title("Human accuracy on m2sv (200-item subset)", fontsize=9)
ax.margins(x=0.16)
save(fig, "human_accuracy_dist")
print("\nwrote human_accuracy_dist")

# ---- Fig 2: Cohen's kappa agreement heatmap -------------------------------
humans_sorted = sorted(NAIVE, key=lambda e: LABEL[e])  # A2..A8
raters = [("A1 (expert)", EXP)] + [(LABEL[e], ans[e]) for e in humans_sorted] + \
         [("Gemini-3-Pro", GEM), ("Qwen-plus", QWEN)]
names = [n for n, _ in raters]
M = np.full((len(raters), len(raters)), np.nan)
for i, (_, di) in enumerate(raters):
    for j, (_, dj) in enumerate(raters):
        M[i, j] = 1.0 if i == j else kappa(di, dj)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(M, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
for i in range(len(names)):
    for j in range(len(names)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if M[i, j] > 0.25 else "white")
ax.set_title("Inter-rater agreement (Cohen's $\\kappa$)", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\kappa$")
save(fig, "agreement_kappa")
print("wrote agreement_kappa")

# ---- Fig 3: non-circular difficulty by #options ---------------------------
def per_problem_acc(group):
    byk = {2: [], 3: [], 4: []}
    for pid in P:
        k = len(P[pid]["options"])
        if k not in byk:
            continue
        rs = [1 if ans[e].get(pid) == P[pid]["answer"] else 0 for e in group if pid in ans[e]]
        if rs:
            byk[k].append(np.mean(rs))
    return {k: (np.mean(v) if v else np.nan, len(v)) for k, v in byk.items()}

hum = per_problem_acc(ENGAGED)
gem = {k: (np.mean([1 if GEM.get(pid) == P[pid]["answer"] else 0
            for pid in P if len(P[pid]["options"]) == k]), 0) for k in (2, 3, 4)}
ks = [2, 3, 4]; x = np.arange(len(ks)); w = 0.38
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.bar(x - w/2, [hum[k][0] for k in ks], w, label="Engaged humans (n=6)", color="#2e6fdb")
ax.bar(x + w/2, [gem[k][0] for k in ks], w, label="Gemini-3-Pro", color="#e67e22")
ax.plot(x, [1/k for k in ks], "k--", lw=.9, marker="o", ms=3, label="random")
for xi, k in zip(x, ks):
    ax.text(xi - w/2, hum[k][0] + .02, f"{hum[k][0]:.0%}", ha="center", fontsize=7)
    ax.text(xi + w/2, gem[k][0] + .02, f"{gem[k][0]:.0%}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"{k} options\n({hum[k][1]} probs)" for k in ks], fontsize=8)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.25)])
ax.set_title("Difficulty by candidate count (non-circular)", fontsize=9)
ax.legend(fontsize=7, frameon=False)
save(fig, "difficulty_by_options")
print("wrote difficulty_by_options")

# ---- Fig 4: de-circularized difficulty by human RT tertiles ---------------
# Difficulty = per-problem mean RT across the 3 engaged annotators who completed
# all 200 (kappa>=0.5). Accuracy is then measured separately for humans/models,
# so difficulty and the accuracy curve do not come from one annotator.
ENG3 = [aid for aid in ENGAGED if completed[aid]]   # engaged full-completers
prt, ha = {}, {}
for pid in P:
    rts = [rtp[a][pid] for a in ENG3 if pid in rtp[a]]
    cors = [1 if ans[a][pid] == P[pid]["answer"] else 0 for a in ENG3 if pid in ans[a]]
    if rts:
        prt[pid] = float(np.mean(rts)); ha[pid] = float(np.mean(cors))
order = sorted(prt, key=lambda p: prt[p]); t = len(order) // 3
buckets = [("Easy", order[:t]), ("Medium", order[t:2*t]), ("Hard", order[2*t:])]
def macc(d, ids): return float(np.mean([1 if d.get(p) == P[p]["answer"] else 0 for p in ids]))
H = [float(np.mean([ha[p] for p in ids])) for _, ids in buckets]
G = [macc(GEM, ids) for _, ids in buckets]
Q = [macc(QWEN, ids) for _, ids in buckets]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.plot(x, H, "-o", color="#1e8449", label="Human (engaged, n=3)")
ax.plot(x, G, "-s", color="#e67e22", label="Gemini-3-Pro")
ax.plot(x, Q, "-^", color="#7f8c8d", label="Qwen3-VL")
for xi, (h, g) in enumerate(zip(H, G)):
    ax.text(xi, h + .03, f"{h:.0%}", ha="center", fontsize=7, color="#1e8449")
    ax.text(xi, g - .07, f"{g:.0%}", ha="center", fontsize=7, color="#e67e22")
ax.set_xticks(x); ax.set_xticklabels(["Easy", "Medium", "Hard"])
ax.set_ylabel("Accuracy"); ax.set_ylim(0.2, 1.0)
ax.set_yticks(np.arange(.2, 1.01, .2)); ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(.2, 1.01, .2)])
ax.set_title("Accuracy vs.\\ human-perceived difficulty", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower left")
save(fig, "human_difficulty_decircular")
print(f"wrote human_difficulty_decircular  (H={[f'{h:.0%}' for h in H]} G={[f'{g:.0%}' for g in G]})")
