"""Build the committable, anonymized human-baseline data bundle.

Authors-only step: reads the private human-eval DB (PII emails) plus the expert
and model predictions, strips all identifiers, and writes
``analysis/human_baseline_data.json`` --- the single self-contained input that
``human_baseline_figs.py`` consumes so readers can reproduce the figures without
any private data.

Inputs (gitignored / local):
  /tmp/eval-prod.db                  human responses (pull from the VM)
  analysis/participants.local.json   {naive: [...emails...]}
  results/manual.json                expert annotator predictions
  results/{gemini-3-pro,qwen3-vl-plus}.json   model predictions
  human_eval/problems.json           gold answers + option counts

Output (committed, no PII):
  analysis/human_baseline_data.json
"""
import json, sqlite3, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = "/tmp/eval-prod.db"

cfg = json.loads((ROOT / "analysis/participants.local.json").read_text())
NAIVE = cfg["naive"]
P = {p["id"]: p for p in json.loads((ROOT / "human_eval/problems.json").read_text())["problems"]}
EXPERT = {str(r["id"]): str(r["pred"]).strip().upper()
          for r in json.loads((ROOT / "results/manual.json").read_text())["results"]}

def model(path):
    return {str(r["id"]): str(r.get("pred", "")).strip().upper()
            for r in json.loads((ROOT / path).read_text())["results"]}
MODELS = {"Gemini-3-Pro": model("results/gemini-3-pro.json"),
          "Qwen3-VL": model("results/qwen3-vl-plus.json"),
          "Qwen3-VL-235B-Thinking": model("past_results/2025-10-23/qwen3-vl-235b-a22b-thinking.json")}

conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
resp = {e: conn.execute(
            "SELECT problem_id, selected, client_elapsed_ms FROM responses WHERE email=?",
            (e,)).fetchall()
        for e in NAIVE}

def acc(e):
    cm = [r for r in resp[e] if r["problem_id"] in P]
    return sum(1 for r in cm if r["selected"] == P[r["problem_id"]]["answer"]) / len(cm)

# Analyze only annotators who completed all 200 items, so every figure is computed
# over the same item set. Two partial annotators (17 and 76 items) are excluded for
# comparability; their small overlap makes per-pair kappa unstable.
COMPLETERS = [e for e in NAIVE if len([r for r in resp[e] if r["problem_id"] in P]) >= 200]

# Anonymize: expert = A1; naive annotators = A2.. by descending accuracy.
anon = {e: f"A{i+2}" for i, e in enumerate(sorted(COMPLETERS, key=acc, reverse=True))}

bundle = {
    "_about": "Anonymized m2sv human-baseline data. No PII. See analysis/human_baseline_figs.py.",
    "problems": {pid: {"answer": p["answer"], "n_options": len(p["options"]),
                       "azimuths": p.get("meta", {}).get("azimuths", [])}
                 for pid, p in P.items()},
    "expert": EXPERT,
    "models": MODELS,
    "annotators": {
        anon[e]: {
            "completed_all": len([r for r in resp[e] if r["problem_id"] in P]) >= 200,
            "responses": [{"pid": r["problem_id"], "selected": r["selected"],
                           "elapsed_ms": r["client_elapsed_ms"]}
                          for r in resp[e] if r["problem_id"] in P],
        }
        for e in COMPLETERS
    },
}
out = ROOT / "analysis/human_baseline_data.json"
out.write_text(json.dumps(bundle, indent=0))
print(f"wrote {out}  ({len(bundle['annotators'])} annotators, "
      f"{sum(len(a['responses']) for a in bundle['annotators'].values())} responses, no PII)")
