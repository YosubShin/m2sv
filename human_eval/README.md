# M2SV Human-Eval Webapp

Hostable replacement for the Streamlit `review_webapp.py`. Collects human answers
(with per-problem timing) on the 200-problem set from `results/manual.json`.

## Layout
- `build_dataset.py` — freeze the 200 problems + copy images into `static/img/`
- `app.py` — FastAPI server (`/api/start`, `/api/next`, `/api/answer`, `/api/me`)
- `db.py` — SQLite schema/helpers (`eval.db`, WAL)
- `export.py` — dump responses to CSV + per-participant `evaluate_vlm` JSON
- `static/` — single-page keyboard-driven frontend
- `problems.json` — frozen set (committed); images are regenerated, not committed

## Setup (on the VM)
```bash
pip install -r human_eval/requirements.txt
# Regenerate the frozen images from the local dataset (problems.json is committed):
python human_eval/build_dataset.py
# Serve:
python -m uvicorn human_eval.app:app --host 0.0.0.0 --port 8000
```
Put it behind nginx/caddy (TLS) or a tunnel and share the URL.

## Design notes
- **Identity:** participant email. Returning email resumes; each problem answered once.
- **Order:** per-user random shuffle, seeded by email (deterministic resume).
- **Timing:** browser measures load→submit (`client_elapsed_ms`); tab-hidden time
  tracked separately (`hidden_ms`). Server also logs `served_at`/`answered_at`.
- **No feedback** shown (no correctness) to avoid learning effects.
- **Global `problem_id`** preserved for cross-participant comparison.

## Export for analysis
```bash
python human_eval/export.py --out human_eval/export
```
Per-participant JSONs match the `evaluate_vlm` format, so the `analysis/` scripts
run unchanged.
