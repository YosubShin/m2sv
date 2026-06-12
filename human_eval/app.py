"""FastAPI server for the M2SV human-eval webapp.

Run:
  pip install fastapi uvicorn
  python -m uvicorn human_eval.app:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /api/start    {email}                  -> upsert participant, progress
  GET  /api/next?email=                        -> next unanswered problem (no gold)
  POST /api/answer   {email, problem_id, ...}  -> grade + store, progress
  GET  /api/me?email=                          -> progress
Static frontend served at /.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# App revision: bump when participant-facing task conditions change so cohorts
# stay separable in analysis. rev 1 = original (no instructions/feedback);
# rev 2 = instructions + practice + post-answer feedback + you-vs-AI score.
REVISION = 2

# ---- Load frozen problem set once at startup -------------------------------
_raw = json.loads((HERE / "problems.json").read_text())["problems"]
PROBLEMS: dict[str, dict] = {p["id"]: p for p in _raw}
PROBLEM_IDS: list[str] = [p["id"] for p in _raw]
TOTAL = len(PROBLEM_IDS)

# Per-problem AI data: correctness (head-to-head score), answer + reasoning (reveal).
_ai = json.loads((HERE / "ai_scores.json").read_text())
AI_MODEL: str = _ai.get("model", "AI")
AI_CORRECT: dict[str, bool] = _ai.get("correct", {})
AI_ANSWER: dict[str, str] = _ai.get("answer", {})
AI_REASONING: dict[str, str] = _ai.get("reasoning", {})

app = FastAPI(title="M2SV Human Eval")
db.init_db()


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email address")
    return email


def shuffled_order(email: str) -> list[str]:
    """Deterministic per-user permutation of all problem ids.

    Seeded by email so a returning participant reconstructs the same order and
    resumes exactly where they left off.
    """
    seed = int(hashlib.sha256(email.encode()).hexdigest(), 16) % (2**32)
    order = list(PROBLEM_IDS)
    random.Random(seed).shuffle(order)
    return order


def client_view(problem: dict) -> dict:
    """Strip the gold answer before sending to the browser."""
    return {
        "id": problem["id"],
        "question": problem["question"],
        "options": problem["options"],
        "image_map": problem["image_map"],
        "image_sv": problem["image_sv"],
    }


# ---- API models ------------------------------------------------------------
class StartReq(BaseModel):
    email: str


class AnswerReq(BaseModel):
    email: str
    problem_id: str
    selected: str
    client_elapsed_ms: Optional[int] = None
    hidden_ms: int = 0
    served_at: Optional[float] = None
    flagged: bool = False
    # Client declares which task condition it's running. Old clients omit this
    # and are recorded as rev 1, so answers from a not-yet-refreshed browser
    # stay correctly tagged with the condition the participant actually saw.
    revision: int = 1


# Hide the you-vs-AI head-to-head until the participant has answered this many,
# so it doesn't anchor their first answers and lands as a "reveal".
AI_REVEAL_AT = 10


def score_for(conn, email: str) -> dict:
    """Running head-to-head over the problems this participant has answered."""
    rows = conn.execute(
        "SELECT problem_id, is_correct FROM responses WHERE email = ?", (email,)
    ).fetchall()
    answered = len(rows)
    you = sum(r["is_correct"] for r in rows)
    unlocked = answered >= AI_REVEAL_AT
    s = {
        "answered": answered,
        "you_correct": you,
        "ai_model": AI_MODEL,
        "ai_unlocked": unlocked,
        "ai_reveal_at": AI_REVEAL_AT,
    }
    if unlocked:  # only send the AI number once unlocked
        s["ai_correct"] = sum(1 for r in rows if AI_CORRECT.get(r["problem_id"]))
    return s


# ---- Endpoints -------------------------------------------------------------
@app.post("/api/start")
def start(req: StartReq, user_agent: str = "") -> dict:
    email = normalize_email(req.email)
    with db.connect() as conn:
        db.upsert_participant(conn, email, user_agent)
        prog = db.progress(conn, email)
        sc = score_for(conn, email)
    return {"email": email, "total": TOTAL, "score": sc, **prog}


@app.get("/api/next")
def next_problem(email: str) -> dict:
    email = normalize_email(email)
    with db.connect() as conn:
        done = db.answered_ids(conn, email)
        prog = db.progress(conn, email)
        sc = score_for(conn, email)
    for pid in shuffled_order(email):
        if pid not in done:
            return {
                "problem": client_view(PROBLEMS[pid]),
                "position": prog["done"] + 1,
                "total": TOTAL,
                "served_at": time.time(),
                "score": sc,
                **prog,  # done, total_ms, correct — keeps header stats accurate
            }
    return {"problem": None, "total": TOTAL, "finished": True, "score": sc, **prog}


@app.post("/api/answer")
def answer(req: AnswerReq) -> dict:
    email = normalize_email(req.email)
    problem = PROBLEMS.get(req.problem_id)
    if problem is None:
        raise HTTPException(status_code=404, detail="Unknown problem_id")

    selected = (req.selected or "").strip().upper()
    if selected not in [o.upper() for o in problem["options"]]:
        raise HTTPException(status_code=400, detail="selected not among options")

    is_correct = selected == problem["answer"]
    with db.connect() as conn:
        position = db.progress(conn, email)["done"] + 1
        inserted = db.record_answer(
            conn,
            email=email,
            problem_id=req.problem_id,
            selected=selected,
            is_correct=is_correct,
            client_elapsed_ms=req.client_elapsed_ms,
            hidden_ms=max(0, req.hidden_ms),
            served_at=req.served_at,
            position=position,
            flagged=req.flagged,
            revision=req.revision,
        )
        prog = db.progress(conn, email)
        sc = score_for(conn, email)
    # Reveal correctness so the client can show feedback.
    resp = {
        "ok": True,
        "duplicate": not inserted,
        "total": TOTAL,
        "was_correct": is_correct,  # distinct key; prog also carries a 'correct' count
        "correct_answer": problem["answer"],
        "score": sc,
        **prog,
    }
    # Once the AI is unlocked (>= reveal threshold), also reveal what it answered
    # on THIS problem and its reasoning.
    if sc.get("ai_unlocked") and req.problem_id in AI_ANSWER:
        resp["ai_reveal"] = {
            "model": AI_MODEL,
            "answer": AI_ANSWER[req.problem_id],
            "was_correct": AI_CORRECT.get(req.problem_id, False),
            "reasoning": AI_REASONING.get(req.problem_id, ""),
        }
    return resp


@app.get("/api/me")
def me(email: str) -> dict:
    email = normalize_email(email)
    with db.connect() as conn:
        prog = db.progress(conn, email)
        sc = score_for(conn, email)
    return {"email": email, "total": TOTAL, "score": sc, **prog}


@app.get("/practice.json")
def practice() -> FileResponse:
    return FileResponse(HERE / "practice.json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=STATIC), name="static")
