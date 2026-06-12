"use strict";

const $ = (id) => document.getElementById(id);
const LETTERS = ["A", "B", "C", "D", "E"];
// Task condition this client implements:
//   rev 3 = instructions + practice + feedback + score + AI answer & reasoning reveal.
const REVISION = 3;

const state = {
  email: null,
  problem: null,
  servedAt: null,        // server timestamp echoed back for cross-check
  loadedAt: null,        // performance.now() when both images finished loading
  selected: null,        // chosen letter
  hiddenMs: 0,           // accumulated tab-hidden time for current problem
  hiddenSince: null,
  rafId: null,
  imgsPending: 0,
  phase: "answer",       // "answer" -> selecting; "feedback" -> reviewing result
  isPractice: false,
  practice: null,        // cached practice problem
  aiRevealAvailable: false,
};

// ---- Timing helpers --------------------------------------------------------
function rawElapsedMs() {
  if (state.loadedAt == null) return 0;
  return performance.now() - state.loadedAt;
}
function currentHiddenMs() {
  let h = state.hiddenMs;
  if (state.hiddenSince != null) h += performance.now() - state.hiddenSince;
  return h;
}
function fmtSec(ms) { return (ms / 1000).toFixed(1) + "s"; }
function fmtClock(ms) {
  const s = Math.round(ms / 1000);
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60);
  return m + "m " + (s % 60) + "s";
}

function tickTimer() {
  $("timer").textContent = fmtSec(Math.max(0, rawElapsedMs() - currentHiddenMs()));
  state.rafId = requestAnimationFrame(tickTimer);
}

// Track "away" time from two signals: tab visibility + window focus.
function markAway() {
  if (state.hiddenSince == null) state.hiddenSince = performance.now();
}
function markBack() {
  if (state.hiddenSince != null) {
    state.hiddenMs += performance.now() - state.hiddenSince;
    state.hiddenSince = null;
  }
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) markAway(); else markBack();
});
window.addEventListener("blur", markAway);
window.addEventListener("focus", markBack);

// ---- API -------------------------------------------------------------------
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

function updateProgress(p) {
  const done = p.done || 0, total = p.total || 200;
  $("progress-text").textContent = `${done} / ${total}`;
  $("progress-fill").style.width = (100 * done / total) + "%";
  $("total-time").textContent = fmtClock(p.total_ms || 0);
  $("avg").textContent = done > 0 ? fmtSec((p.total_ms || 0) / done) : "–";
}

function updateScore(sc) {
  if (!sc) return;
  $("score-you").textContent = `You: ${sc.you_correct}/${sc.answered}`;
  const ai = $("score-ai");
  const at = sc.ai_reveal_at || 10;
  if (sc.ai_unlocked) {
    ai.classList.remove("muted");
    ai.title = `How ${sc.ai_model} scored on the problems you've seen`;
    ai.textContent = `AI: ${sc.ai_correct}/${sc.answered}`;
  } else {
    ai.classList.add("muted");
    const left = Math.max(0, at - (sc.answered || 0));
    ai.title = `The AI's score and its reasoning unlock after ${at} answers`;
    ai.textContent = left > 0
      ? `AI: 🔒 ${left} more to unlock`
      : "AI: 🔒";
  }
}

// Light, safe rendering of the AI's reasoning (escape, then **bold** + newlines).
function escapeHtml(s) {
  return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
function renderReasoning(s) {
  return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");
}

// ---- Render a problem ------------------------------------------------------
function renderOptions(options) {
  const box = $("options");
  box.innerHTML = "";
  options.forEach((opt, i) => {
    const letter = LETTERS[i];
    const el = document.createElement("div");
    el.className = "option";
    el.dataset.letter = letter;
    el.innerHTML = `<span class="num">${i + 1}</span><span>${letter}</span>`;
    el.addEventListener("click", () => selectOption(letter));
    box.appendChild(el);
  });
}

function selectOption(letter) {
  if (state.phase !== "answer" || !state.problem) return;
  if (!state.problem.options.map((o) => o.toUpperCase()).includes(letter)) return;
  state.selected = letter;
  document.querySelectorAll(".option").forEach((el) => {
    el.classList.toggle("selected", el.dataset.letter === letter);
  });
  $("next-btn").disabled = false;
}

function startTimerWhenLoaded() {
  state.imgsPending = 2;
  const onLoad = () => {
    if (--state.imgsPending === 0) {
      state.loadedAt = performance.now();
      cancelAnimationFrame(state.rafId);
      tickTimer();
    }
  };
  const map = $("img-map"), sv = $("img-sv");
  map.onload = sv.onload = onLoad;
  map.onerror = sv.onerror = onLoad;
  map.src = state.problem.image_map;
  sv.src = state.problem.image_sv;
}

// Reset the per-problem UI/timing state before showing a new problem.
function resetForNewProblem() {
  cancelAnimationFrame(state.rafId);
  $("timer").textContent = "0.0s";
  state.selected = null;
  state.hiddenMs = 0;
  state.hiddenSince = (document.hidden || !document.hasFocus()) ? performance.now() : null;
  state.loadedAt = null;
  state.phase = "answer";
  $("next-btn").disabled = true;
  $("next-btn").innerHTML = "Submit &amp; Next ⏎";
  $("feedback").classList.add("hidden");
  $("feedback").className = "hidden";
  $("ai-panel").classList.add("hidden");
  $("stage").classList.remove("with-ai");
  state.aiRevealAvailable = false;
}

async function loadNext() {
  resetForNewProblem();
  const data = await api(`/api/next?email=${encodeURIComponent(state.email)}`);
  updateProgress(data);
  updateScore(data.score);
  if (!data.problem) {
    finish(data);
    return;
  }
  state.problem = data.problem;
  state.servedAt = data.served_at;
  renderOptions(data.problem.options);
  startTimerWhenLoaded();
}

// ---- Feedback --------------------------------------------------------------
function showFeedback(correct, correctLetter, extraHtml, aiReveal) {
  state.phase = "feedback";
  cancelAnimationFrame(state.rafId);
  document.querySelectorAll(".option").forEach((el) => {
    if (el.dataset.letter === correctLetter) el.classList.add("correct");
    if (!correct && el.dataset.letter === state.selected) el.classList.add("wrong");
  });
  const fb = $("feedback");
  fb.className = correct ? "good" : "bad";
  const head = correct ? "✓ Correct!" : `✗ Not quite — the answer was ${correctLetter}.`;
  let html = `<b>${head}</b>${extraHtml || ""}`;

  // AI reveal (only once unlocked; server omits it before then). The reasoning
  // trace stays hidden behind an opt-in toggle so it isn't intrusive.
  state.aiRevealAvailable = !!aiReveal;
  const panel = $("ai-panel");
  panel.classList.add("hidden");
  if (aiReveal) {
    const ok = aiReveal.was_correct;
    html += ` · <span class="ai-inline ${ok ? "ok" : "no"}">🤖 ${aiReveal.model}: `
      + `${aiReveal.answer} ${ok ? "✓" : "✗"}</span>`
      + ` <button type="button" id="ai-toggle" class="linkbtn">show reasoning</button>`;
    panel.innerHTML =
      `<div class="ai-panel-head"><span class="ai-head ${ok ? "ok" : "no"}">🤖 ${aiReveal.model} answered `
      + `<b>${aiReveal.answer}</b> — ${ok ? "✓ correct" : "✗ wrong"}</span>`
      + `<button type="button" id="ai-close" class="linkbtn">✕ close</button></div>`
      + `<div class="ai-reasoning">${renderReasoning(aiReveal.reasoning)}</div>`;
  }
  html += ` <span class="muted">Press Enter to continue.</span>`;
  fb.innerHTML = html;
  if (aiReveal) {
    $("ai-toggle").addEventListener("click", toggleAiPanel);
    $("ai-close").addEventListener("click", toggleAiPanel);
  }

  $("next-btn").disabled = false;
  $("next-btn").innerHTML = state.isPractice ? "Start the real task →" : "Next ⏎";
}

// Toggle the AI reasoning overlay (opt-in; dismissible).
function toggleAiPanel() {
  if (!state.aiRevealAvailable) return;
  const hidden = $("ai-panel").classList.toggle("hidden");
  $("stage").classList.toggle("with-ai", !hidden);  // make room beside the photos
  const btn = $("ai-toggle");
  if (btn) btn.textContent = hidden ? "show reasoning" : "hide reasoning";
}

// Single entry point for Enter / button: submit while answering, advance while reviewing.
function advance() {
  if (state.phase === "feedback") {
    if (state.isPractice) startRealTask();
    else loadNext();
    return;
  }
  submit();
}

async function submit() {
  if (state.phase !== "answer" || !state.selected || state.loadedAt == null) return;
  cancelAnimationFrame(state.rafId);

  if (state.isPractice) {
    const ok = state.selected === state.practice.answer;
    showFeedback(ok, state.practice.answer,
      ` <span class="muted">${state.practice.explanation || ""}</span>`);
    return;
  }

  const payload = {
    email: state.email,
    problem_id: state.problem.id,
    selected: state.selected,
    client_elapsed_ms: Math.round(rawElapsedMs()),
    hidden_ms: Math.round(currentHiddenMs()),
    served_at: state.servedAt,
    revision: REVISION,
  };
  $("next-btn").disabled = true;
  let res;
  try {
    res = await api("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    $("next-btn").disabled = false;
    alert("Could not save answer: " + e.message);
    return;
  }
  updateProgress(res);
  updateScore(res.score);
  showFeedback(res.was_correct, res.correct_answer, "", res.ai_reveal);
}

function finish(p) {
  $("app").classList.add("hidden");
  $("done-screen").classList.remove("hidden");
  const sc = p.score || {};
  const you = sc.answered ? ` You answered ${sc.you_correct}/${sc.answered} correctly` +
    (sc.ai_unlocked ? ` (AI: ${sc.ai_correct}/${sc.answered}).` : ".") : "";
  $("done-summary").textContent =
    `You completed ${p.done} problems in ${fmtClock(p.total_ms || 0)}.${you} You can close this tab.`;
}

// ---- Practice --------------------------------------------------------------
async function showPractice() {
  state.isPractice = true;
  $("score-box").classList.add("hidden");          // no score during practice
  $("mode-badge").classList.remove("hidden");
  if (!state.practice) state.practice = await api("/practice.json");
  resetForNewProblem();
  state.problem = state.practice;
  renderOptions(state.practice.options);
  startTimerWhenLoaded();
}

function startRealTask() {
  state.isPractice = false;
  localStorage.setItem("m2sv_practiced_" + state.email, "1");
  $("mode-badge").classList.add("hidden");
  $("score-box").classList.remove("hidden");
  loadNext();
}

// ---- Keyboard --------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  if ($("app").classList.contains("hidden")) return;
  if (e.key === "Enter") { e.preventDefault(); advance(); return; }
  if ((e.key === "r" || e.key === "R") && state.phase === "feedback") { toggleAiPanel(); return; }
  if (state.phase !== "answer") return;
  const n = parseInt(e.key, 10);
  if (!Number.isNaN(n) && n >= 1 && state.problem && n <= state.problem.options.length) {
    selectOption(LETTERS[n - 1]);
  }
});
$("next-btn").addEventListener("click", advance);

// ---- Start -----------------------------------------------------------------
$("email-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!$("agree").checked) {
    $("welcome-err").textContent = "Please confirm you are 18+ and agree to participate.";
    return;
  }
  const email = $("email").value.trim().toLowerCase();
  try {
    const data = await api("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
    state.email = data.email;
    localStorage.setItem("m2sv_email", data.email);
    $("welcome").classList.add("hidden");
    $("app").classList.remove("hidden");
    // First-timers (no prior answers, haven't practiced) get the practice round.
    const practiced = localStorage.getItem("m2sv_practiced_" + data.email);
    if ((data.done || 0) === 0 && !practiced) showPractice();
    else loadNext();
  } catch (err) {
    $("welcome-err").textContent = err.message;
  }
});

// Gate the Start button on the consent checkbox.
$("agree").addEventListener("change", () => {
  $("start-btn").disabled = !$("agree").checked;
});

// Pre-fill last email for convenience.
const saved = localStorage.getItem("m2sv_email");
if (saved) $("email").value = saved;
