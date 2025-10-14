import os
import io
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import argparse
import re
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("evaluate_vlm")


def read_image_bytes(path: str) -> bytes:
    p = Path(path)
    return p.read_bytes()


def encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def to_image_bytes_and_mime(value, repo_root: Path) -> tuple[bytes, str]:
    """Accepts a path string, Image feature dict, or PIL Image; returns bytes, mime."""
    import mimetypes
    try:
        from PIL.Image import Image as PILImage
    except Exception:
        PILImage = None  # type: ignore

    # dict with path from datasets Image(decode=False)
    if isinstance(value, dict) and "path" in value:
        path = value["path"]
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return read_image_bytes(path), mime
    # string relative path
    if isinstance(value, str):
        path = str((repo_root / value).resolve())
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return read_image_bytes(path), mime
    # PIL image object
    if PILImage is not None and isinstance(value, PILImage):
        buf = io.BytesIO()
        # Prefer PNG to avoid recompression artifacts
        value.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    raise TypeError(f"Unsupported image value type: {type(value)}")


def build_prompt(question: str, options: List[str]) -> str:
    opts = "\n".join(f"{chr(ord('A') + i)}" for i, _ in enumerate(options))
    instructions = (
        "You will be given two images: a labeled overhead map and a street-view photo.\n"
        "Choose which labeled direction on the map corresponds to the direction in which the street view photo was taken.\n"
        "Answer with a single letter only (A, B, C, ...)."
    )
    return f"{instructions}\n\n{question}\n\nOptions:\n{opts}"


# -------- Providers --------

class ProviderError(Exception):
    pass


class OpenAIProvider:
    def __init__(self, model: str, api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
    def infer(self, prompt: str, images: List[tuple[bytes, str]]) -> str:
        # Use Chat Completions multimodal schema with image_url data URIs
        content: List[Dict] = [{"type": "text", "text": prompt}]
        for data, mime in images:
            b64 = encode_b64(data)
            data_uri = f"data:{mime};base64,{b64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_uri, "detail": "auto"},
            })
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()


class GeminiProvider:
    def __init__(self, model: str, api_key: str | None = None):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        self.genai = genai
        self.model = genai.GenerativeModel(model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
    def infer(self, prompt: str, images: List[tuple[bytes, str]]) -> str:
        # Pass raw parts as {mime_type, data} which works across genai versions
        parts = [prompt]
        for data, mime in images:
            parts.append({"mime_type": mime, "data": data})
        resp = self.model.generate_content(parts)
        return resp.text.strip()


class ClaudeProvider:
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
    def infer(self, prompt: str, images: List[tuple[bytes, str]]) -> str:
        # Anthropic Messages API expects content blocks of type "image" with base64 source
        content_blocks: List[Dict] = []
        for data, mime in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": encode_b64(data),
                },
            })
        # Add the instruction text as a content block
        content_blocks.append({"type": "text", "text": prompt})
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": content_blocks}],
        )
        return msg.content[0].text.strip()


def normalize_letter(text: str, num_options: int) -> str:
    """Return a single option letter if confidently present.

    Priority:
    1) Exact single-letter response (ignoring surrounding whitespace).
    2) Letter inside \boxed{X} (case-insensitive).
    3) Explicit conclusion phrases like "answer is X" or "final answer: X" (also supports "is:").
    4) Last non-empty line is effectively just a styled single letter (e.g., **B**, (C), `A`, "C.").
    5) As a weaker fallback, accept phrases like "choose X", "option X", "arrow X" unless preceded by elimination/negation context.
    Otherwise returns empty string to avoid false positives from prose.
    """
    if text is None:
        return ""
    t = text.strip()
    if not t:
        return ""

    def is_valid_letter(ch: str) -> str:
        if not ch:
            return ""
        ch_u = ch.upper()
        idx = ord(ch_u) - ord("A")
        return ch_u if 0 <= idx < num_options else ""

    # 1) Exact single letter
    m = re.fullmatch(r"\s*([A-Za-z])\s*", t)
    if m:
        ch = is_valid_letter(m.group(1))
        if ch:
            return ch

    # 2) \boxed{X}
    m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", t, flags=re.IGNORECASE)
    if m:
        ch = is_valid_letter(m.group(1))
        if ch:
            return ch

    # 2b) Repeated-letter outputs like "C. C" or "B B" as the entire response
    m = re.fullmatch(r"\s*([A-Za-z])\s*[\.-:;,]?\s*\1\s*\.?\s*", t)
    if m:
        ch = is_valid_letter(m.group(1))
        if ch:
            return ch

    # 3) Prefer explicit conclusion phrases anywhere in text (prefer the last such mention)
    explicit_answer_patterns = [
        r"(?:\bthe\s+answer\b|\banswer\b)\s*(?:is\s*[:=]?|[:=])\s*([A-Za-z])\b",
        r"\bfinal\s*(?:answer)?\s*(?:is\s*[:=]?|[:=])\s*([A-Za-z])\b",
    ]
    explicit_candidates: list[str] = []
    for pat in explicit_answer_patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            explicit_candidates.append(m.group(1))
    for raw in reversed(explicit_candidates):
        ch = is_valid_letter(raw)
        if ch:
            return ch

    # 4) Last non-empty line: accept if it's effectively just a single styled letter
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        # If the last line itself contains an explicit phrase, re-use explicit logic on it for precision
        for pat in explicit_answer_patterns:
            m2 = re.search(pat, last, flags=re.IGNORECASE)
            if m2:
                ch = is_valid_letter(m2.group(1))
                if ch:
                    return ch
        # Repeated-letter on last line like "C. C"
        mrep = re.fullmatch(r"\s*([A-Za-z])\s*[\.-:;,]?\s*\1\s*\.?\s*", last)
        if mrep:
            ch = is_valid_letter(mrep.group(1))
            if ch:
                return ch
        # Strip typical wrappers and styling around a lone letter
        stripped = re.sub(r"[\s\*`_~\-–—\(\)\[\]\{\}\"'.:;,!]+", "", last)
        # If what's left is a single letter, accept it
        if re.fullmatch(r"[A-Za-z]", stripped):
            ch = is_valid_letter(stripped)
            if ch:
                return ch

    # 5) Weaker fallback: ambiguous phrases choose/option/arrow X, but avoid elimination contexts
    ambiguous_patterns = [
        r"\bchoose\s*([A-Za-z])\b",
        r"\b(?:option|choice|arrow)\s*([A-Za-z])\b",
    ]
    last_ch = ""
    for pat in ambiguous_patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            start = m.start()
            context = t[max(0, start-50):start].lower()
            if any(neg in context for neg in ["eliminate", "eliminates", "eliminated", "eliminating", "not ", "isn't", "is not", "avoid", "eliminates option", "eliminate option"]):
                continue
            ch = is_valid_letter(m.group(1))
            if ch:
                last_ch = ch
    if last_ch:
        return last_ch

    return ""


def evaluate_split(dataset_path: str, provider: str, model: str, limit: int | None = None, resume_from: str | None = None) -> Dict:
    ds = load_dataset(dataset_path, split="train") if os.path.isdir(dataset_path) else load_dataset(dataset_path)["train"]

    # The dataset we produce has relative paths inside repo; resolve to disk paths
    repo_root = Path(dataset_path) if os.path.isdir(dataset_path) else Path(".")

    # Build provider
    if provider == "openai":
        runner = OpenAIProvider(model)
    elif provider == "gemini":
        runner = GeminiProvider(model)
    elif provider == "claude":
        runner = ClaudeProvider(model)
    else:
        raise ValueError("provider must be one of: openai, gemini, claude")

    # Resume state
    processed_ids = set()
    results = []
    correct = 0
    total = 0
    empty_preds = 0
    if resume_from and Path(resume_from).exists():
        try:
            prev = json.loads(Path(resume_from).read_text())
            prev_results = prev.get("results", prev if isinstance(prev, list) else [])
            for r in prev_results:
                rid = str(r.get("id", ""))
                if rid:
                    processed_ids.add(rid)
            results.extend(prev_results)
            correct = sum(1 for r in prev_results if r.get("correct") is True)
            total = len(prev_results)
            logger.info(f"Resuming from {resume_from}: already have {total} results, {correct} correct")
        except Exception as e:
            logger.warning(f"Failed to load resume file {resume_from}: {e}")

    added = 0
    # Determine how many new items to add this run
    if limit is not None:
        # When resuming, interpret --limit as TOTAL desired count
        # i.e., add (limit - already_processed) more, but not below 0
        additional_quota = max(0, limit - total)
    else:
        # No limit specified -> process all remaining
        additional_quota = None
    for i, row in enumerate(ds):
        rid = str(row.get("id", i))
        if processed_ids and rid in processed_ids:
            continue
        if additional_quota is not None and added >= additional_quota:
            break
        # Resolve images to bytes + mime (handles PIL Image, dict with path, or str path)
        image_map_val = row.get("image_map") or (row.get("images") or [None, None])[0]
        image_sv_val = row.get("image_sv") or (row.get("images") or [None, None])[1]
        map_bytes, map_mime = to_image_bytes_and_mime(image_map_val, repo_root)
        sv_bytes, sv_mime = to_image_bytes_and_mime(image_sv_val, repo_root)

        prompt = build_prompt(row["question"], row["options"])
        try:
            pred_raw = runner.infer(prompt, [(map_bytes, map_mime), (sv_bytes, sv_mime)])
        except Exception as e:
            logger.warning(f"Inference failed for idx={i}: {e}")
            pred_raw = ""
        pred = normalize_letter(pred_raw, len(row["options"]))
        gold = str(row["answer"]).strip().upper()
        is_correct = pred == gold
        cflag = bool(is_correct)
        correct += int(cflag)
        total += 1
        if pred == "":
            empty_preds += 1
        results.append({
            "id": rid,
            "pred": pred,
            "gold": gold,
            "raw": pred_raw,
            "correct": cflag,
        })
        added += 1

    acc = correct / total if total else 0.0
    logger.info(f"Accuracy: {acc:.3f} ({correct}/{total}) | empty preds: {empty_preds}")
    return {"accuracy": acc, "correct": correct, "total": total, "empty": empty_preds, "results": results}


def reparse_results(dataset_path: str, results_path: str, limit: int | None = None) -> Dict:
    """Recompute predictions from existing results JSON by re-normalizing raw strings.

    This avoids re-querying any model. We look up option counts from the dataset
    (by id when available; otherwise by row index) and re-derive preds using
    normalize_letter.
    """
    ds = load_dataset(dataset_path, split="train") if os.path.isdir(dataset_path) else load_dataset(dataset_path)["train"]

    # Build id -> option count and index -> option count
    id_to_num_opts: Dict[str, int] = {}
    idx_to_num_opts: Dict[int, int] = {}
    for i, row in enumerate(ds):
        n = len(row.get("options", []))
        idx_to_num_opts[i] = n
        rid = str(row.get("id", i))
        id_to_num_opts[rid] = n

    data = json.loads(Path(results_path).read_text())
    results = data.get("results", data if isinstance(data, list) else [])

    new_results = []
    correct = 0
    total = 0
    empty_preds = 0
    for i, r in enumerate(results):
        if limit is not None and i >= limit:
            break
        rid = str(r.get("id", i))
        num_opts = id_to_num_opts.get(rid, idx_to_num_opts.get(i, 26))
        raw = r.get("raw", "")
        gold = str(r.get("gold", "")).strip().upper()
        pred = normalize_letter(raw, num_opts)
        is_correct = pred == gold
        correct += int(is_correct)
        total += 1
        if pred == "":
            empty_preds += 1
        nr = dict(r)
        nr["pred"] = pred
        nr["correct"] = is_correct
        new_results.append(nr)

    acc = correct / total if total else 0.0
    logger.info(f"Reparsed accuracy: {acc:.3f} ({correct}/{total}) | empty preds: {empty_preds}")
    return {"accuracy": acc, "correct": correct, "total": total, "empty": empty_preds, "results": new_results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on a two-image multiple-choice dataset")
    parser.add_argument("dataset", help="Path or HF repo id to dataset")
    parser.add_argument("--provider", choices=["openai", "gemini", "claude"], required=False)
    parser.add_argument("--model", required=False, help="Model id for the chosen provider")
    parser.add_argument("--limit", type=int, default=None, help="Eval only first N rows")
    parser.add_argument("--out", default=None, help="Optional JSON output path for results")
    parser.add_argument("--reparse-result", action="store_true", help="Re-parse answers from --out JSON without querying any model")
    parser.add_argument("--resume", action="store_true", help="Resume from the JSON at --out; append new rows after the last processed id")
    parser.add_argument("--self-test", action="store_true", help="Run built-in normalization tests and exit")
    args = parser.parse_args()

    if args.self_test:
        # Simple self-test suite for normalize_letter
        cases = [
            ("The answer is C.", 4, "C"),
            ("Final answer: B", 3, "B"),
            ("answer is: A", 3, "A"),
            ("**B**", 4, "B"),
            ("(C)", 4, "C"),
            ("C.", 4, "C"),
            ("C. C", 4, "C"),
            ("B. B", 3, "B"),
            ("Choose A", 3, "A"),
            ("We eliminate option C. The answer is B", 3, "B"),
            ("Point C is the corner house... Therefore, the direction is A.\n\nA", 3, "A"),
            ("The green line (B) ... The answer is: B", 3, "B"),
            ("\\boxed{ C }", 4, "C"),
            ("Eliminate B; choose C.", 3, "C"),
            ("Eliminate C; choose B.", 3, "B"),
            ("We should not pick C. Final answer: A", 3, "A"),
        ]
        failures = []
        for text, n, expected in cases:
            got = normalize_letter(text, n)
            if got != expected:
                failures.append((text, expected, got))
        if failures:
            logger.error(f"Self-test failed {len(failures)} case(s):")
            for text, exp, got in failures:
                logger.error(f"Expected {exp}, got {got} for text: {text!r}")
            raise SystemExit(1)
        logger.info("Self-test passed: normalize_letter")
        raise SystemExit(0)

    if args.reparse_result:
        if not args.out:
            parser.error("--reparse-result requires --out to point to an existing results JSON")
        if not Path(args.out).exists():
            parser.error(f"Results file not found: {args.out}")
        metrics = reparse_results(args.dataset, args.out, args.limit)
    else:
        if not args.provider or not args.model:
            parser.error("--provider and --model are required unless --reparse-result is given")
        resume_path = None
        if args.resume:
            if not args.out:
                parser.error("--resume requires --out to read prior results from and write updates to")
            resume_path = args.out if Path(args.out).exists() else None
            if resume_path is None:
                logger.info("No existing results at --out; starting fresh run")
        metrics = evaluate_split(args.dataset, args.provider, args.model, args.limit, resume_from=resume_path)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
