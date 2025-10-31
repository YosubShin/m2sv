import os
import io
import json
import base64
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import re
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    instructions = (
        "You will be given two images: (1) a north-up overhead map with arrows labeled A, B, C, ... and (2) a street-view photo.\n"
        "Rules:\n"
        "- The camera location is the same for all options: the center of the intersection.\n"
        "- Each letter corresponds to facing outward from that center along the arrow of that label.\n"
        "- The small circles near labels are markers only; they are not camera locations.\n"
        "- The map and photo may be captured years apart. Ignore transient objects (cars, people).\n"
        "Think step by step to compare the street-view with the map (buildings, angles, lanes, landmarks).\n"
        "On the final line, output only: Final answer: \\boxed{X} where X is a single letter (A, B, C, ...)."
    )
    return f"{instructions}\n\n{question}"


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


class QwenProvider:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        # DashScope-compatible OpenAI client
        from openai import OpenAI
        api_key_eff = api_key or os.getenv("DASHSCOPE_API_KEY")
        base_url_eff = base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.client = OpenAI(api_key=api_key_eff, base_url=base_url_eff)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
    def infer(self, prompt: str, images: List[tuple[bytes, str]]) -> str:
        # Use the same Chat Completions multimodal format
        content: List[Dict] = []
        for data, mime in images:
            b64 = encode_b64(data)
            data_uri = f"data:{mime};base64,{b64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_uri},
            })
        content.append({"type": "text", "text": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            stream=False,
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

    # 2) LaTeX boxed letter (allow optional math delimiters like $...$, \(...\), \[...\])
    # Prefer the LAST valid match to avoid earlier instructional placeholders like \boxed{X}
    boxed_pattern = r"(?:\\\(|\\\[|\$)?\s*(?:\\boxed|\\fbox)\s*\{\s*([A-Za-z])\s*\}\s*(?:\\\)|\\\]|\$)?"
    boxed_candidates: list[str] = []
    for m in re.finditer(boxed_pattern, t, flags=re.IGNORECASE):
        boxed_candidates.append(m.group(1))
    for raw in reversed(boxed_candidates):
        ch = is_valid_letter(raw)
        if ch:
            return ch

    # 2b) Repeated-letter outputs like "C. C" or "B B" as the entire response
    m = re.fullmatch(r"\s*([A-Za-z])\s*[\.-:;,]?\s*\1\s*\.?\s*", t)
    if m:
        ch = is_valid_letter(m.group(1))
        if ch:
            return ch

    # 3) Prefer explicit conclusion phrases anywhere in text (prefer the last such mention)
    # Accept variations with markdown styling around the letter and various punctuation
    styled_letter = r"[\s`*_~\(\[\{]*([A-Za-z])[\s`*_~\)\]\}]*"
    explicit_answer_patterns = [
        rf"(?:\bthe\s+answer\b|\banswer\b)\s*(?:is\s*[:=]?|[:=])\s*{styled_letter}\b",
        rf"\bfinal\s*(?:answer)?\s*(?:is\s*[:=]?|[:=])\s*{styled_letter}\b",
        rf"\bfinal\s*answer\s*[:=]\s*{styled_letter}\b",
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


def evaluate_split(dataset_path: str, provider: str, model: str, limit: int | None = None, resume_from: str | None = None, out_path: str | None = None, workers: int = 1, pbar_disable: bool = False, use_openai_batch: bool = False, openai_batch_options: Dict[str, Any] | None = None) -> Dict:
    ds = load_dataset(dataset_path, split="train") if os.path.isdir(dataset_path) else load_dataset(dataset_path)["train"]

    # The dataset we produce has relative paths inside repo; resolve to disk paths
    repo_root = Path(dataset_path) if os.path.isdir(dataset_path) else Path(".")

    # Validate provider early
    if provider not in {"openai", "gemini", "claude", "qwen"}:
        raise ValueError("provider must be one of: openai, gemini, claude, qwen")
    if provider == "openai" and use_openai_batch:
        return evaluate_split_openai_batch(
            ds=ds,
            repo_root=repo_root,
            model=model,
            limit=limit,
            resume_from=resume_from,
            out_path=out_path,
            pbar_disable=pbar_disable,
            batch_options=openai_batch_options or {},
        )

    # Thread-local provider (many SDK clients are not thread-safe)
    _thread_local = threading.local()

    def get_runner():
        if not hasattr(_thread_local, "runner"):
            if provider == "openai":
                _thread_local.runner = OpenAIProvider(model)
            elif provider == "gemini":
                _thread_local.runner = GeminiProvider(model)
            elif provider == "claude":
                _thread_local.runner = ClaudeProvider(model)
            else:
                _thread_local.runner = QwenProvider(model)
        return getattr(_thread_local, "runner")

    # Precompute option counts for id and index to support accurate baselines when resuming
    id_to_num_opts: Dict[str, int] = {}
    idx_to_num_opts: Dict[int, int] = {}
    id_to_index: Dict[str, int] = {}
    for i, row in enumerate(ds):
        n = len(row.get("options", []))
        idx_to_num_opts[i] = n
        rid = str(row.get("id", i))
        id_to_num_opts[rid] = n
        id_to_index[rid] = i

    # Resume state
    processed_ids = set()
    results = []
    correct = 0
    total = 0
    empty_preds = 0
    random_expectation_sum = 0.0
    option_count_hist: Dict[int, int] = {}
    pending_batch_state: Dict[str, Any] | None = None
    pending_rows_state: List[Dict[str, Any]] | None = None
    pending_batch_state: Dict[str, Any] | None = None
    pending_rows_state: List[Dict[str, Any]] | None = None
    pending_batch_state: Dict[str, Any] | None = None
    pending_rows_state: List[Dict[str, Any]] | None = None
    pending_batch_state: Dict[str, Any] | None = None
    pending_rows_state: List[Dict[str, Any]] | None = None
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
            # Initialize baseline and histogram from previous results
            for i, r in enumerate(prev_results):
                rid = str(r.get("id", i))
                n = id_to_num_opts.get(rid, idx_to_num_opts.get(i, 26))
                option_count_hist[n] = option_count_hist.get(n, 0) + 1
                random_expectation_sum += (1.0 / n) if n and n > 0 else 0.0
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

    # Build list of indices to process this run
    indices_to_process: List[int] = []
    for i, row in enumerate(ds):
        rid = str(row.get("id", i))
        if processed_ids and rid in processed_ids:
            continue
        if additional_quota is not None and len(indices_to_process) >= additional_quota:
            break
        indices_to_process.append(i)

    pbar = tqdm(total=len(indices_to_process), desc=f"Evaluating ({provider}:{model})", unit="ex", disable=pbar_disable)

    def _process_one(idx: int) -> tuple[int, str, str, int, str]:
        row = ds[idx]
        rid_local = str(row.get("id", idx))
        # Resolve images to bytes + mime (handles PIL Image, dict with path, or str path)
        image_map_val = row.get("image_map") or (row.get("images") or [None, None])[0]
        image_sv_val = row.get("image_sv") or (row.get("images") or [None, None])[1]
        map_bytes, map_mime = to_image_bytes_and_mime(image_map_val, repo_root)
        sv_bytes, sv_mime = to_image_bytes_and_mime(image_sv_val, repo_root)
        prompt = build_prompt(row["question"], row["options"])
        try:
            pred_raw_local = get_runner().infer(prompt, [(map_bytes, map_mime), (sv_bytes, sv_mime)])
        except Exception as e:
            logger.warning(f"Inference failed for idx={idx}: {e}")
            pred_raw_local = ""
        num_opts_local = len(row["options"])
        gold_local = str(row["answer"]).strip().upper()
        return idx, rid_local, pred_raw_local, num_opts_local, gold_local

    # Parallel inference, serialize aggregation and snapshotting in main thread
    if workers is None or workers < 1:
        workers = 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_process_one, idx): idx for idx in indices_to_process}
        for fut in as_completed(future_to_idx):
            idx, rid, pred_raw, num_opts, gold = fut.result()
            option_count_hist[num_opts] = option_count_hist.get(num_opts, 0) + 1
            random_expectation_sum += (1.0 / num_opts) if num_opts > 0 else 0.0
            pred = normalize_letter(pred_raw, num_opts)
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
            pbar.update(1)

            # Incremental checkpointing
            if out_path:
                acc = correct / total if total else 0.0
                rand_baseline = (random_expectation_sum / total) if total else 0.0
                snapshot = {
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                    "empty": empty_preds,
                    "random_baseline": rand_baseline,
                    "option_count_hist": option_count_hist,
                    "results": results,
                }
                outp = Path(out_path)
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_text(json.dumps(snapshot, indent=2))

    # Deterministic ordering of results by dataset row index (fallback by id)
    def _result_sort_key(r: Dict) -> tuple:
        rid = str(r.get("id", ""))
        idx = id_to_index.get(rid)
        if idx is not None:
            return (0, idx)
        try:
            return (1, int(rid))
        except Exception:
            return (2, rid)

    try:
        results.sort(key=_result_sort_key)
    except Exception:
        # If sorting fails for any reason, keep existing order
        pass

    acc = correct / total if total else 0.0
    rand_baseline = (random_expectation_sum / total) if total else 0.0
    logger.info(
        f"[{provider}:{model}] dataset={dataset_path} | Accuracy: {acc:.3f} ({correct}/{total}) | empty preds: {empty_preds} | random baseline: {rand_baseline:.3f}"
    )
    try:
        pbar.close()
    except Exception:
        pass
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "empty": empty_preds,
        "random_baseline": rand_baseline,
        "option_count_hist": option_count_hist,
        "results": results,
    }


def evaluate_split_openai_batch(
    ds,
    repo_root: Path,
    model: str,
    limit: int | None,
    resume_from: str | None,
    out_path: str | None,
    pbar_disable: bool,
    batch_options: Dict[str, Any],
) -> Dict:
    """Evaluate dataset rows using the OpenAI Batch API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for --openai-batch. Install with `pip install openai`."
        ) from exc

    poll_interval = float(batch_options.get("poll_interval", 10.0))
    timeout_val = batch_options.get("timeout", 3600.0)
    timeout = float(timeout_val) if timeout_val is not None else None
    completion_window = batch_options.get("completion_window", "24h")
    metadata = batch_options.get("metadata")
    delete_remote_files = batch_options.get("delete_remote_files", True)
    keep_files = bool(batch_options.get("keep_files", False))
    requests_path_cfg = batch_options.get("requests_path")
    results_path_cfg = batch_options.get("results_path")
    api_key = batch_options.get("api_key") or os.getenv("OPENAI_API_KEY")
    base_url = batch_options.get("base_url") or os.getenv("OPENAI_BASE_URL") or batch_options.get("api_base")
    organization = batch_options.get("organization") or os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")

    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    if organization:
        client_kwargs["organization"] = organization
    client = OpenAI(**client_kwargs)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("batch_options['metadata'] must be a dict when provided")

    id_to_num_opts: Dict[str, int] = {}
    idx_to_num_opts: Dict[int, int] = {}
    id_to_index: Dict[str, int] = {}
    for i, row in enumerate(ds):
        n = len(row.get("options", []))
        idx_to_num_opts[i] = n
        rid = str(row.get("id", i))
        id_to_num_opts[rid] = n
        id_to_index[rid] = i

    processed_ids = set()
    results = []
    correct = 0
    total = 0
    empty_preds = 0
    random_expectation_sum = 0.0
    option_count_hist: Dict[int, int] = {}
    pending_batch_state: Dict[str, Any] | None = None
    pending_rows_state: List[Dict[str, Any]] | None = None
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
            for i, r in enumerate(prev_results):
                rid = str(r.get("id", i))
                n = id_to_num_opts.get(rid, idx_to_num_opts.get(i, 26))
                option_count_hist[n] = option_count_hist.get(n, 0) + 1
                random_expectation_sum += (1.0 / n) if n and n > 0 else 0.0
            logger.info(f"(batch) Resuming from {resume_from}: already have {total} results, {correct} correct")
            pending_batch_state = prev.get("pending_batch")
            pending_rows_state_raw = prev.get("pending_rows")
            if isinstance(pending_rows_state_raw, list):
                pending_rows_state = []
                for entry in pending_rows_state_raw:
                    if isinstance(entry, dict) and "idx" in entry and "custom_id" in entry:
                        pending_rows_state.append(entry)
        except Exception as e:
            logger.warning(f"(batch) Failed to load resume file {resume_from}: {e}")

    added = 0
    if limit is not None:
        additional_quota = max(0, limit - total)
    else:
        additional_quota = None

    indices_to_process: List[int] = []
    for i, row in enumerate(ds):
        rid = str(row.get("id", i))
        if processed_ids and rid in processed_ids:
            continue
        if additional_quota is not None and len(indices_to_process) >= additional_quota:
            break
        indices_to_process.append(i)

    final_statuses = {"completed", "failed", "cancelled", "expired"}
    if pending_batch_state and pending_rows_state:
        stored_indices: List[int] = []
        for entry in pending_rows_state:
            raw_idx = entry.get("idx")
            try:
                idx_val = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if idx_val >= 0:
                stored_indices.append(idx_val)
        if stored_indices:
            if indices_to_process and indices_to_process != stored_indices:
                logger.warning(f"[batch] Pending batch indices differ from current selection; using stored indices from pending batch ({len(stored_indices)} rows)")
            indices_to_process = stored_indices
    elif pending_batch_state and pending_batch_state.get("status") not in final_statuses:
        logger.info("[batch] Pending batch present but without stored row metadata; using current selection for resume")

    if not indices_to_process:
        acc = correct / total if total else 0.0
        rand_baseline = (random_expectation_sum / total) if total else 0.0
        logger.info(
            f"[batch] No new rows to process | Accuracy: {acc:.3f} ({correct}/{total}) | empty preds: {empty_preds} | random baseline: {rand_baseline:.3f}"
        )
        return {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "empty": empty_preds,
            "random_baseline": rand_baseline,
            "option_count_hist": option_count_hist,
            "results": results,
        }

    requests_path: Path | None = None
    batch_requests: List[Dict[str, Any]] = []
    pending_rows: List[Dict[str, Any]] = []
    batch: Any = None  # type: ignore
    batch_job_id: str | None = None
    input_file_id: str | None = None
    reusing_existing_batch = False
    build_requests = True

    if pending_batch_state and pending_batch_state.get("job_id"):
        batch_job_id = str(pending_batch_state["job_id"])
        requests_path_str = pending_batch_state.get("requests_path")
        if requests_path_str:
            try:
                requests_path = Path(requests_path_str)
            except Exception:
                requests_path = None
        input_file_id = pending_batch_state.get("input_file_id")
        try:
            batch = client.batches.retrieve(batch_job_id)
            logger.info(f"[batch] Resuming existing job {batch_job_id} status={batch.status}")
        except Exception as exc:
            logger.warning(f"[batch] Failed to retrieve existing batch {batch_job_id}: {exc}; creating a new job")
            batch = None
            batch_job_id = None
            pending_batch_state = None
        else:
            if batch.status in final_statuses:
                if batch.status == "completed":
                    reusing_existing_batch = True
                    build_requests = False
                else:
                    logger.warning(f"[batch] Previous batch {batch_job_id} ended with status={batch.status}; creating a new job")
                    batch = None
                    batch_job_id = None
                    pending_batch_state = None
            else:
                reusing_existing_batch = True
                build_requests = False
    elif pending_batch_state:
        logger.warning("[batch] Pending batch metadata missing job_id; creating a new job")
        pending_batch_state = None

    if reusing_existing_batch and pending_rows_state:
        pending_rows = pending_rows_state
    else:
        for idx in indices_to_process:
            row = ds[idx]
            rid_local = str(row.get("id", idx))
            custom_id = f"idx-{idx}"
            num_opts = len(row["options"])
            gold = str(row["answer"]).strip().upper()
            meta_entry = {
                "idx": idx,
                "custom_id": custom_id,
                "rid": rid_local,
                "num_opts": num_opts,
                "gold": gold,
            }
            pending_rows.append(meta_entry)
            if build_requests:
                image_map_val = row.get("image_map") or (row.get("images") or [None, None])[0]
                image_sv_val = row.get("image_sv") or (row.get("images") or [None, None])[1]
                map_bytes, map_mime = to_image_bytes_and_mime(image_map_val, repo_root)
                sv_bytes, sv_mime = to_image_bytes_and_mime(image_sv_val, repo_root)
                prompt = build_prompt(row["question"], row["options"])
                content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                for data, mime in [(map_bytes, map_mime), (sv_bytes, sv_mime)]:
                    b64 = encode_b64(data)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"},
                    })
                batch_requests.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": content,
                            }
                        ],
                    },
                })

    def _write_checkpoint(pending_batch: Dict[str, Any] | None, pending_rows_snapshot: List[Dict[str, Any]] | None) -> None:
        if not out_path:
            return
        snapshot: Dict[str, Any] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "empty": empty_preds,
            "random_baseline": (random_expectation_sum / total) if total else 0.0,
            "option_count_hist": option_count_hist,
            "results": results,
        }
        if pending_batch:
            snapshot["pending_batch"] = pending_batch
            if pending_rows_snapshot is not None:
                snapshot["pending_rows"] = pending_rows_snapshot
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(snapshot, indent=2))

    if build_requests:
        logger.info(f"[batch] Submitting {len(batch_requests)} requests to OpenAI Batch (model={model})")
        if not batch_requests:
            raise RuntimeError("[batch] No requests generated for submission")
        if requests_path_cfg:
            requests_path = Path(requests_path_cfg)
            requests_path.parent.mkdir(parents=True, exist_ok=True)
            request_file_handle = requests_path.open("w", encoding="utf-8")
        else:
            request_file_handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
            requests_path = Path(request_file_handle.name)
        with request_file_handle:
            for entry in batch_requests:
                request_file_handle.write(json.dumps(entry))
                request_file_handle.write("\n")

        with requests_path.open("rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")
        input_file_id = input_file.id

        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata,
        )
        batch_job_id = batch.id
        logger.info(f"[batch] Created job {batch_job_id} status={batch.status}")

        pending_batch_payload = {
            "job_id": batch_job_id,
            "status": batch.status,
            "input_file_id": input_file_id,
            "output_file_id": getattr(batch, "output_file_id", None),
            "requests_path": str(requests_path) if requests_path else None,
            "row_indices": [meta["idx"] for meta in pending_rows],
            "model": model,
        }
        _write_checkpoint(pending_batch_payload, pending_rows)
    elif reusing_existing_batch:
        batch_job_id = batch_job_id or (batch.id if batch is not None else None)
        if pending_batch_state:
            pending_batch_state["status"] = getattr(batch, "status", pending_batch_state.get("status"))
            pending_batch_state["output_file_id"] = getattr(batch, "output_file_id", pending_batch_state.get("output_file_id"))
            _write_checkpoint(pending_batch_state, pending_rows if pending_rows else pending_rows_state)

    if batch is None or batch_job_id is None:
        raise RuntimeError("[batch] Failed to initialize batch job")

    start_time = time.time()
    while batch.status not in final_statuses:
        if timeout is not None and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Batch job {batch_job_id} did not complete within {timeout} seconds (last status={batch.status})")
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch_job_id)
        logger.info(f"[batch] Polled job {batch_job_id} status={batch.status}")
        if pending_batch_state or build_requests:
            interim_payload = {
                "job_id": batch_job_id,
                "status": batch.status,
                "input_file_id": input_file_id,
                "output_file_id": getattr(batch, "output_file_id", None),
                "requests_path": str(requests_path) if requests_path else None,
                "row_indices": [meta["idx"] for meta in pending_rows],
                "model": model,
            }
            _write_checkpoint(interim_payload, pending_rows if pending_rows else pending_rows_state)

    if batch.status != "completed":
        _write_checkpoint(None, None)
        raise RuntimeError(f"Batch job {batch_job_id} ended with status={batch.status}")
    if not getattr(batch, "output_file_id", None):
        _write_checkpoint(None, None)
        raise RuntimeError(f"Batch job {batch_job_id} completed but no output_file_id was returned")

    def _read_all(stream_obj) -> bytes:
        if stream_obj is None:
            return b""
        if hasattr(stream_obj, "read"):
            data = stream_obj.read()
            if isinstance(data, str):
                return data.encode("utf-8")
            return data
        if hasattr(stream_obj, "iter_bytes"):
            return b"".join(stream_obj.iter_bytes())
        if isinstance(stream_obj, bytes):
            return stream_obj
        if isinstance(stream_obj, bytearray):
            return bytes(stream_obj)
        raise TypeError(f"Unsupported stream object type: {type(stream_obj)}")

    output_stream = client.files.content(batch.output_file_id)
    output_bytes = _read_all(output_stream)
    if not output_bytes:
        raise RuntimeError(f"Batch job {batch.id} returned empty output content")

    results_path_final: Path | None = None
    if results_path_cfg:
        results_path_final = Path(results_path_cfg)
        results_path_final.parent.mkdir(parents=True, exist_ok=True)
        results_path_final.write_bytes(output_bytes)
    elif keep_files:
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False)
        tmp.write(output_bytes)
        tmp.flush()
        tmp.close()
        results_path_final = Path(tmp.name)

    if delete_remote_files:
        if input_file_id:
            try:
                client.files.delete(input_file_id)
            except Exception as exc:
                logger.debug(f"[batch] Failed to delete input file {input_file_id}: {exc}")
        output_file_id_current = getattr(batch, "output_file_id", None)
        if not results_path_cfg and not keep_files and output_file_id_current:
            try:
                client.files.delete(output_file_id_current)
            except Exception as exc:
                logger.debug(f"[batch] Failed to delete output file {output_file_id_current}: {exc}")

    text_data = output_bytes.decode("utf-8")
    response_map: Dict[str, Dict[str, Any]] = {}
    for line in text_data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as err:
            logger.warning(f"[batch] Skipping malformed JSONL line: {err} | line={line[:120]}...")
            continue
        cid = payload.get("custom_id")
        if cid:
            response_map[cid] = payload

    if len(response_map) != len(pending_rows):
        missing = {meta["custom_id"] for meta in pending_rows} - set(response_map.keys())
        if missing:
            logger.warning(f"[batch] Missing {len(missing)} responses from batch output: {sorted(list(missing))[:5]}")

    def _extract_text(message: Dict[str, Any]) -> str:
        if not message:
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(str(block["text"]))
                    elif block.get("type") == "output_text" and "text" in block:
                        parts.append(str(block["text"]))
            return "".join(parts).strip()
        return ""

    pbar = tqdm(total=len(pending_rows), desc=f"Evaluating (openai-batch:{model})", unit="ex", disable=pbar_disable)
    for meta in pending_rows:
        payload = response_map.get(meta["custom_id"])
        pred_raw = ""
        if not payload:
            logger.warning(f"[batch] No response payload for custom_id={meta['custom_id']}")
        else:
            error_block = payload.get("error")
            if error_block:
                logger.warning(f"[batch] Error for {meta['custom_id']}: {error_block}")
            else:
                response = payload.get("response") or {}
                status_code = response.get("status_code")
                if status_code != 200:
                    logger.warning(f"[batch] Non-200 status for {meta['custom_id']}: {status_code}")
                body = response.get("body") or {}
                choices = body.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    pred_raw = _extract_text(message)
                else:
                    logger.warning(f"[batch] No choices returned for {meta['custom_id']}")

        num_opts = meta["num_opts"]
        option_count_hist[num_opts] = option_count_hist.get(num_opts, 0) + 1
        random_expectation_sum += (1.0 / num_opts) if num_opts > 0 else 0.0
        pred = normalize_letter(pred_raw, num_opts)
        is_correct = pred == meta["gold"]
        correct += int(is_correct)
        total += 1
        if pred == "":
            empty_preds += 1
        results.append({
            "id": meta["rid"],
            "pred": pred,
            "gold": meta["gold"],
            "raw": pred_raw,
            "correct": bool(is_correct),
        })
        added += 1
        pbar.update(1)
    pbar.close()

    def _result_sort_key(r: Dict) -> tuple:
        rid = str(r.get("id", ""))
        idx = id_to_index.get(rid)
        if idx is not None:
            return (0, idx)
        try:
            return (1, int(rid))
        except Exception:
            return (2, rid)

    try:
        results.sort(key=_result_sort_key)
    except Exception:
        pass

    acc = correct / total if total else 0.0
    rand_baseline = (random_expectation_sum / total) if total else 0.0
    logger.info(
        f"[batch] Accuracy: {acc:.3f} ({correct}/{total}) | empty preds: {empty_preds} | random baseline: {rand_baseline:.3f} | new rows: {added}"
    )

    if not keep_files and not requests_path_cfg and requests_path:
        try:
            requests_path.unlink(missing_ok=True)
        except TypeError:
            if requests_path.exists():
                requests_path.unlink()
    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "empty": empty_preds,
        "random_baseline": rand_baseline,
        "option_count_hist": option_count_hist,
        "results": results,
        "batch_job_id": batch.id,
        "batch_status": batch.status,
        "batch_input_file_id": input_file_id,
        "batch_output_file_id": batch.output_file_id,
    }
    if requests_path_cfg or keep_files:
        metrics["batch_input_path"] = str(requests_path)
    if results_path_final is not None:
        metrics["batch_output_path"] = str(results_path_final)
    if batch.metadata:
        metrics["batch_metadata"] = batch.metadata

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(metrics, indent=2))

    return metrics


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
    id_to_index: Dict[str, int] = {}
    for i, row in enumerate(ds):
        n = len(row.get("options", []))
        idx_to_num_opts[i] = n
        rid = str(row.get("id", i))
        id_to_num_opts[rid] = n
        id_to_index[rid] = i

    data = json.loads(Path(results_path).read_text())
    results = data.get("results", data if isinstance(data, list) else [])

    new_results = []
    correct = 0
    total = 0
    empty_preds = 0
    random_expectation_sum = 0.0
    option_count_hist: Dict[int, int] = {}
    for i, r in enumerate(results):
        if limit is not None and i >= limit:
            break
        prev_pred = r.get("pred")
        prev_raw = r.get("raw")
        if (prev_pred is None or str(prev_pred).strip() == "") and (prev_raw is None or str(prev_raw).strip() == ""):
            continue
        rid = str(r.get("id", i))
        num_opts = id_to_num_opts.get(rid, idx_to_num_opts.get(i, 26))
        option_count_hist[num_opts] = option_count_hist.get(num_opts, 0) + 1
        random_expectation_sum += (1.0 / num_opts) if num_opts > 0 else 0.0
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

    # Deterministic ordering by dataset row index
    def _result_sort_key(r: Dict) -> tuple:
        rid = str(r.get("id", ""))
        idx = id_to_index.get(rid)
        if idx is not None:
            return (0, idx)
        try:
            return (1, int(rid))
        except Exception:
            return (2, rid)

    try:
        new_results.sort(key=_result_sort_key)
    except Exception:
        pass

    acc = correct / total if total else 0.0
    rand_baseline = (random_expectation_sum / total) if total else 0.0
    logger.info(
        f"[reparse] dataset={dataset_path} source={results_path} | Accuracy: {acc:.3f} ({correct}/{total}) | empty preds: {empty_preds} | random baseline: {rand_baseline:.3f}"
    )
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "empty": empty_preds,
        "random_baseline": rand_baseline,
        "option_count_hist": option_count_hist,
        "results": new_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on a two-image multiple-choice dataset")
    parser.add_argument("dataset", help="Path or HF repo id to dataset")
    parser.add_argument("--provider", choices=["openai", "gemini", "claude", "qwen"], required=False)
    parser.add_argument("--model", required=False, help="Model id for the chosen provider")
    parser.add_argument("--limit", type=int, default=None, help="Eval only first N rows")
    parser.add_argument("--out", default=None, help="Optional JSON output path for results")
    parser.add_argument("--reparse-result", action="store_true", help="Re-parse answers from --out JSON without querying any model")
    parser.add_argument("--resume", action="store_true", help="Resume from the JSON at --out; append new rows after the last processed id")
    parser.add_argument("--self-test", action="store_true", help="Run built-in normalization tests and exit")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads per evaluation")
    parser.add_argument("--openai-batch", action="store_true", help="Use OpenAI's Batch API for OpenAI provider evaluations")
    parser.add_argument("--batch-poll-interval", type=float, default=10.0, help="Seconds between status polls when waiting on OpenAI batch jobs")
    parser.add_argument("--batch-timeout", type=float, default=3600.0, help="Maximum seconds to wait for OpenAI batch completion (<=0 disables timeout)")
    parser.add_argument("--batch-completion-window", default="24h", help="Desired completion window passed to the Batch API (e.g., 24h or 48h)")
    parser.add_argument("--batch-metadata", default=None, help="Optional JSON object to attach as metadata to the Batch job")
    parser.add_argument("--batch-requests-path", default=None, help="If set, write the generated batch request JSONL to this path")
    parser.add_argument("--batch-results-path", default=None, help="If set, write the downloaded batch result JSONL to this path")
    parser.add_argument("--batch-keep-files", action="store_true", help="Keep generated batch request/result JSONL files when a temp path would otherwise be used")
    parser.add_argument("--batch-keep-remote", action="store_true", help="Do not delete OpenAI uploaded files after the batch completes")
    # Multi-config support: --eval can be passed multiple times: provider,model,out
    parser.add_argument("--eval", dest="evals", action="append", default=None, help="Provider,model,out triple, e.g., 'openai,gpt-4o,results/gpt-4o.json'. Can be provided multiple times.")
    parser.add_argument("--eval-file", default=None, help="Path to a text file with one 'provider,model,out' per line")
    parser.add_argument("--parallel-configs", type=int, default=1, help="Run up to N provider/model configs concurrently")
    args = parser.parse_args()

    batch_metadata = None
    if args.batch_metadata:
        try:
            batch_metadata = json.loads(args.batch_metadata)
        except json.JSONDecodeError as err:
            parser.error(f"Failed to parse --batch-metadata as JSON: {err}")
        if batch_metadata is not None and not isinstance(batch_metadata, dict):
            parser.error("--batch-metadata must be a JSON object")
    batch_timeout = None
    if args.batch_timeout is not None:
        if args.batch_timeout <= 0:
            batch_timeout = None
        else:
            batch_timeout = args.batch_timeout
    openai_batch_options = {
        "poll_interval": args.batch_poll_interval,
        "timeout": batch_timeout,
        "completion_window": args.batch_completion_window,
        "metadata": batch_metadata,
        "requests_path": args.batch_requests_path,
        "results_path": args.batch_results_path,
        "keep_files": args.batch_keep_files,
        "delete_remote_files": not args.batch_keep_remote,
    }

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
            ("* steps...\n\nFinal answer: \\boxed{B}", 4, "B"),
            ("Some bullets...\nFinal answer: $ \\boxed{ d } $", 5, "D"),
            ("Reasoning...\nFinal answer: \\fbox{ a }", 3, "A"),
            ("The final line must be in the format \"Final answer: \\boxed{X}\".\nFinal answer: \\boxed{B}", 4, "B"),
            ("Eliminate B; choose C.", 3, "C"),
            ("Eliminate C; choose B.", 3, "B"),
            ("We should not pick C. Final answer: A", 3, "A"),
            ("\n\n*   **Final Answer:** The final answer is **A**.", 4, "A"),
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
        # Write updated metrics back to the same results file
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))
    else:
        # Build list of configs to run
        configs: List[tuple[str, str, str, bool]] = []  # (provider, model, out, resume?)
        if args.evals or args.eval_file:
            triples: List[str] = []
            if args.evals:
                triples.extend(args.evals)
            if args.eval_file:
                text = Path(args.eval_file).read_text()
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    triples.append(line)
            for t in triples:
                parts = [p.strip() for p in t.split(",")]
                if len(parts) != 3:
                    parser.error(f"Invalid --eval entry: {t!r}. Expected 'provider,model,out'.")
                p, m, o = parts
                res_flag = args.resume and Path(o).exists()
                configs.append((p, m, o, res_flag))
        else:
            # Single-config mode via legacy flags
            if not args.provider or not args.model:
                parser.error("--provider and --model are required unless --reparse-result or --eval/--eval-file is given")
            if not args.out:
                parser.error("Single-config mode requires --out for output path")
            res_flag = args.resume and Path(args.out).exists()
            configs.append((args.provider, args.model, args.out, res_flag))

        # Execute configs, optionally in parallel

        def run_one_config(p: str, m: str, o: str, do_resume: bool) -> tuple[str, Dict]:
            resume_path = o if do_resume else None
            if resume_path is None and args.resume:
                logger.info(f"No existing results at {o}; starting fresh run for {p}:{m}")
            use_batch = args.openai_batch and p == "openai"
            metrics_local = evaluate_split(
                args.dataset,
                p,
                m,
                args.limit,
                resume_from=resume_path,
                out_path=o,
                workers=args.workers,
                pbar_disable=(args.parallel_configs > 1),
                use_openai_batch=use_batch,
                openai_batch_options=openai_batch_options if use_batch else None,
            )
            # Ensure write final metrics to the specific out
            Path(o).parent.mkdir(parents=True, exist_ok=True)
            Path(o).write_text(json.dumps(metrics_local, indent=2))
            return o, metrics_local

        max_cfg_workers = args.parallel_configs if args.parallel_configs and args.parallel_configs > 1 else 1
        if max_cfg_workers == 1 or len(configs) == 1:
            for (p, m, o, do_resume) in configs:
                run_one_config(p, m, o, do_resume)
        else:
            with ThreadPoolExecutor(max_workers=max_cfg_workers) as pool:
                futures = [pool.submit(run_one_config, p, m, o, do_resume) for (p, m, o, do_resume) in configs]
                for fut in as_completed(futures):
                    fut.result()


if __name__ == "__main__":
    main()
