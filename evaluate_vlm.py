import os
import io
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict
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


def evaluate_split(dataset_path: str, provider: str, model: str, limit: int | None = None, resume_from: str | None = None, out_path: str | None = None, workers: int = 1, pbar_disable: bool = False) -> Dict:
    ds = load_dataset(dataset_path, split="train") if os.path.isdir(dataset_path) else load_dataset(dataset_path)["train"]

    # The dataset we produce has relative paths inside repo; resolve to disk paths
    repo_root = Path(dataset_path) if os.path.isdir(dataset_path) else Path(".")

    # Validate provider early
    if provider not in {"openai", "gemini", "claude"}:
        raise ValueError("provider must be one of: openai, gemini, claude")

    # Thread-local provider (many SDK clients are not thread-safe)
    _thread_local = threading.local()

    def get_runner():
        if not hasattr(_thread_local, "runner"):
            if provider == "openai":
                _thread_local.runner = OpenAIProvider(model)
            elif provider == "gemini":
                _thread_local.runner = GeminiProvider(model)
            else:
                _thread_local.runner = ClaudeProvider(model)
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
    parser.add_argument("--provider", choices=["openai", "gemini", "claude"], required=False)
    parser.add_argument("--model", required=False, help="Model id for the chosen provider")
    parser.add_argument("--limit", type=int, default=None, help="Eval only first N rows")
    parser.add_argument("--out", default=None, help="Optional JSON output path for results")
    parser.add_argument("--reparse-result", action="store_true", help="Re-parse answers from --out JSON without querying any model")
    parser.add_argument("--resume", action="store_true", help="Resume from the JSON at --out; append new rows after the last processed id")
    parser.add_argument("--self-test", action="store_true", help="Run built-in normalization tests and exit")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads per evaluation")
    # Multi-config support: --eval can be passed multiple times: provider,model,out
    parser.add_argument("--eval", dest="evals", action="append", default=None, help="Provider,model,out triple, e.g., 'openai,gpt-4o,results/gpt-4o.json'. Can be provided multiple times.")
    parser.add_argument("--eval-file", default=None, help="Path to a text file with one 'provider,model,out' per line")
    parser.add_argument("--parallel-configs", type=int, default=1, help="Run up to N provider/model configs concurrently")
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
            metrics_local = evaluate_split(args.dataset, p, m, args.limit, resume_from=resume_path, out_path=o, workers=args.workers, pbar_disable=(args.parallel_configs > 1))
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
