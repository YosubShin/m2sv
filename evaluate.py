#!/usr/bin/env python3
"""
Evaluation script for Qwen3-VL models on image + text multiple-choice datasets.

Usage:
    python eval/qwen3_vl_eval.py --config eval/configs/qwen3_vl_m2sv.yaml
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
from dataclasses import MISSING, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import math
import io
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from m2sv_eval_utils import format_prompt, normalize_letter

Image.MAX_IMAGE_PIXELS = None

# Disable HF Transfer fallback (not reliable on KOA cluster)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
def _save_outputs(results: List[Dict[str, Any]], summary: Dict[str, Any], primary_dir: str) -> None:
    """Save outputs to primary_dir only."""
    os.makedirs(primary_dir, exist_ok=True)
    results_path = os.path.join(primary_dir, "predictions.csv")
    summary_path = os.path.join(primary_dir, "summary.json")

    pd.DataFrame(results).to_csv(results_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved predictions to: {results_path}")
    print(f"Saved summary to: {summary_path}")


def _cfg_default(field_name: str):
    field_info = EvalConfig.__dataclass_fields__[field_name]  # type: ignore[attr-defined]
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field_info.default_factory()
    raise AttributeError(f"No default value for field '{field_name}'")


@dataclass
class EvalConfig:
    model_name: str
    dataset_name: str = "yosubshin/m2sv"
    dataset_split: str = "train"
    generation_max_new_tokens: int = 128
    generation_temperature: float = 0.1
    generation_repetition_penalty: float = 1.0
    batch_size: int = 1
    limit: Optional[int] = None
    output_dir: str = "./eval/results/qwen3_vl_m2sv"
    save_predictions: bool = True
    # vLLM settings (local engine)
    vllm_model: Optional[str] = None  # override model name/path if needed
    max_pixels: int = 640 * 640
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.95
    vllm_max_model_len: int = 32768
    vllm_max_num_seqs: int = 16
    vllm_dtype: str = "bfloat16"
    vllm_enforce_eager: bool = False
    vllm_enable_chunked_prefill: bool = True
    vllm_enable_prefix_caching: bool = True
    vllm_block_size: int = 16
    vllm_swap_space: int = 2
    vllm_disable_custom_all_reduce: bool = True
    vllm_limit_mm_per_prompt: Dict[str, int] = field(
        default_factory=lambda: {"image": 4, "video": 1}
    )
    vllm_max_num_batched_tokens: Optional[int] = None
    vllm_processor: Optional[str] = None
    vllm_seed: int = 42


def load_config(path: str) -> EvalConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    model_cfg = raw.get("model", {}) or {}
    dataset_cfg = raw.get("dataset", {})
    generation_cfg = raw.get("generation", {}) or {}
    output_cfg = raw.get("output", {}) or {}

    inference_cfg_raw = raw.get("inference", {}) or {}
    if "vllm" in inference_cfg_raw and isinstance(inference_cfg_raw["vllm"], dict):
        vllm_cfg = inference_cfg_raw["vllm"]
        inference_cfg = {k: v for k, v in inference_cfg_raw.items() if k != "vllm"}
    else:
        vllm_cfg = inference_cfg_raw
        inference_cfg = inference_cfg_raw

    model_name = model_cfg.get("model_name") or os.environ.get("MODEL_NAME")
    if not model_name:
        raise ValueError(
            "Model name must be provided via config.model.model_name or MODEL_NAME environment variable."
        )

    cfg = EvalConfig(
        model_name=model_name,
        dataset_name=dataset_cfg.get("name", "yosubshin/m2sv"),
        dataset_split=dataset_cfg.get("split", "train"),
        generation_max_new_tokens=generation_cfg.get("max_new_tokens", 128),
        generation_temperature=generation_cfg.get("temperature", 0.1),
        generation_repetition_penalty=generation_cfg.get("repetition_penalty", 1.0),
        batch_size=int(
            (
                inference_cfg.get("batch_size")
                if inference_cfg.get("batch_size") is not None
                else generation_cfg.get("batch_size", os.environ.get("BATCH_SIZE", 1))
            )
        ),
        limit=generation_cfg.get("limit"),
        output_dir=output_cfg.get("dir", os.environ.get("KOA_RESULTS_DIR", "./eval/results/qwen3_vl_m2sv")),
        save_predictions=output_cfg.get("save_predictions", True),
        vllm_model=vllm_cfg.get("model"),
        max_pixels=int(dataset_cfg.get("max_pixels", 480 * 480) or (480 * 480)),
        vllm_tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1) or 1),
        vllm_gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.95) or 0.95),
        vllm_max_model_len=int(vllm_cfg.get("max_model_len", 32768) or 32768),
        vllm_max_num_seqs=int(vllm_cfg.get("max_num_seqs", 16) or 16),
        vllm_dtype=vllm_cfg.get("dtype", "bfloat16"),
        vllm_enforce_eager=bool(vllm_cfg.get("enforce_eager", False)),
        vllm_enable_chunked_prefill=bool(vllm_cfg.get("enable_chunked_prefill", True)),
        vllm_enable_prefix_caching=bool(vllm_cfg.get("enable_prefix_caching", True)),
        vllm_block_size=int(vllm_cfg.get("block_size", 16) or 16),
        vllm_swap_space=int(vllm_cfg.get("swap_space", 2) or 2),
        vllm_disable_custom_all_reduce=bool(vllm_cfg.get("disable_custom_all_reduce", True)),
        vllm_limit_mm_per_prompt=vllm_cfg.get("limit_mm_per_prompt") or _cfg_default("vllm_limit_mm_per_prompt"),
        vllm_max_num_batched_tokens=(
            int(vllm_cfg["max_num_batched_tokens"]) if vllm_cfg.get("max_num_batched_tokens") is not None else None
        ),
        vllm_processor=vllm_cfg.get("processor"),
        vllm_seed=int(vllm_cfg.get("seed", 42) or 42),
    )
    return cfg


def _build_config_from_cli(args: argparse.Namespace) -> EvalConfig:
    def pick(value, field_name):
        return value if value is not None else _cfg_default(field_name)

    output_dir = args.output_dir or os.environ.get(
        "KOA_RESULTS_DIR", pick(None, "output_dir")
    )
    save_predictions = False if args.no_save_predictions else pick(
        None, "save_predictions"
    )

    cfg = EvalConfig(
        model_name=args.model,
        dataset_name=args.dataset or pick(None, "dataset_name"),
        dataset_split=args.split or pick(None, "dataset_split"),
        generation_max_new_tokens=pick(args.max_new_tokens, "generation_max_new_tokens"),
        generation_temperature=pick(args.temperature, "generation_temperature"),
        generation_repetition_penalty=pick(
            args.repetition_penalty, "generation_repetition_penalty"
        ),
        batch_size=pick(args.batch_size, "batch_size"),
        limit=args.limit if args.limit is not None else pick(None, "limit"),
        output_dir=output_dir,
        save_predictions=save_predictions,
        vllm_model=args.vllm_model or args.model,
        max_pixels=pick(args.max_pixels, "max_pixels"),
        vllm_tensor_parallel_size=pick(
            args.tensor_parallel_size, "vllm_tensor_parallel_size"
        ),
        vllm_gpu_memory_utilization=pick(
            args.gpu_memory_utilization, "vllm_gpu_memory_utilization"
        ),
        vllm_max_model_len=pick(args.max_model_len, "vllm_max_model_len"),
        vllm_max_num_seqs=pick(args.max_num_seqs, "vllm_max_num_seqs"),
        vllm_dtype=pick(None, "vllm_dtype"),
        vllm_enforce_eager=pick(None, "vllm_enforce_eager"),
        vllm_enable_chunked_prefill=pick(None, "vllm_enable_chunked_prefill"),
        vllm_enable_prefix_caching=pick(None, "vllm_enable_prefix_caching"),
        vllm_block_size=pick(None, "vllm_block_size"),
        vllm_swap_space=pick(None, "vllm_swap_space"),
        vllm_disable_custom_all_reduce=pick(None, "vllm_disable_custom_all_reduce"),
        vllm_limit_mm_per_prompt=pick(None, "vllm_limit_mm_per_prompt"),
        vllm_max_num_batched_tokens=pick(
            args.max_num_batched_tokens, "vllm_max_num_batched_tokens"
        ),
        vllm_processor=args.vllm_processor
        if args.vllm_processor is not None
        else pick(None, "vllm_processor"),
        vllm_seed=pick(args.seed, "vllm_seed"),
    )
    return cfg


def _apply_cli_overrides(cfg: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    if args.model:
        cfg.model_name = args.model
    if args.dataset:
        cfg.dataset_name = args.dataset
    if args.split:
        cfg.dataset_split = args.split
    if args.max_new_tokens is not None:
        cfg.generation_max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.generation_temperature = args.temperature
    if args.repetition_penalty is not None:
        cfg.generation_repetition_penalty = args.repetition_penalty
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.limit is not None:
        cfg.limit = args.limit
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_save_predictions:
        cfg.save_predictions = False
    if args.vllm_model:
        cfg.vllm_model = args.vllm_model
    if args.max_pixels is not None:
        cfg.max_pixels = args.max_pixels
    if args.tensor_parallel_size is not None:
        cfg.vllm_tensor_parallel_size = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        cfg.vllm_gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_model_len is not None:
        cfg.vllm_max_model_len = args.max_model_len
    if args.max_num_seqs is not None:
        cfg.vllm_max_num_seqs = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        cfg.vllm_max_num_batched_tokens = args.max_num_batched_tokens
    if args.vllm_processor:
        cfg.vllm_processor = args.vllm_processor
    if args.seed is not None:
        cfg.vllm_seed = args.seed
    return cfg


def _shrink_image(image: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale an image if it exceeds max_pixels."""
    if max_pixels is None or max_pixels <= 0:
        return image
    width, height = image.size
    total_pixels = width * height
    if total_pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / float(total_pixels))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.LANCZOS)


def _load_image_value(value: Any) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        img_copy = value.copy()
        if hasattr(value, "close"):
            value.close()
        return img_copy.convert("RGB")
    return None


def _collect_images(item: Dict[str, Any], max_pixels: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    keys = ["image_map", "image_sv"]
    for key in keys:
        image = _load_image_value(item.get(key))
        if image is not None:
            images.append(_shrink_image(image, max_pixels))
    if not images:
        extra = item.get("images")
        if isinstance(extra, list):
            for value in extra:
                image = _load_image_value(value)
                if image is not None:
                    images.append(_shrink_image(image, max_pixels))
    return images


def _count_options_field(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return len(parsed)
        except Exception:
            cleaned = [tok.strip() for tok in value.strip("[]()") .split(",") if tok.strip()]
            return len(cleaned)
    return 0


def _load_existing_predictions(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    existing: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = dict(row)
            # Normalize the id and correct flag
            rec_id = str(record.get("id", ""))
            if rec_id:
                processed_ids.add(rec_id)
            correct_raw = record.get("correct")
            if isinstance(correct_raw, str):
                record["correct"] = correct_raw.strip().lower() in {"true", "1", "yes"}
            existing.append(record)
    return existing, processed_ids


def _format_prompt_with_images(
    processor: AutoProcessor, prompt: str, image_count: int
) -> str:
    user_content: List[Dict[str, Any]] = []
    for idx in range(image_count):
        user_content.append({"type": "image", "image": f"local_image_{idx}"})
    user_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": user_content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_vllm_engine(cfg: EvalConfig):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit(
            "vLLM is required for this evaluation. Please install vllm>=0.4.0."
        ) from exc

    model_path = cfg.vllm_model or cfg.model_name
    processor_name = cfg.vllm_processor or model_path

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": cfg.vllm_tensor_parallel_size,
        "gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
        "max_model_len": cfg.vllm_max_model_len,
        "max_num_seqs": cfg.vllm_max_num_seqs,
        "trust_remote_code": True,
        "dtype": cfg.vllm_dtype,
        "enforce_eager": cfg.vllm_enforce_eager,
        "enable_chunked_prefill": cfg.vllm_enable_chunked_prefill,
        "enable_prefix_caching": cfg.vllm_enable_prefix_caching,
        "block_size": cfg.vllm_block_size,
        "swap_space": cfg.vllm_swap_space,
        "disable_custom_all_reduce": cfg.vllm_disable_custom_all_reduce,
    }
    if cfg.vllm_limit_mm_per_prompt:
        llm_kwargs["limit_mm_per_prompt"] = cfg.vllm_limit_mm_per_prompt
    if cfg.vllm_max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = cfg.vllm_max_num_batched_tokens

    print("\n[2/4] Initializing vLLM engine...")
    llm = LLM(**llm_kwargs)
    print("  vLLM engine ready.")

    processor_kwargs = {"trust_remote_code": True}
    if cfg.max_pixels:
        processor_kwargs["max_pixels"] = cfg.max_pixels
    try:
        processor = AutoProcessor.from_pretrained(processor_name, **processor_kwargs)
    except TypeError:
        # fallback for processors that don't accept max_pixels
        processor_kwargs.pop("max_pixels", None)
        processor = AutoProcessor.from_pretrained(processor_name, **processor_kwargs)

    return llm, processor, model_path


def _build_sampling_params(cfg: EvalConfig):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=cfg.generation_temperature,
        max_tokens=cfg.generation_max_new_tokens,
        min_tokens=1,
        seed=cfg.vllm_seed,
        repetition_penalty=cfg.generation_repetition_penalty,
    )


def _prepare_llm_request(
    processor: AutoProcessor,
    cfg: EvalConfig,
    item: Dict[str, Any],
) -> Any:
    prompt = format_prompt(item["question"], item["options"])
    images = _collect_images(item, cfg.max_pixels)
    if not images:
        raise ValueError("Sample missing both scene and map images.")

    formatted_prompt = _format_prompt_with_images(processor, prompt, len(images))

    if images:
        return {
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": images},
        }
    return formatted_prompt


def _flush_vllm_batch(
    llm,
    sampling_params,
    batch_inputs: List[Any],
    batch_meta: List[tuple[int, Dict[str, Any], str]],
) -> tuple[int, List[Dict[str, Any]]]:
    produced: List[Dict[str, Any]] = []
    if not batch_inputs:
        return 0, produced
    try:
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    except Exception as exc:
        for idx, item, sample_id in batch_meta:
            produced.append(
                {
                    "id": sample_id,
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "ground_truth": item.get("answer"),
                    "prediction": "ERROR",
                    "raw_response": str(exc),
                    "correct": False,
                }
            )
        return len(batch_meta), produced

    if len(outputs) != len(batch_meta):
        mismatch_msg = (
            f"vLLM returned {len(outputs)} outputs for {len(batch_meta)} inputs."
        )
        for _, item, sample_id in batch_meta:
            produced.append(
                {
                    "id": sample_id,
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "ground_truth": item.get("answer"),
                    "prediction": "ERROR",
                    "raw_response": mismatch_msg,
                    "correct": False,
                }
            )
        return len(batch_meta), produced

    for (_, item, sample_id), output in zip(batch_meta, outputs):
        response_text = (
            output.outputs[0].text.strip() if output and output.outputs else ""
        )
        prediction = normalize_letter(response_text, len(item["options"]))
        ground_truth = item["answer"]
        is_correct = prediction == ground_truth
        produced.append(
            {
                "id": sample_id,
                "question": item.get("question"),
                "options": item.get("options"),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "raw_response": response_text,
                "correct": is_correct,
            }
        )
    return 0, produced


def evaluate_vllm(cfg: EvalConfig) -> Dict[str, Any]:
    print("Configuration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Dataset: {cfg.dataset_name} ({cfg.dataset_split})")
    print(f"  Output dir: {cfg.output_dir}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    existing_results, processed_ids = _load_existing_predictions(predictions_path)
    header_written = predictions_path.exists() and predictions_path.stat().st_size > 0
    csv_fieldnames: Optional[List[str]] = (
        list(existing_results[0].keys()) if existing_results else None
    )
    if existing_results:
        print(
            f"[resume] Loaded {len(existing_results)} existing predictions from {predictions_path}. "
            "Will skip those IDs."
        )

    print("\n[1/4] Loading dataset...")
    dataset: Dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.limit is not None:
        dataset = dataset.select(range(min(cfg.limit, len(dataset))))
    total = len(dataset)
    print(f"  Samples: {total}")

    llm, processor, resolved_model = _load_vllm_engine(cfg)
    sampling_params = _build_sampling_params(cfg)

    print("\n[3/4] Running evaluation with vLLM...")
    new_results: List[Dict[str, Any]] = []
    errors = 0
    batch_inputs: List[Any] = []
    batch_meta: List[tuple[int, Dict[str, Any], str]] = []
    batch_size = max(1, cfg.batch_size)

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating", total=total)):
        sample_id = str(item.get("id", idx))
        if sample_id in processed_ids:
            continue
        try:
            llm_input = _prepare_llm_request(processor, cfg, item)
            batch_inputs.append(llm_input)
            batch_meta.append((idx, item, sample_id))
        except Exception as exc:
            errors += 1
            new_results.append(
                {
                    "id": sample_id,
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "ground_truth": item.get("answer"),
                    "prediction": "ERROR",
                    "raw_response": str(exc),
                    "correct": False,
                }
            )
            csv_fieldnames, header_written = _append_predictions(
                [new_results[-1]], predictions_path, csv_fieldnames, header_written
            )
            processed_ids.add(sample_id)
        if len(batch_inputs) >= batch_size:
            err_count, produced = _flush_vllm_batch(
                llm, sampling_params, batch_inputs, batch_meta
            )
            errors += err_count
            new_results.extend(produced)
            if produced:
                csv_fieldnames, header_written = _append_predictions(
                    produced, predictions_path, csv_fieldnames, header_written
                )
            processed_ids.update(meta_id for _, _, meta_id in batch_meta)
            batch_inputs = []
            batch_meta = []

    # Flush any remaining samples
    err_count, produced = _flush_vllm_batch(llm, sampling_params, batch_inputs, batch_meta)
    errors += err_count
    new_results.extend(produced)
    if produced:
        csv_fieldnames, header_written = _append_predictions(
            produced, predictions_path, csv_fieldnames, header_written
        )
    processed_ids.update(meta_id for _, _, meta_id in batch_meta)

    all_results = existing_results + new_results

    correct = sum(1 for r in all_results if r.get("correct"))
    empty_predictions = sum(
        1 for r in all_results if not (str(r.get("prediction", "")).strip())
    )
    random_expectation = 0.0
    for r in all_results:
        num_opts = _count_options_field(r.get("options"))
        if num_opts > 0:
            random_expectation += 1.0 / num_opts
    accuracy = correct / total if total else 0.0
    random_baseline = random_expectation / total if total else 0.0
    summary = {
        "model": cfg.model_name,
        "dataset": cfg.dataset_name,
        "split": cfg.dataset_split,
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": float(accuracy),
        "random_baseline": float(random_baseline),
        "empty_predictions": empty_predictions,
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "vllm_model_path": resolved_model,
            "max_new_tokens": cfg.generation_max_new_tokens,
            "temperature": cfg.generation_temperature,
            "repetition_penalty": cfg.generation_repetition_penalty,
            "limit": cfg.limit,
            "tensor_parallel_size": cfg.vllm_tensor_parallel_size,
            "gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
            "max_model_len": cfg.vllm_max_model_len,
            "max_num_seqs": cfg.vllm_max_num_seqs,
            "seed": cfg.vllm_seed,
            "errors": errors,
        },
    }

    print("\n[4/4] Evaluation complete.")
    print("=" * 80)
    for key, value in summary.items():
        if key != "config":
            print(f"{key.title().replace('_', ' ')}: {value}")
    print("=" * 80)
    print(f"Random guess baseline: {random_baseline:.2%}")
    print(f"Empty predictions: {empty_predictions}")

    if cfg.save_predictions:
        _save_outputs(all_results, summary, cfg.output_dir)

    return summary


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    if os.environ.get("VLLM_MODEL"):
        cfg.vllm_model = os.environ.get("VLLM_MODEL")
    if os.environ.get("VLLM_PROCESSOR"):
        cfg.vllm_processor = os.environ.get("VLLM_PROCESSOR")
    if os.environ.get("VLLM_TENSOR_PARALLEL_SIZE"):
        try:
            cfg.vllm_tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE"))
        except Exception:
            pass
    if os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"):
        try:
            cfg.vllm_gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"))
        except Exception:
            pass
    if os.environ.get("VLLM_MAX_MODEL_LEN"):
        try:
            cfg.vllm_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN"))
        except Exception:
            pass
    if os.environ.get("VLLM_MAX_NUM_SEQS"):
        try:
            cfg.vllm_max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS"))
        except Exception:
            pass
    if os.environ.get("VLLM_SEED"):
        try:
            cfg.vllm_seed = int(os.environ.get("VLLM_SEED"))
        except Exception:
            pass
    if os.environ.get("KOA_EVAL_BATCH_SIZE"):
        try:
            cfg.batch_size = int(os.environ.get("KOA_EVAL_BATCH_SIZE"))
        except Exception:
            pass

    return evaluate_vllm(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL models.")
    parser.add_argument(
        "-c", "--config", help="Path to config YAML file (optional)."
    )
    parser.add_argument("--model", help="Model name or path for evaluation.")
    parser.add_argument("--dataset", help="Dataset name or path (HF dataset).")
    parser.add_argument("--split", help="Dataset split to evaluate.")
    parser.add_argument(
        "--max-new-tokens", type=int, help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Repetition penalty applied during decoding.",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for evaluation.")
    parser.add_argument("--limit", type=int, help="Limit the number of samples.")
    parser.add_argument("--output-dir", help="Directory to store evaluation outputs.")
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Disable writing predictions/summary to disk.",
    )
    parser.add_argument(
        "--vllm-model",
        help="Optional override for the model path used by vLLM (defaults to --model).",
    )
    parser.add_argument("--vllm-processor", help="Override processor checkpoint.")
    parser.add_argument(
        "--tensor-parallel-size", type=int, help="vLLM tensor parallel size."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Fraction of GPU memory vLLM is allowed to use (0-1).",
    )
    parser.add_argument(
        "--max-model-len", type=int, help="Maximum context length allowed by vLLM."
    )
    parser.add_argument(
        "--max-num-seqs", type=int, help="Maximum concurrent sequences in vLLM."
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help="Override vLLM max_num_batched_tokens.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Resize images so total pixels are <= this value (default 480*480).",
    )
    parser.add_argument("--seed", type=int, help="Sampling seed passed to vLLM.")
    return parser.parse_args()


def _append_predictions(
    rows: List[Dict[str, Any]],
    path: Path,
    csv_fieldnames: Optional[List[str]],
    header_written: bool,
) -> tuple[Optional[List[str]], bool]:
    if not rows:
        return csv_fieldnames, header_written
    if csv_fieldnames is None:
        csv_fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        if not header_written:
            writer.writeheader()
            header_written = True
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in csv_fieldnames})
    return csv_fieldnames, header_written


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        cfg = _apply_cli_overrides(cfg, args)
    else:
        if not args.model:
            raise SystemExit("Provide --model when --config is not supplied.")
        cfg = _build_config_from_cli(args)
    evaluate(cfg)


if __name__ == "__main__":
    main()
