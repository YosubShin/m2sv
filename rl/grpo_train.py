"""GRPO-based RL pipeline for M2SV using multimodal Qwen models.

This script mirrors the TRL GRPO VLM example while adapting it to the M2SV
map-plus-street-view setting. The default model is Qwen/Qwen3-VL-4B-Instruct
with optional vLLM rollouts for faster sampling.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Dict

# Ensure repository root is on sys.path when run from subdirectories.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from m2sv_eval_utils import format_prompt, normalize_letter
from transformers import AutoModelForVision2Seq, AutoProcessor, HfArgumentParser
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
_DEBUG_REWARDS_LOGGED = False


@dataclass
class ScriptArguments:
    """Command-line arguments for GRPO training."""

    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    dataset_name: str = "yosubshin/m2sv-20k"
    dataset_split: str = "train"
    output_dir: str = "outputs/grpo-qwen3vl-m2sv"
    max_train_samples: int | None = None

    beta: float = 0.03
    num_generations: int = 8
    generation_batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.02
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 4
    num_train_epochs: float = 1.0
    max_prompt_length: int | None = None
    max_completion_length: int = 2048
    lr_scheduler_type: str = "cosine"

    logging_steps: int = 1
    save_steps: int = 250
    save_total_limit: int = 3
    seed: int = 42
    report_to: str = "wandb"
    wandb_project: str = "m2sv-grpo"
    wandb_run_name: str | None = None

    use_vllm_rollout: bool = False
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int | None = None
    vllm_dtype: str | None = None
    vllm_trust_remote_code: bool | None = None

    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True


@dataclass
class DataKeys:
    prompt: str = "prompt"
    images: str = "images"
    answer: str = "answer"
    options: str = "options"


def build_messages(question: str, options: Sequence[str]) -> list[dict[str, Any]]:
    """Format the multimodal prompt as chat messages with two image placeholders."""

    option_lines = "\n".join(options)
    prompt_text = format_prompt(f"{question}\n\nOptions:\n{option_lines}", list(options))
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},  # image_map placeholder
                {"type": "image"},  # image_sv placeholder
            ],
        }
    ]


def preprocess_dataset(processor: Any, args: ScriptArguments, keys: DataKeys):
    """Load and convert the M2SV dataset to GRPO-ready columns."""

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_train_samples:
        dataset = dataset.select(range(args.max_train_samples))

    def _map_row(example: dict[str, Any]):
        prompt = build_messages(example["question"], example["options"])
        return {
            keys.prompt: prompt,
            keys.images: [example["image_map"], example["image_sv"]],
            keys.answer: example["answer"],
            keys.options: example["options"],
            "id": example.get("id", ""),
        }

    mapped = dataset.map(
        _map_row,
        remove_columns=dataset.column_names,
    )
    return mapped


def compute_rewards(
    *,
    prompts: list[Any] | None = None,
    completions: List[List[Dict[str, Any]]] | None = None,
    completions_ids: Any | None = None,  # unused, accepted for signature compatibility
    trainer_state: Any | None = None,  # unused, accepted for signature compatibility
    **kwargs: Any,
) -> List[float]:
    """Return GRPO rewards (list of floats), matching TRL's reward_fn contract."""

    _ = prompts, completions_ids, trainer_state  # silence unused warnings

    # TRL may pass completions via keyword arguments; fall back to those if needed.
    completions = completions
    answers = list(kwargs.get("answer"))
    options = list(kwargs.get("options"))

    # Ensure answers/options list lengths match the number of samples
    if not answers and completions:
        answers = [""] * (len(completions) if not isinstance(completions[0], list) else len(completions))
    if not options and answers:
        options = [[] for _ in answers]

    def score(sample: str, gold: str, opts: Sequence[str]) -> float:
        def _to_text(s: Any) -> str:
            if isinstance(s, str):
                return s
            if isinstance(s, dict):
                content = s.get("content")
                if isinstance(content, list):
                    parts: list[str] = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                            parts.append(c["text"])
                    if parts:
                        return "\n".join(parts)
                if isinstance(content, str):
                    return content
            return str(s)

        pred = normalize_letter(_to_text(sample), num_options=len(opts))
        return 1.0 if pred == (gold or "").strip().upper() else 0.0

    rewards: List[float] = []
    if completions and isinstance(completions[0], list):
        # Nested completions: one list per prompt; answers/options should align per prompt.
        for comps, gold, opts in zip(completions, answers, options):
            rewards.extend(score(c, gold, opts) for c in comps)
    else:
        rewards.extend(
            score(c, a, o)
            for c, a, o in zip(completions, answers, options if options else [[]] * len(completions))
        )

    if not rewards:
        rewards = [0.0 for _ in range(len(completions) if completions else 1)]

    # Debug: log the first batch at the start of training to inspect completions/targets.
    global _DEBUG_REWARDS_LOGGED
    if not _DEBUG_REWARDS_LOGGED:
        logger.info(
            "[debug] completions sample=%s | answers=%s | options=%s | reward=%s",
            completions[0],
            answers[0],
            options[0],
            rewards[0],
        )
        _DEBUG_REWARDS_LOGGED = True

    return rewards


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    keys = DataKeys()

    if args.report_to == "wandb" and args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = preprocess_dataset(processor, args, keys)

    generation_kwargs = {
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
        "max_new_tokens": args.max_completion_length,
    }

    config_kwargs = dict(
        output_dir=args.output_dir,
        beta=args.beta,
        num_generations=args.num_generations,
        generation_batch_size=max(args.generation_batch_size, args.num_generations),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        generation_kwargs=generation_kwargs,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.wandb_run_name,
    )

    # Align with TRL's optional vLLM integration for faster rollouts when available.
    if args.use_vllm_rollout:
        config_fields = getattr(GRPOConfig, "__dataclass_fields__", {})
        if "use_vllm" in config_fields:
            config_kwargs["use_vllm"] = True
        if "vllm_kwargs" in config_fields:
            config_kwargs["vllm_kwargs"] = {
                "tensor_parallel_size": None,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_model_len": args.vllm_max_model_len,
                "dtype": args.vllm_dtype,
                "trust_remote_code": (
                    args.vllm_trust_remote_code
                    if args.vllm_trust_remote_code is not None
                    else args.trust_remote_code
                ),
            }

    training_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=compute_rewards,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    logger.info("Saving model and processor to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
