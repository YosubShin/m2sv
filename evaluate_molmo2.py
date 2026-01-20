#!/usr/bin/env python3
"""
Hard-coded Molmo2 evaluation script (matches Untitled5.ipynb workflow).
"""

from __future__ import annotations

import csv
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from m2sv_eval_utils import format_prompt, normalize_letter


def _free_memory() -> None:
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "inputs" in globals():
        del globals()["inputs"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model_and_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )
    # Required for correct batch generation with decoder-only models.
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, processor


def _run_batch_inference(
    batch_items: List[Dict[str, Any]],
    model,
    processor,
    max_new_tokens: int,
) -> List[str]:
    batch_messages = []
    for item in batch_items:
        question = item["question"]
        options = item.get("options", [])
        prompt_text = format_prompt(question, options)
        batch_messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        dict(type="text", text=prompt_text),
                        dict(type="image", image=item["image_map"]),
                        dict(type="image", image=item["image_sv"]),
                    ],
                }
            ]
        )

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = generated_ids[:, input_len:]
    decoded_texts = processor.batch_decode(
        generated_tokens, skip_special_tokens=True
    )
    return decoded_texts


def main() -> None:
    _free_memory()

    model_id = "allenai/Molmo2-4B"
    dataset_name = "yosubshin/m2sv-20k"
    dataset_split = "validation"
    num_examples = 1000
    batch_size = 64
    max_new_tokens = 4096
    output_dir = os.environ.get("MOLMO2_OUTPUT_DIR", "./eval/results/molmo2_m2sv")

    model, processor = _load_model_and_processor(model_id)

    ds = load_dataset(dataset_name, split=dataset_split)
    subset = ds.select(range(num_examples))

    correct_count = 0
    empty_prediction_count = 0
    total_count = 0
    results: List[Dict[str, Any]] = []

    print(
        f"Running batch evaluation on {num_examples} examples with batch_size={batch_size}..."
    )

    for i in tqdm(range(0, num_examples, batch_size)):
        batch_indices = range(i, min(i + batch_size, num_examples))
        batch_items = [subset[j] for j in batch_indices]

        decoded_texts = _run_batch_inference(
            batch_items=batch_items,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

        for item, text in zip(batch_items, decoded_texts):
            options = item.get("options", [])
            num_opts = len(options) if options else 26
            predicted_letter = normalize_letter(text, num_opts)
            if not predicted_letter:
                empty_prediction_count += 1
            ground_truth = item["answer"]
            is_correct = predicted_letter == ground_truth
            if is_correct:
                correct_count += 1
            total_count += 1
            results.append(
                {
                    "id": item.get("id", i),
                    "prediction": predicted_letter,
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                }
            )

    accuracy = correct_count / total_count if total_count else 0.0
    print(f"\nFinal Accuracy (Batch) on {total_count} examples: {accuracy:.2%}")
    print(f"Empty predictions: {empty_prediction_count}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    predictions_path = Path(output_dir) / "predictions.csv"
    summary_path = Path(output_dir) / "summary.json"

    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = (
            list(results[0].keys())
            if results
            else ["id", "prediction", "ground_truth", "correct"]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary = {
        "model": model_id,
        "dataset": dataset_name,
        "split": dataset_split,
        "total_samples": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "empty_predictions": empty_prediction_count,
        "accuracy": float(accuracy),
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "num_examples": num_examples,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
