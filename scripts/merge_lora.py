from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base VLM and save the merged model.")
    parser.add_argument("--base_model", required=True, help="Base model name or path (e.g., Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--lora_path", required=True, help="Path to the LoRA checkpoint (adapter)")
    parser.add_argument("--merged_path", required=True, help="Output path for the merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Torch dtype to load base model")
    parser.add_argument("--device_map", default="auto", help="device_map to use when loading the base model")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code when loading models")
    args = parser.parse_args()

    merged_dir = Path(args.merged_path)
    merged_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(base, args.lora_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code).save_pretrained(merged_dir)
    # Save processor/preprocessor config alongside the merged model.
    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
        processor.save_pretrained(merged_dir)
    except Exception as exc:
        print(f"Warning: failed to save processor/preprocessor config: {exc}")

    print("✅ Merge complete.")


if __name__ == "__main__":
    main()
