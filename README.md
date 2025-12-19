# Changelog

## 2025-10-31-v1
- **20k blueprint + clean splits**:
  - Added `blueprints/20k/train-val-20k.jsonl` (20k curated blueprints) plus deduped splits: `train.jsonl` (10k) and `validation.jsonl` (10k).
  - Introduced `create_validation_split.py` to regenerate validation sets without contaminating rows already used for training.
  - Validation size now 10k (previously 1k), shrinking the 95% margin of error from ~±3% to ~±1% for future eval reports.

## 2025-10-30-v1
- **Dataset pipeline refresh**:
  - Split curation into blueprint generation and rendering; released datasets now ship blueprints only to respect Google Maps licensing.
  - Scaled evaluation data from 1k to 11k examples and added an SFT split populated with Gemini 2.5 Pro reasoning traces.
- **Batch evaluation upgrades**:
  - Added OpenAI Batch API integration with resumable checkpoints, error capture, and upload-size logging.
  - Added Qwen-compatible batch flow with test-endpoint mode, automatic request trimming under size caps, and provider-specific payload patches.
  - Introduced CLI flags for test-mode, max JSONL size guardrails, and batch temperature overrides.
- **Diagnostics & resiliency**:
  - Persist and surface batch error outputs alongside metrics when jobs fail.
  - Log the exact upload size for each batch submission to ease debugging when providers throttle or reject payloads.
- Metrics:
  | Model                         | Accuracy | N    | Margin of Error (±%) | 95% Confidence Interval |
  |--------------------------------|-----------|------|----------------------|--------------------------|
  | gpt-5                          | 57.2%     | 1000 | ±3.1%                | [54.1%, 60.3%]          |
  | gemini-2.5-pro                 | 47.2%     | 1000 | ±3.1%                | [44.1%, 50.3%]          |
  | qwen3-vl-8b-instruct           | 35.5%     | 1000 | ±3.0%                | [32.5%, 38.5%]          |
  | qwen3-vl-8b-thinking           | 34.4%     | 1000 | ±2.9%                | [31.5%, 37.3%]          |
  | qwen3-vl-30b-a3b-instruct      | 33.9%     | 1000 | ±2.9%                | [31.0%, 36.8%]          |
  | qwen3-vl-30b-a3b-thinking      | 36.1%     | 1000 | ±3.0%                | [33.1%, 39.1%]          |
  | qwen3-vl-32b-thinking          | 40.7%     | 1000 | ±3.1%                | [37.6%, 43.8%]          |
  | qwen3-vl-235b-a22b-instruct    | 38.1%     | 1000 | ±3.0%                | [35.1%, 41.1%]          |
  | qwen3-vl-235b-a22b-thinking    | 42.7%     | 1000 | ±3.1%                | [39.6%, 45.8%]          |
  | Random baseline                | 31.4%     | 1000 | ±2.9%                | [28.5%, 34.3%]          |
  | Human baseline                 | 88.0%     | 100  | ±6.4%                | [81.6%, 94.4%]          |

  - For qwen3-vl, we used Alibaba cloud's API to evaluate the models.
  - Everything other than human baseline uses the same validation set (N=1000).
  - Human baseline uses smaller validation set (N=100) from Honolulu, HI.

## 2025-10-15-v1
- **Prompt update**: Encourage step-by-step reasoning; require final line in LaTeX-style `\boxed{X}`.
- **Parallel + multi-config evaluation in `evaluate_vlm.py`**:
  - Per-example parallelism via `--workers` (threaded, thread-local provider clients).
  - Batch multiple provider/model runs via repeatable `--eval` or `--eval-file`.
  - Run configs concurrently with `--parallel-configs` (suppresses per-config progress bars when >1).
  - Final logs include provider, model, and dataset.
- Prompt used in `evaluate_vlm.py`:
  ```
  You will be given two images: (1) a north-up overhead map with arrows labeled A, B, C, ... and (2) a street-view photo.
  Rules:
  - The camera location is the same for all options: the center of the intersection.
  - Each letter corresponds to facing outward from that center along the arrow of that label.
  - The small circles near labels are markers only; they are not camera locations.
  - The map and photo may be captured years apart. Ignore transient objects (cars, people).
  Think step by step to compare the street-view with the map (buildings, angles, lanes, landmarks).
  On the final line, output only: Final answer: \boxed{X} where X is a single letter (A, B, C, ...).
  ```
- Metrics
  | Model             | Accuracy |
  |-------------------|----------|
  | gemini-2.5-pro    | 51%      |
  | gemini-2.5-flash  | 47%      |
  | gpt-4o            | 38%      |
  | claude-opus-4.1   | 36.7% (60 samples) |
  | Random baseline   | 31.8%    |
  | Human baseline    | 88%      |

## GRPO training pipeline

- New RL script: `scripts/grpo_train.py` wires the TRL `GRPOTrainer` to the M2SV
  dataset (`yosubshin/m2sv-20k`) with two-image inputs (`image_map` and
  `image_sv`) and wandb logging enabled by default. The script now defaults to
  `Qwen/Qwen3-VL-4B-Instruct`, sets `padding_side="left"`, and uses
  Qwen3-recommended sampling settings (temperature 0.7, top_p 0.8,
  repetition_penalty 1.05). `max_prompt_length` is unset to keep the full
  chat template intact. Optional TRL/vLLM rollouts can be enabled with
  `--use-vllm-rollout` to accelerate sampling; `vllm_gpu_memory_utilization`
  and `vllm_max_model_len` are exposed for tuning.
- The script is fully CLI-driven via `transformers.HfArgumentParser`, so all
  fields in `ScriptArguments` can be overridden with `--flag value` pairs.
- Usage example (swap `--model_name` for other qwen3-vl sizes as needed or add
  `--use_vllm_rollout` for vLLM-backed sampling):

  ```bash
  WANDB_PROJECT=m2sv-grpo \
  python scripts/grpo_train.py \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --output_dir outputs/grpo-qwen3vl-m2sv \
    --dataset_split train \
    --wandb_run_name qwen3vl-grpo
  ```

  When running under Slurm, wrap the same flags, e.g.:

  ```bash
  srun --nodes=1 --gres=gpu:1 \
    python scripts/grpo_train.py --model_name Qwen/Qwen3-VL-4B-Instruct
  ```

## 2025-10-14-v2
- Why we updated the prompt:
  - Models sometimes assumed the map and street-view were captured at the same time and relied on transient cues (cars, people), which can differ by years. We now explicitly instruct to ignore such transient objects.
  - Models sometimes inferred that the camera was located at the label circles instead of the intersection center. We now clarify the camera is fixed at the center and arrows indicate viewing directions; circles are markers only.
- The evaluation prompt used in `evaluate_vlm.py`:
  ```
  You will be given two images: (1) a north-up overhead map with arrows labeled A, B, C, ... and (2) a street-view photo.
  Rules:
  - The camera location is the same for all options: the center of the intersection.
  - Each letter corresponds to facing outward from that center along the arrow of that label.
  - The small circles near labels are markers only; they are not camera locations.
  - The map and photo may be captured years apart. Ignore transient objects (cars, people).
  Respond with just the single letter (A, B, C, ...), no words or punctuation.
  ```
- Metrics
  | Model             | Accuracy |
  |-------------------|----------|
  | gemini-2.5-pro    | 39%      |
  | gpt-4o            | 47%      |
  | gemini-2.5-flash  | 41%      |
  | claude-opus-4.1   | 35%      |
  | Random baseline   | 31.8%    |
  | Human baseline    | 88%      |

## 2025-10-14-v1
- **Initial version**
- **`create_dataset.py`**: Builds a two-image multiple-choice dataset from real intersections.
  - Fetches a Google Static Map centered at each intersection and overlays labeled arrows (A, B, C, ...).
  - Retrieves a Street View image from a nearby panorama at one of the labeled directions.
  - Exports a Hugging Face-style dataset (`train.jsonl`) with fields: `id`, `images` `[map, street_view]`, `question`, `options`, and the correct `answer`.
  - Optional: pushes a typed dataset (with `Image` features) to the Hugging Face Hub.
- **`evaluate_vlm.py`**: Evaluates vision-language models on the dataset.
  - Supports providers: OpenAI, Gemini, Claude; sends two images (map + street view) with a multiple-choice prompt.
  - Robust answer extraction from free-form outputs (single letters, boxed letters, explicit "answer is X", last-line letters, repeated forms like "C. C").
  - Metrics include accuracy, empty prediction count, random-guess baseline (averaged 1/num_options), and an option-count histogram.
  - Features: resume runs (`--resume`), re-parse existing results without querying (`--reparse-result`), and a built-in normalization self-test (`--self-test`).
- Prompt used in `evaluate_vlm.py`:
  ```
    You will be given two images: a labeled overhead map and a street-view photo.
    Choose which labeled direction on the map corresponds to the direction in which the street view photo was taken.
    Answer with a single letter only (A, B, C, ...).
  ```
- Metrics
  | Model             | Accuracy |
  |-------------------|----------|
  | gemini-2.5-pro    | 42%      |
  | gpt-4o            | 41%      |
  | gemini-2.5-flash  | 32%      |
  | Random baseline   | 31.8%    |
  | Human baseline    | 88%      |

# Commands

## Sequential eval
```
python evaluate_vlm.py yosubshin/m2sv --provider openai --model gpt-4o --limit 100 --out results/gpt-4o.json --resume --reparse-result
python evaluate_vlm.py yosubshin/m2sv --provider gemini --model gemini-2.5-flash --limit 100 --out results/gemini-2-5-flash.json --resume --reparse-result
python evaluate_vlm.py yosubshin/m2sv --provider gemini --model gemini-2.5-pro --limit 100 --out results/gemini-2-5-pro.json --resume --reparse-result
python evaluate_vlm.py yosubshin/m2sv --provider claude --model claude-opus-4-1-20250805 --limit 100 --out results/claude-opus-4-1.json --resume --reparse-result
```

## Parallel eval
```
python evaluate_vlm.py yosubshin/m2sv \
  --parallel-configs 4 --workers 4 \
  --eval "openai,gpt-4o,results/gpt-4o.json" \
  --eval "gemini,gemini-2.5-flash,results/gemini-2-5-flash.json" \
  --eval "gemini,gemini-2.5-pro,results/gemini-2-5-pro.json"
```

## Human eval
```
streamlit run review_webapp.py
```

# RL (GRPO) training commands

## 4B model

### Train
```
koa submit scripts/grpo_job.slurm \
  --desc "qwen3-vl-4b-instruct" \
  --env MODEL_NAME=/mnt/lustre/koa/scratch/yosubs/koa-cli/projects/Qwen3-VL/jobs/20251112_223605_qwen3-vl-4b-instruct_lr_5e-6/results/qwen3-vl-4b-instruct/checkpoint-600
```

### Eval
```
koa submit scripts/eval_job.slurm \
  --desc "eval qwen3-vl-4b-instruct" \
  --env MODEL_NAME=/mnt/lustre/koa/scratch/yosubs/koa-cli/projects/map-to-street-view/jobs/20251213_105127_resume_from_0_75epoch/results/grpo
```

### Resume (non-LoRA, 4B)
```
koa submit scripts/grpo_job.slurm \
  --desc "resume qwen3-vl-4b-instruct" \
  --env MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct \
  --env GRPO_RESUME_FROM_CHECKPOINT=/path/to/grpo/checkpoint-XXXX
```

## 8B model (LoRA)

### Train
```
koa submit scripts/grpo_job.slurm \
  --desc "qwen3-vl-8b-instruct-lora" \
  --env GRPO_USE_LORA=TRUE \
  --env GRPO_LEARNING_RATE=2e-5 \
  --env MODEL_NAME=/home/yosubs/koa_scratch/Qwen3-VL/qwen-vl-finetune/output/11k/qwen3-vl-8b-instruct-lora/merged
```

### Eval
```
koa submit scripts/eval_job.slurm \
  --desc "eval qwen3-vl-8b-instruct-lora checkpoint-350" \
  --env MODEL_NAME=/home/yosubs/koa_scratch/koa-cli/projects/map-to-street-view/jobs/20251214_154058_qwen3-vl-8b-instruct-lora/results/grpo/checkpoint-350/merged
```

### Resume (LoRA, 8B)
```
koa submit scripts/grpo_job.slurm \
  --desc "resume qwen3-vl-8b-instruct-lora" \
  --env GRPO_USE_LORA=TRUE \
  --env MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct \  # base model, not merged
  --env GRPO_RESUME_FROM_CHECKPOINT=/path/to/grpo/checkpoint-XXXX  # LoRA + optimizer state
```

Example with separate models for serving vs. training/resume:
```
koa submit scripts/grpo_job.slurm \
  --desc "resume qwen3-vl-8b-instruct-lora" \
  --env GRPO_USE_LORA=TRUE \
  --env GRPO_LEARNING_RATE=2e-5 \
  --env VLLM_MODEL_NAME=/home/yosubs/koa_scratch/koa-cli/projects/map-to-street-view/jobs/20251214_154058_qwen3-vl-8b-instruct-lora/results/grpo/checkpoint-350/merged \
  --env GRPO_MODEL_NAME=/home/yosubs/koa_scratch/koa-cli/projects/map-to-street-view/jobs/20251214_154058_qwen3-vl-8b-instruct-lora/results/grpo/checkpoint-350 \
  --env GRPO_RESUME_FROM_CHECKPOINT=/home/yosubs/koa_scratch/koa-cli/projects/map-to-street-view/jobs/20251214_154058_qwen3-vl-8b-instruct-lora/results/grpo/checkpoint-350
```

## Freeze-and-render dataset workflow

1) Freeze a reproducible blueprint (metadata only, no images yet). This records coordinates, azimuth options, gold label, and the Street View pano id/distance so the dataset can be recreated consistently.

```
python freeze_blueprint.py \
  --out blueprints/20k/train-val-20k.jsonl \
  --total-samples 20000 \
  --per-place-cap 1000 \
  --seed 42 \
  --candidate-multiplier 5 \
  --resume
```

Options:
- `--places-file` (txt or json list) or multiple `--place` args to control geographic diversity; otherwise a default global list is used.
- `--max-sv-distance-m` (default 10m) and `--metadata-radius-m` (default 15m) control Street View eligibility.
- `--dedupe-radius-m` (default 20m) avoids clustering.
 - `--candidate-multiplier` caps the number of candidate nodes checked per city to N× the target for that city (default 10×). This prevents spending excessive time in cities with poor Street View coverage.
 - `--resume` resumes from an existing output JSONL file, appending results incrementally per city and skipping places that already have their target count.

Logging and metrics enhancements in `freeze_blueprint.py`:
- Per city, the script logs: checked, accepted, acceptance rate, elapsed seconds, nodes/sec, and a breakdown of filtering reasons with rates (e.g., `no_metadata`, `metadata_incomplete`, `too_far`, `pano_invalid`, `edges_access_error`, `no_azimuths`, `not_enough_azimuths`, `dedupe_blocked`).

2) Render a dataset from a blueprint (fetch images, overlay arrows, emit HF JSONL):

```
python render_from_blueprint.py blueprints/20k/train.jsonl m2sv-20k-train --output-root data/hf
```

Optional overrides at render time:
- `--override-map-zoom`, `--override-map-type`, `--override-map-size`
- `--override-sv-fov`, `--override-sv-pitch`

Notes:
- Blueprints separate curation (freeze) from rendering (API calls), enabling reproducible splits and cheaper iteration.

3) Merge train and validation splits into a single HF dataset. Use `merge_jsonl_splits.py` to combine two JSONL files into one dataset with `train` and `validation` splits and optionally push to the Hub:

```
python merge_jsonl_splits.py \
  /Users/yosub/co/map-to-street-view/data/hf/m2sv-11k-train/train.jsonl \
  /Users/yosub/co/map-to-street-view/data/hf/m2sv-validation-10k/train.jsonl \
  --out-dir /Users/yosub/co/map-to-street-view/data/hf/m2sv-20k \
  --repo yosubshin/m2sv-20k
```
