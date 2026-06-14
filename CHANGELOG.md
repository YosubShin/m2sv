# Changelog

## 2026-06 — CTB@ICML 2026 camera-ready
- **Multi-annotator human baseline** replacing the single-annotator pilot: 10
  annotators (7 completed all 200) + 1 expert, collected via the hostable
  human-eval web app (`human_eval/`, live at https://m2sv.yosubshin.com).
  - Engaged annotators (excluding one near-chance, low-agreement outlier) average
    **74.0% ± 7.3%**; expert **95%**; inter-annotator agreement Cohen's
    kappa up to **0.76**. Best VLM (Gemini-3-Pro) is **65.2%**.
- **De-circularized difficulty analysis**: response-time difficulty bins and the
  accuracy curve now come from different annotators; human accuracy degrades
  gracefully while models degrade more steeply.
- **Road-azimuth symmetry metric defined** (S = max gap / min gap) and the
  symmetry analysis stratified by candidate count (3-option) to remove the
  #options confound; symmetric intersections are hardest for all models.
- **Reproducible figures**: anonymized data bundle (`analysis/human_baseline_data.json`,
  no PII) + `analysis/human_baseline_figs.py` regenerate every human-baseline
  figure; `analysis/plot_symmetry_accuracy.py` for the model symmetry figure.

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
