# Changelog

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
