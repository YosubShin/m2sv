## Changelog

### 2025-10-15-v1
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