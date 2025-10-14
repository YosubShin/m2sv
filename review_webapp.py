import json
from pathlib import Path
from typing import Dict, List

import streamlit as st
from datasets import load_dataset
from PIL import Image


def load_results(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"accuracy": 0.0, "correct": 0, "total": 0, "results": []}


def save_results(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get_processed_ids(results: Dict) -> set:
    return {str(r.get("id", "")) for r in results.get("results", []) if r.get("id", None) is not None}


def main():
    st.set_page_config(page_title="VLM Manual Evaluation", layout="wide")
    st.title("VLM Manual Evaluation")

    with st.sidebar:
        dataset_path = st.text_input("Dataset path or repo id", value="data/hf/m2sv")
        out_path_str = st.text_input("Results JSON path", value="results/manual.json")
        start_idx = st.number_input("Start index", min_value=0, value=0, step=1)
        decode_images = st.checkbox("Decode images via datasets (recommended)", value=True)
        st.markdown("Use the same JSON format as evaluate_vlm.py. Supports resume if file exists.")

    out_path = Path(out_path_str)
    state = load_results(out_path)
    processed_ids = get_processed_ids(state)

    # Load dataset
    ds = load_dataset(dataset_path, split="train") if Path(dataset_path).exists() else load_dataset(dataset_path)["train"]

    # Root for resolving relative image paths
    repo_root = Path(dataset_path) if Path(dataset_path).exists() else Path(".")

    def resolve_image(val):
        # Accept PIL, dict with path, or string path relative to dataset root
        if isinstance(val, Image.Image):
            return val
        if isinstance(val, dict) and "path" in val:
            return val["path"]
        if isinstance(val, str):
            p = (repo_root / val).resolve()
            return str(p)
        return val

    # Build id->row for quick lookup and determine next index
    id_to_idx = {}
    for i, row in enumerate(ds):
        rid = str(row.get("id", i))
        id_to_idx[rid] = i

    # Find next unprocessed row >= start_idx
    def find_next(start: int) -> int:
        for i in range(start, len(ds)):
            rid = str(ds[i].get("id", i))
            if rid not in processed_ids:
                return i
        return -1

    idx = find_next(start_idx)

    total_done = len(processed_ids)
    total = len(ds)
    st.markdown(f"**Progress:** {total_done} / {total}")
    st.progress(total_done / total if total else 0.0)

    if idx == -1:
        st.success("All items completed!")
        if st.button("Recompute accuracy"):
            # Recompute metrics from saved predictions
            correct = sum(1 for r in state.get("results", []) if r.get("correct") is True)
            total = len(state.get("results", []))
            state["correct"] = correct
            state["total"] = total
            state["accuracy"] = (correct / total) if total else 0.0
            save_results(out_path, state)
            st.rerun()
        return

    row = ds[idx]
    rid = str(row.get("id", idx))
    images = row.get("images")
    image_map = row.get("image_map") or (images[0] if images else None)
    image_sv = row.get("image_sv") or (images[1] if images else None)
    image_map = resolve_image(image_map)
    image_sv = resolve_image(image_sv)
    question = row.get("question", "Where is the Street View photo taken from?")
    options: List[str] = row.get("options", [])
    gold = str(row.get("answer", "")).strip().upper()

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_map, caption="Overhead map with labels", use_container_width=True)
    with col2:
        st.image(image_sv, caption="Street View", use_container_width=True)

    st.subheader("Question")
    st.write(question)

    # Render options as radio (no default selection)
    letters = [chr(ord('A') + i) for i in range(len(options))]
    choices = [f"{l}: {opt}" for l, opt in zip(letters, options)]
    choice = st.radio(
        "Choose one option:",
        options=choices,
        index=None,
        key=f"choice_{rid}",
    )

    raw_answer = st.text_input("Optional notes / raw answer (e.g., \\boxed{A})", value="")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("Submit & Next"):
            pred_letter = ""
            if choice is not None:
                pred_letter = choice.split(":", 1)[0]
            elif raw_answer:
                # Fallback: simple boxed extraction
                import re
                m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", raw_answer)
                if m:
                    pred_letter = m.group(1).upper()

            correct_flag = bool(pred_letter == gold)
            # Append result in evaluate_vlm format
            state.setdefault("results", []).append({
                "id": rid,
                "pred": pred_letter,
                "gold": gold,
                "raw": raw_answer or pred_letter,
                "correct": correct_flag,
            })
            # Update simple metrics
            state["total"] = len(state.get("results", []))
            state["correct"] = sum(1 for r in state["results"] if r.get("correct"))
            state["accuracy"] = (state["correct"] / state["total"]) if state["total"] else 0.0
            save_results(out_path, state)
            st.rerun()

    with col_b:
        st.json({"id": rid, "options": options})


if __name__ == "__main__":
    main()


