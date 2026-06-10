import html
import json
import time
from pathlib import Path
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components
from datasets import load_dataset, load_from_disk
from PIL import Image


FAILURE_CATEGORIES = [
    "Structural symmetry / near-symmetry",
    "Insufficient or missing landmarks",
    "Temporal mismatch (map vs. Street View)",
    "Visual degradation",
    "Viewpoint / orientation misalignment (left–right confusion)",
    "Reasoning error despite sufficient evidence",
    "Other (specify)",
]


def load_results(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"results": []}


def save_results(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get_processed_ids(results: Dict) -> set:
    return {str(r.get("id", "")) for r in results.get("results", []) if r.get("id", None) is not None}


def get_result_index_map(results: Dict) -> Dict[str, int]:
    idx = {}
    for i, r in enumerate(results.get("results", [])):
        rid = r.get("id", None)
        if rid is not None:
            idx[str(rid)] = i
    return idx


def load_failures_csv(path: Path) -> List[Dict]:
    import csv

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = str(row.get("correct", "")).strip().lower()
            if correct in {"true", "1", "yes", "y"}:
                continue
            rid = row.get("id")
            if rid is None:
                continue
            rows.append(row)
    return rows


def main():
    st.set_page_config(page_title="Failure Taxonomy Labeling", layout="wide")
    st.title("Failure Taxonomy Labeling")

    with st.sidebar:
        dataset_path = st.text_input("Dataset path or repo id", value="data/hf/m2sv-11k")
        model_csv_path = st.text_input(
            "Model CSV path",
            value="past_results/2026-01-23/qwen3-vl-8b-instruct-sft-rl.csv",
        )
        out_path_str = st.text_input("Output JSON path", value="results/failure_taxonomy.json")
        start_idx = st.number_input("Start index", min_value=0, value=0, step=1)
        edit_mode = st.checkbox("Include annotated in queue", value=True)
        st.markdown("Label model failures with a single primary category.")

    out_path = Path(out_path_str)
    state = load_results(out_path)
    processed_ids = get_processed_ids(state)
    result_index_map = get_result_index_map(state)

    model_rows = load_failures_csv(Path(model_csv_path))

    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.exists():
        if dataset_path_obj.is_dir() and (dataset_path_obj / "dataset_dict.json").exists():
            ds_dict = load_from_disk(str(dataset_path_obj))
            ds = ds_dict["validation"] if "validation" in ds_dict else ds_dict
        elif dataset_path_obj.is_file():
            ds = load_dataset("json", data_files=str(dataset_path_obj), split="train")
        else:
            validation_files = list(dataset_path_obj.glob("validation.*"))
            if validation_files:
                ds = load_dataset(
                    "json",
                    data_files={"validation": [str(p) for p in validation_files]},
                    split="validation",
                )
            else:
                json_files = list(dataset_path_obj.glob("*.jsonl")) + list(dataset_path_obj.glob("*.json"))
                if json_files:
                    ds = load_dataset(
                        "json",
                        data_files={"validation": [str(p) for p in json_files]},
                        split="validation",
                    )
                else:
                    ds = load_dataset(str(dataset_path_obj), split="validation")
    else:
        ds = load_dataset(dataset_path)["validation"]

    repo_root = Path(dataset_path) if Path(dataset_path).exists() else Path(".")

    def resolve_image(val):
        if isinstance(val, Image.Image):
            return val
        if isinstance(val, dict) and "path" in val:
            return val["path"]
        if isinstance(val, str):
            return str((repo_root / val).resolve())
        return val

    id_to_row = {}
    for row in ds:
        rid = str(row.get("id", row.get("intersection_id", "")))
        if rid:
            id_to_row[rid] = row

    # Filter model rows to those present in the dataset and not yet labeled.
    all_failures = [r for r in model_rows if str(r.get("id")) in id_to_row]
    if edit_mode:
        filtered = all_failures
    else:
        filtered = [r for r in all_failures if str(r.get("id")) not in processed_ids]
    id_to_filtered_idx = {str(r.get("id")): i for i, r in enumerate(filtered)}

    def find_next(start: int) -> int:
        for i in range(start, len(filtered)):
            rid = str(filtered[i].get("id"))
            if rid not in processed_ids:
                return i
        return -1

    idx = find_next(start_idx)
    st.sidebar.markdown("Navigate failures")
    if "nav_idx" not in st.session_state:
        st.session_state["nav_idx"] = idx if idx != -1 else 0
    if filtered:
        st.session_state["nav_idx"] = max(0, min(st.session_state["nav_idx"], len(filtered) - 1))
    nav_cols = st.sidebar.columns([1, 1])
    with nav_cols[0]:
        if st.button("Previous"):
            st.session_state["nav_idx"] = max(0, st.session_state["nav_idx"] - 1)
    with nav_cols[1]:
        if st.button("Next"):
            st.session_state["nav_idx"] = min(len(filtered) - 1, st.session_state["nav_idx"] + 1)

    dropdown_ids = [str(r.get("id")) for r in filtered]
    selected_id = st.sidebar.selectbox("Jump to id", options=dropdown_ids, index=st.session_state["nav_idx"] if dropdown_ids else 0)
    if selected_id in id_to_filtered_idx:
        st.session_state["nav_idx"] = id_to_filtered_idx[selected_id]

    idx = st.session_state["nav_idx"] if filtered else -1
    total_done = len(processed_ids)
    total = len(filtered)
    st.markdown(f"**Progress:** {total_done} / {total}")
    st.progress(total_done / total if total else 0.0)

    if idx == -1:
        st.success("All items completed!")
        return

    model_row = filtered[idx]
    rid = str(model_row.get("id"))

    if st.session_state.get("pending_reset"):
        st.session_state["category_choice"] = None
        st.session_state["comment_input"] = ""
        st.session_state["pending_reset"] = False

    if st.session_state.get("current_rid") != rid:
        st.session_state["current_rid"] = rid
        prev_idx = result_index_map.get(rid)
        if prev_idx is not None:
            prev = state["results"][prev_idx]
            prev_cat = prev.get("category_id")
            if prev_cat:
                st.session_state["category_choice"] = f"{int(prev_cat)}. {FAILURE_CATEGORIES[int(prev_cat) - 1]}"
            else:
                st.session_state["category_choice"] = None
            st.session_state["comment_input"] = prev.get("comment", "")
        else:
            st.session_state["category_choice"] = None
            st.session_state["comment_input"] = ""

    row = id_to_row[rid]
    images = row.get("images")
    image_map = row.get("image_map") or (images[0] if images else None)
    image_sv = row.get("image_sv") or (images[1] if images else None)
    image_map = resolve_image(image_map)
    image_sv = resolve_image(image_sv)

    left, right = st.columns([1, 1])
    with left:
        st.image(image_map, caption="Overhead map", use_column_width=True)
        st.image(image_sv, caption="Street View", use_column_width=True)

    gold = str(model_row.get("ground_truth", "")).strip()
    pred = str(model_row.get("prediction", "")).strip()
    raw_response = model_row.get("raw_response", "")
    with right:
        st.subheader("Model output")
        st.markdown(f"**Gold:** `{gold}`  \n**Prediction:** `{pred}`")
        if raw_response:
            st.markdown("**Model trace**")
            st.markdown(str(raw_response))

    def queue_submit(category_idx: int):
        st.session_state["pending_category"] = category_idx

    def on_choice_change():
        value = st.session_state.get("category_choice")
        if value:
            try:
                idx = int(str(value).split(".", 1)[0].strip())
            except Exception:
                return
            queue_submit(idx)

    def on_shortcut_change():
        key = (st.session_state.get("shortcut_key") or "").strip()
        if key.isdigit():
            val = int(key)
            if 1 <= val <= len(FAILURE_CATEGORIES):
                queue_submit(val)
        st.session_state["shortcut_key"] = ""

    st.subheader("Failure category")
    options = [f"{i + 1}. {name}" for i, name in enumerate(FAILURE_CATEGORIES)]
    st.radio(
        "Select one category (1-7):",
        options=options,
        index=None,
        key="category_choice",
        on_change=on_choice_change,
    )
    st.text_input(
        "Shortcut (press 1-7, then Enter):",
        value="",
        key="shortcut_key",
        on_change=on_shortcut_change,
        max_chars=1,
    )
    components.html(
        """
        <script>
        const el = parent.document.querySelector('input[aria-label="Shortcut (press 1-7, then Enter):"]');
        if (el) { el.focus(); }
        </script>
        """,
        height=0,
    )
    st.text_area("Optional comment", value="", key="comment_input", height=120)

    pending = st.session_state.get("pending_category")
    last_key = "last_submitted_id"
    if pending and st.session_state.get(last_key) != rid:
        entry = {
            "id": rid,
            "category_id": pending,
            "category": FAILURE_CATEGORIES[pending - 1],
            "comment": st.session_state.get("comment_input", ""),
            "model_prediction": pred,
            "gold": gold,
            "model_trace": raw_response,
            "model_csv": str(Path(model_csv_path)),
            "timestamp": time.time(),
        }
        existing_idx = result_index_map.get(rid)
        if existing_idx is None:
            state.setdefault("results", []).append(entry)
        else:
            state["results"][existing_idx] = entry
        save_results(out_path, state)
        st.session_state[last_key] = rid
        st.session_state["pending_category"] = None
        st.session_state["pending_reset"] = True
        st.rerun()

    st.json({"id": rid})


if __name__ == "__main__":
    main()
