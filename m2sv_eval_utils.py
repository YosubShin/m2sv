"""Shared helpers for M2SV evaluation prompts and answer parsing."""

from __future__ import annotations

import re
from typing import List

_PROMPT_INSTRUCTIONS = (
    "You will be given two images: (1) a north-up overhead map with arrows labeled A, B, C, ... "
    "and (2) a street-view photo.\n"
    "Rules:\n"
    "- The camera location is the same for all options: the center of the intersection.\n"
    "- Each letter corresponds to facing outward from that center along the arrow of that label.\n"
    "- The small circles near labels are markers only; they are not camera locations.\n"
    "- The map and photo may be captured years apart. Ignore transient objects (cars, people).\n"
    "Think step by step to compare the street-view with the map (buildings, angles, lanes, landmarks).\n"
    "On the final line, output only: Final answer: \\boxed{X} where X is a single letter (A, B, C, ...)."
)


def format_prompt(question: str, options: List[str]) -> str:
    """Standardized instructions + question for VLM evaluations."""
    question = (question or "").strip()
    return f"{_PROMPT_INSTRUCTIONS}\n\n{question}"


def normalize_letter(text: str, num_options: int) -> str:
    """Return a single option letter if confidently present in the text."""
    if text is None:
        return ""
    t = text.strip()
    if not t:
        return ""

    def is_valid_letter(ch: str) -> str:
        if not ch:
            return ""
        ch_u = ch.upper()
        idx = ord(ch_u) - ord("A")
        return ch_u if 0 <= idx < num_options else ""

    # 1) Exact single letter
    match_single = re.fullmatch(r"\s*([A-Za-z])\s*", t)
    if match_single:
        ch = is_valid_letter(match_single.group(1))
        if ch:
            return ch

    # 2) Boxed / styled letters (allow boxed/fbox and latex delimiters)
    boxed_pattern = (
        r"(?:\\\(|\\\[|\$)?\s*(?:\\boxed|\\fbox)\s*\{\s*([A-Za-z])\s*\}\s*(?:\\\)|\\\]|\$)?"
    )
    boxed_candidates: List[str] = [
        m.group(1) for m in re.finditer(boxed_pattern, t, flags=re.IGNORECASE)
    ]
    for raw in reversed(boxed_candidates):
        ch = is_valid_letter(raw)
        if ch:
            return ch

    # 2b) Repeated-letter outputs like "C. C" or "B B"
    match_repeat = re.fullmatch(r"\s*([A-Za-z])\s*[\.-:;,]?\s*\1\s*\.?\s*", t)
    if match_repeat:
        ch = is_valid_letter(match_repeat.group(1))
        if ch:
            return ch

    # 3) Explicit answer phrases anywhere in the text (prefer the last mention)
    styled_letter = r"[\s`*_~\(\[\{]*([A-Za-z])[\s`*_~\)\]\}]*"
    explicit_patterns = [
        rf"(?:\bthe\s+answer\b|\banswer\b)\s*(?:is\s*[:=]?|[:=])\s*{styled_letter}\b",
        rf"\bfinal\s*(?:answer)?\s*(?:is\s*[:=]?|[:=])\s*{styled_letter}\b",
        rf"\bfinal\s*answer\s*[:=]\s*{styled_letter}\b",
    ]
    explicit_candidates: List[str] = []
    for pattern in explicit_patterns:
        explicit_candidates.extend(
            m.group(1) for m in re.finditer(pattern, t, flags=re.IGNORECASE)
        )
    for raw in reversed(explicit_candidates):
        ch = is_valid_letter(raw)
        if ch:
            return ch

    # 4) Inspect the last non-empty line for a styled single letter
    lines = [line.strip() for line in t.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        for pattern in explicit_patterns:
            match_last = re.search(pattern, last_line, flags=re.IGNORECASE)
            if match_last:
                ch = is_valid_letter(match_last.group(1))
                if ch:
                    return ch

        match_last_repeat = re.fullmatch(
            r"\s*([A-Za-z])\s*[\.-:;,]?\s*\1\s*\.?\s*", last_line
        )
        if match_last_repeat:
            ch = is_valid_letter(match_last_repeat.group(1))
            if ch:
                return ch

        stripped = re.sub(r"[\s\*`_~\-–—\(\)\[\]\{\}\"'.:;,!]+", "", last_line)
        if re.fullmatch(r"[A-Za-z]", stripped):
            ch = is_valid_letter(stripped)
            if ch:
                return ch

    # 5) Weaker fallback: choose/option/arrow phrasing without negation context
    ambiguous_patterns = [
        r"\bchoose\s*([A-Za-z])\b",
        r"\b(?:option|choice|arrow)\s*([A-Za-z])\b",
    ]
    last_candidate = ""
    for pattern in ambiguous_patterns:
        for match in re.finditer(pattern, t, flags=re.IGNORECASE):
            start = match.start()
            context = t[max(0, start - 50):start].lower()
            if any(
                neg in context
                for neg in [
                    "eliminate",
                    "eliminates",
                    "eliminated",
                    "eliminating",
                    "not ",
                    "isn't",
                    "is not",
                    "avoid",
                    "eliminates option",
                    "eliminate option",
                ]
            ):
                continue
            ch = is_valid_letter(match.group(1))
            if ch:
                last_candidate = ch
    if last_candidate:
        return last_candidate

    return ""

