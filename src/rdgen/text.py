"""Sentence splitting and light text utilities (no external NLP dependency)."""
from __future__ import annotations

import hashlib
import re

_ABBREV = {"e.g", "i.e", "dr", "mr", "mrs", "ms", "vs", "etc", "fig", "al", "no", "nr", "st"}
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9ĂÂÎȘȚ\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Rule-based splitter that keeps common abbreviations intact."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SPLIT.split(text)
    out: list[str] = []
    for p in parts:
        if out:
            prev_last = out[-1].rstrip(".").split(" ")[-1].lower()
            if prev_last in _ABBREV:
                out[-1] = f"{out[-1]} {p}"
                continue
        out.append(p)
    return [s.strip() for s in out if s.strip()]


def char_spans_to_units(text: str, units: list[str]) -> list[tuple[int, int]]:
    """Locate each unit's char offsets in ``text`` (in order)."""
    spans, cursor = [], 0
    flat = re.sub(r"\s+", " ", text)
    for u in units:
        i = flat.find(u, cursor)
        if i < 0:
            i = cursor
        spans.append((i, i + len(u)))
        cursor = i + len(u)
    return spans


def stable_id(*parts: str) -> str:
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:12]
