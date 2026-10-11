"""Answer normalisation and correctness (methodology 10.1).

Lexical-only scoring penalised format, not knowledge, in the MedQARo paper.
Here categorical answers are mapped to canonical labels before comparison and
span answers use SQuAD-style normalisation with token F1.
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

from .schema import CATEGORICAL_TYPE, NO, NOT_STATED, YES

# Surface forms mapped to canonical labels. Romanian forms cover MedQARo
# ("Da", "Nu", "NK" = not known).
_LABEL_MAP = {
    YES: {"yes", "y", "true", "da", "present", "entailment", "entailed", "met"},
    NO: {"no", "n", "false", "nu", "absent", "contradiction", "contradicted", "not met"},
    NOT_STATED: {
        "not_stated", "not stated", "not mentioned", "notmentioned", "unknown",
        "unanswerable", "nk", "maybe", "cannot be determined", "insufficient information",
        "nu se stie", "nu se știe", "neprecizat",
    },
}
_SURFACE_TO_LABEL = {s: lab for lab, forms in _LABEL_MAP.items() for s in forms}

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_text(s: str) -> str:
    s = strip_accents(s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def to_label(s: str) -> str | None:
    """Map free text to a canonical categorical label, or None if unclear."""
    t = normalize_text(s.replace("_", " "))
    if t in _SURFACE_TO_LABEL:
        return _SURFACE_TO_LABEL[t]
    first = t.split(" ")[0] if t else ""
    if first in _SURFACE_TO_LABEL and first in {"yes", "no", "da", "nu"}:
        return _SURFACE_TO_LABEL[first]
    for form, lab in _SURFACE_TO_LABEL.items():
        if len(form) > 3 and form in t:
            return lab
    return None


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def is_correct(pred: str, gold: str, answer_type: str, f1_threshold: float = 1.0) -> bool:
    """Binary correctness c(f, R, q).

    Categorical: canonical labels must match. Span: normalised exact match, or
    token F1 >= ``f1_threshold`` when a softer criterion is pre-registered.
    """
    if answer_type == CATEGORICAL_TYPE:
        return to_label(pred) == gold
    if to_label(pred) == NOT_STATED and to_label(gold) == NOT_STATED:
        return True
    if normalize_text(pred) == normalize_text(gold):
        return True
    return f1_threshold < 1.0 and token_f1(pred, gold) >= f1_threshold


def same_answer(a: str, b: str, answer_type: str) -> bool:
    """Whether two predictions are the same answer (used for 'answer changed')."""
    if answer_type == CATEGORICAL_TYPE:
        return to_label(a) == to_label(b)
    la, lb = to_label(a), to_label(b)
    if la == NOT_STATED or lb == NOT_STATED:
        return la == lb
    return normalize_text(a) == normalize_text(b)


def is_abstention(pred: str) -> bool:
    return to_label(pred) == NOT_STATED
