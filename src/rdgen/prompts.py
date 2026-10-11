"""Prompt templates. Question first, then document (MedQARo found Q+E+A beat
E+Q+A), then an answer slot. One template per answer type and language."""
from __future__ import annotations

from .schema import CATEGORICAL_TYPE, Item

_SYSTEM = {
    "en": ("Answer the question using only the document. If the document does not "
           "contain the information, answer 'not stated'."),
    "ro": ("Răspunde la întrebare folosind doar documentul. Dacă documentul nu conține "
           "informația, răspunde 'NK'."),
}
_CAT_HINT = {"en": "Answer with yes, no, or not stated.", "ro": "Răspunde cu Da, Nu sau NK."}
_SPAN_HINT = {"en": "Answer with a short phrase copied from the document, or 'not stated'.",
              "ro": "Răspunde cu o expresie scurtă din document sau 'NK'."}
_EVID_HINT = {"en": "First list the numbers of the supporting sentences after 'Evidence:', then give the answer after 'Answer:'.",
              "ro": "Mai întâi enumeră numerele propozițiilor relevante după 'Evidence:', apoi răspunsul după 'Answer:'."}

LABEL_WORDS = {
    "en": {"yes": "yes", "no": "no", "not_stated": "not stated"},
    "ro": {"yes": "Da", "no": "Nu", "not_stated": "NK"},
}


def build_prompt(item: Item, context: str, lang: str = "en", evidence_first: bool = False) -> str:
    hint = _CAT_HINT[lang] if item.answer_type == CATEGORICAL_TYPE else _SPAN_HINT[lang]
    if evidence_first:
        hint = f"{hint} {_EVID_HINT[lang]}"
    return (f"{_SYSTEM[lang]}\n\nQuestion: {item.question}\n\nDocument:\n{context}\n\n"
            f"{hint}\n")


def target_text(item: Item, lang: str = "en", evidence_first: bool = False) -> str:
    ans = LABEL_WORDS[lang].get(item.answer, item.answer) if item.answer_type == CATEGORICAL_TYPE else item.answer
    if item.answer == "not_stated":
        ans = LABEL_WORDS[lang]["not_stated"]
    if evidence_first:
        ev = ", ".join(str(i) for i in item.evidence) or "none"
        return f"Evidence: {ev}\nAnswer: {ans}"
    return f"Answer: {ans}"
