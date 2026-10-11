"""PubMedQA expert-labelled subset (``ori_pqal.json``).

Context = abstract without its conclusion, split into sentences. No gold
evidence units exist, so the support closure relies on the verifier
(``closure_mode='full'``). ``maybe`` is treated as uncertain and excluded from
record-dependence.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import CATEGORICAL_TYPE, NO, NOT_STATED, TEST_SHIFT, YES, Item
from ..text import split_sentences, stable_id
from . import register

_MAP = {"yes": (YES, "pos"), "no": (NO, "explicit_neg"), "maybe": (NOT_STATED, "uncertain")}


@register("pubmedqa")
def load_pubmedqa(path: str, split: str = TEST_SHIFT, max_items: int | None = None) -> list[Item]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[Item] = []
    for pmid, row in data.items():
        units = [s for ctx in row["CONTEXTS"] for s in split_sentences(ctx)]
        answer, pol = _MAP[row["final_decision"]]
        items.append(Item(
            item_id=f"pubmedqa-{pmid}", domain="pubmedqa", split=split,
            group_id=stable_id("pubmedqa", pmid), units=units, question=row["QUESTION"],
            answer=answer, answer_type=CATEGORICAL_TYPE, evidence=[], polarity=pol,
            meta={"pmid": pmid, "needs_verifier_closure": True},
        ))
        if max_items and len(items) >= max_items:
            break
    return items
