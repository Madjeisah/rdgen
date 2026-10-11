"""Qasper (scientific domain, NLP papers).

Units are paragraphs. Kept answer types: yes/no, unanswerable and extractive.
Free-form (abstractive) answers are skipped because correctness cannot be
scored reliably without a judge. The first annotator's answer is used.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import CATEGORICAL_TYPE, NO, NOT_STATED, SPAN_TYPE, TEST_SHIFT, TRAIN, YES, Item
from ..text import stable_id
from . import register

_TABLE_PREFIX = "FLOAT SELECTED"


def _paper_units(paper: dict) -> list[str]:
    units = [paper.get("abstract", "")]
    for sec in paper.get("full_text", []):
        for para in sec.get("paragraphs", []) or []:
            para = " ".join(para.split())
            if para:
                units.append(para)
    return [u for u in units if u]


def _items_from_file(path: Path, split: str, max_items: int | None) -> list[Item]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[Item] = []
    for pid, paper in data.items():
        units = _paper_units(paper)
        index = {u: k for k, u in enumerate(units)}
        for qa in paper.get("qas", []):
            if not qa.get("answers"):
                continue
            ans = qa["answers"][0]["answer"]
            evidence = [index[" ".join(e.split())] for e in ans.get("evidence", [])
                        if not e.startswith(_TABLE_PREFIX) and " ".join(e.split()) in index]
            if ans.get("unanswerable"):
                answer, atype, pol = NOT_STATED, CATEGORICAL_TYPE, "unanswerable"
            elif ans.get("yes_no") is not None:
                answer = YES if ans["yes_no"] else NO
                atype = CATEGORICAL_TYPE
                pol = "pos" if answer == YES else ("explicit_neg" if evidence else "implicit_neg")
            elif ans.get("extractive_spans"):
                answer, atype, pol = ans["extractive_spans"][0], SPAN_TYPE, "span"
            else:
                continue
            if pol in ("pos", "explicit_neg", "span") and not evidence:
                pol = "unknown_neg" if answer == NO else pol
            items.append(Item(
                item_id=f"qasper-{qa['question_id']}", domain="qasper", split=split,
                group_id=stable_id("qasper", pid), units=units, question=qa["question"],
                answer=answer, answer_type=atype, evidence=evidence, polarity=pol,
                meta={"paper_id": pid},
            ))
            if max_items and len(items) >= max_items:
                return items
    return items


@register("qasper")
def load_qasper(train: str, valid: str, test: str, shift_items: list[Item] | None = None,
                max_items: int | None = None) -> list[Item]:
    """Official Qasper train/dev/test JSON. Shifted test comes from another
    loader (PubMedQA or BioASQ) and is passed in ``shift_items``."""
    from ..schema import TEST_ID, VALID
    items = _items_from_file(Path(train), TRAIN, max_items)
    items += _items_from_file(Path(valid), VALID, max_items)
    items += _items_from_file(Path(test), TEST_ID, max_items)
    for it in shift_items or []:
        it.split = TEST_SHIFT
        items.append(it)
    return items
