"""ContractNLI (legal domain, NDAs).

Units are the annotated contract spans. Each (document, hypothesis) pair
becomes a question. Entailment -> yes (pos), Contradiction -> no
(explicit_neg), NotMentioned -> not_stated (unanswerable).

Shift: the dataset mixes NDAs from EDGAR and from web search. Whether the
released metadata separates them reliably must be checked before use; the
``source_of`` function is configurable for that reason. If no reliable source
split exists, fall back to ``shift='hypothesis'``: hold out a set of
hypotheses (question types) as the shifted test.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from ..schema import CATEGORICAL_TYPE, NO, NOT_STATED, TEST_SHIFT, TRAIN, YES, Item
from ..text import stable_id
from . import group_split, register

_LABEL = {"Entailment": (YES, "pos"), "Contradiction": (NO, "explicit_neg"),
          "NotMentioned": (NOT_STATED, "unanswerable")}


def default_source_of(doc: dict) -> str:
    """Heuristic only: EDGAR filings usually carry accession-style names.
    Verify against the data before relying on it."""
    name = doc.get("file_name", "")
    return "edgar" if re.search(r"\d{10}-\d{2}-\d{6}|ex-?\d+", name, re.I) else "web"


def _read(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["documents"], data["labels"]


@register("contractnli")
def load_contractnli(files: list[str], shift: str = "source", train_source: str = "edgar",
                     heldout_hypotheses: list[str] | None = None,
                     source_of: Callable[[dict], str] = default_source_of,
                     seed: int = 13) -> list[Item]:
    items: list[Item] = []
    for fp in files:
        docs, labels = _read(Path(fp))
        for doc in docs:
            text = doc["text"]
            units = [" ".join(text[s:e].split()) for s, e in doc["spans"]]
            src = source_of(doc)
            gid = stable_id("contractnli", str(doc["id"]))
            for key, ann in doc["annotation_sets"][0]["annotations"].items():
                answer, pol = _LABEL[ann["choice"]]
                hyp = labels[key]["hypothesis"]
                if shift == "source":
                    split = TRAIN if src == train_source else TEST_SHIFT
                else:
                    split = TEST_SHIFT if key in (heldout_hypotheses or []) else TRAIN
                items.append(Item(
                    item_id=f"contractnli-{doc['id']}-{key}", domain="contractnli", split=split,
                    group_id=gid, units=units,
                    question=f"Does the contract state the following? {hyp}",
                    answer=answer, answer_type=CATEGORICAL_TYPE,
                    evidence=sorted(ann.get("spans", [])), polarity=pol,
                    meta={"source": src, "hypothesis_key": key},
                ))
    return group_split(items, seed=seed)
