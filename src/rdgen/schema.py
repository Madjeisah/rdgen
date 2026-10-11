"""Unified item schema shared by every domain.

Every dataset is converted to ``Item`` records: a document split into units
(sentences, paragraphs or contract spans), a question, a gold answer, the
indices of gold evidence units, and a polarity class that decides how the
record-dependence estimators treat the item (see ``POLARITY``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# Categorical answer labels used across domains.
YES, NO, NOT_STATED = "yes", "no", "not_stated"
CATEGORICAL = (YES, NO, NOT_STATED)

# Answer types.
CATEGORICAL_TYPE = "categorical"   # answer is one of CATEGORICAL
SPAN_TYPE = "span"                  # answer is a short free-text span

# Polarity classes (methodology section 10.4).
#   pos           : answer supported by explicit positive evidence  -> in RD
#   explicit_neg  : answer "no" supported by an explicit negation   -> in RD
#   implicit_neg  : answer "no" because nothing is mentioned        -> excluded from RD
#   unanswerable  : gold answer not_stated                          -> abstention analysis only
#   uncertain     : hedged gold label (possible / maybe)            -> excluded from RD
#   span          : span answer with evidence                       -> in RD (prefer substitution)
#   unknown_neg   : "no" whose evidence type is not annotated       -> excluded, flagged
POLARITY = ("pos", "explicit_neg", "implicit_neg", "unanswerable", "uncertain", "span", "unknown_neg")
RD_POLARITIES = ("pos", "explicit_neg", "span")

# Splits.
TRAIN, VALID, TEST_ID, TEST_SHIFT = "train", "valid", "test_id", "test_shift"
SPLITS = (TRAIN, VALID, TEST_ID, TEST_SHIFT)


@dataclass
class Item:
    item_id: str
    domain: str
    split: str
    group_id: str            # patient / document id: the bootstrap cluster
    units: list[str]         # document units in order
    question: str
    answer: str              # categorical label or span text
    answer_type: str         # CATEGORICAL_TYPE or SPAN_TYPE
    evidence: list[int] = field(default_factory=list)
    polarity: str = "pos"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.polarity not in POLARITY:
            raise ValueError(f"unknown polarity {self.polarity!r} for {self.item_id}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r} for {self.item_id}")
        if self.answer_type == CATEGORICAL_TYPE and self.answer not in CATEGORICAL:
            raise ValueError(f"categorical answer {self.answer!r} not in {CATEGORICAL}")
        bad = [i for i in self.evidence if not 0 <= i < len(self.units)]
        if bad:
            raise ValueError(f"evidence indices {bad} out of range for {self.item_id}")

    @property
    def in_rd(self) -> bool:
        return self.polarity in RD_POLARITIES

    def with_units(self, units: list[str], evidence: list[int] | None = None, **meta) -> "Item":
        """Copy with a new document (used by perturbations)."""
        new_meta = dict(self.meta)
        new_meta.update(meta)
        return Item(
            item_id=self.item_id, domain=self.domain, split=self.split,
            group_id=self.group_id, units=list(units), question=self.question,
            answer=self.answer, answer_type=self.answer_type,
            evidence=list(evidence if evidence is not None else []),
            polarity=self.polarity, meta=new_meta,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**d)


def write_jsonl(items: Iterable[Item], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Item]:
    with Path(path).open(encoding="utf-8") as f:
        return [Item.from_dict(json.loads(line)) for line in f if line.strip()]


def iter_split(items: Iterable[Item], split: str) -> Iterator[Item]:
    return (it for it in items if it.split == split)


def summarize(items: list[Item]) -> dict:
    """Counts by split x polarity, for the data statement."""
    out: dict[str, dict[str, int]] = {}
    for it in items:
        out.setdefault(it.split, {}).setdefault(it.polarity, 0)
        out[it.split][it.polarity] += 1
    return out
