"""Augmentation pools for the gate ablation (H2).

In the real study the pool comes from a generator: LLM-written synthetic
records for unseen cancer types, questions generated on unlabelled target-site
text, or English evidence spans projected into Romanian. Any generator works
as long as it writes ``Item`` JSONL (see ``schema.py``).

``prior_contaminated_pool`` builds a controlled pool for tests and the demo.
It mixes two kinds of item, mirroring the failure mode the necessity gate is
meant to remove:

* record-derived items: the document states the answer, so the answer is only
  recoverable by reading;
* prior-consistent items: the answer equals the source-domain majority answer
  and the supporting sentence is absent, so a model learns "answer the usual
  thing" - accurate in-domain, harmful under shift.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Sequence

from ..perturb import delete
from ..schema import TRAIN, Item


def prior_contaminated_pool(items: Sequence[Item], n: int = 400, contamination: float = 0.5,
                            seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    train = [it for it in items if it.split == TRAIN and it.evidence and it.polarity in ("pos", "span")]
    prior: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        if it.split == TRAIN:
            prior[it.meta.get("template", it.question)][it.answer] += 1
    pool: list[Item] = []
    for k in range(n):
        src = rng.choice(train)
        if rng.random() < contamination:
            tmpl = src.meta.get("template", src.question)
            majority = prior[tmpl].most_common(1)[0][0]
            it = delete(src, src.evidence, "pool_prior")
            it.answer = majority
            it.polarity = "implicit_neg" if majority == "no" else "unknown_neg"
            it.meta = {**it.meta, "pool_kind": "prior_consistent"}
        else:
            it = src.with_units(src.units, src.evidence, pool_kind="record_derived")
        it.item_id = f"{src.item_id}#pool{k}"
        it.split = TRAIN
        pool.append(it)
    return pool
