"""Training regimes of the model population (protocol step 1).

  zeroshot      no training
  sft           answer-only supervised fine-tuning
  sft_evidence  target = evidence unit ids, then the answer
  abstain_aug   sft + counterfactual copies with the evidence deleted and the
                gold answer set to not_stated (teaches record-dependence)
  gated         sft + an augmentation pool filtered by the necessity gate
  retrieval     sft with a retrieval context strategy (set in the model config)

``abstain_aug`` and ``gated`` are the levers used to build accuracy-matched
model pairs that differ in record-dependence (the interventional test).
"""
from __future__ import annotations

import random
from typing import Sequence

from ..perturb import delete, support_closure
from ..schema import CATEGORICAL_TYPE, NOT_STATED, TRAIN, Item

REGIMES = ("zeroshot", "sft", "sft_evidence", "abstain_aug", "gated", "retrieval")


def counterfactual_abstentions(items: Sequence[Item], rate: float = 0.3, seed: int = 0,
                               closure_mode: str = "gold") -> list[Item]:
    """Evidence-deleted copies labelled not_stated (positive and span items)."""
    rng = random.Random(seed)
    out = []
    for it in items:
        if it.polarity not in ("pos", "span") or rng.random() >= rate:
            continue
        clo = support_closure(it, closure_mode)
        if not clo:
            continue
        cf = delete(it, clo, "cf_abstain")
        cf.item_id = it.item_id + "#cf"
        cf.answer = NOT_STATED
        cf.answer_type = CATEGORICAL_TYPE if it.answer_type == CATEGORICAL_TYPE else it.answer_type
        cf.polarity = "unanswerable"
        out.append(cf)
    return out


def build_training_set(items: Sequence[Item], regime: str, aug_pool: Sequence[Item] = (),
                       cf_rate: float = 0.3, seed: int = 0) -> list[Item]:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    train = [it for it in items if it.split == TRAIN]
    if regime == "zeroshot":
        return []
    if regime in ("sft", "sft_evidence", "retrieval"):
        return train
    if regime == "abstain_aug":
        return train + counterfactual_abstentions(train, cf_rate, seed)
    if regime == "gated":
        return train + list(aug_pool)
    raise AssertionError
