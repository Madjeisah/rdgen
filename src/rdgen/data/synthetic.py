"""Synthetic domain with known ground truth.

Used to (a) unit-test the estimators, (b) run the whole pipeline end to end
before real data is available, and (c) feed the power simulation.

Each document describes one record with K binary attributes and M valued
attributes. Two things differ between source and target domains, mirroring
the two mechanisms in the theory section:

* prevalence of each binary attribute (the validity of priors shifts), and
* how often absence is written down explicitly (documentation convention).

Gold labels follow a closed-world reading: an attribute that is not
mentioned is "no" (implicit_neg). Explicitly negated attributes are
explicit_neg. Each binary item also carries a ``flip`` entry so the
substitution estimator can be tested on categorical answers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..schema import CATEGORICAL_TYPE, NO, SPAN_TYPE, TEST_SHIFT, TRAIN, YES, Item
from . import group_split, register

_FILLER = [
    "The record was reviewed at the scheduled visit.",
    "Routine observations were documented by the team.",
    "Follow-up was arranged according to the usual protocol.",
    "The summary was prepared at the end of the episode.",
    "No changes to the administrative details were noted.",
    "The case was discussed at the weekly meeting.",
    "Previous documents were consulted for context.",
    "The plan was explained and agreed.",
]
_VALUES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


@dataclass
class DomainSpec:
    name: str
    prevalence: list[float]                 # P(attribute present), one per binary attribute
    explicit_neg_rate: float = 0.5          # P(absence is written down)
    value_weights: list[list[float]] = field(default_factory=list)  # per valued attribute
    filler_units: int = 6
    style: str = "plain"


def default_specs(k: int = 6, m: int = 2, flip_fraction: float = 0.5) -> tuple[DomainSpec, DomainSpec]:
    """Source and target specs whose priors and conventions differ.

    Source prevalences are skewed (so a prior is informative in-domain); in the
    target, ``flip_fraction`` of attributes have their prevalence mirrored, so
    the source prior becomes wrong on them.
    """
    rng = random.Random(7)
    src_prev = [rng.choice([rng.uniform(0.1, 0.25), rng.uniform(0.75, 0.9)]) for _ in range(k)]
    n_flip = int(round(flip_fraction * k))
    tgt_prev = [1 - p if i < n_flip else p for i, p in enumerate(src_prev)]
    w_src = [[8, 4, 2, 1, 1, 1, 1, 1] for _ in range(m)]
    w_tgt = [[1, 1, 2, 4, 8, 1, 1, 1] for _ in range(m)]
    return (DomainSpec("source", src_prev, 0.6, w_src),
            DomainSpec("target", tgt_prev, 0.25, w_tgt, style="terse"))


def _attr_sentence(k: int, present: bool, style: str) -> str:
    if style == "terse":
        return f"Finding A{k}: {'positive' if present else 'negative'}."
    return f"Attribute A{k} is present." if present else f"There is no evidence of attribute A{k}."


def _value_sentence(j: int, v: str, style: str) -> str:
    return f"V{j} = {v}." if style == "terse" else f"The value of V{j} is {v}."


def make_record(rid: str, spec: DomainSpec, split: str, rng: random.Random,
                domain: str = "synthetic", repeat_rate: float = 0.25) -> list[Item]:
    units: list[str] = []
    facts: list[tuple[str, int, object]] = []
    for k, p in enumerate(spec.prevalence):
        present = rng.random() < p
        written = present or rng.random() < spec.explicit_neg_rate
        facts.append(("bin", k, (present, written)))
    for j, w in enumerate(spec.value_weights):
        facts.append(("val", j, rng.choices(_VALUES, weights=w)[0]))
    # interleave facts and filler; some facts are repeated (redundant evidence)
    pieces: list[tuple[str, int | None]] = []
    for kind, k, val in facts:
        if kind == "bin":
            present, written = val
            if written:
                s = _attr_sentence(k, present, spec.style)
                pieces.append((s, k))
                if rng.random() < repeat_rate:
                    pieces.append((s.replace("Attribute", "Again, attribute") if spec.style == "plain" else s, k))
        else:
            s = _value_sentence(k, val, spec.style)
            pieces.append((s, 100 + k))
            if rng.random() < repeat_rate:
                pieces.append((s, 100 + k))
    pieces += [(rng.choice(_FILLER), None) for _ in range(spec.filler_units)]
    rng.shuffle(pieces)
    units = [s for s, _ in pieces]
    keys = [k for _, k in pieces]

    items: list[Item] = []
    for kind, k, val in facts:
        if kind == "bin":
            present, written = val
            ev = [i for i, kk in enumerate(keys) if kk == k]
            answer = YES if present else NO
            pol = "pos" if present else ("explicit_neg" if written else "implicit_neg")
            flip = None
            if ev:
                flip = {"units": {i: _attr_sentence(k, not present, spec.style) for i in ev},
                        "answer": NO if present else YES}
            items.append(Item(
                item_id=f"{domain}-{rid}-A{k}", domain=domain, split=split, group_id=rid,
                units=units, question=f"Is attribute A{k} present?", answer=answer,
                answer_type=CATEGORICAL_TYPE, evidence=ev, polarity=pol,
                meta={"template": f"A{k}", "flip": flip, "spec": spec.name},
            ))
        else:
            ev = [i for i, kk in enumerate(keys) if kk == 100 + k]
            items.append(Item(
                item_id=f"{domain}-{rid}-V{k}", domain=domain, split=split, group_id=rid,
                units=units, question=f"What is the value of V{k}?", answer=str(val),
                answer_type=SPAN_TYPE, evidence=ev, polarity="span",
                meta={"template": f"V{k}", "spec": spec.name},
            ))
    return items


@register("synthetic")
def load_synthetic(n_source: int = 400, n_target: int = 200, seed: int = 0,
                   specs: tuple[DomainSpec, DomainSpec] | None = None,
                   domain: str = "synthetic") -> list[Item]:
    src, tgt = specs or default_specs()
    rng = random.Random(seed)
    items: list[Item] = []
    for r in range(n_source):
        items += make_record(f"s{r}", src, TRAIN, rng, domain)
    group_split(items, seed=seed)
    for r in range(n_target):
        items += make_record(f"t{r}", tgt, TEST_SHIFT, rng, domain)
    return items
