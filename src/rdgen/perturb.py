"""Document perturbations (methodology 10.2-10.3).

* ``support_closure``   - every unit that supports the answer, so repeated
                          facts are all removed (redundancy-aware deletion).
* ``delete``            - remove a set of units.
* ``control_indices``   - a random, position-matched set of non-supporting
                          units of the same size (the deletion control).
* ``sufficiency_item``  - evidence alone, padded with filler from other
                          documents to the original length.
* ``substitute``        - change the answer-bearing value in every mention and
                          record the answer the document now implies.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

from .normalize import normalize_text
from .schema import CATEGORICAL_TYPE, SPAN_TYPE, Item
from .verify import Verifier

CLOSURE_MODES = ("top1", "gold", "full")


def support_closure(item: Item, mode: str = "full", verifier: Verifier | None = None,
                    tau: float = 0.5, retrieve_k: int = 8) -> list[int]:
    """Indices of units supporting the gold answer.

    top1 : first gold evidence unit only (the naive ablation)
    gold : all gold evidence units
    full : gold + units containing the span answer + verifier-positive units
           among the top ``retrieve_k`` units retrieved for the question
    """
    if mode not in CLOSURE_MODES:
        raise ValueError(mode)
    gold = sorted(set(item.evidence))
    if mode == "top1":
        return gold[:1]
    if mode == "gold":
        return gold
    closure = set(gold)
    if item.answer_type == SPAN_TYPE:
        a = normalize_text(item.answer)
        if a:
            pat = re.compile(rf"(?<!\w){re.escape(a)}(?!\w)")
            closure |= {i for i, u in enumerate(item.units) if pat.search(normalize_text(u))}
    if verifier is not None and item.units:
        from .context import retrieve
        cand = retrieve(item.units, f"{item.question} {item.answer}", retrieve_k)
        scores = verifier.support([item.units[i] for i in cand], item.question, item.answer)
        closure |= {i for i, s in zip(cand, scores) if s >= tau}
    return sorted(closure)


def delete(item: Item, idx: Sequence[int], tag: str) -> Item:
    drop = set(idx)
    units = [u for i, u in enumerate(item.units) if i not in drop]
    remap = {}
    n = 0
    for i in range(len(item.units)):
        if i not in drop:
            remap[i] = n
            n += 1
    ev = [remap[i] for i in item.evidence if i in remap]
    return item.with_units(units, ev, perturbation=tag, deleted=sorted(drop))


@dataclass
class ControlDraw:
    indices: list[int]
    shortfall: int            # how many could not be matched (too few non-support units)


def control_indices(item: Item, closure: Sequence[int], rng: random.Random) -> ControlDraw:
    """Random non-supporting units, matched in number and approximate position."""
    n = len(item.units)
    pool = [i for i in range(n) if i not in set(closure)]
    chosen: list[int] = []
    for c in closure:
        if not pool:
            break
        # nearest available positions, random tie-break
        dist = sorted(pool, key=lambda j: (abs(j - c), rng.random()))
        pick = dist[0] if rng.random() < 0.5 or len(dist) == 1 else dist[min(1, len(dist) - 1)]
        chosen.append(pick)
        pool.remove(pick)
    return ControlDraw(sorted(chosen), len(closure) - len(chosen))


def sufficiency_item(item: Item, closure: Sequence[int], filler_pool: Sequence[str],
                     rng: random.Random) -> Item:
    """Evidence units at their original positions; everything else replaced by
    filler from other documents, so length and position are preserved."""
    keep = set(closure)
    units = [u if i in keep else rng.choice(filler_pool) for i, u in enumerate(item.units)]
    return item.with_units(units, sorted(keep), perturbation="sufficiency")


def build_filler_pool(items: Sequence[Item], max_units: int = 5000, seed: int = 0) -> list[str]:
    """Units from documents of the same domain that are evidence for *no*
    question about their document. A unit that answers another question about
    the same record is not neutral filler: it would plant a competing fact."""
    rng = random.Random(seed)
    evidence_text: set[str] = set()
    docs: dict[tuple[str, int], list[str]] = {}
    for it in items:
        evidence_text.update(it.units[i] for i in it.evidence)
        docs.setdefault((it.group_id, len(it.units)), it.units)
    pool = [u for units in docs.values() for u in units if u not in evidence_text]
    rng.shuffle(pool)
    return pool[:max_units]


@dataclass
class Substitution:
    item: Item
    expected: str


_MONTHS = ("january february march april may june july august september october "
           "november december jan feb mar apr jun jul aug sep sept oct nov dec").split()
_NUMWORDS = set("zero one two three four five six seven eight nine ten eleven twelve "
                "hundred thousand million billion first second third".split())


def answer_shape(ans: str) -> str:
    """Coarse type of a span answer, used to pick a plausible substitute.

    A substitute of a different type ("1997" -> "the Treaty of Paris") makes the
    edited document incoherent, and a model that declines to repeat it is not
    thereby shown to be guessing. Matching the type keeps substitution a test of
    reading.
    """
    a = ans.strip()
    low = a.lower()
    toks = low.replace(",", " ").split()
    if re.fullmatch(r"(1[0-9]|20)\d\d", a):
        return "year"
    if any(t.strip(".") in _MONTHS for t in toks) and any(ch.isdigit() for ch in a):
        return "date"
    if re.fullmatch(r"[$£€]?\s*[\d.,]+\s*(%|percent|million|billion|thousand)?", low) or (
            toks and all(t in _NUMWORDS or t.replace(".", "").isdigit() for t in toks)):
        return "number"
    words = [w for w in re.findall(r"[A-Za-z][\w'-]*", a)]
    if words and all(w[0].isupper() for w in words if len(w) > 3):
        return "name"
    return "phrase"


def _length_bucket(ans: str) -> int:
    n = len(ans.split())
    return 0 if n <= 1 else (1 if n <= 3 else 2)


def pick_substitute(gold: str, alt_answers: Sequence[str], rng: random.Random,
                    avoid_text: str = "") -> str | None:
    """Alternative answer of the same shape (and, if possible, length) that does
    not already occur in the document."""
    g = normalize_text(gold)
    avoid = avoid_text.lower()
    pool = [a for a in alt_answers
            if normalize_text(a) and normalize_text(a) != g and a.lower() not in avoid]
    if not pool:
        return None
    shape, lb = answer_shape(gold), _length_bucket(gold)
    same = [a for a in pool if answer_shape(a) == shape]
    closest = [a for a in same if _length_bucket(a) == lb] or same
    return rng.choice(closest) if closest else None


def substitute(item: Item, closure: Sequence[int], alt_answers: Sequence[str],
               rng: random.Random) -> Substitution | None:
    """Change the answer-bearing value everywhere it occurs.

    Span answers: replace the gold string with an alternative answer of the
    same template in every closure unit. Categorical answers: only when the
    item carries a domain-specific ``meta['flip']`` (rewritten units and the
    new label), since generic negation rewriting is unreliable.
    """
    if item.answer_type == CATEGORICAL_TYPE:
        flip = item.meta.get("flip")
        if not flip:
            return None
        units = list(item.units)
        for k, txt in flip["units"].items():
            units[int(k)] = txt
        return Substitution(item.with_units(units, item.evidence, perturbation="substitution"),
                            flip["answer"])
    if not closure:
        return None
    alt = pick_substitute(item.answer, alt_answers, rng, avoid_text=" ".join(item.units))
    if alt is None:          # no plausible substitute of the same type: skip, do not force one
        return None
    pat = re.compile(re.escape(item.answer), re.I)
    units = list(item.units)
    changed = False
    for i in closure:
        new = pat.sub(lambda _m: alt, units[i])
        changed |= new != units[i]
        units[i] = new
    # also replace stray mentions outside the closure so no copy survives
    for i, u in enumerate(units):
        if i not in closure and pat.search(u):
            units[i] = pat.sub(lambda _m: alt, u)
    if not changed:
        return None
    return Substitution(item.with_units(units, item.evidence, perturbation="substitution"), alt)
