"""Fidelity gates for augmentation data (methodology 10.7).

A candidate training item (synthetic, projected or generated on target-site
text) is kept only if

  1. entailment:  some unit of its document supports its answer
                  (verifier score >= tau_e), and
  2. necessity:   no probe model recovers the answer once the support
                  closure is deleted - i.e. the answer is not guessable
                  from priors.

``gate_mode`` selects the ablation arm: none | entailment | necessity | both.
The verifier and probes passed here must differ from those used to measure
record-dependence in the prediction study (Goodhart guard).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models.base import Answerer
from ..normalize import is_correct
from ..perturb import delete, support_closure
from ..schema import Item
from ..verify import Verifier

GATE_MODES = ("none", "entailment", "necessity", "both")


@dataclass
class GateReport:
    n_in: int
    n_kept: int
    n_fail_entail: int
    n_fail_necessity: int


def apply_gate(pool: Sequence[Item], mode: str, verifier: Verifier | None,
               probes: Sequence[Answerer] = (), tau_e: float = 0.5,
               closure_mode: str = "full") -> tuple[list[Item], GateReport]:
    if mode not in GATE_MODES:
        raise ValueError(mode)
    if mode == "none":
        return list(pool), GateReport(len(pool), len(pool), 0, 0)
    keep = [True] * len(pool)
    fail_e = fail_n = 0
    if mode in ("entailment", "both"):
        if verifier is None:
            raise ValueError("entailment gate needs a verifier")
        for k, it in enumerate(pool):
            if it.polarity in ("unanswerable", "implicit_neg"):
                continue          # nothing to entail; necessity still applies
            scores = verifier.support(it.units, it.question, it.answer)
            if not scores or max(scores) < tau_e:
                keep[k] = False
                fail_e += 1
    if mode in ("necessity", "both"):
        if not probes:
            raise ValueError("necessity gate needs at least one probe model")
        idx = [k for k, ok in enumerate(keep) if ok]
        ablated = []
        for k in idx:
            it = pool[k]
            clo = support_closure(it, closure_mode, verifier)
            ablated.append(delete(it, clo, "gate") if clo else it)
        for probe in probes:
            preds = probe.predict(ablated)
            for k, it, p in zip(idx, ablated, preds):
                if not keep[k]:
                    continue
                recovered = is_correct(p.answer, pool[k].answer, pool[k].answer_type)
                # unanswerable items are *supposed* to be answered not_stated
                if recovered and pool[k].polarity not in ("unanswerable",):
                    keep[k] = False
                    fail_n += 1
    kept = [it for it, ok in zip(pool, keep) if ok]
    return kept, GateReport(len(pool), len(kept), fail_e, fail_n)
