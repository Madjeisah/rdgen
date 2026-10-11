"""Measurement ablations (protocol step 6).

Each *arm* is one way of measuring record-dependence. All arms are scored on
the same model and the same in-domain test items, so differences between arms
come from the estimator alone. A ``CachingAnswerer`` makes this cheap: every
distinct document is answered once, so the original predictions (and any
perturbed document two arms share) are never recomputed.

Arms (``ARMS``), each answering one reviewer question:

  primary          closure=full, union control correction, deletion on pos+span (pre-registered RD)
  no_control       same, without the control                       "is it just input disruption?"
  control_difference  control removed by plain subtraction         "does the correction model matter?"
  control_xor      control as cancelling binary flips              (alternative correction model)
  closure_top1     delete only the first gold evidence unit        "does redundancy handling matter?"
  closure_gold     delete gold units only (no verifier expansion)  "does the verifier add anything?"
  del_explicit_neg also delete on explicit negatives               "why handle polarity separately?"
  del_all_pol      deletion on every polarity with evidence        (most naive polarity handling)
  substitution     TR: answer follows a changed value              "an independent estimator"
  sufficiency      SUF: evidence + filler keeps the answer         "evidence is enough?"
  verifier_alt     primary with a second verifier in the closure   "sensitivity to the verifier"
  contextcite      ContextCite-style contributive share (no gold)  "do we need gold evidence?"

Each arm yields one model-level score; ``analysis/ablation.py`` then re-runs
the pre-registered prediction analysis with every arm as the predictor.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Sequence

import pandas as pd

from .baselines.contextcite import contextcite_dependence
from .estimators import ProbeConfig, chance_rates, point, probe_model, summarize_probe
from .models.base import Answerer, Prediction
from .schema import Item
from .verify import Verifier


class CachingAnswerer:
    """Wraps an answerer; identical (question, document) pairs are answered once."""

    def __init__(self, inner: Answerer):
        self.inner = inner
        self.name, self.family = inner.name, getattr(inner, "family", "")
        self._pred: dict[str, Prediction] = {}
        self._lp: dict[tuple[str, str], float] = {}
        self.calls = self.hits = 0

    @staticmethod
    def _key(it: Item) -> str:
        h = hashlib.sha1()
        h.update(it.item_id.split("#")[0].encode())
        h.update(it.question.encode())
        h.update("\x1e".join(it.units).encode())
        return h.hexdigest()

    def predict(self, items: Sequence[Item]) -> list[Prediction]:
        keys = [self._key(it) for it in items]
        todo = {}
        for k, it in zip(keys, items):
            if k not in self._pred and k not in todo:
                todo[k] = it
        self.calls += len(items)
        self.hits += len(items) - len(todo)
        if todo:
            for k, p in zip(todo, self.inner.predict(list(todo.values()))):
                self._pred[k] = p
        return [self._pred[k] for k in keys]

    def logprob(self, items: Sequence[Item], answers: Sequence[str]) -> list[float]:
        keys = [(self._key(it), a) for it, a in zip(items, answers)]
        todo = {k: (it, a) for k, it, a in zip(keys, items, answers) if k not in self._lp}
        if todo:
            vals = self.inner.logprob([v[0] for v in todo.values()], [v[1] for v in todo.values()])
            self._lp.update(dict(zip(todo, vals)))
        return [self._lp[k] for k in keys]


@dataclass
class Arm:
    name: str
    metric: str                        # which summary metric is the arm's score
    probe: ProbeConfig | None = None   # None -> derived from another arm's probe
    from_arm: str | None = None
    use_alt_verifier: bool = False
    description: str = ""


def default_arms(base: ProbeConfig | None = None) -> list[Arm]:
    b = base or ProbeConfig()
    return [
        Arm("primary", "RD", b, description="pre-registered RD"),
        Arm("no_control", "RD_raw", from_arm="primary", description="no control correction"),
        Arm("control_difference", "RD_diff", from_arm="primary", description="control by subtraction"),
        Arm("control_xor", "RD_xor", from_arm="primary", description="control as cancelling flips"),
        Arm("substitution", "TR", from_arm="primary", description="substitution estimator"),
        Arm("sufficiency", "SUF", from_arm="primary", description="evidence-only sufficiency"),
        Arm("closure_top1", "RD", replace(b, closure_mode="top1"), description="delete first evidence unit only"),
        Arm("closure_gold", "RD", replace(b, closure_mode="gold"), description="gold evidence only"),
        Arm("del_explicit_neg", "RD", replace(b, deletion_polarities=("pos", "span", "explicit_neg")),
            description="deletion also on explicit negatives"),
        Arm("del_all_pol", "RD", replace(b, deletion_polarities=("pos", "span", "explicit_neg", "unknown_neg"),
                                         include_polarities=("pos", "span", "explicit_neg", "unknown_neg")),
            description="deletion on every polarity with evidence"),
        Arm("verifier_alt", "RD", b, use_alt_verifier=True, description="second verifier in closure"),
        Arm("contextcite", "cc_dependence", description="ContextCite-style, no gold evidence"),
    ]


ARMS = [a.name for a in default_arms()]


@dataclass
class AblationResult:
    scores: dict[str, float]
    detail: dict[str, pd.DataFrame] = field(default_factory=dict)
    cache_hit_rate: float = 0.0


def run_measurement_ablations(model: Answerer, test_items: Sequence[Item], train_items: Sequence[Item],
                              verifier: Verifier | None, alt_verifier: Verifier | None = None,
                              arms: Sequence[Arm] | None = None, n_boot: int = 0,
                              contextcite_items: int = 100, seed: int = 0) -> AblationResult:
    """Score every arm for one model. Returns ``scores[arm] = value``."""
    arms = list(arms or default_arms(ProbeConfig(seed=seed)))
    cm = CachingAnswerer(model)
    chance = chance_rates(train_items) if train_items else None
    summaries: dict[str, pd.DataFrame] = {}
    scores: dict[str, float] = {}
    filler = train_items or test_items
    for arm in arms:
        if arm.name == "contextcite":
            scores[arm.name] = contextcite_dependence(cm, test_items, seed=seed,
                                                      max_items=contextcite_items)["cc_dependence"]
            continue
        if arm.probe is None:
            continue
        if arm.use_alt_verifier and alt_verifier is None:
            continue
        ver = alt_verifier if arm.use_alt_verifier else verifier
        df = probe_model(cm, test_items, arm.probe, ver, filler_items=filler)
        summaries[arm.name] = summarize_probe(df, chance, n_boot=n_boot,
                                              correction=arm.probe.control_correction)
        scores[arm.name] = point(summaries[arm.name], arm.metric)
    for arm in arms:
        if arm.from_arm and arm.from_arm in summaries:
            scores[arm.name] = point(summaries[arm.from_arm], arm.metric)
    rate = cm.hits / cm.calls if cm.calls else 0.0
    return AblationResult(scores, summaries, rate)
