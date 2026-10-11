"""A mock answerer with a known mixture of reading and guessing.

For each item the model either *reads* (with probability ``reliance``) or
*guesses from a prior* learned on the source training split. The choice is a
deterministic hash of (model, item), so it is identical for the original and
perturbed versions of the same item - exactly the property the deletion and
substitution estimators rely on. Ground-truth record-dependence is therefore
known, which lets the tests check that the estimators recover it.

Reading follows the document under a closed-world convention: a mentioned
attribute is answered from its sentence; an unmentioned one is answered "no"
(or ``not_stated`` when ``abstain_when_silent``). Span values are copied from
their sentence, or ``not_stated`` when absent.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from ..schema import CATEGORICAL_TYPE, NO, NOT_STATED, TRAIN, YES, Item
from .base import Prediction

_POS = re.compile(r"attribute a(\d+) is present|finding a(\d+): positive", re.I)
_NEG = re.compile(r"no evidence of attribute a(\d+)|finding a(\d+): negative", re.I)
_VAL = re.compile(r"(?:the value of )?v(\d+)\s*(?:is|=)\s*([a-z]+)", re.I)


def _u01(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


@dataclass
class MockAnswerer:
    name: str
    family: str = "mock"
    reliance: float = 0.7            # P(read) per item
    read_noise: float = 0.03         # error rate when reading
    abstain_when_silent: bool = False
    read_conf: float = 0.92
    # Sensitivity to input disruption: the probability of changing the answer
    # when a fraction f of the document was deleted is min(1, disruption * f),
    # regardless of *which* units were deleted. Real LLMs show this; it is the
    # nuisance the random-deletion control subtracts.
    disruption: float = 0.0
    priors: dict = field(default_factory=dict)   # template -> Counter of answers

    def fit_prior(self, items: Sequence[Item]) -> "MockAnswerer":
        counts: dict[str, Counter] = defaultdict(Counter)
        for it in items:
            if it.split == TRAIN:
                counts[it.meta.get("template", it.question)][it.answer] += 1
        self.priors = dict(counts)
        return self

    # -- reading --------------------------------------------------------
    @staticmethod
    def _read(it: Item, abstain: bool) -> str:
        tmpl = it.meta.get("template", "")
        text = " ".join(it.units)
        if it.answer_type == CATEGORICAL_TYPE and tmpl.startswith("A"):
            k = tmpl[1:]
            pos = any((m.group(1) or m.group(2)) == k for m in _POS.finditer(text))
            neg = any((m.group(1) or m.group(2)) == k for m in _NEG.finditer(text))
            if pos and not neg:
                return YES
            if neg and not pos:
                return NO
            if pos and neg:
                return YES
            return NOT_STATED if abstain else NO
        if tmpl.startswith("V"):
            k = tmpl[1:]
            for m in _VAL.finditer(text):
                if m.group(1) == k:
                    return m.group(2).lower()
            return NOT_STATED
        return NOT_STATED

    def _prior(self, it: Item) -> tuple[str, float]:
        c = self.priors.get(it.meta.get("template", it.question))
        if not c:
            return (NO if it.answer_type == CATEGORICAL_TYPE else NOT_STATED), 0.5
        ans, n = c.most_common(1)[0]
        return ans, n / sum(c.values())

    def _disrupt(self, it: Item, ans: str) -> str:
        deleted = it.meta.get("deleted") or []
        if not self.disruption or not deleted:
            return ans
        frac = len(deleted) / (len(it.units) + len(deleted))
        key = f"{len(it.units)}:{','.join(map(str, deleted))}"
        if _u01(self.name, it.item_id, "disrupt", key) < min(1.0, self.disruption * frac):
            # degrade rather than flip: broken input pushes towards an
            # uninformative answer, it does not restore a deleted fact
            return NOT_STATED if ans != NOT_STATED else ("unknown" if it.answer_type != CATEGORICAL_TYPE else NO)
        return ans

    def _decide(self, it: Item) -> tuple[str, float, bool]:
        reads = _u01(self.name, it.item_id, "read") < self.reliance
        if reads:
            ans = self._read(it, self.abstain_when_silent)
            if _u01(self.name, it.item_id, "noise") < self.read_noise:
                ans = {YES: NO, NO: YES}.get(ans, NOT_STATED)
            return self._disrupt(it, ans), self.read_conf, True
        ans, freq = self._prior(it)
        return self._disrupt(it, ans), max(0.5, min(0.99, freq)), False

    def predict(self, items: Sequence[Item]) -> list[Prediction]:
        out = []
        for it in items:
            ans, conf, _ = self._decide(it)
            probs = {}
            if it.answer_type == CATEGORICAL_TYPE:
                rest = (1 - conf) / 2
                probs = {lab: (conf if lab == ans else rest) for lab in (YES, NO, NOT_STATED)}
            out.append(Prediction(ans, conf, probs))
        return out

    def logprob(self, items: Sequence[Item], answers: Sequence[str]) -> list[float]:
        out = []
        for it, a in zip(items, answers):
            ans, conf, _ = self._decide(it)
            p = conf if ans == a else (1 - conf) / 2
            out.append(math.log(max(p, 1e-6)))
        return out
