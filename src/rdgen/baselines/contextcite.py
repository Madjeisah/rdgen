"""ContextCite-style contributive dependence (Cohen-Wang et al., NeurIPS 2024).

For each item: draw ``n_ablations`` random masks that keep each unit with
probability 1/2, score the log-probability of the model's original answer
under each masked document, fit a sparse linear surrogate (Lasso) from mask
to score, and read the weights as each unit's contribution.

Model-level baseline score = mean over items of the share of the full-context
log-odds explained by the context (sum of positive weights relative to the
gap between full and empty context), clipped to [0, 1]. Unlike RD it uses no
gold evidence and no deletion control, which is why it is a baseline.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
from sklearn.linear_model import Lasso

from ..models.base import Answerer
from ..schema import Item


def _logit(lp: float) -> float:
    p = min(max(math.exp(lp), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def contextcite_dependence(model: Answerer, items: Sequence[Item], n_ablations: int = 32,
                           alpha: float = 0.01, seed: int = 0, max_items: int = 200) -> dict:
    rng = random.Random(seed)
    items = list(items)[:max_items]
    base = model.predict(items)
    scores = []
    for it, pr in zip(items, base):
        n = len(it.units)
        if n == 0:
            continue
        masks = [[rng.random() < 0.5 for _ in range(n)] for _ in range(n_ablations)]
        variants = [it.with_units([u for u, keep in zip(it.units, m) if keep]) for m in masks]
        full_empty = [it, it.with_units([])]
        lps = model.logprob(variants + full_empty, [pr.answer] * (len(variants) + 2))
        y = np.array([_logit(v) for v in lps[:-2]])
        X = np.array(masks, dtype=float)
        if np.ptp(y) < 1e-9:
            scores.append(0.0)
            continue
        w = Lasso(alpha=alpha, max_iter=5000).fit(X, y).coef_
        gap = _logit(lps[-2]) - _logit(lps[-1])
        share = float(np.clip(w[w > 0].sum() / abs(gap), 0, 1)) if abs(gap) > 1e-6 else 0.0
        scores.append(share)
    return {"cc_dependence": float(np.mean(scores)) if scores else float("nan"), "cc_n": len(scores)}
